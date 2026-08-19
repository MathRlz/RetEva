"""LLM-as-judge: multi-aspect scoring of retrieval + answer quality per query trace."""

from __future__ import annotations

import json
import threading
import logging
from typing import Any, Dict, List, Optional

from ..llm_client.client import LLMClient

logger = logging.getLogger("evaluator.judge")

# Per-aspect rubric lines injected into the prompt (only the configured aspects are asked).
ASPECT_DESCRIPTIONS = {
    "relevance": "Are the retrieved documents relevant to the question?",
    "faithfulness": "Is the answer supported by (grounded in) the retrieved documents?",
    "correctness": "Is the answer factually correct?",
    "completeness": "Does the answer fully address the question?",
    "clarity": "Is the answer clear and well-structured?",
    "accuracy": "Is the stated information accurate?",
    "factuality": "Are the stated facts true?",
}


def _parse_json_verdict(content: str) -> Dict[str, Any]:
    """Parse a strict-JSON verdict from the LLM, stripping markdown fences / prose if needed."""
    try:
        return json.loads(content)
    except Exception:
        import re

        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.MULTILINE)
        m = re.search(r"\{.*\}", stripped, re.DOTALL)
        if m:
            stripped = m.group(0)
        try:
            return json.loads(stripped)
        except Exception as exc2:
            raise RuntimeError(
                f"LLM judge returned non-JSON. Expected strict JSON, got: {content[:500]}"
            ) from exc2


def _retrieved_block(trace: Dict[str, Any], *, include_doc_text: bool, top_k: int) -> str:
    """Render the top-k retrieved docs for the prompt — their text when available + requested,
    else their ids (the weak fallback)."""
    items = (trace.get("retrieved") or [])[:top_k]
    lines = []
    for rank, it in enumerate(items, start=1):
        if not isinstance(it, dict):
            continue
        key = str(it.get("doc_key", ""))
        text = it.get("text") if include_doc_text else None
        lines.append(f"  {rank}. [{key}] {text}" if text else f"  {rank}. {key}")
    return "\n".join(lines) if lines else "  (none retrieved)"


def build_judge_prompt(
    trace: Dict[str, Any],
    *,
    aspects: List[str],
    judge_mode: str,
    reference_mode: str,
    include_doc_text: bool,
    top_k: int,
    system_prompt: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
) -> tuple[str, str]:
    """(system, user) prompt for a multi-aspect judgment of one trace. ``user_prompt_template``,
    when given, is ``.format(...)``-ed with question/retrieved/answer/reference/aspects."""
    question = trace.get("question", "")
    retrieved = _retrieved_block(trace, include_doc_text=include_doc_text, top_k=top_k)
    answer = trace.get("generated_answer", "")
    reference = trace.get("reference_answer", "")
    aspect_lines = "\n".join(f"  - {a}: {ASPECT_DESCRIPTIONS.get(a, a)}" for a in aspects)

    sys_p = system_prompt or (
        "You are an expert evaluator for a medical question-answering retrieval system. "
        "Score each requested aspect from 0.0 (worst) to 1.0 (best). Be strict and consistent. "
        "Respond with STRICT JSON only — no prose, no markdown."
    )

    if user_prompt_template:
        user_p = user_prompt_template.format(
            question=question, retrieved=retrieved, answer=answer, reference=reference,
            aspects=", ".join(aspects),
        )
    else:
        parts = [f"Question:\n  {question}", f"\nRetrieved documents:\n{retrieved}"]
        if judge_mode in ("answer_quality", "both"):
            parts.append(f"\nGenerated answer:\n  {answer or '(none)'}")
        if reference_mode == "graded" and reference:
            parts.append(f"\nReference answer (grade against this):\n  {reference}")
        parts.append(f"\nScore each aspect from 0.0 to 1.0:\n{aspect_lines}")
        parts.append(
            '\nReturn STRICT JSON: {"aspect_scores": {<aspect>: <0..1>, ...}, '
            '"overall": <0..1>, "verdict": "PASS"|"FAIL", "reason": "<short>"}'
        )
        user_p = "\n".join(parts)
    return sys_p, user_p


def _mean(xs: List[float]) -> float:
    """Mean of ``xs``, 0.0 for empty (the judge reports 0 when no case scored an aspect)."""
    return sum(xs) / len(xs) if xs else 0.0


def _aggregate(aspect_scores: Dict[str, float], config) -> float:
    """Combine per-aspect scores into an overall score per ``score_aggregation``."""
    vals = list(aspect_scores.values())
    if not vals:
        return 0.0
    agg = getattr(config, "score_aggregation", "average")
    if agg == "min":
        return min(vals)
    if agg == "max":
        return max(vals)
    if agg == "weighted" and getattr(config, "aspect_weights", None):
        w = config.aspect_weights
        return sum(aspect_scores.get(a, 0.0) * w.get(a, 0.0) for a in aspect_scores)
    return sum(vals) / len(vals)


#: Appended to the system prompt on a re-ask. The observed parse failure is a reply that ran out
#: of context mid-object (`{"aspect_scores": {`), so the fix is a shorter one, not a longer wait.
_TERSE_RETRY = (
    "\nReply with JSON ONLY — no prose, no markdown. Keep \"reason\" under 20 words."
)


def judge_trace(trace: Dict[str, Any], *, client: LLMClient, config) -> Dict[str, Any]:
    """Judge one trace across ``config.judge_aspects`` in a single LLM call. Returns
    ``{overall, aspect_scores, verdict, reason}`` (scores in [0, 1]).

    An unparsable reply is re-asked ONCE with a terser instruction: a judged case is expensive,
    and losing it to a cut-off `reason` silently shrinks the judged set."""
    aspects = list(config.judge_aspects)
    sys_p, user_p = build_judge_prompt(
        trace, aspects=aspects, judge_mode=config.judge_mode,
        reference_mode=config.reference_mode, include_doc_text=config.include_doc_text,
        top_k=config.judge_top_k, system_prompt=config.system_prompt,
        user_prompt_template=config.user_prompt_template,
    )
    content = client.call(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        use_cache=True,
    )
    try:
        verdict = _parse_json_verdict(content)
    except RuntimeError:
        logger.warning(
            "judge reply for query_id=%s did not parse — re-asking for terse JSON",
            trace.get("query_id"),
        )
        # use_cache=False: never re-serve the reply that just failed to parse.
        content = client.call(
            [{"role": "system", "content": sys_p + _TERSE_RETRY},
             {"role": "user", "content": user_p}],
            use_cache=False,
        )
        verdict = _parse_json_verdict(content)
    raw = verdict.get("aspect_scores") or {}
    aspect_scores = {a: float(raw[a]) for a in aspects if a in raw}
    overall = (
        _aggregate(aspect_scores, config) if aspect_scores
        else float(verdict.get("overall", 0.0))
    )
    threshold = getattr(config, "pass_threshold", 0.5)
    pass_default = "PASS" if overall >= threshold else "FAIL"
    return {
        "overall": overall,
        "aspect_scores": aspect_scores,
        "verdict": str(verdict.get("verdict", pass_default)).upper(),
        "reason": str(verdict.get("reason", "")),
    }


def run_llm_judging(traces: List[Dict[str, Any]], judge_config) -> Dict[str, Any]:
    """Run the multi-aspect LLM judge over the (first ``max_cases``) traces. One failed trace is
    dropped-and-logged so the stage never aborts. Returns the aggregate report incl. per-case
    ``details`` ([{query_id, judge:{overall, aspect_scores, verdict, reason}}])."""
    from ..llm_client.parallel import map_completions

    max_cases = getattr(judge_config, "max_cases", 0)
    if max_cases < 0:
        raise RuntimeError("LLM judge max_cases must be >= 0 when judge is enabled")
    aspects = list(judge_config.judge_aspects)
    client = LLMClient(judge_config.to_llm_config(), component="judge")

    selected = traces if max_cases == 0 else traces[:max_cases]

    failures: Dict[str, int] = {}
    failures_lock = threading.Lock()

    def _one(trace: Dict[str, Any]):
        try:
            verdict = judge_trace(trace, client=client, config=judge_config)
        except Exception as exc:  # noqa: BLE001 — drop-and-log one bad case, keep the rest (M3)
            logger.warning(
                "judge failed for query_id=%s (%s: %s); skipping this case",
                trace.get("query_id"), type(exc).__name__, exc,
            )
            with failures_lock:
                kind = type(exc).__name__
                failures[kind] = failures.get(kind, 0) + 1
            return None
        return {"query_id": trace.get("query_id"), "judge": verdict}

    # Ordered + error-isolated: a dropped case still just disappears from `details`.
    judged = map_completions(
        selected, _one,
        workers=getattr(judge_config, "concurrency", 1),
        desc="Judging", unit="case",
    )
    details: List[Dict[str, Any]] = [d for d in judged if d is not None]
    failed = len(selected) - len(details)
    if failed:
        # `cases` counts successes, so without this line a shortfall is invisible in the report
        # and a single warning per case is lost in a long log.
        logger.warning(
            "judged %d/%d traces — %d failed (%s)", len(details), len(selected), failed,
            ", ".join(f"{k} {v}" for k, v in sorted(failures.items())) or "unknown",
        )
    else:
        logger.info("judged %d/%d traces (0 failed)", len(details), len(selected))

    overalls = [d["judge"]["overall"] for d in details]
    aspect_means = {
        a: _mean([d["judge"]["aspect_scores"][a]
                  for d in details if a in d["judge"]["aspect_scores"]])
        for a in aspects
    }
    return {
        "model": judge_config.model,
        "api_base": judge_config.get_api_base(),
        "judge_mode": judge_config.judge_mode,
        "aspects": aspects,
        "cases": len(details),
        # cases + failed_cases == attempted: a judged set smaller than the trace set is a fact
        # the report has to carry, not something to infer from the log.
        "attempted": len(selected),
        "failed_cases": failed,
        "failures": dict(sorted(failures.items())),
        "mean_score": _mean(overalls),
        "aspect_means": aspect_means,
        "pass_rate": (
            sum(1 for d in details if d["judge"]["verdict"] == "PASS") / len(details)
            if details else 0.0
        ),
        "details": details,
    }
