"""LLM-as-judge utilities for retrieval evaluation traces."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..llm.client import LLMClient


def _extract_retrieved_doc_keys(trace: Dict[str, Any]) -> List[str]:
    items = trace.get("retrieved", [])
    return [str(it.get("doc_key", "")) for it in items if isinstance(it, dict)]


def _parse_json_verdict(content: str) -> Dict[str, Any]:
    """Parse strict JSON verdict from LLM, stripping markdown fences if needed."""
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


def judge_trace_with_openai_compatible(
    trace: Dict[str, Any],
    *,
    client: LLMClient,
    judge_mode: str = "retrieval",
    system_prompt: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
) -> Dict[str, Any]:
    """Judge a single trace and return structured verdict.

    Supports judge_mode: "retrieval", "answer_quality", or "both".
    """
    if judge_mode == "retrieval":
        return _judge_retrieval(trace, system_prompt, user_prompt_template, client)

    if judge_mode == "answer_quality":
        return _judge_answer_quality(trace, system_prompt, user_prompt_template, client)

    if judge_mode == "both":
        r = _judge_retrieval(trace, system_prompt, user_prompt_template, client)
        a = _judge_answer_quality(trace, system_prompt, user_prompt_template, client)
        combined_score = (r["score"] + a["score"]) / 2.0
        return {
            "score": combined_score,
            "verdict": "PASS" if combined_score >= 0.5 else "FAIL",
            "reason": f"retrieval: {r['reason']} | answer: {a['reason']}",
            "retrieval": r,
            "answer_quality": a,
        }

    raise ValueError(f"Unknown judge_mode: {judge_mode!r}")


def _judge_retrieval(
    trace: Dict[str, Any],
    system_prompt: Optional[str],
    user_prompt_template: Optional[str],
    client: LLMClient,
) -> Dict[str, Any]:
    relevant = trace.get("relevant", {})
    retrieved = _extract_retrieved_doc_keys(trace)

    sys_p = system_prompt or "Respond with strict JSON only."
    if user_prompt_template:
        user_p = user_prompt_template.format(
            question=trace.get("question", ""),
            expected=json.dumps(relevant, ensure_ascii=False),
            retrieved=json.dumps(retrieved, ensure_ascii=False),
        )
    else:
        user_p = (
            "You are evaluating retrieval quality for a medical QA benchmark. "
            "Given expected relevant doc ids and retrieved doc ids, score relevance.\n"
            "Return strict JSON with fields: score (0..1), verdict (PASS|FAIL), reason (short).\n"
            f"Expected relevant: {json.dumps(relevant, ensure_ascii=False)}\n"
            f"Retrieved: {json.dumps(retrieved, ensure_ascii=False)}"
        )

    content = client.call([{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}])
    verdict = _parse_json_verdict(content)
    if "score" not in verdict or "verdict" not in verdict:
        raise RuntimeError(f"LLM judge JSON missing required fields: {verdict}")
    return {
        "score": float(verdict["score"]),
        "verdict": str(verdict["verdict"]),
        "reason": str(verdict.get("reason", "")),
    }


def _judge_answer_quality(
    trace: Dict[str, Any],
    system_prompt: Optional[str],
    user_prompt_template: Optional[str],
    client: LLMClient,
) -> Dict[str, Any]:
    from .prompts import ANSWER_QUALITY_SYSTEM_PROMPT, ANSWER_QUALITY_USER_TEMPLATE

    generated = trace.get("generated_answer", "")
    reference = trace.get("reference_answer", "")

    if not generated:
        return {"score": 0.0, "verdict": "FAIL", "reason": "no generated answer"}
    if not reference:
        return {"score": 0.0, "verdict": "FAIL", "reason": "no reference answer"}

    sys_p = system_prompt or ANSWER_QUALITY_SYSTEM_PROMPT
    user_p = (user_prompt_template or ANSWER_QUALITY_USER_TEMPLATE).format(
        question=trace.get("question", ""),
        reference_answer=reference,
        generated_answer=generated,
    )

    content = client.call([{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}])
    verdict = _parse_json_verdict(content)
    if "score" not in verdict or "verdict" not in verdict:
        raise RuntimeError(f"LLM judge JSON missing required fields: {verdict}")
    return {
        "score": float(verdict["score"]),
        "verdict": str(verdict["verdict"]),
        "reason": str(verdict.get("reason", "")),
    }


def run_llm_judging(
    traces: List[Dict[str, Any]],
    judge_config,
    *,
    judge_mode: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
) -> Dict[str, Any]:
    """Run LLM-as-judge on top traces. Raises on any critical configuration/runtime error."""
    import logging as _logging
    from tqdm import tqdm

    _mode = judge_mode or getattr(judge_config, "judge_mode", "retrieval")
    max_cases = getattr(judge_config, "max_cases", 0)
    if max_cases < 0:
        raise RuntimeError("LLM judge max_cases must be >= 0 when judge is enabled")

    client = LLMClient(judge_config.to_llm_config())

    _sys_p = system_prompt or getattr(judge_config, "system_prompt", None)
    _user_t = user_prompt_template or getattr(judge_config, "user_prompt_template", None)

    selected = traces if max_cases == 0 else traces[:max_cases]
    outputs: List[Dict[str, Any]] = []
    for trace in tqdm(selected, desc="Judging", unit="case"):
        verdict = judge_trace_with_openai_compatible(
            trace,
            client=client,
            judge_mode=_mode,
            system_prompt=_sys_p,
            user_prompt_template=_user_t,
        )
        outputs.append({"query_id": trace.get("query_id"), "judge": verdict})

    scores: List[float] = []
    for row in outputs:
        judge_data = row.get("judge")
        if isinstance(judge_data, dict) and "score" in judge_data:
            scores.append(float(judge_data["score"]))
    return {
        "model": judge_config.model,
        "api_base": judge_config.get_api_base(),
        "judge_mode": _mode,
        "cases": len(outputs),
        "mean_score": (sum(scores) / len(scores)) if scores else 0.0,
        "details": outputs,
    }


def judge_multi_aspect(
    query: str,
    retrieved_text: str,
    aspects: List[str],
    *,
    client: LLMClient,
    chain_of_thought: bool = False,
    score_aggregation: str = "average",
    aspect_weights: Dict[str, float] = None,
) -> Dict[str, Any]:
    """Judge a retrieved document across multiple aspects."""
    import re
    from .prompts import get_multi_aspect_prompt

    system_prompt, user_prompt = get_multi_aspect_prompt(
        query, retrieved_text, aspects, chain_of_thought
    )
    response_text = client.call(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    )

    aspect_scores = {}
    for aspect in aspects:
        match = re.search(rf"{aspect}.*?(\d+)", response_text, re.IGNORECASE)
        aspect_scores[aspect] = float(match.group(1)) if match else 3.0

    if score_aggregation == "weighted" and aspect_weights:
        overall = sum(
            aspect_scores.get(a, 0) * aspect_weights.get(a, 0) for a in aspects
        )
    elif score_aggregation == "min":
        overall = min(aspect_scores.values())
    elif score_aggregation == "max":
        overall = max(aspect_scores.values())
    else:
        overall = sum(aspect_scores.values()) / len(aspect_scores)

    return {
        "aspect_scores": aspect_scores,
        "overall_score": overall,
        "raw_response": response_text if chain_of_thought else None,
    }


def judge_single_aspect(
    query: str,
    retrieved_text: str,
    aspect: str = "relevance",
    *,
    client: LLMClient,
    chain_of_thought: bool = False,
    few_shot_examples: List[Dict] = None,
) -> Dict[str, Any]:
    """Judge a retrieved document on a single aspect."""
    import re
    from .prompts import get_judge_prompt

    system_prompt, user_prompt = get_judge_prompt(
        query, retrieved_text, aspect, chain_of_thought, few_shot_examples
    )
    response_text = client.call(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    ).strip()

    score_match = re.search(r"(\d+)", response_text)
    score = float(score_match.group(1)) if score_match else 3.0

    return {
        "score": score,
        "aspect": aspect,
        "reasoning": response_text if chain_of_thought else None,
    }
