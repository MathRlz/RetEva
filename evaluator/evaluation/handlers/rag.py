"""RAG stage handlers: answer generation + finalize (LLM judge, traces, latency).

Each handler
registers itself via ``@register_stage_handler`` at import time.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger
from ...judge import run_llm_judging
from ...metrics import per_speaker_breakdown, judge_calibration
from ..helpers import _payload_to_key
from ..answer_gen import generate_answers
from ..executor.state import RunState
from .retrieval import _retrieved_from_bus
from ._common import is_asr_text_retrieval, retrieval_ran

logger = get_logger(__name__)


def _answer_corpus_lookup(s: "RunState", results_with_scores) -> dict:
    """doc_id → payload, from the full corpus + the retrieved payloads (richer metadata).
    Used to find reference answers even for retrieval misses. The corpus base is built once
    per run (cached on the state — get_corpus rebuilds N dicts each call); the retrieved
    overlay is rebuilt per call (it is branch-specific)."""
    base = s.corpus_lookup_base
    if base is None:
        base = {}
        if hasattr(s.dataset, "get_corpus"):
            for doc in s.dataset.get_corpus():
                base[str(doc.get("doc_id", ""))] = doc
        s.corpus_lookup_base = base
    corpus_lookup = dict(base)
    for idx, result in enumerate(results_with_scores):
        for payload, _ in result:
            corpus_lookup[str(payload.get("doc_id", payload.get("id", idx)))] = payload
    return corpus_lookup


def _generate_answer_details(
    s: "RunState", results, all_relevant, results_with_scores, query_ids
) -> dict:
    """Answer-GENERATION node: generate answers (if enabled); return query_id → detail map
    (the SAME detail dicts stored in ``results['answer_generation']``, so the answer_metrics
    node can enrich them in place). Scoring is the answer_metrics node's job."""
    cfg = s.answer_gen_config
    if cfg is None or not getattr(cfg, "enabled", False):
        return {}

    s.cb("phase_5_answer_gen", 0, s.total, "Phase 5: Answer generation")
    # Bus-only (M1d-2): the effective (most-processed) query in ASR modes (QUERY_TEXT_CHAIN).
    query_texts = (
        s.input("query_text", default=[])
        if is_asr_text_retrieval(s)
        else s.get_artifact("reference_transcription", default=[])
    )
    answer_results = generate_answers(
        traces_data=(query_ids, all_relevant, results_with_scores),
        all_query_texts=query_texts,
        corpus_lookup=_answer_corpus_lookup(s, results_with_scores),
        config=cfg,
    )
    results["answer_generation"] = answer_results
    logger.info("Answer generation complete — %d cases", answer_results["cases"])
    return {d["query_id"]: d for d in answer_results["details"]}


def _build_query_traces(
    s: "RunState",
    results,
    all_relevant,
    wer_scores,
    cer_scores,
    per_query_recall5,
    ans_detail_by_qid,
    results_with_scores,
    query_ids,
) -> None:
    """Build per-query traces (+ per-speaker breakdown) when trace_limit > 0."""
    if not (s.trace_limit > 0 and results_with_scores):
        return

    limit = min(s.trace_limit, len(results_with_scores))
    references = s.get_artifact(
        "reference_transcription", default=[]
    )  # bus-only (M1d-2)
    from ...utils.progress import progress_iter

    traces = []
    for i in progress_iter(range(limit), "Building query traces", total=limit, unit="query"):
        retrieved = [
            {
                "doc_key": _payload_to_key(payload),
                "score": float(score),
                # doc text for the LLM judge (falls back to the key when absent)
                "text": (
                    payload.get("text", payload.get("content", ""))
                    if isinstance(payload, dict) else ""
                ),
            }
            for payload, score in results_with_scores[i]
        ]
        query_id = query_ids[i] if i < len(query_ids) else str(i)
        ans_detail = ans_detail_by_qid.get(query_id, {})
        # Read metadata WITHOUT decoding audio (`s.dataset[i]` loads the waveform + requires
        # an audio_path — which the oracle branch has none of). The question objects carry it.
        questions = getattr(s.dataset, "questions", None)
        if questions is not None and i < len(questions):
            sample_meta = getattr(questions[i], "metadata", {}) or {}
        elif questions is None and i < len(s.dataset):
            sample_meta = s.dataset[i].get("metadata", {})
        else:
            sample_meta = {}
        trace_entry: Dict[str, Any] = {
            "query_id": query_id,
            "relevant": all_relevant[i] if i < len(all_relevant) else {},
            "retrieved": retrieved,
            "question": (
                references[i]
                if i < len(references)
                else ans_detail.get("question", "")
            ),
            "generated_answer": ans_detail.get("generated_answer", ""),
            "reference_answer": ans_detail.get("reference_answer", ""),
            "metadata": sample_meta,
        }
        if i < len(wer_scores):
            trace_entry["per_query_wer"] = wer_scores[i]
            trace_entry["per_query_cer"] = cer_scores[i]
        if i < len(per_query_recall5):
            trace_entry["recall_at_5"] = per_query_recall5[i]
        traces.append(trace_entry)
    results["query_traces"] = traces

    per_speaker = per_speaker_breakdown(traces)
    if per_speaker is not None:
        results["per_speaker"] = per_speaker


def _run_judge(
    s: "RunState", results, all_relevant, per_query_recall5, retrieved_keys
) -> None:
    """Judge node: run the LLM judge over query traces (if enabled) + calibration."""
    cfg = s.judge_config
    if cfg is None or not getattr(cfg, "enabled", False):
        return
    if "query_traces" not in results:
        raise RuntimeError("LLM judge requires query traces; set trace_limit > 0")

    s.cb("phase_6_judge", 0, s.total, "Phase 6: LLM judge")
    judge_mode = getattr(cfg, "judge_mode", "retrieval")
    logger.info(
        "Running LLM judge — mode=%s model=%s cases=%d",
        judge_mode,
        getattr(cfg, "model", "unknown"),
        getattr(cfg, "max_cases", -1),
    )
    judge_results = run_llm_judging(results["query_traces"], cfg)
    results["llm_judge"] = judge_results

    # Judge calibration: correlate per-query judge overall score with IR metrics.
    judge_scores = [
        d.get("judge", {}).get("overall", float("nan"))
        for d in judge_results.get("details", [])
    ]
    calibration = judge_calibration(
        judge_scores, per_query_recall5, retrieved_keys, all_relevant
    )
    if calibration:
        results.update(calibration)
        logger.info(
            "Judge calibration — vs_MRR=%.3f vs_Recall5=%.3f",
            results.get("judge_vs_MRR_correlation", float("nan")),
            results.get("judge_vs_Recall5_correlation", float("nan")),
        )

    logger.info(
        "Judge complete — cases=%d mean_score=%.4f pass_rate=%.1f%%",
        judge_results["cases"],
        judge_results["mean_score"],
        100.0
        * sum(
            1
            for r in judge_results["details"]
            if r.get("judge", {}).get("verdict") == "PASS"
        )
        / max(len(judge_results["details"]), 1),
    )


@register_stage_handler("generate", self_timed=True)
def _stage_generate(s: RunState) -> None:
    """Answer-generation node: RAG answer generation only (retrieval modes). Writes
    ``answer_generation`` into ``s.results`` + the per-query detail map onto
    ``s.ans_detail_by_qid`` (read by the build_query_traces node). Conditional on
    ``answer_generation.enabled``."""
    if not retrieval_ran(s):
        return
    results_with_scores, _, query_ids = _retrieved_from_bus(s)
    s.ans_detail_by_qid = _generate_answer_details(
        s, s.results, s.metrics_all_relevant, results_with_scores, query_ids
    )
    s.put_artifact("generated_answers", {"generated": "answer_generation" in s.results})


def _stage_answer_metrics(s: RunState) -> None:
    """Answer-quality comparison node: score the generated answers vs their reference
    answers + retrieved context (ROUGE / hallucination / dose-safety / context-recall),
    enriching the answer_generation details in place + the mean_* aggregates the report
    reads. Was bundled inside answer generation."""
    from ..answer_gen import score_answers

    answer_results = s.results.get("answer_generation")
    if not isinstance(answer_results, dict) or not answer_results.get("details"):
        return
    results_with_scores, _, query_ids = _retrieved_from_bus(s)
    score_answers(
        answer_results,
        traces_data=(query_ids, s.metrics_all_relevant, results_with_scores),
        corpus_lookup=_answer_corpus_lookup(s, results_with_scores),
        config=s.answer_gen_config,
    )
    s.put_artifact("answer_scores", {"mean_rougeL": answer_results.get("mean_rougeL")})
    logger.info(
        "Answer metrics — mean ROUGE-L: %s",
        (
            f"{answer_results['mean_rougeL']:.4f}"
            if answer_results.get("mean_rougeL") is not None
            else "n/a"
        ),
    )


def _stage_build_query_traces(s: RunState) -> None:
    """Explicit trace builder: assemble the per-query traces from the retrieved docs, the
    generated answers (``s.ans_detail_by_qid``, empty when no answer_gen ran) and the
    per-item WER/recall. Always present in retrieval modes when tracing is on — so the
    judge + report read the traces without the old ``traces_built`` state machine."""
    if not retrieval_ran(s):
        return
    # In a branched run this node is expanded per branch; the report carries a single
    # top-level `query_traces`, so the first branch (topological order) builds it and the
    # rest are no-ops — matching the single-trace semantics of the former shared flag.
    if "query_traces" in s.results:
        return
    results_with_scores, _, query_ids = _retrieved_from_bus(s)
    _build_query_traces(
        s,
        s.results,
        s.metrics_all_relevant,
        s.wer_scores,
        s.cer_scores,
        s.per_query_recall5,
        s.ans_detail_by_qid,
        results_with_scores,
        query_ids,
    )
    s.put_artifact("query_traces", {"built": "query_traces" in s.results})


def _stage_answer_judge(s: RunState) -> None:
    """LLM-judge comparison node: scores the query traces (built upstream) vs the judge
    rubric + calibrates against IR metrics. Present only when the judge is enabled."""
    if not retrieval_ran(s):
        return
    _, retrieved_keys, _ids = _retrieved_from_bus(s)
    _run_judge(s, s.results, s.metrics_all_relevant, s.per_query_recall5, retrieved_keys)
    details = (s.results.get("llm_judge") or {}).get("details") or []
    _publish_judge_scores(s, details)


def _publish_judge_scores(s: RunState, details: list) -> None:
    """Publish the per-query judge outputs as keyed ItemSets so the (reference-free) judge
    metrics score them through the normal path: ``judge_scores`` (overall), ``judge_pass``
    (1.0/0.0 → judge_pass_rate), and ``judge_aspect_<a>`` for each configured aspect. The
    finalize node folds these into the report via ``attach_judge_metrics`` (J3)."""
    from ..item_set import ItemSet

    ids = [str(d["query_id"]) for d in details]
    s.put_items("judge_scores", ItemSet(ids, [d["judge"]["overall"] for d in details]))
    s.put_items(
        "judge_pass",
        ItemSet(ids, [1.0 if d["judge"]["verdict"] == "PASS" else 0.0 for d in details]),
    )
    for aspect in s.judge_config.judge_aspects:
        pairs = [
            (str(d["query_id"]), d["judge"]["aspect_scores"][aspect])
            for d in details if aspect in d["judge"]["aspect_scores"]
        ]
        if pairs:
            s.put_items(
                f"judge_aspect_{aspect}",
                ItemSet([p[0] for p in pairs], [p[1] for p in pairs]),
            )


def _stage_finalize(s: RunState) -> None:
    """Terminal node: latency summary + trace/report close-out. Trace building moved to the
    explicit build_query_traces node; the LLM judge to answer_judge."""
    s.stage_times["total_s"] = time.perf_counter() - s.t_total
    s.results["latency"] = s.stage_times
    # Judge metrics (J3): the judge node is downstream of the report assembler, so its per-query
    # scores are merged into the report here, at the terminal node, via the metric registry.
    from .metrics import attach_judge_metrics

    attach_judge_metrics(s)
    _attach_traces_to_report(s)
    logger.info(
        "Stage latency — asr=%.1fs embed=%.1fs retrieve=%.1fs total=%.1fs",
        s.stage_times.get("asr_s", 0),
        s.stage_times.get("embedding_s", 0),
        s.stage_times.get("retrieval_s", 0),
        s.stage_times["total_s"],
    )


# Per-query trace / answer-gen keys producers write top-level (working state during the run);
# consolidated into ``report['traces']`` and dropped from the output (the report is the single
# source — see drop_mirrored_top_level_keys).
_TRACE_KEYS = ("query_traces", "retrieval_failure_analysis", "answer_generation")
# LLM-judge outputs (the judge runs in finalize) → consolidated into ``report['judge']``.
_JUDGE_KEYS = (
    "llm_judge",
    "judge_vs_MRR_correlation",
    "judge_vs_Recall5_correlation",
)


def _attach_traces_to_report(s: "RunState") -> None:
    """Consolidate per-query traces + failure analysis + answer-gen details into
    ``report['traces']`` and the LLM-judge results + calibration into ``report['judge']``. The
    terminal node owns this, so every producer (metrics → answer_gen → judge) has written its
    part; the top-level copies are then dropped at the output boundary."""
    report = s.results.get("report")
    if not isinstance(report, dict):
        return
    traces = {k: s.results[k] for k in _TRACE_KEYS if k in s.results}
    if traces:
        report["traces"] = traces
    judge = {k: s.results[k] for k in _JUDGE_KEYS if k in s.results}
    if judge:
        report["judge"] = judge


def drop_mirrored_top_level_keys(results: dict) -> None:
    """Output-boundary G5 cutover: once traces/judge are in ``report['traces']``/``['judge']``,
    drop the duplicate top-level keys so the report is the single source in the returned result.
    Only drops keys actually mirrored (no data loss if the report wasn't assembled). Run after
    the whole graph (sinks read the top-level working copies during the run)."""
    report = results.get("report")
    if not isinstance(report, dict):
        return
    mirrored = set(report.get("traces", {})) | set(report.get("judge", {}))
    for key in (*_TRACE_KEYS, *_JUDGE_KEYS):
        if key in mirrored:
            results.pop(key, None)
