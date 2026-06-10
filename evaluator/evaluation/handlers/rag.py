"""RAG stage handlers: answer generation + finalize (LLM judge, traces, latency).

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X7). Each handler
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
from .retrieval import _rehydrate_retrieved

logger = get_logger(__name__)


def _generate_answer_details(s: "RunState", results, all_relevant) -> dict:
    """Answer-generation node: run answer generation (if enabled); return query_id → detail map."""
    cfg = s.answer_gen_config
    if cfg is None or not getattr(cfg, "enabled", False):
        return {}

    s.cb("phase_5_answer_gen", 0, s.total, "Phase 5: Answer generation")
    query_texts = (
        s.all_hypotheses if s.mode == "asr_text_retrieval" else s.all_ground_truth
    )

    # Build corpus_lookup from the full corpus so reference answers can be found
    # even for retrieval misses (not just retrieved docs).
    corpus_lookup: dict = {}
    if hasattr(s.dataset, "get_corpus"):
        for doc in s.dataset.get_corpus():
            corpus_lookup[str(doc.get("doc_id", ""))] = doc
    # Overlay with retrieved payloads (may carry richer runtime metadata).
    for idx, result in enumerate(s.all_results_with_scores):
        for payload, _ in result:
            corpus_lookup[str(payload.get("doc_id", payload.get("id", idx)))] = payload

    answer_results = generate_answers(
        traces_data=(s.all_query_ids, all_relevant, s.all_results_with_scores),
        all_query_texts=query_texts,
        corpus_lookup=corpus_lookup,
        config=cfg,
    )
    results["answer_generation"] = answer_results
    logger.info(
        "Answer generation complete — %d cases, mean ROUGE-L: %s",
        answer_results["cases"],
        (
            f"{answer_results['mean_rougeL']:.4f}"
            if answer_results["mean_rougeL"] is not None
            else "n/a"
        ),
    )
    return {d["query_id"]: d for d in answer_results["details"]}


def _build_query_traces(
    s: "RunState",
    results,
    all_relevant,
    wer_scores,
    cer_scores,
    per_query_recall5,
    ans_detail_by_qid,
) -> None:
    """Build per-query traces (+ per-speaker breakdown) when trace_limit > 0."""
    if not (s.trace_limit > 0 and s.all_results_with_scores):
        return

    limit = min(s.trace_limit, len(s.all_results_with_scores))
    traces = []
    for i in range(limit):
        retrieved = [
            {"doc_key": _payload_to_key(payload), "score": float(score)}
            for payload, score in s.all_results_with_scores[i]
        ]
        query_id = s.all_query_ids[i] if i < len(s.all_query_ids) else str(i)
        ans_detail = ans_detail_by_qid.get(query_id, {})
        sample_meta = s.dataset[i].get("metadata", {}) if i < len(s.dataset) else {}
        trace_entry: Dict[str, Any] = {
            "query_id": query_id,
            "relevant": all_relevant[i] if i < len(all_relevant) else {},
            "retrieved": retrieved,
            "question": (
                s.all_ground_truth[i]
                if i < len(s.all_ground_truth)
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


def _run_judge(s: "RunState", results, all_relevant, per_query_recall5) -> None:
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
    judge_results = run_llm_judging(results["query_traces"], cfg, judge_mode=judge_mode)
    results["llm_judge"] = judge_results

    # Judge calibration: correlate per-query judge score with IR metrics.
    judge_scores = [
        d.get("judge", {}).get("score", float("nan"))
        for d in judge_results.get("details", [])
    ]
    calibration = judge_calibration(
        judge_scores, per_query_recall5, s.all_retrieved, all_relevant
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


@register_stage_handler("answer_gen", self_timed=True)
def _stage_answer_gen(s: RunState) -> None:
    """Answer-generation node: RAG answer generation + per-query traces (retrieval modes only).

    Reads the IR intermediates the metrics stage stored on the state; writes
    ``answer_generation`` and ``query_traces`` into ``s.results``.
    """
    if s.mode == "asr_only":
        return
    _rehydrate_retrieved(s)  # the retrieved artifact from the bound producer (R4c)
    _t = time.perf_counter()
    s.ans_detail_by_qid = _generate_answer_details(s, s.results, s.metrics_all_relevant)
    _build_query_traces(
        s,
        s.results,
        s.metrics_all_relevant,
        s.wer_scores,
        s.cer_scores,
        s.per_query_recall5,
        s.ans_detail_by_qid,
    )
    s.stage_times["answer_gen_s"] = s.stage_times.get("answer_gen_s", 0.0) + (
        time.perf_counter() - _t
    )


@register_stage_handler("finalize", self_timed=True)
def _stage_finalize(s: RunState) -> None:
    """Judge node: LLM judge (retrieval modes) + latency summary. Terminal node."""
    if s.mode != "asr_only":
        _t = time.perf_counter()
        _run_judge(s, s.results, s.metrics_all_relevant, s.per_query_recall5)
        s.stage_times["judge_s"] = s.stage_times.get("judge_s", 0.0) + (
            time.perf_counter() - _t
        )
    s.stage_times["total_s"] = time.perf_counter() - s.t_total
    s.results["latency"] = s.stage_times
    _attach_traces_to_report(s)
    logger.info(
        "Stage latency — asr=%.1fs embed=%.1fs retrieve=%.1fs total=%.1fs",
        s.stage_times.get("asr_s", 0),
        s.stage_times.get("embedding_s", 0),
        s.stage_times.get("retrieval_s", 0),
        s.stage_times["total_s"],
    )


# Trace keys the legacy path writes top-level that the results page consumes; G2 mirrors them
# into the report so the UI (G4) can read them from one place ahead of the legacy-path deletion.
_TRACE_KEYS = ("query_traces", "retrieval_failure_analysis", "answer_generation")
# LLM-judge outputs (the judge runs in finalize before this attach); G3 mirrors them into the
# report so the report is the single source for judge metrics too, ready for the G5 cutover.
_JUDGE_KEYS = (
    "llm_judge",
    "judge_vs_MRR_correlation",
    "judge_vs_Recall5_correlation",
)


def _attach_traces_to_report(s: "RunState") -> None:
    """Mirror per-query traces + failure analysis + answer-gen details into ``report['traces']``
    (G2) and the LLM-judge results + calibration into ``report['judge']`` (G3). The terminal node
    owns this, so every producer (metrics → answer_gen → judge) has written its part. Additive:
    the legacy top-level keys stay until the G5 cutover."""
    report = s.results.get("report")
    if not isinstance(report, dict):
        return
    traces = {k: s.results[k] for k in _TRACE_KEYS if k in s.results}
    if traces:
        report["traces"] = traces
    judge = {k: s.results[k] for k in _JUDGE_KEYS if k in s.results}
    if judge:
        report["judge"] = judge
