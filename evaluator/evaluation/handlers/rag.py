"""RAG stage handlers: answer generation + finalize (LLM judge, traces, latency).

Each handler registers itself via ``@register_stage_handler`` at import time.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger
from ...judge import run_llm_judging
from ...metrics import per_speaker_breakdown, judge_calibration
from ..helpers import _payload_to_key
from ..answer_gen import generate_answers
from ..executor.state import RunState
from .retrieval import _retrieved_from_bus
from ._common import _relevant_from_bus, publish_keyed_or_plain
from ...utils.progress import progress_iter
from ..answer_gen import score_answers
from ..item_set import ItemSet
from ..aggregate import flatten_report, reduce_scores
from ..metric_registry import compute_metrics

logger = get_logger(__name__)


def _answer_corpus_lookup(s: "RunState", results_with_scores) -> dict:
    """doc_id → payload, from the full corpus + retrieved payloads (richer metadata) —
    used to find reference answers even for retrieval misses. The corpus base is cached
    on the state; the retrieved overlay is rebuilt per call (it is branch-specific)."""
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
    # Resolved before execution: global ⊕ node params; the node's presence forces enabled.
    cfg = s.resolved_config(default=s.answer_gen_config)
    # Stash it for the (separate) answer_metrics node — resolved_config() is keyed by the
    # CURRENT node, and scoring runs under a different node id, so it can't see this node's
    # own override without this handoff (see _stage_answer_metrics).
    s.answer_gen_resolved_config = cfg
    if cfg is None or not getattr(cfg, "enabled", False):
        return {}

    s.cb("step_5_answer_gen", 0, s.total, "Step 5: Answer generation")
    # Bus-only: query_text is the single honest source for the effective query — an
    # audio-only graph with no ASR node has none, which we make LOUD instead of silently
    # emitting zero answers.
    query_texts = s.input("query_text", default=[])
    if not query_texts:
        logger.warning(
            "Answer generation enabled but no query_text on the bus — producing 0 answers. "
            "The query is query_text-only; an audio-only graph needs an ASR node to supply it."
        )
        return {}
    answer_results = generate_answers(
        traces_data=(query_ids, all_relevant, results_with_scores),
        all_query_texts=query_texts,
        corpus_lookup=_answer_corpus_lookup(s, results_with_scores),
        config=cfg,
        # `context_source: full_text` ranks article chunks against the question with the run's
        # own text embedder (same model + cache as retrieval); unused otherwise.
        embedder=s.text_embedding_pipeline,
    )
    results["answer_generation"] = answer_results
    # Bus-published too (multi-variant): a graph with N answer_gen nodes would otherwise
    # have `_stage_answer_metrics`/finalize read whichever variant's dict happens to be in the
    # shared `s.results` at that point — sibling-looked-up via the `generated_answers` producer
    # id (there's no declared `answer_generation` port, so no direct `get_artifact` binding).
    s.put_artifact("answer_generation", answer_results)
    logger.info("Answer generation complete — %d cases", answer_results["cases"])
    return {d["query_id"]: d for d in answer_results["details"]}


def _build_query_traces(
    s: "RunState",
    results,
    all_relevant,
    results_with_scores,
    query_ids,
) -> None:
    """Build per-query traces (+ per-speaker breakdown), at most ``trace_limit`` of them
    (0 = no limit). Per-item scores and answer details are joined by query id — WER is
    published in reference order and recall in retrieved order, so a positional pairing
    would cross them."""
    if not results_with_scores:
        return

    limit = (
        len(results_with_scores) if s.trace_limit <= 0
        else min(s.trace_limit, len(results_with_scores))
    )
    references = s.get_artifact(
        "reference_transcription", default=[]
    )  # bus-only
    wer_items = s.keyed_items("per_query_wer")
    cer_items = s.keyed_items("per_query_cer")
    recall_items = s.keyed_items("per_query_recall5")
    ans_items = s.keyed_items("generated_answers")

    def _by_id(items, qid):
        return items.value_for(qid) if items is not None and items.has(qid) else None

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
        ans_detail = _by_id(ans_items, query_id) or {}
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
        wer = _by_id(wer_items, query_id)
        if wer is not None:
            trace_entry["per_query_wer"] = wer
            cer = _by_id(cer_items, query_id)
            if cer is not None:
                trace_entry["per_query_cer"] = cer
        recall = _by_id(recall_items, query_id)
        if recall is not None:
            trace_entry["recall_at_5"] = recall
        traces.append(trace_entry)
    results["query_traces"] = traces

    per_speaker = per_speaker_breakdown(traces)
    if per_speaker is not None:
        results["per_speaker"] = per_speaker


def _run_judge(
    s: "RunState", all_relevant, per_query_recall5, retrieved_keys, cfg
) -> Optional[dict]:
    """Judge node: run the LLM judge over query traces (if enabled) + calibration. ``cfg`` is
    the node-overlaid judge config. Built + returned, not mutated in place — multi-variant
    callers publish it on the bus under their own node id."""
    if cfg is None or not getattr(cfg, "enabled", False):
        return None
    # Read off the artifact bus, not `s.results` — the bus survives the metrics/report node's
    # `s.results = results` rebuild regardless of node order (see build_query_traces).
    traces = s.get_artifact("query_traces", default=None)
    if traces is None:
        raise RuntimeError(
            "LLM judge requires query traces, and none were built — add a "
            "`build_query_traces` node to the graph (and wire it into answer_judge)."
        )

    s.cb("step_6_judge", 0, s.total, "Step 6: LLM judge")
    judge_mode = getattr(cfg, "judge_mode", "retrieval")
    logger.info(
        "Running LLM judge — mode=%s model=%s cases=%d",
        judge_mode,
        getattr(cfg, "model", "unknown"),
        getattr(cfg, "max_cases", -1),
    )
    judge_results = run_llm_judging(traces, cfg)
    out: dict = {"llm_judge": judge_results}

    # Judge calibration: correlate per-query judge overall score with IR metrics.
    judge_scores = [
        d.get("judge", {}).get("overall", float("nan"))
        for d in judge_results.get("details", [])
    ]
    calibration = judge_calibration(
        judge_scores, per_query_recall5, retrieved_keys, all_relevant
    )
    if calibration:
        out.update(calibration)
        logger.info(
            "Judge calibration — vs_MRR=%.3f vs_Recall5=%.3f",
            out.get("judge_vs_MRR_correlation", float("nan")),
            out.get("judge_vs_Recall5_correlation", float("nan")),
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
    return out


def _retrieved_or_text_ids(s: "RunState"):
    """``_retrieved_from_bus``, with a query_text fallback for retrieval-free graphs: the
    no-retrieval baseline has no ``retrieved`` artifact, so ids fall back to the keyed
    ``query_text``/``reference_transcription`` ItemSet, with empty result lists (exactly
    what that graph means). A graph WITH retrieval is unaffected."""
    results_with_scores, retrieved_keys, query_ids = _retrieved_from_bus(s)
    if results_with_scores or query_ids:
        return results_with_scores, retrieved_keys, query_ids
    # Whichever per-query keyed stream THIS node has bound (answer_gen binds query_text;
    # build_query_traces / answer_metrics / judge bind reference_transcription).
    for name in ("query_text", "reference_transcription"):
        keyed = s.keyed_items(name)
        if keyed is not None and keyed.ids:
            ids = [str(i) for i in keyed.ids]
            return [[] for _ in ids], [[] for _ in ids], ids
    return results_with_scores, retrieved_keys, query_ids


@register_stage_handler("generate", self_timed=True)
def _stage_generate(s: RunState) -> None:
    """Answer-generation node: generate answers for every query — with the retrieved docs
    as context in RAG graphs, or with NO context in a retrieval-free graph (the parametric
    no-retrieval baseline). Writes ``answer_generation`` into ``s.results`` and publishes
    the per-query detail map as the ``generated_answers`` ItemSet, which the trace node
    id-joins. Conditional on ``answer_generation.enabled``."""
    results_with_scores, _, query_ids = _retrieved_or_text_ids(s)
    detail_by_qid = _generate_answer_details(
        s, s.results, _relevant_from_bus(s), results_with_scores, query_ids
    )
    publish_keyed_or_plain(
        s, "generated_answers", list(detail_by_qid.values()), list(detail_by_qid.keys())
    )


def _decision_labels(s: RunState) -> dict:
    """query_id → the dataset's yes/no/maybe label, from the keyed ``short_answers`` ItemSet
    (id-joined, so a dropped item can't shift labels onto the wrong queries). Empty when the
    graph does not wire the artifact into this node."""
    items = s.keyed_items("short_answers")
    if items is None:
        return {}
    return {str(qid): str(val) for qid, val in zip(items.ids, items.values)}


def _fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


def _stage_answer_metrics(s: RunState) -> None:
    """Answer-quality comparison node: scores generated answers vs reference answers +
    retrieved context, enriching the answer_generation details in place + the mean_*
    aggregates the report reads."""

    # Sibling lookup, not `s.results` (multi-variant): `answer_generation` has no declared
    # port of its own, so it rides alongside the bound `generated_answers` producer.
    answer_results = s.sibling_artifact("generated_answers", "answer_generation")
    if not isinstance(answer_results, dict) or not answer_results.get("details"):
        return
    results_with_scores, _, query_ids = _retrieved_or_text_ids(s)
    score_answers(
        answer_results,
        traces_data=(query_ids, _relevant_from_bus(s), results_with_scores),
        corpus_lookup=_answer_corpus_lookup(s, results_with_scores),
        # The SAME resolved config generation used (stashed by _generate_answer_details) —
        # not the flat default, so a node-level answer_gen override affects its own scoring.
        config=getattr(s, "answer_gen_resolved_config", None) or s.answer_gen_config,
        decision_gt=_decision_labels(s),
    )
    # Publish every aggregate score_answers computed — the report node binds this artifact and
    # merges it in. Publishing only mean_rougeL silently dropped the rest, and the report never
    # read even that (the metrics node rebuilds `results`, discarding what ran before it).
    s.put_artifact("answer_scores", {
        k: v for k, v in answer_results.items()
        if k.startswith("mean_") or k == "decision_unknown_rate"
    })
    logger.info(
        "Answer metrics — decision accuracy: %s (unknown %s) | mean ROUGE-L: %s",
        _fmt(answer_results.get("mean_decision_accuracy")),
        _fmt(answer_results.get("decision_unknown_rate")),
        (
            f"{answer_results['mean_rougeL']:.4f}"
            if answer_results.get("mean_rougeL") is not None
            else "n/a"
        ),
    )


def _stage_build_query_traces(s: RunState) -> None:
    """Explicit trace builder: assemble the per-query traces from the retrieved docs, the
    generated answers (the keyed ``generated_answers`` ItemSet, empty when no answer_gen
    ran) and the per-item WER/recall score artifacts — all id-joined off the bus.
    Always present in retrieval modes when tracing is on — so the judge + report read the
    traces."""
    # No dedup-by-shared-key guard here (multi-variant): a graph with N build_query_traces
    # nodes (one per compared variant) must build ALL N — each publishes its own traces on the
    # bus keyed by its own node id below. `s.results["query_traces"]` is written-then-read-back
    # as scratch space within this single call only (never read across nodes).
    results_with_scores, _, query_ids = _retrieved_or_text_ids(s)
    if not query_ids:
        # Nothing to key traces by — say WHY (which inputs this node is bound to and what
        # was actually on the bus) instead of leaving the judge to fail 2 nodes later
        # with "no traces were built".
        bindings = list(getattr(s.current_node, "bindings", ()) or ())
        logger.warning(
            "build_query_traces '%s': no per-query ids — no retrieved artifact and no keyed "
            "query_text/reference_transcription on the bus; publishing no traces. bindings=%s",
            getattr(s.current_node, "id", "?"), bindings,
        )
        return
    _build_query_traces(
        s,
        s.results,
        _relevant_from_bus(s),
        results_with_scores,
        query_ids,
    )
    # Publish the actual traces on the bus (not just a built/not-built marker) — downstream
    # readers (judge, finalize) use this instead of `s.results["query_traces"]` because the
    # metrics/report node replaces `s.results` wholesale and would silently wipe it otherwise.
    traces = s.results.get("query_traces")
    if traces is not None:
        s.put_artifact("query_traces", traces)


def _stage_answer_judge(s: RunState) -> None:
    """LLM-judge comparison node: scores the query traces (built upstream) vs the judge
    rubric + calibrates against IR metrics. Present only when the judge is enabled."""
    cfg = s.resolved_config(default=s.judge_config)
    if cfg is not None and not cfg.enabled:
        return  # a per-branch {enabled: false} judge node — skip this branch's judging
    _, retrieved_keys, _ids = _retrieved_or_text_ids(s)
    judge_out = _run_judge(
        s, _relevant_from_bus(s),
        list(s.get_artifact("per_query_recall5", default=[])), retrieved_keys, cfg,
    )
    if judge_out is None:
        return
    # Bus-published (multi-variant): `finalize` sibling-looks this up via the (already
    # bound) judge_pass/judge_scores producer — see `attach_judge_metrics`.
    s.put_artifact("judge_summary", judge_out)
    details = (judge_out.get("llm_judge") or {}).get("details") or []
    _publish_judge_scores(s, details, cfg)


def _publish_judge_scores(s: RunState, details: list, cfg=None) -> None:
    """Publish the per-query judge outputs as keyed ItemSets so the (reference-free) judge
    metrics score them through the normal path: ``judge_scores`` (overall), ``judge_pass``
    (1.0/0.0 → judge_pass_rate), and ``judge_aspect_<a>`` for each configured aspect. The
    finalize node folds these into the report via ``attach_judge_metrics``."""

    ids = [str(d["query_id"]) for d in details]
    s.put_items("judge_scores", ItemSet(ids, [d["judge"]["overall"] for d in details]))
    s.put_items(
        "judge_pass",
        ItemSet(ids, [1.0 if d["judge"]["verdict"] == "PASS" else 0.0 for d in details]),
    )
    for aspect in (cfg if cfg is not None else s.judge_config).judge_aspects:
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
    """Terminal node: assembles THIS node's own report (metrics + traces + judge + latency)
    from its bound producers and publishes it on the bus as ``run_report`` (multi-variant —
    a graph with N finalize nodes, one per compared variant, must not share one mutable
    ``s.results``; `run_graph` collects every `run_report` at the end). Trace building lives in
    the explicit build_query_traces node; the LLM judge in answer_judge."""
    own_report: Dict[str, Any] = dict(s.get_artifact("metrics", default={}))
    s.stage_times["total_s"] = time.perf_counter() - s.t_total
    # Whole-run values (total wall time, per-node timing) — not meaningfully "per variant",
    # the same numbers land in every finalize node's own_report.
    own_report["latency"] = dict(s.stage_times)
    # Per-node wall time — every node that ran, including the LLM stages the bucket-based
    # `latency` never covered (answer generation and the judge were simply absent).
    own_report["node_latency"] = dict(s.node_times)
    # Judge metrics: the judge node is downstream of the report assembler, so its per-query
    # scores are merged into the report here, at the terminal node, via the metric registry.

    attach_judge_metrics(s, own_report)
    _attach_traces_to_report(s, own_report)
    s.put_artifact("run_report", own_report)
    logger.info(
        "Stage latency — asr=%.1fs embed=%.1fs retrieve=%.1fs total=%.1fs",
        s.stage_times.get("asr_s", 0),
        s.stage_times.get("embedding_s", 0),
        s.stage_times.get("retrieval_s", 0),
        s.stage_times["total_s"],
    )
    if s.node_times:
        # Slowest first: on an LLM run answer_gen/answer_judge dominate, and that is exactly
        # what the bucket summary above cannot show.
        logger.info(
            "Node latency — %s",
            "  ".join(
                f"{node}={secs:.1f}s"
                for node, secs in sorted(
                    s.node_times.items(), key=lambda kv: -kv[1]
                )
            ),
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


def _attach_traces_to_report(s: "RunState", own_report: dict) -> None:
    """Consolidate per-query traces + failure analysis + answer-gen details into
    ``own_report['report']['traces']``, and the LLM-judge results + calibration into
    ``['judge']``. Read via the bus or a sibling lookup (this node's OWN producers only,
    never a global scan); the mirrored top-level copies are dropped at the output boundary
    (`drop_mirrored_top_level_keys`)."""
    report = own_report.get("report")
    if not isinstance(report, dict):
        return
    traces = {}
    # retrieval_failure_analysis rides along as a top-level key of the bound `metrics`
    # artifact itself (metrics_diagnostics.py) — already in own_report, no separate lookup.
    if "retrieval_failure_analysis" in own_report:
        traces["retrieval_failure_analysis"] = own_report["retrieval_failure_analysis"]
    answer_generation = s.sibling_artifact("generated_answers", "answer_generation")
    if answer_generation is not None:
        traces["answer_generation"] = answer_generation
    # query_traces off the bus (own binding), not `s.results` — see build_query_traces / _run_judge.
    bus_traces = s.get_artifact("query_traces", default=None)
    if bus_traces is not None:
        traces["query_traces"] = bus_traces
    if traces:
        report["traces"] = traces
    judge_bound_name = "judge_pass" if s._producers("judge_pass") else "judge_scores"
    judge = s.sibling_artifact(judge_bound_name, "judge_summary")
    if judge:
        report["judge"] = judge


def drop_mirrored_top_level_keys(results: dict) -> None:
    """Output boundary: once traces/judge are in ``report['traces']``/``['judge']``, drop
    the duplicate top-level keys so the report is the single source. Only drops keys
    actually mirrored (no data loss if the report wasn't assembled); run after the whole
    graph (sinks read the top-level working copies during the run)."""
    report = results.get("report")
    if not isinstance(report, dict):
        return
    mirrored = set(report.get("traces", {})) | set(report.get("judge", {}))
    for key in (*_TRACE_KEYS, *_JUDGE_KEYS):
        if key in mirrored:
            results.pop(key, None)


def attach_judge_metrics(s: "RunState", own_report: dict) -> None:
    """Merge the LLM-judge per-query scores into ``own_report``'s report as registry metrics.
    The judge node runs *downstream* of the report assembler, so its keyed ``judge_*``
    ItemSets are not on the bus when the report is first scored — this terminal pass merges
    them in via the same registry + reducer. No-op when the judge did not run, so a
    judge-off run (the parity default) is byte-identical. ``own_report`` is THIS finalize
    node's own dict (multi-variant) — never the shared ``s.results``."""
    report = own_report.get("report") if own_report else None
    if not isinstance(report, dict):
        return

    # the judge ItemSets are read through the finalize node's DECLARED bindings
    # (judge_scores / judge_pass / every judge_aspect_* is a declared optional input now).
    judge_arts: Dict[str, ItemSet] = {}
    for art, pid in getattr(s.current_node, "bindings", ()):
        if not (art in ("judge_scores", "judge_pass") or art.startswith("judge_aspect_")):
            continue
        if s.ctx.has(pid, art):
            val = s.ctx.get(pid, art)
            if isinstance(val, ItemSet):
                judge_arts[art] = val  # last bound producer wins
    if not judge_arts:
        return

    # collect_all → exactly the judge_* metrics fire (only judge artifacts are present);
    # the config's `metrics:` allowlist applies here too.
    scores = compute_metrics(judge_arts, only=s.metric_allowlist)
    if not scores:
        return
    # Attach to the run's mode branch when present (single-branch), else the first branch.
    branches = report.setdefault("branches", {})
    target = s.mode if s.mode in branches else next(iter(branches), s.mode)
    branch = branches.setdefault(target, {})
    for name, item_scores in scores.items():
        branch[name] = reduce_scores(
            item_scores, with_ci=getattr(s, "compute_confidence_intervals", False)
        )
    own_report.update(flatten_report(report))
