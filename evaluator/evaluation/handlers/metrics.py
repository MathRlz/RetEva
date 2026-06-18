"""Metrics stage handlers: per-branch aggregate report + the single-branch metrics node.

Registers the ``metrics`` and ``aggregate`` handlers at import time. The metric **registry**
is the one place metrics are computed (Phase 4 L3 removed the legacy double-compute path); the
flat ``WER``/``MRR``/``Recall@k``/... keys are report-derived aliases (`_derive_bare_keys`).
Diagnostics the report does not carry + the per-item intermediates the rag/judge stages consume
are computed from the aligned per-item state. WER/CER score the ASR hypothesis (`query_text`,
immutable since Phase 4) — no separate raw_query_text snapshot.

Two concerns are extracted to siblings: the provenance assembly (``metrics_provenance``) and the
IR diagnostics (``metrics_diagnostics``); this core owns the report assembly + the typed metric
nodes (transcription / retrieval / metrics / aggregate).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger
from ..helpers import _search_results_to_keys
from ..executor.state import RunState
from ..result_schema import RunResults
from .retrieval import _retrieved_from_bus
from ._common import (
    asr_ran,
    retrieval_ran,
    is_asr_text_retrieval,
    _ctx_first,
    _reference_transcriptions,
)
from .metrics_provenance import _run_provenance, _record_model_info
from .metrics_diagnostics import _ir_diagnostics
from ...metrics import (
    wer_recall_correlation,
    word_error_rate,
    character_error_rate,
)
from ...metrics.ir import recall_at_k

logger = get_logger(__name__)


@register_stage_handler("measure", self_timed=True)
def _stage_measure(s: RunState) -> None:
    """The ``measure`` operator: dispatch by family / trace to the typed comparison,
    report-assembler, trace-builder, or judge body (all unchanged; answer/trace/judge live
    in the rag handlers, alignment in the embedding handlers)."""
    from .embedding import _stage_embedding_alignment_metrics
    from .rag import (
        _stage_answer_judge,
        _stage_answer_metrics,
        _stage_build_query_traces,
    )
    from ._dispatch import dispatch_operator

    return dispatch_operator("measure", {
        "transcription_metrics": _stage_transcription_metrics,
        "retrieval_metrics": _stage_retrieval_metrics,
        "metrics": _stage_metrics,
        "embedding_alignment_metrics": _stage_embedding_alignment_metrics,
        "answer_metrics": _stage_answer_metrics,
        "build_query_traces": _stage_build_query_traces,
        "answer_judge": _stage_answer_judge,
    }, s)


def _branch_of(producer_id: str) -> str:
    """Branch label encoded in a node id (``retrieval@corr`` → ``corr``; else ``main``)."""
    return producer_id.split("@", 1)[1] if "@" in producer_id else "main"


def _is_ir_metric(name: str) -> bool:
    """IR (retrieval) metric names — dropped from the report when the dataset join is disjoint
    (B5), since with no relevant doc in the corpus they would be a misleading 0, not 'n/a'.
    """
    return name in ("mrr", "map") or name.startswith(("recall@", "precision@", "ndcg@"))


def _drop_ir_if_disabled(s: "RunState", scores: Dict[str, Any]) -> Dict[str, Any]:
    """Remove IR metrics from a branch's score set when `s.disable_ir_metrics` (B5)."""
    if not getattr(s, "disable_ir_metrics", False):
        return scores
    return {k: v for k, v in scores.items() if not _is_ir_metric(k)}


# ── Shared reduction utilities (M7): one registry-scoring + provenance + attach path
# used by both the single-branch metrics node and the multi-branch aggregate. ──


def _branch_scores(s: "RunState", artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Score one branch's artifacts via the metric registry (collect_all), with the
    B5 IR gate applied — the single scoring call both report paths share."""
    from ..metric_registry import compute_metrics

    return _drop_ir_if_disabled(s, compute_metrics(artifacts, collect_all=True))


def _attach_report(results, report) -> None:
    """Attach the keyed report + surface its flattened keys to the leaderboard (M4)."""
    from ..aggregate import flatten_report

    results["report"] = report
    results.update(flatten_report(report))


def _retrieval_wer_impact(
    per_branch: Dict[str, Dict[str, Any]], baseline: str
) -> Dict[str, Dict[str, Any]]:
    """Retrieval-WER-Impact: recall lost vs the oracle baseline, per recall@k —
    `Recall(baseline) − Recall(branch)` over id-aligned items (the degradation
    ASR/correction imposes on retrieval). M3; extracted from the aggregate (M7)."""
    base_scores = per_branch.get(baseline, {})
    impact: Dict[str, Dict[str, Any]] = {}
    for branch, scores in per_branch.items():
        if branch == baseline:
            continue
        per_metric: Dict[str, Any] = {}
        for name in scores:
            if not name.startswith("recall@") or name not in base_scores:
                continue
            ids, base_vals, branch_vals = base_scores[name].align(scores[name])
            if ids:
                losses = [float(b) - float(v) for b, v in zip(base_vals, branch_vals)]
                per_metric[name] = {
                    "mean": sum(losses) / len(losses),
                    "n": len(losses),
                }
        if per_metric:
            impact[f"{branch}_vs_{baseline}"] = per_metric
    return impact


def _stage_aggregate(s: RunState) -> None:
    """Terminal report builder (W6/A9): per-branch metrics + cross-branch deltas.

    Scans the RunContext for every ``retrieved`` producer (one per branch, per-producer
    keyed), scores each against the shared ground-truth artifacts via the metric registry,
    and builds the report with paired deltas vs a baseline branch. Owns ``results['report']``
    when present (supersedes the single-branch metrics-node report)."""
    from ..item_set import ItemSet
    from ..aggregate import build_report

    relevant = _ctx_first(s, "relevant_docs")
    reference = _ctx_first(s, "reference_text")
    # ASR-quality metrics pair against the spoken transcription when the asr node
    # published it (M1c-3 reconciliation of the M1a trap); reference_text (question)
    # remains the fallback — identical on TTS-bridge datasets where spoken == question.
    ref_transcription = _ctx_first(s, "reference_transcription")
    if isinstance(ref_transcription, ItemSet):
        reference = ref_transcription
    # Per-branch query_text = the branch's ASR hypothesis for WER/CER. query_text is
    # immutable (correction/optimization emit distinct names) so the only producers are
    # asr / dataset_source — no correction preference needed (Phase 4).
    qtext_by_branch: Dict[str, ItemSet] = {}
    shared_qtext: Optional[ItemSet] = None
    qtext_slots = [(p, n) for p, n in s.ctx.slots() if n == "query_text"]
    for pid, _ in qtext_slots:
        val = s.ctx.get(pid, "query_text")
        if not isinstance(val, ItemSet):
            continue
        shared_qtext = val
        qtext_by_branch.setdefault(_branch_of(pid), val)
    only_shared = shared_qtext if len(qtext_slots) == 1 else None

    # Items dropped per-item upstream (T1) are excluded from the keyed report so a
    # placeholder/empty result never reaches a metric — the report measures survivors only.
    dropped_ids = s.drop_sink.all_dropped_ids()
    per_branch: Dict[str, Dict[str, ItemSet]] = {}
    branch_ids: Dict[str, set] = {}
    for pid, art in list(s.ctx.slots()):
        if art != "retrieved":
            continue
        retrieved = s.ctx.get(pid, art)
        if not isinstance(retrieved, ItemSet):
            continue
        if dropped_ids:
            retrieved = retrieved.filter(lambda i, _v: i not in dropped_ids)
        branch = _branch_of(pid)
        branch_ids[branch] = set(retrieved.ids)
        # retrieved values are per-query (payload, score) lists → doc-id keys for IR metrics
        keyed = retrieved.with_values(
            [_search_results_to_keys(r) for r in retrieved.values]
        )
        artifacts: Dict[str, ItemSet] = {"retrieved": keyed}
        if relevant is not None:
            artifacts["relevant_docs"] = relevant
        if reference is not None:
            artifacts["reference_text"] = reference
            # query_text for WER/CER: this branch's ASR hypothesis, else the lone shared one.
            qt = qtext_by_branch.get(branch, only_shared)
            if qt is not None:
                artifacts["query_text"] = qt
        scores = _branch_scores(s, artifacts)
        if scores:
            per_branch[branch] = scores
    if not per_branch:
        return
    # Baseline branch for deltas: the oracle "ref" branch when present (Retrieval-WER-Impact
    # is measured against it), else "main", else the first by name.
    baseline = next(
        (b for b in ("ref", "main") if b in per_branch), sorted(per_branch)[0]
    )
    # Per-branch drops vs the full run (union of all branches' query ids): which items each
    # branch lost, so the shrinking paired denominator (S1) is auditable at report level (S2).
    full_ids: set = set().union(*branch_ids.values()) if branch_ids else set()
    dropped_by_branch = {
        b: sorted(full_ids - ids) for b, ids in branch_ids.items() if full_ids - ids
    }
    report = build_report(
        per_branch,
        baseline=baseline,
        provenance=_run_provenance(s, dropped_by_branch),
        with_ci=True,
    )
    # Cross-stage: per-branch Pearson(WER, Recall@5) — does worse ASR cost retrieval? (M2)
    # (wer_recall_correlation is imported at module top.)
    for branch, scores in per_branch.items():
        wer, rec = scores.get("wer"), scores.get("recall@5")
        if wer is None or rec is None:
            continue
        ids, wer_vals, rec_vals = wer.align(rec)
        corr = wer_recall_correlation(wer_vals, rec_vals)
        if corr is not None:
            report["branches"][branch]["wer_recall_correlation"] = {
                "mean": corr,
                "n": len(ids),
            }
    impact = _retrieval_wer_impact(per_branch, baseline)
    if impact:
        report["retrieval_wer_impact"] = impact
    if s.disable_ir_metrics and s.join_warning:
        report["join_warning"] = s.join_warning  # B5: why IR metrics are absent
    _attach_report(s.results, report)
    logger.info(
        "aggregate: %d branch(es) %s, baseline=%s",
        len(per_branch),
        sorted(per_branch),
        baseline,
    )


def _asr_hypothesis(s: RunState) -> list:
    """The ASR hypothesis from the bus (``query_text``). It is immutable — correction /
    optimization publish distinct names — so this is always the un-rewritten ASR output
    WER/CER score against (no raw_query_text snapshot needed since Phase 4)."""
    return list(s.get_artifact("query_text", default=[]))


def _wer_cer_pair(pair):
    """Pure ``(WER, CER)`` for one ``(reference, hypothesis)`` pair. Top-level (not a closure)
    so the ``process`` CPU-stage backend can pickle it (Roadmap 4b)."""
    gt_text, hyp_text = pair
    return word_error_rate(gt_text, hyp_text), character_error_rate(gt_text, hyp_text)


def _asr_item_scores(s: RunState):
    """Per-item raw-ASR WER/CER lists (consumed by the answer-gen / judge / trace stages).

    The per-item WER/CER map is a pure, CPU-bound, order-preserving fold, so it runs through the
    4b ``parallel_map`` (``sync`` by default → byte-identical to the serial loop; ``thread`` /
    ``process`` parallelize the GIL-bound edit-distance work)."""
    if not asr_ran(s):  # only ASR modes carry WER/CER
        return [], []
    from ..executor.cpu_parallel import parallel_map, resolve_cpu_backend

    pairs = list(zip(_reference_transcriptions(s), _asr_hypothesis(s)))
    backend, workers = resolve_cpu_backend(s.config)
    scored = parallel_map(_wer_cer_pair, pairs, backend=backend, workers=workers)
    return [w for w, _ in scored], [c for _, c in scored]


# registry metric name -> legacy flat bare key. Exactly the legacy key set (no
# Precision@k / ceer / raw_wer here: those stay registry-only in report["branches"]).
_BARE_KEY_FOR = {
    "wer": "WER",
    "cer": "CER",
    "mrr": "MRR",
    "map": "MAP",
    "recall@1": "Recall@1",
    "recall@5": "Recall@5",
    "recall@10": "Recall@10",
    "ndcg@1": "NDCG@1",
    "ndcg@5": "NDCG@5",
    "ndcg@10": "NDCG@10",
}


def _derive_bare_keys(results: "RunResults", scores: Dict[str, Any], asr_mode: bool) -> None:
    """Surface the registry per-item scores as the legacy flat bare keys (report-derived
    aliases, L3) — the registry is the single scalar source. WER/CER are gated to ASR runs
    (``asr_mode``; legacy parity: audio_emb/audio_text carry no WER). Means are identical to the
    deleted legacy scalars (smoke-confirmed: single-branch WER 0.6541 == registry wer)."""
    for name, items in scores.items():
        key = _BARE_KEY_FOR.get(name)
        if key is None:
            continue
        if name in ("wer", "cer") and not asr_mode:
            continue
        vals = [float(v) for _, v in items]
        results[key] = (sum(vals) / len(vals)) if vals else 0.0


def _stage_transcription_metrics(s: RunState) -> None:
    """Comparison node: the ASR hypothesis (``query_text``) vs the spoken GT
    (``reference_transcription``) → per-item WER/CER. Sets ``s.wer_scores``/``s.cer_scores``
    (read by the report + the rag/judge/trace stages) and publishes a summary artifact."""
    s.wer_scores, s.cer_scores = _asr_item_scores(s)
    n = len(s.wer_scores)
    s.put_artifact(
        "transcription_scores",
        {
            "wer_mean": (sum(s.wer_scores) / n) if n else None,
            "cer_mean": (sum(s.cer_scores) / n) if n else None,
            "n": n,
        },
    )


def _stage_retrieval_metrics(s: RunState) -> None:
    """Comparison node: the retrieved docs vs the GT relevance (``relevant_docs``) →
    per-item recall@5 + the aligned relevance the IR diagnostics / answer-gen / judge read
    (``s.metrics_all_relevant`` / ``s.per_query_recall5``)."""
    _results_with_scores, retrieved_keys, _ids = _retrieved_from_bus(s)
    all_relevant = list(s.get_artifact("relevant_docs", default=[])) or [
        {str(gt): 1} for gt in _reference_transcriptions(s)
    ]
    recall5 = [recall_at_k(r, rel, 5) for r, rel in zip(retrieved_keys, all_relevant)]
    s.metrics_all_relevant = all_relevant
    s.per_query_recall5 = recall5
    n = len(recall5)
    s.put_artifact(
        "retrieval_scores",
        {"recall@5_mean": (sum(recall5) / n) if n else None, "n": n},
    )


def _stage_metrics(s: RunState) -> None:
    """Report assembler: registry-native scalar report → results (single scalar source, L3).

    No longer a god-node — the per-comparison computation lives in the typed
    ``transcription_metrics`` / ``retrieval_metrics`` nodes (Phase 5); this node assembles
    the report from the registry (the flat WER/MRR/Recall@k/... keys) + the per-item
    intermediates those nodes set. Numerically identical to the former combined node."""
    _results_with_scores, retrieved_keys, _ids = _retrieved_from_bus(s)
    _t_phase = time.perf_counter()
    s.cb("phase_4_metrics", 0, s.total, "Computing metrics")

    results: RunResults = RunResults()
    results["pipeline_mode"] = s.mode
    results["phased"] = True
    results["oracle_mode"] = s.oracle_mode

    _record_model_info(results, s)

    # Registry report + report-derived flat bare keys (the single scalar source).
    _attach_registry_report(s, results, retrieved_keys)

    # Diagnostics the registry report does not carry (from the per-item state the typed
    # metric nodes set upstream).
    if retrieval_ran(s) and retrieved_keys:
        _ir_diagnostics(
            results,
            s,
            s.metrics_all_relevant,
            s.per_query_recall5,
            s.wer_scores,
            retrieved_keys,
        )

    s.stage_times["metrics_s"] = time.perf_counter() - _t_phase
    s.results = results


def _attach_registry_report(
    s: RunState, results: "RunResults", retrieved_keys: list
) -> None:
    """Compute metrics via the registry + aggregate and attach a keyed ``report`` (W4b).

    Builds keyed ``ItemSet``s from the run's aligned per-item state and runs the metric
    registry → ``build_report``. Numerically identical to the legacy headline scalars
    (parity-proven), but keyed + branch-shaped — the basis for cross-branch deltas (W6).
    Additive: the legacy result keys are untouched. No-op when query ids do not align.
    """
    # Per-item identity rides the keyed bus artifacts (M1d-2): the effective query text
    # in ASR modes, the spoken reference in audio modes. A plain (non-keyed) publish
    # means ids did not align — same no-op condition as the legacy all_query_ids check.
    keyed = s.keyed_items("query_text") or s.keyed_items("reference_transcription")
    if keyed is None:
        return
    ids = [str(i) for i in keyed.ids]
    if not ids or len(set(ids)) != len(ids):
        return
    from ..item_set import ItemSet
    from ..aggregate import build_report

    n = len(ids)
    artifacts: Dict[str, ItemSet] = {}

    # Align each per-item list to the query-id order. Lists at least as long as ``ids`` are
    # trimmed to the first n (matches the legacy zip-truncation leniency — e.g. a checkpoint
    # resume can leave a longer, batch-overlapping hypotheses list); shorter lists are skipped.
    def _keyed(values, name):
        if values is not None and len(values) >= n:
            artifacts[name] = ItemSet(ids, list(values)[:n])

    # Bus-first: the ASR hypothesis in ASR modes (query_text is immutable since Phase 4, so
    # this is the un-rewritten output `wer`/`cer` score); the spoken reference in audio modes
    # (legacy parity — there the "query" scored by text metrics is the GT).
    reference = _reference_transcriptions(s)
    query_text = (
        s.get_artifact("query_text")
        if is_asr_text_retrieval(s)
        else reference
    )
    _keyed(query_text, "query_text")
    # ASR-quality reference = the spoken transcription, never question_text (M1a guard).
    _keyed(reference, "reference_text")
    if retrieved_keys:
        _keyed(retrieved_keys, "retrieved")
    relevance = list(s.get_artifact("relevant_docs", default=[])) or [
        {str(gt): 1} for gt in reference
    ]
    if "retrieved" in artifacts:
        _keyed(relevance, "relevant_docs")
    if not artifacts:
        return
    # Exclude per-item drops (T1) so a placeholder/empty result never reaches a metric.
    dropped_ids = s.drop_sink.all_dropped_ids()
    if dropped_ids:
        artifacts = {
            name: items.filter(lambda i, _v: i not in dropped_ids)
            for name, items in artifacts.items()
        }
    scores = _branch_scores(s, artifacts)
    if scores:
        _derive_bare_keys(results, scores, asr_ran(s))
        # The report branch key + ``pipeline_mode`` echo carry the run's mode *label* — the
        # executed graph's identity (``s.mode``), which the node set alone can't reconstruct
        # (audio_emb vs audio_text share a graph). Behavior is graph-derived; only the label is.
        report = build_report(
            {s.mode: scores}, provenance=_run_provenance(s), with_ci=True
        )
        _attach_report(results, report)
