"""Metrics stage handlers: per-branch aggregate report + the single-branch metrics node.

The metric registry is the single place metrics are computed; other flat keys are
report-derived aliases.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger
from ..helpers import _branch_of, _search_results_to_keys
from ..executor.state import RunState
from ..result_schema import RunResults
from .retrieval import _retrieved_from_bus
from ._common import (
    asr_hypothesis,
    _ctx_first,
    _reference_transcriptions,
    _relevant_from_bus,
    publish_keyed_or_plain,
)
from .metrics_provenance import _run_provenance, _record_model_info
from .metrics_diagnostics import _ir_diagnostics
from ...metrics import (
    wer_recall_correlation,
    word_error_rate,
    character_error_rate,
)
from ...metrics.ir import recall_at_k
from .embedding import _stage_embedding_alignment_metrics
from .rag import (
    _stage_answer_judge,
    _stage_answer_metrics,
    _stage_build_query_traces,
)
from ._dispatch import dispatch_operator
from ..item_set import ItemSet
from ..aggregate import build_report, flatten_report, reduce_scores, rollup_variants
from ..metric_registry import compute_metrics
from ..executor.cpu_parallel import run_per_item

logger = get_logger(__name__)


@register_stage_handler("measure", self_timed=True)
def _stage_measure(s: RunState) -> None:
    """The ``measure`` operator: dispatches by family / trace to the typed handler."""

    return dispatch_operator("measure", {
        "transcription_metrics": _stage_transcription_metrics,
        "retrieval_metrics": _stage_retrieval_metrics,
        "metrics": _stage_metrics,
        "embedding_alignment_metrics": _stage_embedding_alignment_metrics,
        "answer_metrics": _stage_answer_metrics,
        "build_query_traces": _stage_build_query_traces,
        "answer_judge": _stage_answer_judge,
    }, s)


def _branched_items(s: "RunState", artifact: str, *, branch_exclusive: bool = False):
    """``(by_branch, only_shared)`` for an artifact's bound ItemSets. The second element is
    None once the artifact is branched, so a branch never reads another branch's copy.
    ``branch_exclusive`` marks an artifact whose absence in a branch is meaningful (e.g.
    ``corrected_query_text`` with correction disabled) — sharing it would score the baseline
    as if corrected, reading a single-corrector comparison as exactly zero effect.
    """

    published = [pid for pid in s._producers(artifact) if s.ctx.has(pid, artifact)]
    by_branch: Dict[str, Any] = {}
    shared = None
    for pid in published:
        val = s.ctx.get(pid, artifact)
        if isinstance(val, ItemSet):
            shared = val
            by_branch.setdefault(_branch_of(pid), val)
    lone_global = len(published) == 1 and not (
        branch_exclusive and "@" in published[0]
    )
    return by_branch, (shared if lone_global else None)


def _is_ir_metric(name: str) -> bool:
    """IR (retrieval) metric names — dropped from the report when the dataset join is disjoint,
    since with no relevant doc in the corpus they would be a misleading 0, not 'n/a'.
    """
    return name in ("mrr", "map") or name.startswith(("recall@", "precision@", "ndcg@"))


def _drop_ir_if_disabled(s: "RunState", scores: Dict[str, Any]) -> Dict[str, Any]:
    if not s.disable_ir_metrics:
        return scores
    return {k: v for k, v in scores.items() if not _is_ir_metric(k)}


# Shared by the single-branch metrics node and the multi-branch aggregate.


def _corrected_metrics_enabled(s: "RunState") -> bool:
    """Reads the query_correction node's stashed resolved config, not `s.resolved_config()`
    (this runs under a different node, which can't see it); falls back to the flat default
    when no query_correction node ran this branch."""
    cfg = getattr(s, "query_correction_resolved_config", None) or s.query_correction_config
    return bool(getattr(cfg, "corrected_metrics", False))


def _branch_scores(s: "RunState", artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Score one branch's artifacts via the metric registry — the single scoring call both
    report paths share. Rolls up lineage variants to parent level here so every consumer
    pairs and reduces on the cluster-safe unit."""

    scores = compute_metrics(
        artifacts,
        include_opt_in=_corrected_metrics_enabled(s),
        only=s.metric_allowlist,
    )
    how = s.variant_rollup
    return _drop_ir_if_disabled(
        s, {name: rollup_variants(items, how) for name, items in scores.items()}
    )


def _attach_report(results, report) -> None:
    """Attach the keyed report + surface its flattened keys to the leaderboard."""

    results["report"] = report
    results.update(flatten_report(report))


def _retrieval_wer_impact(
    per_branch: Dict[str, Dict[str, Any]], baseline: str
) -> Dict[str, Dict[str, Any]]:
    """Retrieval-WER-Impact: recall lost vs the oracle baseline, per recall@k —
    `Recall(baseline) − Recall(branch)` over id-aligned items (the degradation
    ASR/correction imposes on retrieval)."""
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


def _collect_branch_artifacts(s: RunState):
    """Assemble + score each branch's artifacts (id-JOINED, never trimmed — the
    multi-branch path's alignment semantics). Returns ``(per_branch, branch_ids)``."""

    relevant = _ctx_first(s, "relevant_docs")
    reference = _ctx_first(s, "reference_text")
    # ASR-quality metrics pair against the spoken transcription when the asr node
    # published it; reference_text (question) remains the fallback — identical on
    # TTS-bridge datasets where spoken == question.
    ref_transcription = _ctx_first(s, "reference_transcription")
    if isinstance(ref_transcription, ItemSet):
        reference = ref_transcription
    # Per-branch query_text = the branch's ASR hypothesis for WER/CER. query_text is
    # immutable (correction/optimization emit distinct names) so the only producers are
    # asr / dataset_source — no correction preference needed.
    qtext_by_branch, only_shared = _branched_items(s, "query_text")

    # Opt-in: the branch's corrected text, for the corrected_* metrics (an
    # uncorrected branch scores none — its artifacts simply lack the input).
    corrected_by_branch: Dict[str, ItemSet] = {}
    only_shared_corrected: Optional[ItemSet] = None
    if _corrected_metrics_enabled(s):
        corrected_by_branch, only_shared_corrected = _branched_items(
            s, "corrected_query_text", branch_exclusive=True
        )

    # Items dropped per-item upstream are excluded from the keyed report so a
    # placeholder/empty result never reaches a metric — the report measures survivors only.
    dropped_ids = s.drop_sink.all_dropped_ids()
    per_branch: Dict[str, Dict[str, ItemSet]] = {}
    branch_ids: Dict[str, set] = {}
    # Read the branches' retrieved sets through the aggregate's DECLARED bindings; binding
    # order = node order, so per-branch newest-producer-wins holds.
    for pid in s._producers("retrieved"):
        if not s.ctx.has(pid, "retrieved"):
            continue
        retrieved = s.ctx.get(pid, "retrieved")
        if not isinstance(retrieved, ItemSet):
            continue
        if dropped_ids:
            retrieved = retrieved.filter(lambda i, _v: i not in dropped_ids)
        branch = _branch_of(pid)
        branch_ids[branch] = set(retrieved.ids)
        keyed = retrieved.with_values(
            [_search_results_to_keys(r) for r in retrieved.values]
        )
        artifacts: Dict[str, ItemSet] = {"retrieved": keyed}
        if relevant is not None:
            artifacts["relevant_docs"] = relevant
        if reference is not None:
            artifacts["reference_transcription"] = reference
            # query_text for WER/CER: this branch's ASR hypothesis, else the lone shared one.
            qt = qtext_by_branch.get(branch, only_shared)
            if qt is not None:
                artifacts["query_text"] = qt
            cq = corrected_by_branch.get(branch, only_shared_corrected)
            if cq is not None:
                artifacts["corrected_query_text"] = cq
        scores = _branch_scores(s, artifacts)
        if scores:
            per_branch[branch] = scores
    return per_branch, branch_ids


def _attach_cross_stage_diagnostics(report, per_branch, baseline, s: RunState) -> None:
    """WER↔Recall correlation, Retrieval-WER-Impact, and the join warning."""
    # Cross-stage: per-branch Pearson(WER, Recall@5) — does worse ASR cost retrieval?
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
        report["join_warning"] = s.join_warning


def _stage_aggregate(s: RunState) -> None:
    """Terminal report builder: per-branch metrics + cross-branch deltas. Owns
    ``results['report']`` when present (supersedes the single-branch metrics-node report)."""

    per_branch, branch_ids = _collect_branch_artifacts(s)
    if not per_branch:
        return
    # Baseline branch for deltas: the oracle "ref" branch when present (Retrieval-WER-Impact
    # is measured against it), else "main", else the first by name.
    baseline = next(
        (b for b in ("ref", "main") if b in per_branch), sorted(per_branch)[0]
    )
    # Per-branch drops vs the full run (union of all branches' query ids): which items each
    # branch lost, so the shrinking paired denominator is auditable at report level.
    full_ids: set = set().union(*branch_ids.values()) if branch_ids else set()
    dropped_by_branch = {
        b: sorted(full_ids - ids) for b, ids in branch_ids.items() if full_ids - ids
    }
    report = build_report(
        per_branch,
        baseline=baseline,
        provenance=_run_provenance(s, dropped_by_branch),
        with_ci=getattr(s, "compute_confidence_intervals", False),
    )
    _attach_cross_stage_diagnostics(report, per_branch, baseline, s)
    _attach_report(s.results, report)
    logger.info(
        "aggregate: %d branch(es) %s, baseline=%s",
        len(per_branch),
        sorted(per_branch),
        baseline,
    )


def _asr_hypothesis(s: RunState) -> list:
    """The ASR hypothesis bound to this node; immutable (correction/optimization publish
    distinct names), so always the un-rewritten output WER/CER scores against."""
    return asr_hypothesis(s)


def _wer_cer_pair(pair):
    """Pure ``(WER, CER)`` for one ``(reference, hypothesis)`` pair. Top-level (not a closure)
    so the ``process`` CPU-stage backend can pickle it."""
    gt_text, hyp_text = pair
    return word_error_rate(gt_text, hyp_text), character_error_rate(gt_text, hyp_text)


def _asr_item_scores(s: RunState):
    """Per-item raw-ASR WER/CER lists — a pure, order-preserving fold (consumed by
    answer-gen/judge/trace)."""
    hyp = _asr_hypothesis(s)
    if not hyp:  # only a node fed by an ASR hypothesis carries WER/CER
        return [], []

    pairs = list(zip(_reference_transcriptions(s), hyp))
    scored = run_per_item(s, _wer_cer_pair, pairs)
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
    """Surfaces registry per-item scores as legacy flat bare keys; WER/CER are gated to
    ASR runs (audio_emb/audio_text carry no WER)."""
    for name, items in scores.items():
        key = _BARE_KEY_FOR.get(name)
        if key is None:
            continue
        if name in ("wer", "cer") and not asr_mode:
            continue
        vals = [float(v) for _, v in items]
        if vals:  # a metric with zero items must not surface as a real 0.0
            results[key] = sum(vals) / len(vals)


def _stage_transcription_metrics(s: RunState) -> None:
    """Comparison node: the ASR hypothesis (``query_text``) vs the spoken GT
    (``reference_transcription``) → per-item WER/CER, published KEYED
    (``per_query_wer``/``per_query_cer``) so the trace/diagnostics consumers id-join them,
    plus the summary artifact."""
    wer_scores, cer_scores = _asr_item_scores(s)
    # explicit None check: an empty-but-present query_text ItemSet must not
    # fall through to the reference set (ItemSet is falsy when empty)
    keyed = s.keyed_items("query_text")
    if keyed is None:
        keyed = s.keyed_items("reference_transcription")
    ids = keyed.ids if keyed is not None else []
    publish_keyed_or_plain(s, "per_query_wer", wer_scores, ids)
    publish_keyed_or_plain(s, "per_query_cer", cer_scores, ids)
    n = len(wer_scores)
    s.put_artifact(
        "transcription_scores",
        {
            "wer_mean": (sum(wer_scores) / n) if n else None,
            "cer_mean": (sum(cer_scores) / n) if n else None,
            "n": n,
        },
    )


def _stage_retrieval_metrics(s: RunState) -> None:
    """Comparison node: the retrieved docs vs the GT relevance (``relevant_docs``) →
    per-item recall@5, published KEYED (``per_query_recall5``) + the summary
    artifact. Relevance itself stays a bus read (``_relevant_from_bus``)."""
    _results_with_scores, retrieved_keys, ids = _retrieved_from_bus(s)
    all_relevant = _relevant_from_bus(s)
    recall5 = [recall_at_k(r, rel, 5) for r, rel in zip(retrieved_keys, all_relevant)]
    if len(recall5) < len(ids):
        # zip truncated (relevance shorter than retrieved): a positional ids[:n] slice
        # could attribute scores to the wrong queries — publish plain instead.
        logger.warning(
            "retrieval metrics: %d retrieved vs %d relevance entries; "
            "publishing per_query_recall5 without ids",
            len(ids), len(recall5),
        )
        publish_keyed_or_plain(s, "per_query_recall5", recall5, [])
    else:
        publish_keyed_or_plain(s, "per_query_recall5", recall5, ids)
    n = len(recall5)
    s.put_artifact(
        "retrieval_scores",
        {"recall@5_mean": (sum(recall5) / n) if n else None, "n": n},
    )


def _stage_metrics(s: RunState) -> None:
    """Report assembler: registry-native scalar report → results (single scalar source).
    Assembles from the registry + the per-item intermediates the typed metric nodes set."""
    _results_with_scores, retrieved_keys, _ids = _retrieved_from_bus(s)
    _t_step = time.perf_counter()
    s.cb("step_4_metrics", 0, s.total, "Computing metrics")

    results: RunResults = {}
    results["pipeline_mode"] = s.mode
    results["phased"] = True
    results["oracle_mode"] = s.oracle_mode

    _record_model_info(results, s)

    _attach_registry_report(s, results, retrieved_keys)

    # Diagnostics the registry report does not carry, from the per-item score artifacts the
    # typed metric nodes publish (get_artifact unwraps them to their published-order lists).
    if retrieved_keys:
        _ir_diagnostics(
            results,
            s,
            _relevant_from_bus(s),
            list(s.get_artifact("per_query_recall5", default=[])),
            list(s.get_artifact("per_query_wer", default=[])),
            retrieved_keys,
        )

    # Answer-quality aggregates from the answer_metrics node (mean_rougeL & co). They ride an
    # artifact because this assembler REBUILDS `results` — anything a pre-metrics node wrote
    # into s.results is discarded, which is how these were computed, logged, then lost.
    answer_scores = s.get_artifact("answer_scores", default=None)
    if isinstance(answer_scores, dict):
        results.update({k: v for k, v in answer_scores.items() if v is not None})

    s.stage_times["metrics_s"] = time.perf_counter() - _t_step
    s.results = results
    # Also published keyed by this node's own id: a graph with N `metrics` nodes would
    # otherwise each silently overwrite the shared `s.results` slot. `finalize` reads its
    # own bound producer via `s.get_artifact("metrics")` (binding-scoped, not this assignment).
    s.put_artifact("metrics", results)


def _build_keyed_artifacts(s: RunState, retrieved_keys: list) -> Dict[str, Any]:
    """Single-branch artifact assembly: positionally TRIMMED to the keyed query-id order
    (legacy zip-truncation leniency), deliberately different from the multi-branch path's
    id-join in ``_collect_branch_artifacts``. Returns {} when per-item identity is
    unavailable (no keyed publish / duplicate ids)."""

    # Per-item identity rides the keyed bus artifacts: the effective query text
    # when an asr node feeds this branch, the spoken reference otherwise. A plain (non-keyed) publish
    # means ids did not align — same no-op condition as the legacy all_query_ids check.
    keyed = s.keyed_items("query_text")
    if keyed is None:
        keyed = s.keyed_items("reference_transcription")
    if keyed is None:
        return {}
    ids = [str(i) for i in keyed.ids]
    if not ids or len(set(ids)) != len(ids):
        return {}

    n = len(ids)
    artifacts: Dict[str, ItemSet] = {}

    # Trimmed to the first n (legacy zip-truncation leniency — e.g. a checkpoint resume can
    # leave a longer, batch-overlapping hypotheses list); shorter lists are skipped.
    def _keyed(values, name):
        if values is None:
            return
        if len(values) < n:
            logger.warning(
                "metrics: %s has only %d values for %d query ids — skipping it (the "
                "metrics that consume it will be absent from the report)",
                name, len(values), n,
            )
            return
        if len(values) > n:
            logger.warning(
                "metrics: %s has %d values for %d query ids — trimming to the ids "
                "(a keyed publish should already align; check the producer)",
                name, len(values), n,
            )
        artifacts[name] = ItemSet(ids, list(values)[:n])

    # Bus-first: this node's own ASR hypothesis when fed one, else the reference — mixed
    # graphs (ASR arm next to text/latent arms) score each branch by what really fed it.
    reference = _reference_transcriptions(s)
    query_text = asr_hypothesis(s) or reference
    _keyed(query_text, "query_text")
    # ASR-quality reference = the spoken transcription, never question_text.
    _keyed(reference, "reference_transcription")
    if retrieved_keys:
        _keyed(retrieved_keys, "retrieved")
    relevance = _relevant_from_bus(s)
    if "retrieved" in artifacts:
        _keyed(relevance, "relevant_docs")
    # Opt-in corrected text: read bus-first; compute_metric id-aligns it, no trim needed.
    if _corrected_metrics_enabled(s) and "reference_transcription" in artifacts:
        corrected = _ctx_first(s, "corrected_query_text")
        if isinstance(corrected, ItemSet):
            artifacts["corrected_query_text"] = corrected
    if not artifacts:
        return {}
    # Exclude per-item drops so a placeholder/empty result never reaches a metric.
    dropped_ids = s.drop_sink.all_dropped_ids()
    if dropped_ids:
        artifacts = {
            name: items.filter(lambda i, _v: i not in dropped_ids)
            for name, items in artifacts.items()
        }
    return artifacts


def _attach_registry_report(
    s: RunState, results: "RunResults", retrieved_keys: list
) -> None:
    """Computes metrics via the registry and attaches a keyed ``report``; numerically
    identical to the legacy headline scalars (parity-proven) and additive (legacy keys
    untouched). No-op when query ids do not align.
    """

    artifacts = _build_keyed_artifacts(s, retrieved_keys)
    if not artifacts:
        return
    scores = _branch_scores(s, artifacts)
    if scores:
        _derive_bare_keys(results, scores, bool(asr_hypothesis(s)))
        # branch key + ``pipeline_mode`` carry the run's mode *label* (``s.mode``) — the node
        # set alone can't reconstruct it (audio_emb vs audio_text share a graph); only the
        # label is mode-derived, not behavior.
        report = build_report(
            {s.mode: scores}, provenance=_run_provenance(s),
            with_ci=getattr(s, "compute_confidence_intervals", False),
        )
        _attach_report(results, report)
