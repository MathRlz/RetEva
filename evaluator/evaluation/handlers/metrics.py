"""Metrics stage handlers: per-branch aggregate report + the single-branch metrics node.

Registers the ``metrics`` and ``aggregate`` handlers at import time. The metric **registry**
is the one place metrics are computed (Phase 4 L3 removed the legacy double-compute path); the
flat ``WER``/``MRR``/``Recall@k``/... keys are report-derived aliases (`_derive_bare_keys`).
Diagnostics the report does not carry + the per-item intermediates the rag/judge stages consume
are computed from the aligned per-item state. Raw ASR text is read from the ctx bus
(`raw_query_text`, T4) — no ``RunState`` mirror.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger
from ..helpers import _search_results_to_keys
from ..executor.state import RunState
from ..result_schema import RunResults
from .retrieval import _rehydrate_retrieved
from ...metrics import (
    first_relevant_rank_distribution,
    wer_recall_correlation,
    categorize_failures,
    embedding_alignment,
    word_error_rate,
    character_error_rate,
)
from ...metrics.ir import recall_at_k, reciprocal_rank, ndcg_at_k
from ...analysis.errors import analyze_retrieval_failures
from ...analysis.significance import bootstrap_confidence_interval

logger = get_logger(__name__)

# Bootstrap confidence-interval settings.
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_ITERATIONS = 1000
MIN_SAMPLES_FOR_CI = 20


def _branch_of(producer_id: str) -> str:
    """Branch label encoded in a node id (``retrieval@corr`` → ``corr``; else ``main``)."""
    return producer_id.split("@", 1)[1] if "@" in producer_id else "main"


def _ctx_first(s: RunState, name: str) -> Any:
    """The first ``ItemSet`` published anywhere for artifact ``name`` (shared GT)."""
    for pid, art in s.ctx.slots():
        if art == name:
            return s.ctx.get(pid, name)
    return None


def _collect_cache_stats(s: RunState) -> Optional[Dict[str, Any]]:
    """Per-stage cache hit/miss counts for the provenance block (T3): which artifacts were
    reused vs recomputed. Best-effort — a pipeline without stats is just omitted."""
    out: Dict[str, Any] = {}
    for attr, label in (
        ("asr_pipeline", "asr"),
        ("text_embedding_pipeline", "text_embedding"),
        ("audio_embedding_pipeline", "audio_embedding"),
    ):
        pipe = getattr(s, attr, None)
        if pipe is not None and hasattr(pipe, "get_cache_stats"):
            try:
                stats = pipe.get_cache_stats()
                if stats:
                    out[label] = stats
            except Exception:
                pass
    return out or None


def _llm_cost_summary() -> Optional[Dict[str, Any]]:
    """The run's accumulated LLM token/latency cost for the provenance block (T8)."""
    from ...llm.cost import COST

    return COST.summary()


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


@register_stage_handler("aggregate", self_timed=True)
def _stage_aggregate(s: RunState) -> None:
    """Terminal report builder (W6/A9): per-branch metrics + cross-branch deltas.

    Scans the RunContext for every ``retrieved`` producer (one per branch, per-producer
    keyed), scores each against the shared ground-truth artifacts via the metric registry,
    and builds the report with paired deltas vs a baseline branch. Owns ``results['report']``
    when present (supersedes the single-branch metrics-node report)."""
    from ..item_set import ItemSet
    from ..metric_registry import compute_metrics
    from ..aggregate import build_report

    relevant = _ctx_first(s, "relevant_docs")
    reference = _ctx_first(s, "reference_text")
    # Per-branch query_text (the effective query feeding retrieval) for WER/CER. Prefer a
    # correction producer over asr within a branch; remember the lone shared producer (M1).
    qtext_by_branch: Dict[str, ItemSet] = {}
    shared_qtext: Optional[ItemSet] = None
    qtext_slots = [(p, n) for p, n in s.ctx.slots() if n == "query_text"]
    for pid, _ in qtext_slots:
        val = s.ctx.get(pid, "query_text")
        if not isinstance(val, ItemSet):
            continue
        shared_qtext = val
        b = _branch_of(pid)
        if b not in qtext_by_branch or "correction" in pid:
            qtext_by_branch[b] = val
    only_shared = shared_qtext if len(qtext_slots) == 1 else None

    # Raw (pre-correction) ASR text per branch, for raw_wer/raw_cer (L1). Published by the
    # asr node; never republished by correction, so it stays the uncorrected hypothesis.
    raw_qtext_by_branch: Dict[str, ItemSet] = {}
    shared_raw_qtext: Optional[ItemSet] = None
    raw_slots = [(p, n) for p, n in s.ctx.slots() if n == "raw_query_text"]
    for pid, _ in raw_slots:
        val = s.ctx.get(pid, "raw_query_text")
        if not isinstance(val, ItemSet):
            continue
        shared_raw_qtext = val
        raw_qtext_by_branch[_branch_of(pid)] = val
    only_shared_raw = shared_raw_qtext if len(raw_slots) == 1 else None

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
            # query_text for WER/CER: this branch's own, else the lone shared producer.
            qt = qtext_by_branch.get(branch, only_shared)
            if qt is not None:
                artifacts["query_text"] = qt
            # raw_query_text for raw_wer/raw_cer (L1): branch's own asr output, else shared.
            rqt = raw_qtext_by_branch.get(branch, only_shared_raw)
            if rqt is not None:
                artifacts["raw_query_text"] = rqt
        scores = _drop_ir_if_disabled(s, compute_metrics(artifacts, collect_all=True))
        if scores:
            per_branch[branch] = scores
    if not per_branch:
        return
    # Baseline branch for deltas: the oracle "ref" branch when present (Retrieval-WER-Impact
    # is measured against it), else "main", else the first by name.
    baseline = next(
        (b for b in ("ref", "main") if b in per_branch), sorted(per_branch)[0]
    )
    from ..provenance import build_provenance

    seed = getattr(getattr(s.config, "audio_synthesis", None), "seed", None)
    # Per-branch drops vs the full run (union of all branches' query ids): which items each
    # branch lost, so the shrinking paired denominator (S1) is auditable at report level (S2).
    full_ids: set = set().union(*branch_ids.values()) if branch_ids else set()
    dropped_by_branch = {
        b: sorted(full_ids - ids) for b, ids in branch_ids.items() if full_ids - ids
    }
    provenance = build_provenance(
        s.config,
        seed=seed,
        timing=dict(s.stage_times),
        dropped_by_branch=dropped_by_branch or None,
        dropped_by_node=dict(s.drop_sink.by_node) or None,
        cache_stats=_collect_cache_stats(s),
        cost=_llm_cost_summary(),
    )
    report = build_report(
        per_branch, baseline=baseline, provenance=provenance, with_ci=True
    )
    # Cross-stage: per-branch Pearson(WER, Recall@5) — does worse ASR cost retrieval? (M2)
    from ...metrics.diagnostics import wer_recall_correlation

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
    # Retrieval-WER-Impact: recall lost vs the oracle baseline = Recall(ref) − Recall(branch),
    # per recall@k (the degradation ASR/correction imposes on retrieval). M3.
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
    if impact:
        report["retrieval_wer_impact"] = impact
    if s.disable_ir_metrics and s.join_warning:
        report["join_warning"] = s.join_warning  # B5: why IR metrics are absent
    s.results["report"] = report
    from ..aggregate import flatten_report

    s.results.update(flatten_report(report))  # surface to leaderboard (M4)
    logger.info(
        "aggregate: %d branch(es) %s, baseline=%s",
        len(per_branch),
        sorted(per_branch),
        baseline,
    )


def _record_model_info(results: "RunResults", s: RunState) -> None:
    """Record model names + audio<->text embedding alignment (metadata, not metrics)."""
    if s.mode in ("asr_text_retrieval", "asr_only"):
        results["asr"] = s.asr_pipeline.model.name()
    if s.mode == "asr_text_retrieval":
        results["embedder"] = s.text_embedding_pipeline.model.name()
    elif s.mode == "audio_text_retrieval":
        results["audio_embedder"] = s.audio_embedding_pipeline.model.name()
        results["text_embedder"] = s.text_embedding_pipeline.model.name()
        try:
            alignment = embedding_alignment(
                s.audio_emb_for_alignment, s.text_emb_for_alignment
            )
            if alignment is not None:
                results["embedding_alignment"] = alignment
                logger.info(
                    "Embedding alignment - cosine mean=%.4f std=%.4f",
                    alignment["audio_text_cosine_mean"],
                    alignment["audio_text_cosine_std"],
                )
        except Exception as exc:
            logger.warning("Embedding alignment computation failed: %s", exc)


def _raw_query_text(s: RunState) -> list:
    """Raw (pre-correction) ASR text from the bus (``raw_query_text``, published by the asr
    node, T4). Falls back to ``query_text`` for non-ASR modes / direct callers where the asr
    node didn't publish it (there raw == effective query)."""
    return list(
        s.get_artifact("raw_query_text", default=s.get_artifact("query_text", []))
    )


def _asr_item_scores(s: RunState):
    """Per-item raw-ASR WER/CER lists (consumed by the answer-gen / judge / trace stages)."""
    wer_scores: List[float] = []
    cer_scores: List[float] = []
    if s.mode not in ("asr_text_retrieval", "asr_only"):
        return wer_scores, cer_scores
    for gt_text, hyp_text in zip(s.all_ground_truth, _raw_query_text(s)):
        wer_scores.append(word_error_rate(gt_text, hyp_text))
        cer_scores.append(character_error_rate(gt_text, hyp_text))
    return wer_scores, cer_scores


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


def _derive_bare_keys(results: "RunResults", scores: Dict[str, Any], mode: str) -> None:
    """Surface the registry per-item scores as the legacy flat bare keys (report-derived
    aliases, L3) — the registry is the single scalar source. WER/CER are gated to ASR modes
    (legacy parity: audio_emb/audio_text carry no WER). Means are identical to the deleted
    legacy scalars (smoke-confirmed: single-branch WER 0.6541 == registry wer)."""
    asr_mode = mode in ("asr_text_retrieval", "asr_only")
    for name, items in scores.items():
        key = _BARE_KEY_FOR.get(name)
        if key is None:
            continue
        if name in ("wer", "cer") and not asr_mode:
            continue
        vals = [float(v) for _, v in items]
        results[key] = (sum(vals) / len(vals)) if vals else 0.0


def _ir_diagnostics(results, s, all_relevant, recall5, wer_scores) -> None:
    """Retrieval diagnostics not carried by the registry report (correlation, rank
    distribution, failure rate/analysis, flat CIs). Same functions/values as the old path.
    """
    corr = wer_recall_correlation(wer_scores, recall5)
    if corr is not None:
        results["wer_recall5_correlation"] = corr

    rank_dist, failure_rate = first_relevant_rank_distribution(
        s.all_retrieved, all_relevant
    )
    results["first_relevant_rank_distribution"] = rank_dist
    results["retrieval_failure_rate"] = failure_rate

    if s.trace_limit > 0:
        _attach_failure_analysis(
            results, s, all_relevant, rank_dist, recall5, wer_scores
        )

    if s.compute_confidence_intervals and len(s.all_retrieved) >= MIN_SAMPLES_FOR_CI:
        ci_inputs = {
            "MRR": [
                reciprocal_rank(r, rel) for r, rel in zip(s.all_retrieved, all_relevant)
            ],
            "Recall@5": recall5,
            "NDCG@5": [
                ndcg_at_k(r, rel, 5) for r, rel in zip(s.all_retrieved, all_relevant)
            ],
        }
        for name, ci_scores in ci_inputs.items():
            try:
                results[f"{name}_ci"] = bootstrap_confidence_interval(
                    ci_scores, alpha=BOOTSTRAP_ALPHA, n_bootstrap=BOOTSTRAP_ITERATIONS
                )
            except Exception as exc:
                logger.warning("CI computation failed for %s: %s", name, exc)


def _attach_failure_analysis(
    results, s, all_relevant, rank_dist, recall5, wer_scores
) -> None:
    """Retrieval failure-mode decomposition (only when traces are enabled)."""
    query_texts = (
        s.all_hypotheses if s.mode == "asr_text_retrieval" else s.all_ground_truth
    )
    details = [
        {
            "query": query_texts[i],
            "retrieved": s.all_retrieved[i],
            "relevant": all_relevant[i],
        }
        for i in range(len(s.all_retrieved))
    ]
    analysis = analyze_retrieval_failures({"details": details}, top_k=s.k)
    if wer_scores and len(wer_scores) == len(recall5):
        corpus_doc_ids = None
        if hasattr(s.dataset, "get_corpus"):
            try:
                corpus_doc_ids = {
                    str(doc.get("doc_id", "")) for doc in s.dataset.get_corpus()
                }
                corpus_doc_ids.discard("")
            except Exception:
                corpus_doc_ids = None
        analysis["failure_categories"] = categorize_failures(
            wer_scores,
            recall5,
            rank_dist,
            all_relevant=all_relevant,
            corpus_doc_ids=corpus_doc_ids,
        )
    results["retrieval_failure_analysis"] = analysis


@register_stage_handler("metrics", self_timed=True)
def _stage_metrics(s: RunState) -> None:
    """Terminal node: registry-native metrics -> results (single scalar source, L3).

    The metric registry (via ``_attach_registry_report`` -> ``_derive_bare_keys``) is the
    single place the scalar metrics are computed; the flat WER/MRR/Recall@k/... keys are
    report-derived aliases. Per-item intermediates the rag/judge stages consume + diagnostics
    the report does not carry are computed here from the aligned per-item state (same
    functions/values as the removed legacy path)."""
    _rehydrate_retrieved(s)
    _t_phase = time.perf_counter()
    s.cb("phase_4_metrics", 0, s.total, "Computing metrics")

    results: RunResults = RunResults()
    results["pipeline_mode"] = s.mode
    results["phased"] = True
    results["oracle_mode"] = s.oracle_mode

    _record_model_info(results, s)

    # Per-item intermediates consumed by the answer-gen / judge / trace stages.
    s.wer_scores, s.cer_scores = _asr_item_scores(s)
    if s.mode != "asr_only":
        all_relevant = s.all_relevance or [{str(gt): 1} for gt in s.all_ground_truth]
        recall5 = [
            recall_at_k(r, rel, 5) for r, rel in zip(s.all_retrieved, all_relevant)
        ]
        s.metrics_all_relevant = all_relevant
        s.per_query_recall5 = recall5

    # Registry report + report-derived flat bare keys (the single scalar source).
    _attach_registry_report(s, results)

    # Diagnostics the registry report does not carry (computed from per-item state).
    if s.mode != "asr_only" and s.all_retrieved:
        _ir_diagnostics(
            results, s, s.metrics_all_relevant, s.per_query_recall5, s.wer_scores
        )

    s.stage_times["metrics_s"] = time.perf_counter() - _t_phase
    s.results = results


def _attach_registry_report(s: RunState, results: "RunResults") -> None:
    """Compute metrics via the registry + aggregate and attach a keyed ``report`` (W4b).

    Builds keyed ``ItemSet``s from the run's aligned per-item state and runs the metric
    registry → ``build_report``. Numerically identical to the legacy headline scalars
    (parity-proven), but keyed + branch-shaped — the basis for cross-branch deltas (W6).
    Additive: the legacy result keys are untouched. No-op when query ids do not align.
    """
    ids = [str(i) for i in (s.all_query_ids or [])]
    if not ids or len(set(ids)) != len(ids):
        return
    from ..item_set import ItemSet
    from ..metric_registry import compute_metrics
    from ..aggregate import build_report

    n = len(ids)
    artifacts: Dict[str, ItemSet] = {}

    # Align each per-item list to the query-id order. Lists at least as long as ``ids`` are
    # trimmed to the first n (matches the legacy zip-truncation leniency — e.g. a checkpoint
    # resume can leave a longer, batch-overlapping hypotheses list); shorter lists are skipped.
    def _keyed(values, name):
        if values is not None and len(values) >= n:
            artifacts[name] = ItemSet(ids, list(values)[:n])

    query_text = (
        s.all_hypotheses if s.mode == "asr_text_retrieval" else s.all_ground_truth
    )
    _keyed(query_text, "query_text")
    # Raw ASR text for raw_wer/raw_cer (L1). Read the pre-correction snapshot from the bus
    # (raw_query_text, published by the asr node, T4); falls back to the effective query for
    # non-ASR modes / direct callers (where raw == query_text).
    _keyed(_raw_query_text(s) or query_text, "raw_query_text")
    _keyed(s.all_ground_truth, "reference_text")
    if s.all_retrieved:
        _keyed(s.all_retrieved, "retrieved")
    relevance = s.all_relevance or [{str(gt): 1} for gt in s.all_ground_truth]
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
    scores = _drop_ir_if_disabled(s, compute_metrics(artifacts, collect_all=True))
    if scores:
        _derive_bare_keys(results, scores, s.mode)
        from ..provenance import build_provenance

        seed = getattr(getattr(s.config, "audio_synthesis", None), "seed", None)
        provenance = build_provenance(
            s.config,
            seed=seed,
            timing=dict(s.stage_times),
            dropped_by_node=dict(s.drop_sink.by_node) or None,
            cache_stats=_collect_cache_stats(s),
            cost=_llm_cost_summary(),
        )
        report = build_report({s.mode: scores}, provenance=provenance, with_ci=True)
        results["report"] = report
        from ..aggregate import flatten_report

        results.update(flatten_report(report))  # surface to leaderboard (M4)
