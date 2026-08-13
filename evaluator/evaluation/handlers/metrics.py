"""Metrics stage handlers: per-branch aggregate report + the single-branch metrics node.

Registers the ``metrics`` and ``aggregate`` handlers at import time. The metric **registry**
is the one place metrics are computed; the
flat ``WER``/``MRR``/``Recall@k``/... keys are report-derived aliases (`_derive_bare_keys`).
Diagnostics the report does not carry + the per-item intermediates the rag/judge stages consume
are computed from the aligned per-item state. WER/CER score the ASR hypothesis (`query_text`,
immutable) — no separate raw_query_text snapshot.

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


def _branched_items(s: "RunState", artifact: str, *, branch_exclusive: bool = False):
    """``(by_branch, only_shared)`` for an artifact's bound-and-published ItemSets: each
    branch's own value, plus the lone value when a single producer serves every branch —
    None once the artifact is branched, so a branch never reads another branch's copy.
    binding-scoped (the aggregate declares + binds what it reads).

    A lone producer is normally shared: CSE collapses an identical prefix across branches
    into ONE node (``asr@asr`` feeding every branch), so a branch-namespaced id does not
    imply exclusivity. ``branch_exclusive`` marks an artifact whose ABSENCE in a branch is
    meaningful — ``corrected_query_text``: a branch with correction disabled has no
    corrector node at all, so a lone namespaced publisher belongs to its own branch only.
    Sharing it would score the uncorrected baseline as if it had been corrected, making a
    single-corrector comparison read as exactly zero effect (found by an Ollama smoke run).
    """
    from ..item_set import ItemSet

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
    """IR (retrieval) metric names — dropped from the report when the dataset join is disjoint
    (B5), since with no relevant doc in the corpus they would be a misleading 0, not 'n/a'.
    """
    return name in ("mrr", "map") or name.startswith(("recall@", "precision@", "ndcg@"))


def _drop_ir_if_disabled(s: "RunState", scores: Dict[str, Any]) -> Dict[str, Any]:
    """Remove IR metrics from a branch's score set when `s.disable_ir_metrics` (B5)."""
    if not s.disable_ir_metrics:
        return scores
    return {k: v for k, v in scores.items() if not _is_ir_metric(k)}


# ── Shared reduction utilities (M7): one registry-scoring + provenance + attach path
# used by both the single-branch metrics node and the multi-branch aggregate. ──


def _corrected_metrics_enabled(s: "RunState") -> bool:
    """C7 opt-in: corrected-text + drug-aware (rx) metrics fire only when the run's
    query_correction config sets ``corrected_metrics`` (reports/m1c otherwise unchanged)."""
    return bool(
        getattr(s.query_correction_config, "corrected_metrics", False)
    )


def _branch_scores(s: "RunState", artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Score one branch's artifacts via the metric registry (collect-all, or the config's
    `metrics:` allowlist when set — B1), with the B5 IR gate applied — the single scoring
    call both report paths share. The C7 opt-in metrics are computed only when the run
    enables ``corrected_metrics`` (or names them in the allowlist). Lineage variants are
    rolled up to parent level here so every consumer (report, retrieval-WER-impact, the
    WER↔recall correlation) pairs and reduces on the cluster-safe unit."""
    from ..aggregate import rollup_variants
    from ..metric_registry import compute_metrics

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
    """Attach the keyed report + surface its flattened keys to the leaderboard (M4)."""
    from ..aggregate import flatten_report

    results["report"] = report
    results.update(flatten_report(report))


def attach_judge_metrics(s: "RunState") -> None:
    """Merge the LLM-judge per-query scores into the report as registry metrics (J3).

    The judge node runs *downstream* of the report assembler (and, in a branched run, of the
    terminal ``aggregate``), so its keyed ``judge_*`` ItemSets are not on the bus when the
    report is first scored. This terminal pass — called from the finalize node, after every
    report producer has run — scores those ItemSets through the SAME metric registry + reducer
    and merges the scalars into ``report['branches'][<branch>]`` + the flat leaderboard keys.
    No-op when the judge did not run (no ``judge_*`` ItemSets published), so a judge-off run
    (the parity default) is byte-identical."""
    report = s.results.get("report") if s.results else None
    if not isinstance(report, dict):
        return
    from ..item_set import ItemSet

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
    from ..metric_registry import compute_metrics
    from ..aggregate import flatten_report, reduce_scores

    # collect_all → exactly the judge_* metrics fire (only judge artifacts are present);
    # the config's `metrics:` allowlist (B1) applies here too.
    scores = compute_metrics(judge_arts, only=s.metric_allowlist)
    if not scores:
        return
    # The judge runs once globally (single query_traces); attach to the run's mode branch when
    # present (single-branch), else the first branch (a branched run's primary).
    branches = report.setdefault("branches", {})
    target = s.mode if s.mode in branches else next(iter(branches), s.mode)
    branch = branches.setdefault(target, {})
    for name, item_scores in scores.items():
        branch[name] = reduce_scores(
            item_scores, with_ci=getattr(s, "compute_confidence_intervals", False)
        )
    s.results.update(flatten_report(report))


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


def _collect_branch_artifacts(s: RunState):
    """Assemble + score each branch's artifacts (id-JOINED, never trimmed — the
    multi-branch path's alignment semantics). Returns ``(per_branch, branch_ids)``."""
    from ..item_set import ItemSet

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
    qtext_by_branch, only_shared = _branched_items(s, "query_text")

    # C7 (opt-in): the branch's corrected text, for the corrected_* metrics (an
    # uncorrected branch scores none — its artifacts simply lack the input).
    corrected_by_branch: Dict[str, ItemSet] = {}
    only_shared_corrected: Optional[ItemSet] = None
    if _corrected_metrics_enabled(s):
        corrected_by_branch, only_shared_corrected = _branched_items(
            s, "corrected_query_text", branch_exclusive=True
        )

    # Items dropped per-item upstream (T1) are excluded from the keyed report so a
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
        # retrieved values are per-query (payload, score) lists → doc-id keys for IR metrics
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
    # Cross-stage: per-branch Pearson(WER, Recall@5) — does worse ASR cost retrieval? (M2)
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


def _stage_aggregate(s: RunState) -> None:
    """Terminal report builder (W6/A9): per-branch metrics + cross-branch deltas.

    Reads every bound ``retrieved`` producer (one per branch, per-producer keyed),
    scores each against the shared ground-truth artifacts via the metric registry,
    and builds the report with paired deltas vs a baseline branch. Owns ``results['report']``
    when present (supersedes the single-branch metrics-node report)."""
    from ..aggregate import build_report

    per_branch, branch_ids = _collect_branch_artifacts(s)
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

    The per-item WER/CER map is a pure, CPU-bound, order-preserving fold."""
    if not asr_ran(s):  # only ASR modes carry WER/CER
        return [], []
    from ..executor.cpu_parallel import run_per_item

    pairs = list(zip(_reference_transcriptions(s), _asr_hypothesis(s)))
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
    """Surface the registry per-item scores as the legacy flat bare keys (report-derived
    aliases) — the registry is the single scalar source. WER/CER are gated to ASR runs
    (``asr_mode``; audio_emb/audio_text carry no WER)."""
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
    """Report assembler: registry-native scalar report → results (single scalar source, L3).

    The per-comparison computation lives in the typed ``transcription_metrics`` /
    ``retrieval_metrics`` nodes; this node assembles the report from the registry (the
    flat WER/MRR/Recall@k/... keys) + the per-item intermediates those nodes set."""
    _results_with_scores, retrieved_keys, _ids = _retrieved_from_bus(s)
    _t_phase = time.perf_counter()
    s.cb("phase_4_metrics", 0, s.total, "Computing metrics")

    results: RunResults = {}
    results["pipeline_mode"] = s.mode
    results["phased"] = True
    results["oracle_mode"] = s.oracle_mode

    _record_model_info(results, s)

    # Registry report + report-derived flat bare keys (the single scalar source).
    _attach_registry_report(s, results, retrieved_keys)

    # Diagnostics the registry report does not carry, from the per-item score artifacts the
    # typed metric nodes publish (get_artifact unwraps them to their published-order lists).
    if retrieval_ran(s) and retrieved_keys:
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

    s.stage_times["metrics_s"] = time.perf_counter() - _t_phase
    s.results = results


def _build_keyed_artifacts(s: RunState, retrieved_keys: list) -> Dict[str, Any]:
    """The single-branch report path's artifact assembly: positionally TRIMMED to the
    keyed query-id order (the legacy zip-truncation leniency — deliberately different
    from the multi-branch path's id-join in ``_collect_branch_artifacts``).

    Returns {} when per-item identity is unavailable (no keyed publish / duplicate ids)
    — the caller's no-op condition."""
    from ..item_set import ItemSet

    # Per-item identity rides the keyed bus artifacts (M1d-2): the effective query text
    # in ASR modes, the spoken reference in audio modes. A plain (non-keyed) publish
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

    # Align each per-item list to the query-id order. Lists at least as long as ``ids`` are
    # trimmed to the first n (matches the legacy zip-truncation leniency — e.g. a checkpoint
    # resume can leave a longer, batch-overlapping hypotheses list); shorter lists are skipped.
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

    # Bus-first: the ASR hypothesis in ASR modes (query_text is immutable, so
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
    _keyed(reference, "reference_transcription")
    if retrieved_keys:
        _keyed(retrieved_keys, "retrieved")
    relevance = _relevant_from_bus(s)
    if "retrieved" in artifacts:
        _keyed(relevance, "relevant_docs")
    # C7 (opt-in): the corrected text for the corrected_* metrics. Read off the bus — the
    # metrics node reads it bus-first; compute_metric id-aligns, no trim.
    if _corrected_metrics_enabled(s) and "reference_transcription" in artifacts:
        corrected = _ctx_first(s, "corrected_query_text")
        if isinstance(corrected, ItemSet):
            artifacts["corrected_query_text"] = corrected
    if not artifacts:
        return {}
    # Exclude per-item drops (T1) so a placeholder/empty result never reaches a metric.
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
    """Compute metrics via the registry + aggregate and attach a keyed ``report`` (W4b).

    Builds keyed ``ItemSet``s from the run's aligned per-item state and runs the metric
    registry → ``build_report``. Numerically identical to the legacy headline scalars
    (parity-proven), but keyed + branch-shaped — the basis for cross-branch deltas (W6).
    Additive: the legacy result keys are untouched. No-op when query ids do not align.
    """
    from ..aggregate import build_report

    artifacts = _build_keyed_artifacts(s, retrieved_keys)
    if not artifacts:
        return
    scores = _branch_scores(s, artifacts)
    if scores:
        _derive_bare_keys(results, scores, asr_ran(s))
        # The report branch key + ``pipeline_mode`` echo carry the run's mode *label* — the
        # executed graph's identity (``s.mode``), which the node set alone can't reconstruct
        # (audio_emb vs audio_text share a graph). Behavior is graph-derived; only the label is.
        report = build_report(
            {s.mode: scores}, provenance=_run_provenance(s),
            with_ci=getattr(s, "compute_confidence_intervals", False),
        )
        _attach_report(results, report)
