"""Query-side stage handlers: correction, augmentation, LLM optimization.

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X7). Each handler
registers itself via ``@register_stage_handler`` at import time.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from tqdm import tqdm

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger
from ..helpers import _search_results_to_keys
from ..executor.state import RunState
from ...models.retrieval.query.optimization import (
    rewrite_query,
    generate_hypothetical_document,
    decompose_query,
    generate_multi_queries,
    combine_retrieval_results,
)

logger = get_logger(__name__)


def _run_query_optimization_phase(
    mode,
    all_hypotheses,
    all_ground_truth,
    all_results_with_scores,
    query_opt_config,
    text_embedding_pipeline,
    retrieval_pipeline,
    k,
):
    """Apply query optimization to the per-query texts before retrieval.

    rewrite/hyde transform the query text in place. decompose/multi_query expand
    each query, retrieve per sub-query, and merge — fully bypassing the normal
    embedding + retrieval phases.

    Returns (all_hypotheses, all_ground_truth, all_results_with_scores,
    all_retrieved, bypassed). all_retrieved is a list only when bypassed, else None.
    """
    method = query_opt_config.method
    query_texts_src = (
        all_hypotheses if mode == "asr_text_retrieval" else all_ground_truth
    )
    all_retrieved = None
    bypassed = False

    if method in ("rewrite", "hyde"):
        fn = rewrite_query if method == "rewrite" else generate_hypothetical_document
        optimized = []
        # NOTE: rewrite_query supports context-aware iterative refinement via its
        # `context=` arg (config.use_initial_context / context_top_k), but this single
        # pass calls it without context — that capability is not wired here yet.
        # Wiring it would require an initial retrieval to feed top-k docs back in.
        for q in tqdm(query_texts_src, desc=f"Query optimization ({method})"):
            try:
                optimized.append(fn(q, query_opt_config))
            except Exception as exc:
                logger.warning("Query optimization failed for %r: %s", q[:80], exc)
                optimized.append(q)
        if mode == "asr_text_retrieval":
            all_hypotheses = optimized
        else:
            all_ground_truth = optimized
        logger.info(
            "Query optimization complete: %d queries transformed", len(optimized)
        )

    elif method in ("decompose", "multi_query"):
        if text_embedding_pipeline is None or retrieval_pipeline is None:
            logger.warning(
                "Query optimization %s requires text_embedding + retrieval pipeline — skipping",
                method,
            )
        else:
            fn = decompose_query if method == "decompose" else generate_multi_queries
            all_results_with_scores = []
            all_retrieved = []
            for q in tqdm(query_texts_src, desc=f"Query optimization ({method})"):
                try:
                    sub_qs = fn(q, query_opt_config)
                except Exception as exc:
                    logger.warning("Query expansion failed for %r: %s", q[:80], exc)
                    sub_qs = [q]
                sub_embs = np.array(text_embedding_pipeline.process_batch(sub_qs))
                sub_results = retrieval_pipeline.search_batch(
                    sub_embs, k, query_texts=sub_qs
                )
                merged = combine_retrieval_results(
                    sub_results,
                    strategy=query_opt_config.combine_strategy,
                    k=k,
                )
                all_results_with_scores.append(merged)
                all_retrieved.append(_search_results_to_keys(merged))
            bypassed = True
            logger.info(
                "Query optimization (%s) complete: %d queries expanded and retrieved",
                method,
                len(all_results_with_scores),
            )

    return (
        all_hypotheses,
        all_ground_truth,
        all_results_with_scores,
        all_retrieved,
        bypassed,
    )


def _node_correction_config(s: "RunState") -> Any:
    """The correction config for the current node: a transient `QueryCorrectionConfig` built
    from `current_node.params` (per-branch divergence, R2) when present, else the global one.
    """
    params = getattr(s.current_node, "params", None) or {}
    if not params:
        return s.query_correction_config
    from ...config.query_correction import QueryCorrectionConfig

    return QueryCorrectionConfig(
        enabled=bool(params.get("enabled", True)),
        method=params.get("method", "rule"),
        replacements=params.get("replacements", {}) or {},
        use_default_rules=bool(params.get("use_default_rules", True)),
        kb_terms=params.get("kb_terms", []) or [],
        kb_max_distance=int(params.get("kb_max_distance", 1)),
    )


@register_stage_handler("query_correction", time_key="correction_s")
def _stage_query_correction(s: "RunState") -> None:
    """Post-ASR query correction node (rule-based domain repair). Rewrites the query
    hypotheses in place and republishes ``query_text`` so the next consumer (query
    optimization / text embedding) reads the corrected text.

    Per-branch divergence (R2): a node's ``params`` build a transient ``QueryCorrectionConfig``
    so correction can be enabled on *one* branch only (the `corr` branch) while others no-op.
    """
    cfg = _node_correction_config(s)
    if cfg is None or not getattr(cfg, "enabled", False):
        return
    if s.query_opt_bypassed:
        return
    from ..query_correction import correct_query_texts, correction_diff
    from ..item_set import ItemSet

    texts = list(s.get_artifact("query_text", default=s.all_hypotheses))
    corrected = correct_query_texts(texts, cfg)
    n_changed = sum(1 for a, b in zip(texts, corrected) if a != b)
    s.all_hypotheses = corrected
    # Republish query_text (keyed when ids align, W2) so downstream embeds the corrected text.
    ids = [str(i) for i in (s.all_query_ids or range(len(corrected)))]
    if len(ids) == len(corrected) and len(set(ids)) == len(ids):
        s.put_items("query_text", ItemSet(ids, corrected))
        # Correction-diff artifact (C1): what the corrector changed, per query — the evidence
        # behind the asr-vs-asr+correction comparison.
        s.put_items("correction_diff", ItemSet(ids, correction_diff(texts, corrected)))
    else:
        s.put_artifact("query_text", corrected)
        s.put_artifact("correction_diff", correction_diff(texts, corrected))
    logger.info(
        "Query correction complete: %d/%d texts changed", n_changed, len(corrected)
    )


@register_stage_handler("augmenter", time_key="correction_s")
def _stage_augmenter(s: "RunState") -> None:
    """Robustness perturbation node (C2): corrupt each query text deterministically (seeded per
    item) — ASR-confusion homophones + dangerous dose/unit swaps — then republish ``query_text``
    so downstream embeds the corrupted query. A branch-divergence source: one branch clean, one
    augmented, and the cross-branch delta quantifies robustness. Params (per node/branch) drive
    which perturbations apply."""
    from ...pipeline.text_augmentation import TextAugmentConfig, TextAugmenter
    from ..item_set import ItemSet
    from ..provenance import DEFAULT_SEED, item_seed

    params = getattr(s.current_node, "params", None) or {}
    node_id = getattr(s.current_node, "id", "augmenter")
    cfg = TextAugmentConfig(
        homophones=bool(params.get("homophones", True)),
        unit_corruption=bool(params.get("unit_corruption", True)),
        char_swap_prob=float(params.get("char_swap_prob", 0.0)),
        max_edits=int(params.get("max_edits", 2)),
    )
    augmenter = TextAugmenter(cfg)
    seed = getattr(getattr(s.config, "audio_synthesis", None), "seed", None)
    base_seed = int(seed) if seed is not None else DEFAULT_SEED

    items = s.get_items("query_text", default=None)
    if isinstance(items, ItemSet):
        ids, texts = items.ids, items.values
    else:
        texts = list(s.get_artifact("query_text", default=s.all_hypotheses))
        ids = [str(i) for i in (s.all_query_ids or range(len(texts)))]

    augmented = [
        augmenter.augment(t, seed=item_seed(base_seed, qid, node_id, 0))
        for qid, t in zip(ids, texts)
    ]
    n_changed = sum(1 for a, b in zip(texts, augmented) if a != b)
    s.all_hypotheses = augmented
    if len(ids) == len(augmented) and len(set(ids)) == len(ids):
        s.put_items("query_text", ItemSet(ids, augmented))
    else:
        s.put_artifact("query_text", augmented)
    logger.info(
        "augmenter '%s': %d/%d query texts perturbed",
        node_id,
        n_changed,
        len(augmented),
    )


@register_stage_handler("query_optimization", time_key="query_opt_s")
def _stage_query_optimization(s: "RunState") -> None:
    """LLM query optimization node (rewrite/HyDE/multi-query). Reads its pipelines +
    config from state; may bypass downstream embedding/retrieval (multi-query path)."""
    if s.query_opt_config is None:
        return
    _run_query_optimization_stage(
        s, s.query_opt_config, s.text_embedding_pipeline, s.retrieval_pipeline
    )
    # Publish the optimized query text so text_embedding (bound to the latest producer)
    # embeds it rather than the raw ASR hypotheses (R4d).
    s.put_artifact("query_text", s.all_hypotheses)


def _run_query_optimization_stage(
    state: "RunState",
    query_opt_config,
    text_embedding_pipeline,
    retrieval_pipeline,
) -> None:
    """Query-optimization node: optimize query texts (may bypass embedding+retrieval)."""
    state.cb(
        "phase_1_5_query_opt",
        0,
        state.total,
        f"Phase 1.5: Query optimization ({query_opt_config.method})",
    )
    _t = time.perf_counter()
    (
        state.all_hypotheses,
        state.all_ground_truth,
        state.all_results_with_scores,
        _qopt_retrieved,
        state.query_opt_bypassed,
    ) = _run_query_optimization_phase(
        state.mode,
        state.all_hypotheses,
        state.all_ground_truth,
        state.all_results_with_scores,
        query_opt_config,
        text_embedding_pipeline,
        retrieval_pipeline,
        state.k,
    )
    if state.query_opt_bypassed:
        state.all_retrieved = _qopt_retrieved
        # The bypass produced final retrieval results in-handler (embedding+retrieval are
        # skipped downstream). Publish them to the ctx bus under this node — so the keyed
        # report path (the branched `aggregate`, which scans ctx `retrieved` slots) sees them
        # instead of relying on the RunState fallback (W2 / qopt-bypass gap fix).
        from .retrieval import _publish_retrieved

        _publish_retrieved(state, state.all_results_with_scores)
    state.stage_times["query_opt_s"] += time.perf_counter() - _t
