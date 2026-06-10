"""Retrieval stage handlers: candidate generation + rerank/refinement.

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X6). Each handler
registers itself via ``@register_stage_handler`` at import time.
"""

from __future__ import annotations

import numpy as np

from ..stage_registry import register_stage_handler
from ...logging_config import get_logger, TimingContext
from ..helpers import _payload_to_key, _search_results_to_keys
from ..item_isolation import isolate_batch
from ..executor.state import RunState
from ..executor.node_pipeline import _node_reranking
from ._common import DEBUG_SAMPLE_LIMIT, MATCH_SYMBOL, MISS_SYMBOL

logger = get_logger(__name__)


def _log_retrieval_debug(
    retrieval_pipeline,
    all_retrieved,
    all_ground_truth,
    all_relevance,
    all_hypotheses,
    all_query_ids,
    all_embeddings,
    mode,
    k,
) -> None:
    """Verbose per-query retrieval inspection (dense + no-rerank only).

    Pure diagnostic logging extracted from the retrieval phase; has no effect on
    results. Logs DB vector stats and a re-search trace for the first 3 queries.
    """
    # Debug: Check vector store properties
    if (
        hasattr(retrieval_pipeline.vector_store, "vectors")
        and retrieval_pipeline.vector_store.vectors is not None
    ):
        db_vectors = retrieval_pipeline.vector_store.vectors
        logger.info(f"DB vectors shape: {db_vectors.shape}")
        logger.info(
            f"DB vectors stats - mean: {db_vectors.mean():.4f}, std: {db_vectors.std():.4f}"
        )
        logger.info(
            f"DB vectors norms - mean: {np.linalg.norm(db_vectors, axis=1).mean():.4f}"
        )
        logger.info(
            f"DB payload count: {len(getattr(retrieval_pipeline.vector_store, 'payloads', []))}"
        )

    if not (len(all_retrieved) > 0 and len(all_ground_truth) > 0):
        return

    can_debug_with_search = (
        retrieval_pipeline.strategy_config.core.mode == "dense"
        and retrieval_pipeline.strategy_config.reranking.mode == "none"
    )
    if not can_debug_with_search:
        logger.info(
            "Skipping dense-only debug re-search for current retrieval mode/reranker"
        )
        return

    logger.info("RETRIEVAL DEBUG SAMPLE (first %d queries):", DEBUG_SAMPLE_LIMIT)

    # Build a doc_id → text lookup from vector store payloads for ground-truth display
    _db_payloads = getattr(retrieval_pipeline.vector_store, "payloads", [])
    _doc_text_lookup: dict = {}
    for _p in _db_payloads:
        if isinstance(_p, dict):
            _did = str(_p.get("doc_id", ""))
            _dtxt = _p.get("text") or _p.get("abstract") or _p.get("title") or ""
            if _did:
                _doc_text_lookup[_did] = str(_dtxt)

    emb_array = (
        np.array(all_embeddings) if isinstance(all_embeddings, list) else all_embeddings
    )
    for i in range(min(DEBUG_SAMPLE_LIMIT, len(all_embeddings))):
        results_with_scores = retrieval_pipeline.search(emb_array[i], k=10)

        # gt_doc_id: the expected document id
        if i < len(all_relevance) and len(all_relevance[i]) > 0:
            gt_doc_id = next(iter(all_relevance[i].keys()))
        else:
            gt_doc_id = str(all_ground_truth[i])

        # query_text: what was actually searched (ASR output or reference)
        if mode == "asr_text_retrieval" and i < len(all_hypotheses):
            query_text = all_hypotheses[i]
        elif i < len(all_ground_truth):
            query_text = str(all_ground_truth[i])
        else:
            query_text = all_query_ids[i] if i < len(all_query_ids) else "?"

        # gt_doc_text: the text of the expected document (from corpus)
        gt_doc_text = _doc_text_lookup.get(gt_doc_id, "")

        logger.info(f"\nQuery {i+1}:")
        logger.info(f"  ASR query text:        '{query_text[:120]}'")
        if mode == "asr_text_retrieval" and i < len(all_ground_truth):
            ref = str(all_ground_truth[i])
            if ref != query_text:
                logger.info(f"  Ground truth question: '{ref[:120]}'")
        logger.info(f"  Expected doc:  [{gt_doc_id}] '{gt_doc_text[:100]}'")
        logger.info(f"  Query emb norm: {np.linalg.norm(emb_array[i]):.4f}")
        logger.info(f"  Top 5 retrieved:")

        for j, (payload, score) in enumerate(results_with_scores[:5], 1):
            doc_id = (
                payload.get("doc_id", "") if isinstance(payload, dict) else str(payload)
            )
            match = MATCH_SYMBOL if doc_id == gt_doc_id else MISS_SYMBOL
            preview = (
                payload.get("text", str(payload))
                if isinstance(payload, dict)
                else str(payload)
            )[:80]
            logger.info(
                f"    {j}. [{match}] Score: {score:.4f} | [{doc_id}] '{preview}'"
            )

        gt_rank = None
        for rank, (payload, score) in enumerate(results_with_scores, 1):
            doc_id = (
                payload.get("doc_id", "") if isinstance(payload, dict) else str(payload)
            )
            if doc_id == gt_doc_id:
                gt_rank = rank
                logger.info(
                    f"  Expected doc found at rank: {gt_rank} (score: {score:.4f})"
                )
                break

        if gt_rank is None:
            logger.info(f"  Expected doc NOT found in top-{k}")
            payload_keys = [_payload_to_key(p) for p in _db_payloads]
            if gt_doc_id in payload_keys:
                logger.info(f"  (Doc exists in DB but outside top-{k})")
            else:
                logger.info(f"  (Doc NOT in DB at all!)")


def _publish_retrieved(s: RunState, results_list: list) -> None:
    """Publish ``retrieved`` as a keyed ``ItemSet`` when query ids align 1:1 (W3);
    otherwise the plain list. ``get_artifact('retrieved')`` returns the list either way, so
    rerank/metrics/answer_gen are unchanged; metric nodes read it keyed via ``get_items``.
    """
    ids = [str(i) for i in (s.all_query_ids or [])]
    if len(ids) == len(results_list) and len(set(ids)) == len(ids):
        from ..item_set import ItemSet

        s.put_items("retrieved", ItemSet(ids, list(results_list)))
    else:
        s.put_artifact("retrieved", results_list)


def _retrieval_query_texts(s: RunState):
    """Query texts feeding retrieval/rerank (mode-dependent); None for audio_emb."""
    if s.mode == "asr_text_retrieval":
        return s.all_hypotheses
    if s.mode == "audio_text_retrieval":
        return s.all_ground_truth
    if s.mode == "audio_emb_retrieval":
        if s.retrieval_pipeline.strategy_config.core.mode != "dense":
            raise ValueError(
                "audio_emb_retrieval supports only retrieval_mode='dense'. "
                "Sparse/hybrid requires query text unavailable in this path."
            )
    return None


def _finalize_retrieval(s: RunState, results_list) -> None:
    """Store final (payload, score) results + retrieved keys; log debug."""
    s.all_results_with_scores = results_list
    s.all_retrieved = [_search_results_to_keys(results) for results in results_list]
    # Publish the `retrieved` artifact (list of per-query (payload, score) lists) under
    # this node's id so downstream reads the right producer (retrieval or a rerank). Same
    # shape as retrieve_candidates output, so rerank instances chain (R4c/D2). Keyed as an
    # ItemSet (W3) when query ids align — get_artifact unwraps to the list for legacy
    # consumers; metric nodes read it keyed via get_items.
    _publish_retrieved(s, results_list)
    logger.info(f"Retrieval complete: {len(s.all_retrieved)} queries")
    _log_retrieval_debug(
        s.retrieval_pipeline,
        s.all_retrieved,
        s.all_ground_truth,
        s.all_relevance,
        s.all_hypotheses,
        s.all_query_ids,
        s.all_embeddings,
        s.mode,
        s.k,
    )


@register_stage_handler("retrieval", time_key="retrieval_s")
def _stage_retrieval(s: RunState) -> None:
    """Candidate generation (dense/sparse/hybrid). When a ``rerank`` node follows
    (refinement configured) this only fetches candidates; otherwise it finalizes."""
    if s.query_opt_bypassed:
        return
    # query_vectors artifact = the retrieval-input embeddings (published by the latest
    # embedding/fusion producer); fall back to all_embeddings for direct callers (R4b).
    query_vectors = s.get_artifact("query_vectors", default=s.all_embeddings)
    if isinstance(query_vectors, list) and len(query_vectors) > 0:
        query_vectors = np.array(query_vectors)
    s.all_embeddings = query_vectors  # keep for retrieval-debug logging
    s.cb("phase_3_retrieval", 0, s.total, "Phase 3: Retrieval")
    # The vector_index artifact is the indexed retrieval pipeline (published by
    # corpus_index); fall back to the shared pipeline for direct callers (R4a).
    rp = s.get_artifact("vector_index", default=s.retrieval_pipeline)
    # Per-node k (R2): a branch may retrieve a different depth.
    params = getattr(s.current_node, "params", None) or {}
    k = int(params.get("k", s.k))
    node_id = getattr(s.current_node, "id", "retrieval")
    with TimingContext("Retrieval Phase", logger):
        s.retrieval_query_texts = _retrieval_query_texts(s)
        qtexts = s.retrieval_query_texts
        nq = len(query_vectors)
        ids = [str(i) for i in (s.all_query_ids or range(nq))]

        def _row_texts(i: int):
            return [qtexts[i]] if qtexts is not None else None

        if rp.needs_refinement or s.rerank_in_graph:
            # Per-item isolation (T1): one query failing to fetch candidates drops out (empty
            # candidate list, recorded) instead of aborting; keyed report excludes it.
            s.retrieval_candidates = isolate_batch(
                ids,
                list(range(nq)),
                batch_fn=lambda _idx: rp.retrieve_candidates(
                    query_vectors, k, query_texts=qtexts
                ),
                item_fn=lambda i: rp.retrieve_candidates(
                    np.asarray(query_vectors[i : i + 1]), k, query_texts=_row_texts(i)
                )[0],
                node_id=node_id,
                placeholder=[],
                sink=s.drop_sink,
            )
            # Publish candidates as `retrieved` so a following rerank reads them (D2).
            s.put_artifact("retrieved", s.retrieval_candidates)
        else:
            _finalize_retrieval(
                s,
                isolate_batch(
                    ids,
                    list(range(nq)),
                    batch_fn=lambda _idx: rp.search_batch(
                        query_vectors, k, query_texts=qtexts
                    ),
                    item_fn=lambda i: rp.search_batch(
                        np.asarray(query_vectors[i : i + 1]),
                        k,
                        query_texts=_row_texts(i),
                    )[0],
                    node_id=node_id,
                    placeholder=[],
                    sink=s.drop_sink,
                ),
            )
    s.cb("phase_3_retrieval", s.total, s.total, "Retrieval complete")


@register_stage_handler("rerank", time_key="retrieval_s")
def _stage_rerank(s: RunState) -> None:
    """Refine retrieved candidates: rerank (cross-encoder / token-overlap) + MMR /
    threshold + truncate to k. Present only when refinement is configured."""
    if s.query_opt_bypassed:
        return
    s.cb("phase_3_5_rerank", 0, s.total, "Phase 3.5: Rerank")
    with TimingContext("Rerank Phase", logger):
        # Read the candidates from this node's bound producer (retrieval, or a prior
        # rerank when chained) — so duplicate rerank instances route correctly (D2).
        candidates = s.get_artifact("retrieved", default=s.retrieval_candidates)
        rp = s.retrieval_pipeline
        params = getattr(s.current_node, "params", None)
        with _node_reranking(rp, params, getattr(s, "service_provider", None)):
            results_list = rp.refine_candidates(
                candidates,
                s.all_embeddings,
                s.k,
                query_texts=s.retrieval_query_texts,
            )
        _finalize_retrieval(s, results_list)
    s.cb("phase_3_5_rerank", s.total, s.total, "Rerank complete")


def _rehydrate_retrieved(s: RunState) -> None:
    """Pull the `retrieved` artifact from the bound producer (retrieval/rerank) onto the
    state's result fields (R4c). Shared by the metrics + answer_gen handlers."""
    results_list = s.get_artifact("retrieved", default=s.all_results_with_scores)
    s.all_results_with_scores = results_list
    s.all_retrieved = [_search_results_to_keys(r) for r in results_list]
