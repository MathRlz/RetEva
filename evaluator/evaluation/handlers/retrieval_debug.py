"""Verbose per-query retrieval inspection — pure diagnostic logging.

Extracted out of ``handlers/retrieval.py`` (post-decomposition audit): it has no effect on
results (logs only), so it lives apart from the retrieval handlers that drive the run. The
``_stage_retrieval`` finalize path calls :func:`log_retrieval_debug` after publishing.
"""

from __future__ import annotations

import numpy as np

from ..helpers import _payload_to_key
from ...logging_config import get_logger
from ._common import DEBUG_SAMPLE_LIMIT, MATCH_SYMBOL, MISS_SYMBOL

logger = get_logger(__name__)


def log_retrieval_debug(
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
