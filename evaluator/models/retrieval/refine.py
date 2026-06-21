"""Post-retrieval refinement strategies: rerank, MMR, threshold filtering.

Pure functions extracted from ``pipeline/retrieval_pipeline.py``; they take the
relevant config slice + state explicitly so the pipeline only orchestrates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ...logging_config import get_logger
from .rag.strategies import DistanceMetric, mmr_rerank, threshold_filter
from .scoring import min_max_norm, payload_key, payload_text, tokenize
from .strategy import PostProcessingConfig, RerankingConfig

if TYPE_CHECKING:
    from .rag.reranker import BaseReranker

logger = get_logger(__name__)


def token_overlap_score(query_text: str, payload: Any) -> float:
    """Jaccard overlap between query and payload token sets."""
    q = set(tokenize(query_text))
    d = set(tokenize(payload_text(payload)))
    if not q or not d:
        return 0.0
    return len(q & d) / max(len(q | d), 1)


def rerank_results(
    query_text: str,
    results: List[Tuple[Any, float]],
    k: int,
    *,
    reranking: RerankingConfig,
    reranker: Optional["BaseReranker"] = None,
) -> List[Tuple[Any, float]]:
    """Apply reranking to initial retrieval results.

    Modes: ``none`` (truncate to k), ``token_overlap`` (Jaccard blended with the
    base score by ``reranking.weight``), ``cross_encoder`` (delegates to the
    *reranker* instance, which takes precedence whenever set).
    """
    if reranking.mode == "none" and reranker is None:
        return results[:k]

    limited = results[: max(k, reranking.top_k)]

    # Cross-encoder reranking takes precedence if reranker is set
    if reranker is not None:
        return reranker.rerank(query_text, limited, top_k=k)

    if reranking.mode == "token_overlap":
        base_scores = {idx: float(score) for idx, (_, score) in enumerate(limited)}
        base_norm = min_max_norm(base_scores)
        rerank_scores = {
            idx: token_overlap_score(query_text, payload)
            for idx, (payload, _) in enumerate(limited)
        }
        rerank_norm = min_max_norm(rerank_scores)

        merged = []
        for idx, (payload, _) in enumerate(limited):
            score = (1.0 - reranking.weight) * base_norm.get(
                idx, 0.0
            ) + reranking.weight * rerank_norm.get(idx, 0.0)
            merged.append((payload, float(score)))

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:k]

    from ...config.types import RERANKER_MODES

    if reranking.mode not in RERANKER_MODES:
        raise ValueError(f"Unsupported reranker_mode: {reranking.mode}")

    return results[:k]


def apply_mmr(
    query_emb: np.ndarray,
    results: List[Tuple[Any, float]],
    k: int,
    *,
    post: PostProcessingConfig,
    index_embeddings: Optional[np.ndarray],
    index_payloads: Optional[List[Any]],
    metric: DistanceMetric,
) -> List[Tuple[Any, float]]:
    """MMR-rerank *results* for diversity using the stored index embeddings."""
    if not post.use_mmr or not results:
        return results[:k]

    if index_embeddings is None or index_payloads is None:
        logger.warning("MMR enabled but index embeddings not stored. Skipping MMR.")
        return results[:k]

    payload_to_idx: Dict[str, int] = {
        payload_key(p): i for i, p in enumerate(index_payloads)
    }

    result_indices = []
    valid_results = []
    for payload, score in results:
        key = payload_key(payload)
        if key in payload_to_idx:
            result_indices.append(payload_to_idx[key])
            valid_results.append((payload, score))

    if not valid_results:
        return results[:k]

    return mmr_rerank(
        query_emb=query_emb,
        results=valid_results,
        doc_embs=index_embeddings[result_indices],
        k=k,
        lambda_param=post.mmr_lambda,
        metric=metric,
    )


def apply_threshold(
    results: List[Tuple[Any, float]], *, post: PostProcessingConfig
) -> List[Tuple[Any, float]]:
    """Drop results below ``post.min_similarity_threshold`` (no-op when unset)."""
    if post.min_similarity_threshold is None:
        return results
    return threshold_filter(results, min_score=post.min_similarity_threshold)
