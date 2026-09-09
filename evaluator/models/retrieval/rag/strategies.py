"""Advanced retrieval strategies: MMR for result diversity, similarity-threshold filtering,
and pluggable distance metrics."""

from enum import Enum
from typing import Any, List, Tuple, Optional

import numpy as np


class DistanceMetric(Enum):
    """Distance/similarity metrics for vector comparison."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2 normalize a vector."""
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return vec
    return vec / norm


def compute_similarity(
    a: np.ndarray,
    b: np.ndarray,
    metric: DistanceMetric = DistanceMetric.COSINE
) -> float:
    """Similarity between two vectors under ``metric`` (higher = more similar)."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()

    if metric == DistanceMetric.COSINE:
        a_norm = _normalize(a)
        b_norm = _normalize(b)
        return float(np.dot(a_norm, b_norm))

    elif metric == DistanceMetric.DOT_PRODUCT:
        return float(np.dot(a, b))

    elif metric == DistanceMetric.EUCLIDEAN:
        # Convert distance to similarity: sim = 1 / (1 + dist)
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_similarity_batch(
    query: np.ndarray,
    candidates: np.ndarray,
    metric: DistanceMetric = DistanceMetric.COSINE
) -> np.ndarray:
    """Similarity between ``query`` and each row of ``candidates`` under ``metric``
    (higher = more similar); returns shape (n,)."""
    query = np.asarray(query, dtype=np.float32).flatten()
    candidates = np.asarray(candidates, dtype=np.float32)

    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)

    if metric == DistanceMetric.COSINE:
        query_norm = _normalize(query)
        cand_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
        cand_norms = np.where(cand_norms < 1e-9, 1.0, cand_norms)
        candidates_norm = candidates / cand_norms
        return candidates_norm @ query_norm

    elif metric == DistanceMetric.DOT_PRODUCT:
        return candidates @ query

    elif metric == DistanceMetric.EUCLIDEAN:
        dists = np.linalg.norm(candidates - query, axis=1)
        return 1.0 / (1.0 + dists)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def mmr_search(
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    k: int,
    lambda_param: float = 0.5,
    metric: DistanceMetric = DistanceMetric.COSINE,
    initial_scores: Optional[np.ndarray] = None,
) -> List[Tuple[int, float]]:
    """Maximal Marginal Relevance search: balances relevance to the query against diversity
    among already-selected documents via
    ``score = lambda * Sim(d, query) - (1-lambda) * max(Sim(d, selected))``.
    ``lambda_param`` near 1.0 prioritizes relevance, near 0.0 prioritizes diversity."""
    query_emb = np.asarray(query_emb, dtype=np.float32).flatten()
    doc_embs = np.asarray(doc_embs, dtype=np.float32)

    if doc_embs.ndim == 1:
        doc_embs = doc_embs.reshape(1, -1)

    n_docs = len(doc_embs)
    if n_docs == 0:
        return []

    k = min(k, n_docs)

    if initial_scores is not None:
        relevance_scores = np.asarray(initial_scores, dtype=np.float32)
    else:
        relevance_scores = compute_similarity_batch(query_emb, doc_embs, metric)

    selected_indices: List[int] = []
    selected_embs: List[np.ndarray] = []
    unselected = set(range(n_docs))

    results: List[Tuple[int, float]] = []

    for _ in range(k):
        if not unselected:
            break

        best_idx = -1
        best_mmr_score = float('-inf')

        for idx in unselected:
            relevance = relevance_scores[idx]

            if selected_embs:
                max_sim_to_selected = max(
                    compute_similarity(doc_embs[idx], sel_emb, metric)
                    for sel_emb in selected_embs
                )
            else:
                max_sim_to_selected = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected_indices.append(best_idx)
            selected_embs.append(doc_embs[best_idx])
            unselected.remove(best_idx)
            results.append((best_idx, float(best_mmr_score)))

    return results


def mmr_rerank(
    query_emb: np.ndarray,
    results: List[Tuple[Any, float]],
    doc_embs: np.ndarray,
    k: int,
    lambda_param: float = 0.5,
    metric: DistanceMetric = DistanceMetric.COSINE,
) -> List[Tuple[Any, float]]:
    """Apply MMR reranking to existing (payload, score) results, using ``doc_embs``
    (same order as ``results``) for diversity scoring."""
    if not results:
        return []

    initial_scores = np.array([score for _, score in results], dtype=np.float32)

    mmr_results = mmr_search(
        query_emb=query_emb,
        doc_embs=doc_embs,
        k=k,
        lambda_param=lambda_param,
        metric=metric,
        initial_scores=initial_scores,
    )

    return [(results[idx][0], score) for idx, score in mmr_results]


def threshold_filter(
    results: List[Tuple[Any, float]],
    min_score: float = 0.5,
) -> List[Tuple[Any, float]]:
    """Filter out results scoring below ``min_score`` (inclusive)."""
    return [(payload, score) for payload, score in results if score >= min_score]
