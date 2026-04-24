"""Advanced retrieval strategies for diversity and quality filtering.

This module provides:
- MaxMarginalRelevance (MMR) for result diversity
- Similarity threshold filtering for quality control  
- Custom distance metrics for flexible similarity computation
"""

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
    """Compute similarity between two vectors using specified metric.
    
    Args:
        a: First vector
        b: Second vector
        metric: Distance metric to use
        
    Returns:
        Similarity score (higher = more similar)
    """
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
    """Compute similarity between query and multiple candidates.
    
    Args:
        query: Query vector of shape (dim,)
        candidates: Candidate vectors of shape (n, dim)
        metric: Distance metric to use
        
    Returns:
        Array of similarity scores of shape (n,)
    """
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
    doc_payloads: Optional[List[Any]] = None,
    initial_scores: Optional[np.ndarray] = None,
) -> List[Tuple[int, float]]:
    """Maximal Marginal Relevance search for diverse results.
    
    MMR balances relevance to the query with diversity among selected documents.
    Score = lambda * Sim(d, query) - (1-lambda) * max(Sim(d, selected))
    
    Args:
        query_emb: Query embedding vector
        doc_embs: Document embedding matrix of shape (n_docs, dim)
        k: Number of results to return
        lambda_param: Balance between relevance (1.0) and diversity (0.0).
            Higher values prioritize relevance, lower values prioritize diversity.
        metric: Distance metric for similarity computation
        doc_payloads: Optional payloads to return with indices
        initial_scores: Optional pre-computed relevance scores to query
        
    Returns:
        List of (doc_index, mmr_score) tuples in selection order
    """
    query_emb = np.asarray(query_emb, dtype=np.float32).flatten()
    doc_embs = np.asarray(doc_embs, dtype=np.float32)
    
    if doc_embs.ndim == 1:
        doc_embs = doc_embs.reshape(1, -1)
    
    n_docs = len(doc_embs)
    if n_docs == 0:
        return []
    
    k = min(k, n_docs)
    
    # Compute relevance scores to query
    if initial_scores is not None:
        relevance_scores = np.asarray(initial_scores, dtype=np.float32)
    else:
        relevance_scores = compute_similarity_batch(query_emb, doc_embs, metric)
    
    # Track selected and unselected indices
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
            
            # Compute max similarity to already selected docs
            if selected_embs:
                max_sim_to_selected = max(
                    compute_similarity(doc_embs[idx], sel_emb, metric)
                    for sel_emb in selected_embs
                )
            else:
                max_sim_to_selected = 0.0
            
            # MMR score
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
    """Apply MMR reranking to existing retrieval results.
    
    Args:
        query_emb: Query embedding vector
        results: List of (payload, score) tuples from initial retrieval
        doc_embs: Embeddings corresponding to results (same order)
        k: Number of results to return
        lambda_param: Balance between relevance and diversity
        metric: Distance metric for similarity computation
        
    Returns:
        Reranked list of (payload, mmr_score) tuples
    """
    if not results:
        return []
    
    # Extract initial scores
    initial_scores = np.array([score for _, score in results], dtype=np.float32)
    
    # Run MMR
    mmr_results = mmr_search(
        query_emb=query_emb,
        doc_embs=doc_embs,
        k=k,
        lambda_param=lambda_param,
        metric=metric,
        initial_scores=initial_scores,
    )
    
    # Map back to payloads
    return [(results[idx][0], score) for idx, score in mmr_results]


def threshold_filter(
    results: List[Tuple[Any, float]],
    min_score: float = 0.5,
) -> List[Tuple[Any, float]]:
    """Filter results below a minimum similarity score.
    
    Args:
        results: List of (payload, score) tuples
        min_score: Minimum score threshold (inclusive)
        
    Returns:
        Filtered list of (payload, score) tuples
    """
    return [(payload, score) for payload, score in results if score >= min_score]


def threshold_filter_with_fallback(
    results: List[Tuple[Any, float]],
    min_score: float = 0.5,
    min_results: int = 1,
) -> List[Tuple[Any, float]]:
    """Filter results below threshold, but ensure minimum number of results.
    
    If filtering would return fewer than min_results, returns the top min_results
    regardless of threshold.
    
    Args:
        results: List of (payload, score) tuples
        min_score: Minimum score threshold
        min_results: Minimum number of results to return
        
    Returns:
        Filtered list of (payload, score) tuples
    """
    filtered = threshold_filter(results, min_score)
    
    if len(filtered) >= min_results:
        return filtered
    
    # Return top min_results if threshold filtering is too aggressive
    return results[:min_results]


def borda_count_fusion(
    rankings: List[List[Tuple[Any, float]]],
    k: int = None
) -> List[Tuple[Any, float]]:
    """Combine rankings using Borda count voting.
    
    Each document gets points based on its rank in each list:
    top document gets n points, second gets n-1, etc.
    
    Args:
        rankings: List of ranked result lists.
        k: Number of top results to return (None = all).
        
    Returns:
        Combined ranking with Borda scores.
    """
    from collections import defaultdict
    
    borda_scores = defaultdict(float)
    item_lookup = {}
    
    for ranking in rankings:
        n = len(ranking)
        for rank, (item, _) in enumerate(ranking):
            # Higher rank = more points
            points = n - rank
            
            # Use string representation as key
            key = str(item)
            borda_scores[key] += points
            item_lookup[key] = item
    
    # Sort by Borda score descending
    sorted_items = sorted(
        borda_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    if k is not None:
        sorted_items = sorted_items[:k]
    
    return [(item_lookup[key], score) for key, score in sorted_items]


def distribution_based_fusion(
    dense_results: List[Tuple[Any, float]],
    sparse_results: List[Tuple[Any, float]],
    method: str = "z_score",
    k: int = None
) -> List[Tuple[Any, float]]:
    """Combine dense and sparse results using score normalization.
    
    Normalizes scores from different systems before combining them,
    handling different score distributions appropriately.
    
    Args:
        dense_results: Dense retrieval results.
        sparse_results: Sparse retrieval results.
        method: Normalization method. Options: "z_score", "min_max", "rank".
        k: Number of results to return (None = all).
        
    Returns:
        Fused results with normalized scores.
    """
    from collections import defaultdict
    import numpy as np
    
    if not dense_results and not sparse_results:
        return []
    
    # Normalize scores
    def normalize_scores(results, method):
        if not results:
            return {}
        
        scores = [s for _, s in results]
        
        if method == "z_score":
            # Z-score normalization
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            if std_score < 1e-9:
                # All scores are the same
                return {str(item): 1.0 for item, _ in results}
            
            normalized = {}
            for item, score in results:
                z_score = (score - mean_score) / std_score
                # Map to [0, 1] using sigmoid
                normalized[str(item)] = 1.0 / (1.0 + np.exp(-z_score))
            
            return normalized
        
        elif method == "min_max":
            # Min-max normalization
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score - min_score < 1e-9:
                return {str(item): 1.0 for item, _ in results}
            
            return {
                str(item): (score - min_score) / (max_score - min_score)
                for item, score in results
            }
        
        elif method == "rank":
            # Rank-based normalization
            n = len(results)
            return {
                str(item): (n - rank) / n
                for rank, (item, _) in enumerate(results)
            }
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    # Normalize both result sets
    dense_norm = normalize_scores(dense_results, method)
    sparse_norm = normalize_scores(sparse_results, method)
    
    # Combine scores
    combined_scores = defaultdict(float)
    item_lookup = {}
    
    for item, _ in dense_results:
        key = str(item)
        combined_scores[key] += dense_norm[key]
        item_lookup[key] = item
    
    for item, _ in sparse_results:
        key = str(item)
        combined_scores[key] += sparse_norm[key]
        item_lookup[key] = item
    
    # Sort by combined score
    sorted_items = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    if k is not None:
        sorted_items = sorted_items[:k]
    
    return [(item_lookup[key], score) for key, score in sorted_items]
