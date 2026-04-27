"""Hybrid retrieval combining dense and sparse methods."""

from typing import Any, Dict, List, Tuple, Optional, TypeVar
import numpy as np

from ....logging_config import get_logger

logger = get_logger(__name__)

K = TypeVar('K')


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[Any, float]]],
    k: int = 60,
    top_n: Optional[int] = None
) -> List[Tuple[Any, float]]:
    """Combine multiple rankings using Reciprocal Rank Fusion (RRF).
    
    RRF score for document d = sum over rankings of 1 / (k + rank(d))
    
    This method is effective for combining rankings from different retrieval
    systems without requiring score normalization.
    
    Args:
        rankings: List of ranked result lists. Each list contains (item, score) tuples.
        k: RRF parameter controlling the impact of lower-ranked documents.
            Higher k gives more weight to top results. Default is 60.
        top_n: Number of top results to return. If None, returns all.
        
    Returns:
        List of (item, rrf_score) tuples sorted by RRF score descending.
    """
    rrf_scores: Dict[Any, float] = {}
    item_lookup: Dict[Any, Any] = {}
    
    for ranking in rankings:
        for rank, (item, _score) in enumerate(ranking, start=1):
            item_key = _get_item_key(item)
            rrf_scores[item_key] = rrf_scores.get(item_key, 0.0) + 1.0 / (k + rank)
            item_lookup[item_key] = item
    
    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    if top_n is not None:
        sorted_items = sorted_items[:top_n]
    
    return [(item_lookup[key], score) for key, score in sorted_items]


def _get_item_key(item: Any) -> str:
    """Extract a hashable key from an item."""
    if isinstance(item, dict):
        if item.get("doc_id") is not None:
            return str(item["doc_id"])
        if item.get("text") is not None:
            return str(item["text"])
    return str(item)


def _min_max_normalize(scores: Dict[K, float]) -> Dict[K, float]:
    """Normalize scores to [0, 1] range using min-max normalization."""
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


class HybridRetriever:
    """Hybrid retriever combining dense and sparse retrieval methods.
    
    Supports two fusion strategies:
    - Weighted linear combination with min-max normalization
    - Reciprocal Rank Fusion (RRF)
    """
    
    def __init__(
        self,
        dense_retriever: Any,
        sparse_retriever: Any,
        dense_weight: float = 0.5,
        fusion_method: str = "weighted",
        rrf_k: int = 60,
    ):
        """Initialize hybrid retriever.
        
        Args:
            dense_retriever: Dense retriever with search(query_embedding, k) method.
            sparse_retriever: Sparse retriever with search(query, k) method.
            dense_weight: Weight for dense scores in weighted fusion (0-1).
                Sparse weight is (1 - dense_weight).
            fusion_method: Fusion strategy - "weighted" or "rrf".
            rrf_k: RRF k parameter (only used when fusion_method="rrf").
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
    
    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        k: int = 10,
        dense_k: Optional[int] = None,
        sparse_k: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """Perform hybrid search combining dense and sparse results.
        
        Args:
            query_embedding: Dense query embedding vector.
            query_text: Query text for sparse retrieval.
            k: Number of final results to return.
            dense_k: Number of dense results to fetch (default: k * 2).
            sparse_k: Number of sparse results to fetch (default: k * 2).
            
        Returns:
            List of (payload, score) tuples sorted by combined score.
        """
        dense_k = dense_k or k * 2
        sparse_k = sparse_k or k * 2
        
        # Get dense results
        dense_results = self.dense_retriever.search(query_embedding, dense_k)
        
        # Get sparse results
        sparse_raw = self.sparse_retriever.search(query_text, sparse_k)
        # Convert sparse results to (payload, score) format
        sparse_results = self._convert_sparse_results(sparse_raw)
        
        # Fuse results
        if self.fusion_method == "rrf":
            return self._fuse_rrf(dense_results, sparse_results, k)
        else:
            return self._fuse_weighted(dense_results, sparse_results, k)
    
    def _convert_sparse_results(
        self, sparse_results: List[Tuple[int, float]]
    ) -> List[Tuple[Any, float]]:
        """Convert sparse (index, score) results to (payload, score)."""
        payloads = []
        for idx, score in sparse_results:
            text = self.sparse_retriever.get_text(idx)
            payloads.append(({"text": text, "doc_id": idx}, score))
        return payloads
    
    def _fuse_weighted(
        self,
        dense_results: List[Tuple[Any, float]],
        sparse_results: List[Tuple[Any, float]],
        k: int,
    ) -> List[Tuple[Any, float]]:
        """Fuse results using weighted linear combination."""
        # Build score dictionaries
        dense_scores = {_get_item_key(p): float(s) for p, s in dense_results}
        sparse_scores = {_get_item_key(p): float(s) for p, s in sparse_results}
        
        # Normalize scores
        dense_norm = _min_max_normalize(dense_scores)
        sparse_norm = _min_max_normalize(sparse_scores)
        
        # Build payload lookup
        payload_by_key: Dict[str, Any] = {}
        for payload, _ in dense_results:
            payload_by_key[_get_item_key(payload)] = payload
        for payload, _ in sparse_results:
            payload_by_key[_get_item_key(payload)] = payload
        
        # Combine scores
        all_keys = set(dense_norm.keys()) | set(sparse_norm.keys())
        merged = {}
        for key in all_keys:
            merged[key] = (
                self.dense_weight * dense_norm.get(key, 0.0)
                + self.sparse_weight * sparse_norm.get(key, 0.0)
            )
        
        # Sort and return top-k
        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(payload_by_key[key], score) for key, score in ranked]
    
    def _fuse_rrf(
        self,
        dense_results: List[Tuple[Any, float]],
        sparse_results: List[Tuple[Any, float]],
        k: int,
    ) -> List[Tuple[Any, float]]:
        """Fuse results using Reciprocal Rank Fusion."""
        return reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
            top_n=k
        )
    
    def search_batch(
        self,
        query_embeddings: np.ndarray,
        query_texts: List[str],
        k: int = 10,
    ) -> List[List[Tuple[Any, float]]]:
        """Perform hybrid search for multiple queries.
        
        Args:
            query_embeddings: Array of dense query embedding vectors.
            query_texts: List of query texts for sparse retrieval.
            k: Number of results per query.
            
        Returns:
            List of result lists, each containing (payload, score) tuples.
        """
        results = []
        for query_emb, query_text in zip(query_embeddings, query_texts):
            results.append(self.search(query_emb, query_text, k))
        return results
