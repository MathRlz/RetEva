"""RAG (Retrieval-Augmented Generation) components for search and retrieval."""

from .sparse import BM25Retriever
from .hybrid import HybridRetriever, reciprocal_rank_fusion
from .reranker import BaseReranker, CrossEncoderReranker
from .strategies import (
    DistanceMetric,
    compute_similarity,
    compute_similarity_batch,
    mmr_search,
    mmr_rerank,
    threshold_filter,
    threshold_filter_with_fallback,
)

__all__ = [
    # Sparse retrieval
    "BM25Retriever",
    
    # Hybrid retrieval
    "HybridRetriever",
    "reciprocal_rank_fusion",
    
    # Reranking
    "BaseReranker",
    "CrossEncoderReranker",
    
    # Advanced strategies
    "DistanceMetric",
    "compute_similarity",
    "compute_similarity_batch",
    "mmr_search",
    "mmr_rerank",
    "threshold_filter",
    "threshold_filter_with_fallback",
]
