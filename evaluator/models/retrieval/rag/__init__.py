"""RAG (Retrieval-Augmented Generation) components for search and retrieval."""

from .hybrid import reciprocal_rank_fusion
from .reranker import BaseReranker, CrossEncoderReranker
from .strategies import (
    DistanceMetric,
    compute_similarity,
    compute_similarity_batch,
    mmr_search,
    mmr_rerank,
    threshold_filter,
)

__all__ = [
    # Hybrid retrieval
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
]
