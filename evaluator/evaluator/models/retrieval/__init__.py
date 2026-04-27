"""Retrieval components for dense, sparse, and hybrid search."""

from .rag.sparse import BM25Retriever
from .rag.hybrid import HybridRetriever, reciprocal_rank_fusion
from .rag.reranker import BaseReranker, CrossEncoderReranker
from .strategy import (
    CoreRetrievalConfig,
    RerankingConfig,
    PostProcessingConfig,
    RetrievalStrategyConfig,
)
from .contracts import (
    ScoredRetrievalResult,
    normalize_search_results,
    normalize_batch_search_results,
)
from .rag.strategies import (
    DistanceMetric,
    compute_similarity,
    compute_similarity_batch,
    mmr_search,
    mmr_rerank,
    threshold_filter,
    threshold_filter_with_fallback,
)
from .fusion_registry import FUSION_REGISTRY, fuse_hybrid_results

__all__ = [
    "BM25Retriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "BaseReranker",
    "CrossEncoderReranker",
    "CoreRetrievalConfig",
    "RerankingConfig",
    "PostProcessingConfig",
    "RetrievalStrategyConfig",
    "ScoredRetrievalResult",
    "normalize_search_results",
    "normalize_batch_search_results",
    # Advanced strategies
    "DistanceMetric",
    "compute_similarity",
    "compute_similarity_batch",
    "mmr_search",
    "mmr_rerank",
    "threshold_filter",
    "threshold_filter_with_fallback",
    "FUSION_REGISTRY",
    "fuse_hybrid_results",
]
