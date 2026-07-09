"""Retrieval components for dense, sparse, and hybrid search."""

from .rag.hybrid import reciprocal_rank_fusion
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
)
from .rag.strategies import (
    DistanceMetric,
    compute_similarity,
    compute_similarity_batch,
    mmr_search,
    mmr_rerank,
    threshold_filter,
)
from .fusion_registry import (
    FUSION_REGISTRY,
    fuse_hybrid_results,
    register_fusion,
    list_fusions,
)

__all__ = [
    "reciprocal_rank_fusion",
    "BaseReranker",
    "CrossEncoderReranker",
    "CoreRetrievalConfig",
    "RerankingConfig",
    "PostProcessingConfig",
    "RetrievalStrategyConfig",
    "ScoredRetrievalResult",
    "normalize_search_results",
    # Advanced strategies
    "DistanceMetric",
    "compute_similarity",
    "compute_similarity_batch",
    "mmr_search",
    "mmr_rerank",
    "threshold_filter",
    "FUSION_REGISTRY",
    "fuse_hybrid_results",
    "register_fusion",
    "list_fusions",
]
