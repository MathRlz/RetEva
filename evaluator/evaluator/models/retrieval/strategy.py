"""Canonical retrieval strategy configuration surface."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CoreRetrievalConfig:
    """Core retrieval strategy settings."""

    mode: str = "dense"  # dense | sparse | hybrid
    hybrid_dense_weight: float = 0.5
    hybrid_fusion_method: str = "weighted"  # weighted | rrf | max_score
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    def validate(self) -> None:
        if self.mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {self.mode}")
        if self.hybrid_fusion_method not in {"weighted", "rrf", "max_score"}:
            raise ValueError(f"Unsupported hybrid fusion method: {self.hybrid_fusion_method}")
        if not 0.0 <= self.hybrid_dense_weight <= 1.0:
            raise ValueError(
                f"hybrid_dense_weight must be in [0, 1], got {self.hybrid_dense_weight}"
            )
        if self.rrf_k <= 0:
            raise ValueError(f"rrf_k must be > 0, got {self.rrf_k}")
        if self.bm25_k1 <= 0:
            raise ValueError(f"bm25_k1 must be > 0, got {self.bm25_k1}")
        if not 0.0 <= self.bm25_b <= 1.0:
            raise ValueError(f"bm25_b must be in [0, 1], got {self.bm25_b}")


@dataclass(frozen=True)
class RerankingConfig:
    """Reranking strategy settings."""

    mode: str = "none"  # none | token_overlap | cross_encoder
    top_k: int = 20
    weight: float = 0.5

    def validate(self) -> None:
        if self.mode not in {"none", "token_overlap", "cross_encoder"}:
            raise ValueError(f"Unsupported reranker mode: {self.mode}")
        if self.top_k <= 0:
            raise ValueError(f"reranker top_k must be > 0, got {self.top_k}")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"reranker weight must be in [0, 1], got {self.weight}")


@dataclass(frozen=True)
class PostProcessingConfig:
    """Post-processing strategy settings."""

    use_mmr: bool = False
    mmr_lambda: float = 0.5
    min_similarity_threshold: Optional[float] = None
    distance_metric: str = "cosine"

    def validate(self) -> None:
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda must be in [0, 1], got {self.mmr_lambda}")
        if self.distance_metric not in {"cosine", "euclidean", "dot_product", "dot"}:
            raise ValueError(
                f"Unknown distance metric: {self.distance_metric}. "
                "Choose from: cosine, euclidean, dot_product"
            )


@dataclass(frozen=True)
class RetrievalStrategyConfig:
    """Unified retrieval strategy surface used by retrieval pipeline."""

    core: CoreRetrievalConfig = CoreRetrievalConfig()
    reranking: RerankingConfig = RerankingConfig()
    post_processing: PostProcessingConfig = PostProcessingConfig()

    def validate(self) -> None:
        self.core.validate()
        self.reranking.validate()
        self.post_processing.validate()

    @classmethod
    def from_vector_db_config(cls, vector_db_config) -> "RetrievalStrategyConfig":
        return cls(
            core=CoreRetrievalConfig(
                mode=vector_db_config.retrieval_mode,
                hybrid_dense_weight=vector_db_config.hybrid_dense_weight,
                hybrid_fusion_method=vector_db_config.hybrid_fusion_method,
                rrf_k=vector_db_config.rrf_k,
                bm25_k1=vector_db_config.bm25_k1,
                bm25_b=vector_db_config.bm25_b,
            ),
            reranking=RerankingConfig(
                mode=vector_db_config.reranker_mode,
                top_k=vector_db_config.reranker_top_k,
                weight=vector_db_config.reranker_weight,
            ),
            post_processing=PostProcessingConfig(
                use_mmr=vector_db_config.use_mmr,
                mmr_lambda=vector_db_config.mmr_lambda,
                min_similarity_threshold=vector_db_config.min_similarity_threshold,
                distance_metric=vector_db_config.distance_metric,
            ),
        )
