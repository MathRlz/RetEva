"""Embedding fusion configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class EmbeddingFusionConfig:
    """Configuration for audio-text embedding fusion: how to combine audio and text
    embedding pipelines for retrieval."""
    enabled: bool = False
    # embedding = fuse the audio + text VECTORS into one retrieval (the fusion node);
    # result = retrieve each stream separately, then fuse the RANKED results (result_fusion).
    level: str = "embedding"  # embedding | result
    audio_weight: float = 0.5  # 0.0-1.0
    text_weight: float = 0.5  # 0.0-1.0
    # weighted (avg) | concatenate (may need dimension_reduction) | max_pool (elementwise
    # max) | average (ignores weights)
    fusion_method: str = "weighted"
    normalize_before_fusion: bool = True  # L2-normalize
    dimension_reduction: Optional[str] = None  # None | pca | random_projection
    target_dim: Optional[int] = None  # required when dimension_reduction is set
    require_same_dimensions: bool = False

    def __post_init__(self):
        if not 0.0 <= self.audio_weight <= 1.0:
            raise ValueError(f"audio_weight must be in [0, 1], got {self.audio_weight}")
        if not 0.0 <= self.text_weight <= 1.0:
            raise ValueError(f"text_weight must be in [0, 1], got {self.text_weight}")

        if self.level not in {"embedding", "result"}:
            raise ValueError(
                f"embedding_fusion.level must be 'embedding' or 'result', got {self.level}"
            )
        valid_methods = {"weighted", "concatenate", "max_pool", "average"}
        if self.fusion_method not in valid_methods:
            raise ValueError(
                f"fusion_method must be one of {valid_methods}, got {self.fusion_method}"
            )

        if self.dimension_reduction is not None:
            valid_reductions = {"pca", "random_projection"}
            if self.dimension_reduction not in valid_reductions:
                raise ValueError(
                    f"dimension_reduction must be one of {valid_reductions} or None, "
                    f"got {self.dimension_reduction}"
                )
            if self.fusion_method != "concatenate":
                raise ValueError(
                    "dimension_reduction only applies when fusion_method='concatenate'"
                )
            if self.target_dim is None:
                raise ValueError(
                    "target_dim must be specified when dimension_reduction is enabled"
                )
