"""Embedding fusion configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class EmbeddingFusionConfig:
    """
    Configuration for audio-text embedding fusion.
    
    When both audio and text embedding pipelines are available, this config
    controls how to fuse the embeddings for improved retrieval performance.
    
    Attributes:
        enabled: Whether embedding fusion is enabled. Default: False.
        audio_weight: Weight for audio embeddings (0.0-1.0). Default: 0.5.
        text_weight: Weight for text embeddings (0.0-1.0). Default: 0.5.
        fusion_method: Method for combining embeddings. Default: "weighted".
            Options:
            - "weighted": Weighted average of normalized embeddings
            - "concatenate": Concatenate embeddings (may require dimension reduction)
            - "max_pool": Element-wise maximum across embeddings
            - "average": Simple average (ignores weights)
        normalize_before_fusion: Whether to L2-normalize embeddings before fusion. Default: True.
        dimension_reduction: Method for reducing concatenated dimensions. Default: None.
            Options: None, "pca", "random_projection", "learned" (future).
        target_dim: Target dimension after reduction (only for concatenate). Default: None.
        require_same_dimensions: If True, require audio and text embeddings to have same dim. Default: False.
    
    Examples:
        >>> # Balanced audio-text fusion
        >>> config = EmbeddingFusionConfig(
        ...     enabled=True,
        ...     audio_weight=0.5,
        ...     text_weight=0.5,
        ...     fusion_method="weighted"
        ... )
        >>> 
        >>> # Audio-dominant fusion
        >>> config = EmbeddingFusionConfig(
        ...     enabled=True,
        ...     audio_weight=0.7,
        ...     text_weight=0.3
        ... )
        >>> 
        >>> # Concatenation with dimension reduction
        >>> config = EmbeddingFusionConfig(
        ...     enabled=True,
        ...     fusion_method="concatenate",
        ...     dimension_reduction="pca",
        ...     target_dim=1024
        ... )
    """
    enabled: bool = False
    audio_weight: float = 0.5
    text_weight: float = 0.5
    fusion_method: str = "weighted"  # weighted | concatenate | max_pool | average
    normalize_before_fusion: bool = True
    dimension_reduction: Optional[str] = None  # None | pca | random_projection
    target_dim: Optional[int] = None
    require_same_dimensions: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.audio_weight <= 1.0:
            raise ValueError(f"audio_weight must be in [0, 1], got {self.audio_weight}")
        if not 0.0 <= self.text_weight <= 1.0:
            raise ValueError(f"text_weight must be in [0, 1], got {self.text_weight}")
        
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
