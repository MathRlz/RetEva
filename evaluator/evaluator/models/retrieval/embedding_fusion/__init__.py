"""Embedding fusion for combining audio and text embeddings.

This module provides functions for fusing embeddings from multiple modalities
(audio and text) to improve retrieval performance in multimodal scenarios.
"""

from typing import Optional, Tuple, Any
import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from ....logging_config import get_logger
from ....config import EmbeddingFusionConfig

logger = get_logger(__name__)


def normalize_embeddings(embeddings: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalize embeddings along specified axis.
    
    Args:
        embeddings: Array of embeddings to normalize, shape (n_samples, dim) or (dim,).
        axis: Axis along which to normalize. Default: -1 (last axis).
        
    Returns:
        Normalized embeddings with the same shape as input.
        
    Examples:
        >>> emb = np.array([[1.0, 2.0, 2.0], [3.0, 4.0, 0.0]])
        >>> normed = normalize_embeddings(emb)
        >>> np.allclose(np.linalg.norm(normed, axis=1), 1.0)
        True
    """
    norms = np.linalg.norm(embeddings, axis=axis, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def reduce_dimensions(
    embeddings: np.ndarray,
    target_dim: int,
    method: str = "pca",
    reducer: Optional[Any] = None
) -> Tuple[np.ndarray, Any]:
    """Reduce embedding dimensionality using specified method.
    
    Args:
        embeddings: Array of embeddings to reduce, shape (n_samples, dim).
        target_dim: Target dimensionality after reduction.
        method: Reduction method - "pca" or "random_projection".
        reducer: Pre-fitted reducer instance. If None, a new one is created and fitted.
        
    Returns:
        Tuple of (reduced_embeddings, fitted_reducer).
        
    Raises:
        ValueError: If method is not supported or target_dim >= current dim.
        
    Examples:
        >>> emb = np.random.randn(100, 2048)
        >>> reduced, reducer = reduce_dimensions(emb, target_dim=1024, method="pca")
        >>> reduced.shape
        (100, 1024)
    """
    if embeddings.shape[1] <= target_dim:
        logger.warning(
            f"Current dimension {embeddings.shape[1]} <= target {target_dim}. "
            f"No reduction performed."
        )
        return embeddings, None
    
    if method == "pca":
        if reducer is None:
            reducer = PCA(n_components=target_dim, random_state=42)
            reduced = reducer.fit_transform(embeddings)
            logger.info(
                f"PCA reduction: {embeddings.shape[1]} -> {target_dim} "
                f"(explained variance: {reducer.explained_variance_ratio_.sum():.3f})"
            )
        else:
            reduced = reducer.transform(embeddings)
    elif method == "random_projection":
        if reducer is None:
            reducer = GaussianRandomProjection(
                n_components=target_dim,
                random_state=42
            )
            reduced = reducer.fit_transform(embeddings)
            logger.info(
                f"Random projection: {embeddings.shape[1]} -> {target_dim}"
            )
        else:
            reduced = reducer.transform(embeddings)
    else:
        raise ValueError(
            f"Unsupported dimension reduction method: {method}. "
            f"Use 'pca' or 'random_projection'."
        )
    
    return reduced, reducer


def _ensure_2d_embeddings(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if audio_embeddings.ndim == 1:
        audio_embeddings = audio_embeddings.reshape(1, -1)
    if text_embeddings.ndim == 1:
        text_embeddings = text_embeddings.reshape(1, -1)
    return audio_embeddings, text_embeddings


def _require_same_dim(audio_dim: int, text_dim: int, method: str) -> None:
    if audio_dim != text_dim:
        raise ValueError(
            f"{method} fusion requires same dimensions, but got "
            f"audio_dim={audio_dim}, text_dim={text_dim}. "
            f"Consider using 'concatenate' with dimension reduction."
        )


def _finalize_fused(
    fused: np.ndarray,
    normalize_before_fusion: bool,
) -> np.ndarray:
    if normalize_before_fusion:
        return normalize_embeddings(fused)
    return fused


def _fuse_weighted(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    config: EmbeddingFusionConfig,
) -> np.ndarray:
    audio_dim = audio_embeddings.shape[1]
    text_dim = text_embeddings.shape[1]
    _require_same_dim(audio_dim, text_dim, "Weighted")
    fused = config.audio_weight * audio_embeddings + config.text_weight * text_embeddings
    return _finalize_fused(fused, config.normalize_before_fusion)


def _fuse_average(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    config: EmbeddingFusionConfig,
) -> np.ndarray:
    audio_dim = audio_embeddings.shape[1]
    text_dim = text_embeddings.shape[1]
    _require_same_dim(audio_dim, text_dim, "Average")
    fused = (audio_embeddings + text_embeddings) / 2.0
    return _finalize_fused(fused, config.normalize_before_fusion)


def _fuse_max_pool(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    config: EmbeddingFusionConfig,
) -> np.ndarray:
    audio_dim = audio_embeddings.shape[1]
    text_dim = text_embeddings.shape[1]
    _require_same_dim(audio_dim, text_dim, "Max pool")
    fused = np.maximum(audio_embeddings, text_embeddings)
    return _finalize_fused(fused, config.normalize_before_fusion)


def _fuse_concatenate(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    config: EmbeddingFusionConfig,
    reducer: Optional[Any] = None,
) -> Tuple[np.ndarray, Optional[Any]]:
    audio_dim = audio_embeddings.shape[1]
    text_dim = text_embeddings.shape[1]
    fused = np.concatenate([audio_embeddings, text_embeddings], axis=1)
    logger.debug(f"Concatenated shape: {fused.shape}")
    if config.dimension_reduction is not None and config.target_dim is not None:
        fused, reducer = reduce_dimensions(
            fused,
            target_dim=config.target_dim,
            method=config.dimension_reduction,
            reducer=reducer,
        )
        logger.info(
            f"Applied {config.dimension_reduction} reduction: "
            f"{audio_dim + text_dim} -> {config.target_dim}"
        )
    else:
        reducer = None
    fused = _finalize_fused(fused, config.normalize_before_fusion)
    return fused, reducer


def fuse_embeddings(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    config: EmbeddingFusionConfig,
    reducer: Optional[Any] = None
) -> Tuple[np.ndarray, Optional[Any]]:
    """Fuse audio and text embeddings using configured fusion method.
    
    Supports multiple fusion strategies:
    - Weighted: Weighted combination of normalized embeddings
    - Concatenate: Concatenate embeddings (optionally with dimension reduction)
    - Max Pool: Element-wise maximum across embeddings
    - Average: Simple average of embeddings
    
    Args:
        audio_embeddings: Audio embedding vectors, shape (n_samples, audio_dim) or (audio_dim,).
        text_embeddings: Text embedding vectors, shape (n_samples, text_dim) or (text_dim,).
        config: EmbeddingFusionConfig specifying fusion parameters.
        reducer: Optional pre-fitted dimension reducer (for concatenate mode).
        
    Returns:
        Tuple of (fused_embeddings, reducer). Reducer is None unless concatenate mode
        with dimension reduction is used.
        
    Raises:
        ValueError: If embeddings have mismatched batch sizes or incompatible dimensions.
        
    Examples:
        >>> audio_emb = np.random.randn(10, 768)
        >>> text_emb = np.random.randn(10, 1024)
        >>> config = EmbeddingFusionConfig(
        ...     enabled=True,
        ...     fusion_method="weighted",
        ...     audio_weight=0.6,
        ...     text_weight=0.4
        ... )
        >>> fused, _ = fuse_embeddings(audio_emb, text_emb, config)
        >>> fused.shape
        (10, 768)
    """
    audio_embeddings, text_embeddings = _ensure_2d_embeddings(
        audio_embeddings,
        text_embeddings,
    )
    
    # Validate batch sizes
    if audio_embeddings.shape[0] != text_embeddings.shape[0]:
        raise ValueError(
            f"Batch size mismatch: audio has {audio_embeddings.shape[0]} samples, "
            f"text has {text_embeddings.shape[0]} samples"
        )
    
    audio_dim = audio_embeddings.shape[1]
    text_dim = text_embeddings.shape[1]
    
    logger.debug(
        f"Fusing embeddings: audio {audio_embeddings.shape}, "
        f"text {text_embeddings.shape}, method={config.fusion_method}"
    )
    
    # Check dimension requirements
    if config.require_same_dimensions and audio_dim != text_dim:
        raise ValueError(
            f"require_same_dimensions=True but audio_dim={audio_dim} != text_dim={text_dim}"
        )
    
    # Normalize if requested
    if config.normalize_before_fusion:
        audio_embeddings = normalize_embeddings(audio_embeddings)
        text_embeddings = normalize_embeddings(text_embeddings)
    
    fusion_method = config.fusion_method
    if fusion_method == "weighted":
        return _fuse_weighted(audio_embeddings, text_embeddings, config), None
    if fusion_method == "average":
        return _fuse_average(audio_embeddings, text_embeddings, config), None
    if fusion_method == "max_pool":
        return _fuse_max_pool(audio_embeddings, text_embeddings, config), None
    if fusion_method == "concatenate":
        return _fuse_concatenate(audio_embeddings, text_embeddings, config, reducer=reducer)
    raise ValueError(f"Unknown fusion method: {fusion_method}")


def validate_fusion_config(
    config: EmbeddingFusionConfig,
    audio_dim: int,
    text_dim: int
) -> None:
    """Validate fusion configuration against embedding dimensions.
    
    Checks that the configuration is compatible with the given embedding dimensions
    and raises descriptive errors if issues are found.
    
    Args:
        config: EmbeddingFusionConfig to validate.
        audio_dim: Dimensionality of audio embeddings.
        text_dim: Dimensionality of text embeddings.
        
    Raises:
        ValueError: If configuration is invalid for the given dimensions.
        
    Examples:
        >>> config = EmbeddingFusionConfig(
        ...     enabled=True,
        ...     fusion_method="weighted"
        ... )
        >>> validate_fusion_config(config, audio_dim=768, text_dim=768)  # OK
        >>> validate_fusion_config(config, audio_dim=768, text_dim=1024)  # Raises
        Traceback (most recent call last):
        ...
        ValueError: ...
    """
    if not config.enabled:
        return
    
    # Check dimension compatibility for different fusion methods
    if config.fusion_method in {"weighted", "average", "max_pool"}:
        if audio_dim != text_dim:
            raise ValueError(
                f"Fusion method '{config.fusion_method}' requires audio and text "
                f"embeddings to have the same dimension, but got audio_dim={audio_dim}, "
                f"text_dim={text_dim}. Consider:\n"
                f"  1. Using fusion_method='concatenate' with dimension_reduction\n"
                f"  2. Padding/projecting embeddings to same dimension\n"
                f"  3. Using different embedding models with matching dimensions"
            )
    
    # Check dimension reduction configuration
    if config.dimension_reduction is not None:
        if config.fusion_method != "concatenate":
            raise ValueError(
                f"dimension_reduction is only applicable with fusion_method='concatenate', "
                f"but got fusion_method='{config.fusion_method}'"
            )
        
        if config.target_dim is None:
            raise ValueError(
                "target_dim must be specified when dimension_reduction is enabled"
            )
        
        concat_dim = audio_dim + text_dim
        if config.target_dim >= concat_dim:
            logger.warning(
                f"target_dim ({config.target_dim}) >= concatenated dimension ({concat_dim}). "
                f"Dimension reduction will have no effect."
            )
    
    # Check weight configuration
    if config.fusion_method == "weighted":
        total_weight = config.audio_weight + config.text_weight
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            logger.warning(
                f"Audio and text weights sum to {total_weight}, not 1.0. "
                f"This may lead to unexpected results."
            )
    
    logger.info(
        f"Fusion config validated: method={config.fusion_method}, "
        f"audio_dim={audio_dim}, text_dim={text_dim}"
    )
