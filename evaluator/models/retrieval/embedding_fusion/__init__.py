"""Fuse audio and text embeddings (weighted, concatenate, max-pool, average) for
multimodal retrieval."""

from typing import Optional, Tuple, Any
import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from ....logging_config import get_logger
from ....config import EmbeddingFusionConfig
from ....utils.numeric import l2_normalize

logger = get_logger(__name__)


def normalize_embeddings(embeddings: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalize embeddings — the canonical :func:`utils.numeric.l2_normalize`, so the fused
    query side and the vector-store corpus side normalize by the exact same rule."""
    return l2_normalize(embeddings, axis)


def reduce_dimensions(
    embeddings: np.ndarray,
    target_dim: int,
    method: str = "pca",
    reducer: Optional[Any] = None
) -> Tuple[np.ndarray, Any]:
    """Reduce embedding dimensionality via PCA or random projection.

    When ``reducer is None`` a new one is fit on ``embeddings`` and returned — **reuse that
    same fitted reducer** for any other embeddings that must live in the same reduced space
    (e.g. a corpus side, or a later run); fitting a second reducer projects into a *different*
    space. The fit is deterministic (``random_state=42``) but data-dependent, so it is not
    comparable across different inputs. The no-op case (``dim <= target_dim``) returns
    ``(embeddings, None)``.
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


def _require_same_dim(
    audio_embeddings: np.ndarray, text_embeddings: np.ndarray, method: str
) -> None:
    """Element-wise fusion needs matching widths; concatenate is the escape hatch."""
    audio_dim = audio_embeddings.shape[1]
    text_dim = text_embeddings.shape[1]
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
    _require_same_dim(audio_embeddings, text_embeddings, "Weighted")
    fused = config.audio_weight * audio_embeddings + config.text_weight * text_embeddings
    return _finalize_fused(fused, config.normalize_before_fusion)


def _fuse_average(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    config: EmbeddingFusionConfig,
) -> np.ndarray:
    _require_same_dim(audio_embeddings, text_embeddings, "Average")
    fused = (audio_embeddings + text_embeddings) / 2.0
    return _finalize_fused(fused, config.normalize_before_fusion)


def _fuse_max_pool(
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    config: EmbeddingFusionConfig,
) -> np.ndarray:
    _require_same_dim(audio_embeddings, text_embeddings, "Max pool")
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
    """Fuse audio and text embeddings via ``config.fusion_method`` ("weighted", "average",
    "max_pool", or "concatenate" with optional dimension reduction). Returns
    ``(fused_embeddings, reducer)``; reducer is None unless concatenate mode reduces
    dimensions."""
    audio_embeddings, text_embeddings = _ensure_2d_embeddings(
        audio_embeddings,
        text_embeddings,
    )

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

    if config.require_same_dimensions and audio_dim != text_dim:
        raise ValueError(
            f"require_same_dimensions=True but audio_dim={audio_dim} != text_dim={text_dim}"
        )

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
    """Validate that ``config`` is compatible with ``audio_dim``/``text_dim``; raises
    ``ValueError`` with a corrective suggestion when it isn't."""
    if not config.enabled:
        return

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
