"""Numeric helpers shared across the retrieval + storage layers."""

import numpy as np

from ..constants import MIN_NORM_THRESHOLD


def l2_normalize(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalize ``vectors`` along ``axis``, flooring the norm at ``MIN_NORM_THRESHOLD``.

    Works for a single vector (1-D) or a batch (2-D); the shape is preserved. The floor means a
    zero / near-zero vector stays ~zero instead of exploding (no division by zero).

    One canonical normalization for *every* embedding — query and corpus, dense and fused — so the
    two sides of a cosine comparison can never drift apart. For any real embedding
    (norm ≫ 1e-12) this is exact unit-normalization.

    Examples:
        >>> emb = np.array([[1.0, 2.0, 2.0], [3.0, 4.0, 0.0]])
        >>> normed = l2_normalize(emb, axis=1)
        >>> bool(np.allclose(np.linalg.norm(normed, axis=1), 1.0))
        True
    """
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    norms = np.maximum(norms, MIN_NORM_THRESHOLD)
    return vectors / norms
