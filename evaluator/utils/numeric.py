"""Numeric helpers shared across the retrieval + storage layers."""

import numpy as np

from ..constants import MIN_NORM_THRESHOLD


def l2_normalize(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalize ``vectors`` along ``axis``, flooring the norm at ``MIN_NORM_THRESHOLD`` so a
    zero/near-zero vector stays ~zero instead of exploding (no division by zero). One canonical
    normalization for every embedding (query/corpus, dense/fused) keeps both sides of a cosine
    comparison from drifting apart.
    """
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    norms = np.maximum(norms, MIN_NORM_THRESHOLD)
    return vectors / norms
