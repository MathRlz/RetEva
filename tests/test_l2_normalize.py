"""Pins the consolidated L2-normalization (audit). One canonical `l2_normalize` replaced 9 copies
of the matrix-floor / single-vector-guard / fusion-`where` rules across the 3 vector stores +
embedding_fusion. On real embeddings (norm >> MIN_NORM_THRESHOLD) all of them are identical — this
test fixes that equivalence so the consolidation can't silently change normalization.
"""

import numpy as np

from evaluator.constants import MIN_NORM_THRESHOLD
from evaluator.utils.numeric import l2_normalize


def test_matrix_and_vector_unit_norm():
    m = np.array([[3.0, 4.0], [5.0, 12.0]])
    assert np.allclose(np.linalg.norm(l2_normalize(m, axis=1), axis=1), 1.0)
    assert np.allclose(np.linalg.norm(l2_normalize(np.array([3.0, 4.0]))), 1.0)


def test_zero_vector_stays_zero():
    # floor (not 1.0) → a zero vector maps to ~zero, never NaN/inf
    assert np.allclose(l2_normalize(np.array([0.0, 0.0, 0.0])), 0.0)
    assert np.allclose(l2_normalize(np.zeros((2, 3)), axis=1), 0.0)


def test_matches_the_old_store_rule_on_real_embeddings():
    """The old store rule was `v / max(norm, eps)` — l2_normalize is identical for any norm that
    exceeds the floor (every real embedding)."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(16, 32)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    expected = v / np.maximum(norms, MIN_NORM_THRESHOLD)
    assert np.allclose(l2_normalize(v, axis=1), expected)


def test_fusion_normalizer_delegates():
    from evaluator.models.retrieval.embedding_fusion import normalize_embeddings

    v = np.array([[1.0, 2.0, 2.0], [3.0, 4.0, 0.0]])
    assert np.allclose(normalize_embeddings(v), l2_normalize(v, -1))


def test_inmemory_store_cosine_round_trip():
    """End-to-end (no models): the consolidated normalize still ranks the nearest vector first."""
    from evaluator.storage.vector_store import InMemoryVectorStore

    store = InMemoryVectorStore()
    store.build(np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), ["east", "north", "west"])
    top = store.search(np.array([0.9, 0.1]), k=1)
    assert top and top[0][0] == "east"
