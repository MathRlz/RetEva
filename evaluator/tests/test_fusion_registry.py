"""Tests for hybrid fusion registry strategies."""

import pytest

from evaluator.models.retrieval.fusion_registry import (
    FUSION_REGISTRY,
    fuse_hybrid_results,
)


def test_fusion_registry_contains_expected_strategies():
    assert {"weighted", "rrf", "max_score"}.issubset(set(FUSION_REGISTRY.keys()))


def test_max_score_fusion_prefers_best_branch_score():
    dense = [({"doc_id": "a"}, 10.0), ({"doc_id": "b"}, 1.0)]
    sparse = [({"doc_id": "b"}, 10.0), ({"doc_id": "a"}, 1.0)]
    results = fuse_hybrid_results(
        "max_score",
        dense,
        sparse,
        dense_weight=0.5,
        top_k=2,
        rrf_k=60,
    )
    ids = [item["doc_id"] for item, _ in results]
    assert set(ids) == {"a", "b"}


def test_unknown_fusion_method_raises():
    with pytest.raises(ValueError, match="Unsupported hybrid fusion method"):
        fuse_hybrid_results(
            "unknown",
            [("a", 1.0)],
            [("a", 1.0)],
            dense_weight=0.5,
            top_k=1,
            rrf_k=60,
        )
