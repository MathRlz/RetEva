"""IR metric primitives (pure functions)."""

import pytest

from evaluator.metrics.ir import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_reciprocal_rank_first_relevant_position():
    assert reciprocal_rank(["a", "b", "c"], {"b": 1}) == pytest.approx(0.5)
    assert reciprocal_rank(["b", "a", "c"], {"b": 1}) == pytest.approx(1.0)
    assert reciprocal_rank(["a", "c"], {"b": 1}) == pytest.approx(0.0)


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"b": 1, "c": 1}, 3) == pytest.approx(2 / 3)
    assert precision_at_k(["b", "a"], {"b": 1}, 2) == pytest.approx(0.5)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b": 1, "c": 1, "d": 1}, 3) == pytest.approx(2 / 3)
    assert recall_at_k(["b", "d"], {"b": 1, "d": 1}, 5) == pytest.approx(1.0)


def test_ndcg_perfect_and_bounds():
    assert ndcg_at_k(["b", "a", "c"], {"b": 1}, 3) == pytest.approx(1.0)
    val = ndcg_at_k(["a", "b", "c"], {"b": 1, "c": 1}, 3)
    assert 0.0 <= val <= 1.0


def test_average_precision_monotonic_cases():
    assert average_precision(["b"], {"b": 1}) == pytest.approx(1.0)
    assert average_precision(["a", "b"], {"b": 1}) == pytest.approx(0.5)


def test_no_relevant_docs_is_zero_not_error():
    assert reciprocal_rank(["a"], {}) == pytest.approx(0.0)
    assert recall_at_k(["a"], {}, 1) == pytest.approx(0.0)
