"""Roadmap 3a groundwork: streaming config + windowed-node partition + window plan + ItemSet.concat."""

import numpy as np
import pytest

from evaluator.config.evaluation import EvaluationConfig
from evaluator.config.streaming import StreamingConfig
from evaluator.evaluation.executor.streaming import (
    CORPUS_KINDS,
    SOURCE_KINDS,
    partition_nodes_for_streaming,
    window_bounds,
)
from evaluator.evaluation.item_set import ItemSet
from evaluator.pipeline.graph.modes import build_stage_graph


# ── StreamingConfig ──────────────────────────────────────────────────


def test_streaming_config_default_off():
    assert StreamingConfig().window_size is None
    assert StreamingConfig().enabled is False
    assert EvaluationConfig().streaming.enabled is False


def test_streaming_config_enabled_and_validation():
    assert StreamingConfig(window_size=256).enabled is True
    with pytest.raises(ValueError, match="window_size"):
        StreamingConfig(window_size=0)
    with pytest.raises(ValueError, match="window_size"):
        StreamingConfig(window_size=-5)


def test_streaming_config_round_trips_through_evaluation_config():
    cfg = EvaluationConfig.from_dict({"streaming": {"window_size": 64}})
    assert cfg.streaming.window_size == 64 and cfg.streaming.enabled


# ── window_bounds ────────────────────────────────────────────────────


def test_window_bounds_partitions_the_range():
    assert window_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert window_bounds(8, 4) == [(0, 4), (4, 8)]


def test_window_bounds_degenerate_cases():
    assert window_bounds(0, 4) == []          # empty dataset → no windows
    assert window_bounds(5, 0) == [(0, 5)]    # non-positive window → whole set
    assert window_bounds(5, 99) == [(0, 5)]   # over-large window → whole set
    assert window_bounds(5, 5) == [(0, 5)]    # exactly n → whole set


def test_window_bounds_cover_every_index_once():
    bounds = window_bounds(23, 7)
    covered = [i for a, b in bounds for i in range(a, b)]
    assert covered == list(range(23))         # contiguous, no gaps/overlaps


# ── partition_nodes_for_streaming ────────────────────────────────────


def test_partition_is_an_exact_three_way_split():
    g = build_stage_graph("asr_text_retrieval", rerank_enabled=True)
    part = partition_nodes_for_streaming(g)
    # every node lands in exactly one role
    assert set(part.all_ids) == {n.id for n in g.nodes}
    assert len(part.all_ids) == len(g.nodes)  # no node duplicated/dropped


def test_partition_places_prelude_windowed_finalize_nodes():
    g = build_stage_graph("asr_text_retrieval", rerank_enabled=True)
    part = partition_nodes_for_streaming(g)
    # prelude = source + whole-corpus embed/index (run once, window-independent)
    assert {"dataset_source", "corpus_embedding", "vector_db"} <= set(part.prelude)
    # windowed = the per-item query producers (run per window)
    assert {"asr", "text_embedding", "retrieval", "rerank"} <= set(part.windowed)
    # finalize = per-item metrics + the report assembler (need the full per-item set)
    assert {"transcription_metrics", "retrieval_metrics", "metrics", "finalize"} <= set(
        part.finalize
    )


def test_source_and_corpus_kinds_are_disjoint():
    assert SOURCE_KINDS.isdisjoint(CORPUS_KINDS)


# ── ItemSet.concat (the windowed accumulator) ───────────────────────


def test_concat_preserves_order_for_list_values():
    a = ItemSet(["q1", "q2"], ["x", "y"])
    b = ItemSet(["q3"], ["z"])
    merged = ItemSet.concat([a, b])
    assert merged.ids == ["q1", "q2", "q3"]
    assert merged.values == ["x", "y", "z"]


def test_concat_stacks_ndarray_values():
    a = ItemSet(["q1"], np.array([[1.0, 2.0]]))
    b = ItemSet(["q2", "q3"], np.array([[3.0, 4.0], [5.0, 6.0]]))
    merged = ItemSet.concat([a, b])
    assert merged.values.shape == (3, 2)
    assert merged.ids == ["q1", "q2", "q3"]


def test_concat_empty_inputs():
    assert ItemSet.concat([]).ids == []
    assert ItemSet.concat([ItemSet.empty(), ItemSet(["q1"], ["a"])]).ids == ["q1"]


def test_concat_rejects_overlapping_windows():
    # overlapping ids across windows would silently double-count — must be a loud error
    with pytest.raises(ValueError, match="duplicate"):
        ItemSet.concat([ItemSet(["q1"], ["a"]), ItemSet(["q1"], ["b"])])
