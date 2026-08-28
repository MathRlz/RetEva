"""Roadmap 4a: cross-run Pareto frontier — pure logic + group query + webapi endpoint."""

from pathlib import Path

import pytest

from evaluator.analysis.pareto import (
    annotate_pareto,
    dominates,
    parse_objectives,
    pareto_frontier,
)
from evaluator.storage import ExperimentStore


# ── pure logic ───────────────────────────────────────────────────────


def test_parse_objectives_defaults_to_max():
    assert parse_objectives("MRR") == [("MRR", "max")]
    assert parse_objectives("MRR:max, latency_ms:min") == [
        ("MRR", "max"), ("latency_ms", "min")
    ]
    with pytest.raises(ValueError):
        parse_objectives("MRR:sideways")
    with pytest.raises(ValueError):
        parse_objectives("")


_OBJ = [("recall", "max"), ("latency", "min")]


def test_dominates_semantics():
    assert dominates({"recall": 0.9, "latency": 10}, {"recall": 0.8, "latency": 12}, _OBJ)
    # a trade-off (better recall, worse latency) does not dominate
    assert not dominates({"recall": 0.9, "latency": 14}, {"recall": 0.8, "latency": 12}, _OBJ)
    # equality is not strict domination
    assert not dominates({"recall": 0.8, "latency": 12}, {"recall": 0.8, "latency": 12}, _OBJ)
    # a missing metric is incomparable
    assert not dominates({"recall": 0.9}, {"recall": 0.8, "latency": 12}, _OBJ)


_ROWS = [
    {"run_id": 1, "metrics": {"recall": 0.9, "latency": 20}},   # best recall, slow
    {"run_id": 2, "metrics": {"recall": 0.7, "latency": 5}},    # worst recall, fast
    {"run_id": 3, "metrics": {"recall": 0.8, "latency": 10}},   # middle
    {"run_id": 4, "metrics": {"recall": 0.6, "latency": 25}},   # dominated by 3
]


def test_pareto_frontier_is_the_non_dominated_set():
    frontier = {r["run_id"] for r in pareto_frontier(_ROWS, _OBJ)}
    assert frontier == {1, 2, 3}  # run 4 is dominated by run 3 (higher recall, lower latency)


def test_pareto_frontier_preserves_input_order():
    assert [r["run_id"] for r in pareto_frontier(_ROWS, _OBJ)] == [1, 2, 3]


def test_annotate_flags_frontier_and_excludes_incomparable():
    rows = _ROWS + [{"run_id": 5, "metrics": {"recall": 0.95}}]  # no latency → excluded
    flags = {r["run_id"]: r["on_frontier"] for r in annotate_pareto(rows, _OBJ)}
    assert flags == {1: True, 2: True, 3: True, 4: False}  # 5 absent (incomparable)


def test_single_objective_frontier_is_the_max():
    frontier = pareto_frontier(_ROWS, [("recall", "max")])
    assert {r["run_id"] for r in frontier} == {1}  # only the top recall is non-dominated


# ── group query + endpoint ───────────────────────────────────────────


def _store_with_group(tmp_path):
    store = ExperimentStore(db_path=str(tmp_path / "leaderboard.sqlite"))
    runs = [
        ("fast", {"MRR": 0.70, "latency_ms": 5.0}),
        ("balanced", {"MRR": 0.80, "latency_ms": 10.0}),
        ("accurate", {"MRR": 0.90, "latency_ms": 20.0}),
        ("dominated", {"MRR": 0.75, "latency_ms": 25.0}),
    ]
    for name, metrics in runs:
        store.upsert_run(
            experiment_name=name, dataset_name="ds", pipeline_mode="asr_text_retrieval",
            metrics=metrics, experiment_group="sweep-A",
        )
    # a run in a different group must not leak in
    store.upsert_run(
        experiment_name="other", dataset_name="ds", pipeline_mode="asr_text_retrieval",
        metrics={"MRR": 0.99, "latency_ms": 1.0}, experiment_group="sweep-B",
    )
    return store


def test_group_runs_returns_only_the_group(tmp_path):
    store = _store_with_group(tmp_path)
    rows = store.group_runs("sweep-A")
    assert {r["experiment_name"] for r in rows} == {"fast", "balanced", "accurate", "dominated"}
    assert all(set(r["metrics"]) == {"MRR", "latency_ms"} for r in rows)


def test_pareto_rows_endpoint_helper(tmp_path):
    from evaluator.analysis.leaderboard_views import pareto_rows

    _store_with_group(tmp_path)
    out = pareto_rows("sweep-A", objectives="MRR:max,latency_ms:min",
                      output_dir=str(tmp_path))
    # the dominated run is off the frontier; the three trade-off runs are on it
    on = {r["experiment_name"]: r["on_frontier"] for r in out["rows"]}
    assert on == {"fast": True, "balanced": True, "accurate": True, "dominated": False}
    assert {r["experiment_name"] for r in out["frontier"]} == {"fast", "balanced", "accurate"}
    assert out["objectives"] == [
        {"metric": "MRR", "direction": "max"},
        {"metric": "latency_ms", "direction": "min"},
    ]


def test_pareto_rows_helper_lives_next_to_store(tmp_path):
    _store_with_group(tmp_path)  # writes leaderboard.sqlite into tmp_path
    assert (Path(tmp_path) / "leaderboard.sqlite").exists()
