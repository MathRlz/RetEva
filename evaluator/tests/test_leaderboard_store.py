"""Tests for SQLite experiment leaderboard storage."""

from pathlib import Path

from evaluator import EvaluationConfig
from evaluator.evaluation.results import EvaluationResults
from evaluator.storage import ExperimentStore


def test_experiment_store_ingest_and_query(tmp_path):
    db_path = tmp_path / "leaderboard.sqlite"
    store = ExperimentStore(str(db_path))

    config = EvaluationConfig()
    config.experiment_name = "run-a"
    config.output_dir = str(tmp_path)
    result = EvaluationResults(
        metrics={"MRR": 0.42, "Recall@5": 0.88},
        config=config,
        metadata={"duration_seconds": 12.3},
    )
    run_id = store.ingest_result(result)
    assert run_id > 0

    rows = store.query_leaderboard(metric_name="MRR", limit=10)
    assert rows
    assert rows[0].experiment_name == "run-a"
    assert rows[0].metric_value == 0.42
    assert Path(db_path).exists()


def test_query_leaderboard_filters_non_numeric_metric(tmp_path):
    store = ExperimentStore(str(tmp_path / "leaderboard.sqlite"))
    config = EvaluationConfig()
    result = EvaluationResults(
        metrics={"MRR": "n/a"},
        config=config,
        metadata={},
    )
    store.ingest_result(result)
    rows = store.query_leaderboard(metric_name="MRR")
    assert rows == []
