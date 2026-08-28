"""C3/F25: leaderboard pre_limit is a bound `?` parameter (not f-string-interpolated SQL).

Functional guard: query_leaderboard still returns the right rows ordered by metric, the
pre-limit binding executes, and a value-bearing filter survives the round-trip.
"""

from evaluator.storage.leaderboard import ExperimentStore


def _store(tmp_path):
    s = ExperimentStore(str(tmp_path / "lb.db"))
    for i, mrr in enumerate([0.1, 0.9, 0.5]):
        s.upsert_run(
            experiment_name=f"exp{i}",
            dataset_name="ds",
            pipeline_mode="asr_text_retrieval",
            metrics={"MRR": mrr},
        )
    # a run missing the metric must be filtered out (not crash the bound query)
    s.upsert_run(
        experiment_name="no_mrr", dataset_name="ds",
        pipeline_mode="asr_text_retrieval", metrics={"Recall@5": 0.3},
    )
    return s


def test_query_leaderboard_orders_by_metric_and_binds_limit(tmp_path):
    s = _store(tmp_path)
    rows = s.query_leaderboard(metric_name="MRR", limit=10)
    assert [r.metric_value for r in rows] == [0.9, 0.5, 0.1]  # sorted desc, no-MRR dropped


def test_query_leaderboard_dataset_filter(tmp_path):
    s = _store(tmp_path)
    assert s.query_leaderboard(metric_name="MRR", dataset_name="ds")
    assert s.query_leaderboard(metric_name="MRR", dataset_name="missing") == []
