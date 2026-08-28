"""Roadmap 4a: the cross-run Pareto UI route + group list (server-rendered, no JS needed)."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from evaluator.storage import ExperimentStore  # noqa: E402
from evaluator.webapi.app import create_app  # noqa: E402


def _seed_group(output_dir):
    store = ExperimentStore(db_path=str(output_dir / "leaderboard.sqlite"))
    for name, metrics in [
        ("fast", {"MRR": 0.70, "latency_ms": 5.0}),
        ("balanced", {"MRR": 0.80, "latency_ms": 10.0}),
        ("accurate", {"MRR": 0.90, "latency_ms": 20.0}),
        ("dominated", {"MRR": 0.75, "latency_ms": 25.0}),
    ]:
        store.upsert_run(
            experiment_name=name, dataset_name="ds", pipeline_mode="asr_text_retrieval",
            metrics=metrics, experiment_group="sweep-A",
        )
    return store


def test_experiment_groups_lists_distinct_groups(tmp_path):
    store = _seed_group(tmp_path)
    store.upsert_run(
        experiment_name="z", dataset_name="ds", pipeline_mode="asr_text_retrieval",
        metrics={"MRR": 0.5}, experiment_group="sweep-B",
    )
    assert store.experiment_groups() == ["sweep-A", "sweep-B"]


def test_pareto_ui_route_renders_frontier(tmp_path):
    _seed_group(tmp_path)
    client = TestClient(create_app())
    resp = client.get(
        "/ui/pareto",
        params={
            "experiment_group": "sweep-A",
            "objectives": "MRR:max,latency_ms:min",
            "output_dir": str(tmp_path),
        },
    )
    assert resp.status_code == 200
    html = resp.text
    # the four runs appear, with the frontier marker on the three non-dominated ones
    for name in ("fast", "balanced", "accurate", "dominated"):
        assert name in html
    assert "★" in html  # at least one frontier star rendered
    assert "Pareto frontier" in html


def test_pareto_ui_route_reports_bad_objectives(tmp_path):
    _seed_group(tmp_path)
    client = TestClient(create_app())
    resp = client.get(
        "/ui/pareto",
        params={"experiment_group": "sweep-A", "objectives": "MRR:sideways",
                "output_dir": str(tmp_path)},
    )
    assert resp.status_code == 200          # rendered, not a 500
    assert "max" in resp.text and "min" in resp.text  # the validation message is shown
