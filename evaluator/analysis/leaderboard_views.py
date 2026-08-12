"""Leaderboard read views: store queries serialized for BOTH the web API/UI and the CLI.

Pure aggregation over ``ExperimentStore`` — lives in analysis/ (not a web router) so the CLI
does not depend on the web layer for what is just data shaping.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from evaluator.storage import ExperimentStore


def leaderboard_rows(
    metric: str = "MRR",
    limit: int = 20,
    output_dir: str = "evaluation_results",
    dataset_name: Optional[str] = None,
    pipeline_mode: Optional[str] = None,
    name: Optional[str] = None,
    model_filters: Optional[Dict[str, str]] = None,
) -> list:
    """Query the leaderboard store and return serializable row dicts.

    Shared by the JSON API and the HTML UI so both read the same shape.
    """
    store = ExperimentStore(db_path=str(Path(output_dir) / "leaderboard.sqlite"))
    rows = store.query_leaderboard(
        metric_name=metric,
        limit=limit,
        dataset_name=dataset_name,
        pipeline_mode=pipeline_mode,
        name=name,
        model_filters=model_filters,
    )
    return [
        {
            "run_id": row.run_id,
            "experiment_name": row.experiment_name,
            "dataset_name": row.dataset_name,
            "pipeline_mode": row.pipeline_mode,
            "metric_value": row.metric_value,
            "duration_seconds": row.duration_seconds,
            "created_at": row.created_at,
        }
        for row in rows
    ]


def pareto_rows(
    experiment_group: str,
    objectives: str = "MRR:max",
    output_dir: str = "evaluation_results",
) -> Dict[str, Any]:
    """The cross-run Pareto view for a run group (Roadmap 4a): every comparable run tagged
    ``on_frontier``, plus the frontier subset. ``objectives`` = ``"MRR:max,latency_ms:min"``."""
    from evaluator.analysis.pareto import annotate_pareto, parse_objectives, pareto_frontier

    objs = parse_objectives(objectives)
    store = ExperimentStore(db_path=str(Path(output_dir) / "leaderboard.sqlite"))
    runs = store.group_runs(experiment_group)
    return {
        "experiment_group": experiment_group,
        "objectives": [{"metric": n, "direction": d} for n, d in objs],
        "rows": annotate_pareto(runs, objs),
        "frontier": pareto_frontier(runs, objs),
    }
