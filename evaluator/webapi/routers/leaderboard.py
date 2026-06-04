"""Leaderboard WebAPI endpoints."""

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from evaluator.storage import ExperimentStore
from evaluator.webapi.schemas import ErrorResponse


def build_leaderboard_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/leaderboard", summary="Query leaderboard")
    def leaderboard(
        metric: str = "MRR",
        limit: int = 20,
        dataset_name: Optional[str] = None,
        pipeline_mode: Optional[str] = None,
        output_dir: str = "evaluation_results",
    ) -> Dict[str, Any]:
        store = ExperimentStore(db_path=str(Path(output_dir) / "leaderboard.sqlite"))
        rows = store.query_leaderboard(
            metric_name=metric,
            limit=limit,
            dataset_name=dataset_name,
            pipeline_mode=pipeline_mode,
        )
        return {
            "metric": metric,
            "rows": [
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
            ],
        }

    @router.get("/api/leaderboard/runs", summary="List leaderboard runs")
    def leaderboard_runs(
        limit: int = 100,
        dataset_name: Optional[str] = None,
        pipeline_mode: Optional[str] = None,
        output_dir: str = "evaluation_results",
    ) -> Dict[str, Any]:
        store = ExperimentStore(db_path=str(Path(output_dir) / "leaderboard.sqlite"))
        runs = store.list_runs(
            limit=limit,
            dataset_name=dataset_name,
            pipeline_mode=pipeline_mode,
        )
        return {"runs": runs}

    @router.get(
        "/api/leaderboard/runs/{run_id}",
        summary="Get run details",
        responses={404: {"model": ErrorResponse}},
    )
    def leaderboard_run(
        run_id: int,
        output_dir: str = "evaluation_results",
    ) -> Dict[str, Any]:
        store = ExperimentStore(db_path=str(Path(output_dir) / "leaderboard.sqlite"))
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    return router
