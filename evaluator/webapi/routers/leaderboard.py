"""Leaderboard WebAPI endpoints."""

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from evaluator.analysis.leaderboard_views import leaderboard_rows, pareto_rows
from evaluator.storage import ExperimentStore


def delete_run_and_cache(
    run_id: int, *, delete_cache: bool = False, output_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """Delete a leaderboard run and, optionally, its cached vector-DB index.

    The vector-DB cache key is read from the run metadata
    (``metadata.cache.load.vector_cache_key``, persisted at ingest time). Used by
    the HTML UI (``/ui/runs/{id}/delete``). Raises 404 if the run is missing.
    """
    store = ExperimentStore(db_path=str(Path(output_dir) / "leaderboard.sqlite"))
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    cache_deleted = False
    if delete_cache:
        load = ((run.get("metadata") or {}).get("cache") or {}).get("load") or {}
        cache_key = load.get("vector_cache_key")
        cache_dir = ((run.get("config") or {}).get("cache") or {}).get("cache_dir")
        if cache_key and cache_dir:
            from evaluator.storage.cache import CacheManager
            cache_deleted = CacheManager(cache_dir=cache_dir).delete_vector_db(cache_key)

    deleted = store.delete_run(run_id)
    return {"run_id": run_id, "deleted": deleted, "cache_deleted": cache_deleted}


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
        return {
            "metric": metric,
            "rows": leaderboard_rows(
                metric=metric, limit=limit, output_dir=output_dir,
                dataset_name=dataset_name, pipeline_mode=pipeline_mode,
            ),
        }

    @router.get(
        "/api/leaderboard/pareto",
        summary="Cross-run Pareto frontier over an experiment group",
    )
    def leaderboard_pareto(
        experiment_group: str,
        objectives: str = "MRR:max",
        output_dir: str = "evaluation_results",
    ) -> Dict[str, Any]:
        try:
            return pareto_rows(experiment_group, objectives=objectives, output_dir=output_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return router
