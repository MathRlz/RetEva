"""Leaderboard WebAPI endpoints."""

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from evaluator.storage import ExperimentStore
from evaluator.webapi.schemas import ErrorResponse


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


def delete_run_and_cache(
    run_id: int, *, delete_cache: bool = False, output_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """Delete a leaderboard run and, optionally, its cached vector-DB index.

    The vector-DB cache key is read from the run metadata
    (``metadata.cache.load.vector_cache_key``, persisted at ingest time). Shared by
    the JSON DELETE endpoint and the HTML UI. Raises 404 if the run is missing.
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

    @router.delete(
        "/api/leaderboard/runs/{run_id}",
        summary="Delete a run (optionally its vector-DB cache)",
        responses={404: {"model": ErrorResponse}},
    )
    def delete_run(
        run_id: int,
        delete_cache: bool = False,
        output_dir: str = "evaluation_results",
    ) -> Dict[str, Any]:
        return delete_run_and_cache(
            run_id, delete_cache=delete_cache, output_dir=output_dir
        )

    return router
