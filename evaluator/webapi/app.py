"""FastAPI application exposing evaluator runtime as HTTP backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from evaluator import ConfigurationError, EvaluationConfig, run_evaluation, run_evaluation_matrix
from evaluator.pipeline import create_pipeline_from_config  # noqa: F401
from evaluator.services import ModelServiceProvider
from evaluator.services.evaluation_service import load_dataset  # noqa: F401
from evaluator.webapi.graph_store import GraphStore
from evaluator.webapi.jobs import JobManager
from evaluator.webapi.ui import build_ui_router
from evaluator.webapi.routers import (
    base as base_router,
    config as config_router,
    datasets as datasets_router,
    graphs as graphs_router,
    introspection as introspection_router,
    jobs as jobs_router,
    leaderboard as leaderboard_router,
    live as live_router,
    models as models_router,
    tts as tts_router,
)


def create_app(
    *,
    evaluation_runner: Callable[[EvaluationConfig], Any] = run_evaluation,
    matrix_runner: Callable[[EvaluationConfig, List[Dict[str, Any]]], Dict[str, Any]] = run_evaluation_matrix,
    provider_factory: Callable[[], ModelServiceProvider] = ModelServiceProvider,
    graph_store: Optional[GraphStore] = None,
) -> FastAPI:
    """Create FastAPI app for evaluator WebUI backend."""
    app = FastAPI(
        title="Evaluator Web API",
        version="0.2.0",
        description="HTTP backend for medical audio retrieval evaluation. "
                    "Supports async evaluation jobs, config management, "
                    "leaderboard queries, and live retrieval.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # A ConfigurationError is always a bad request (unknown/misspelled key, invalid
    # value). Handle it once here so every /api route can let it propagate instead of
    # wrapping each config call in try/except → HTTPException(400). (UI routes render
    # their own HTML error via _prepared_config_or_error and never reach this.)

    @app.exception_handler(ConfigurationError)
    async def _configuration_error_handler(  # noqa: F811 - registered, not called directly
        request: Request, exc: ConfigurationError
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    jobs = JobManager(
        evaluation_runner=evaluation_runner,
        matrix_runner=matrix_runner,
    )

    app.include_router(base_router.build_base_router(provider_factory))
    app.include_router(config_router.build_config_router(provider_factory))
    app.include_router(models_router.build_models_router(provider_factory))
    app.include_router(datasets_router.build_datasets_router())
    app.include_router(graphs_router.build_graphs_router(graph_store or GraphStore()))
    app.include_router(introspection_router.build_introspection_router())
    app.include_router(tts_router.build_tts_router())
    app.include_router(jobs_router.build_jobs_router(jobs))
    app.include_router(leaderboard_router.build_leaderboard_router())
    app.include_router(
        live_router.build_live_router(
            provider_factory,
            pipeline_factory=create_pipeline_from_config,
            dataset_loader=load_dataset,
        )
    )
    app.include_router(build_ui_router(provider_factory, jobs))

    # Mount only when the assets shipped — an install without package data must not
    # take the whole API down (the UI just loses its stylesheet).
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    else:  # pragma: no cover - packaging misconfiguration path
        logging.getLogger(__name__).warning(
            "webapi static assets missing at %s — /static not mounted "
            "(reinstall the package; setup.py ships them via package_data)",
            static_dir,
        )

    return app
