"""Job-related WebAPI endpoints."""

from typing import Dict

from fastapi import APIRouter, HTTPException

from evaluator import ConfigurationError
from evaluator.webapi.form_config import build_validated_run_config, prepare_run_config
from evaluator.webapi.jobs import JobManager
from evaluator.webapi.schemas import (
    ErrorResponse,
    EvaluationJobRequest,
    GraphRunRequest,
    JobSubmitResponse,
)


def build_jobs_router(jobs: JobManager) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/api/jobs/evaluation",
        response_model=JobSubmitResponse,
        summary="Submit evaluation job",
        responses={400: {"model": ErrorResponse}},
    )
    def submit_evaluation(payload: EvaluationJobRequest) -> Dict[str, str]:
        """Submit an async evaluation job; progress is tracked in the /ui job views."""
        try:
            config = prepare_run_config(payload.config, auto_devices=payload.auto_devices)
            job = jobs.submit_evaluation(config)
            return {"job_id": job.job_id}
        except ImportError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post(
        "/api/jobs/from-graph",
        response_model=JobSubmitResponse,
        summary="Run a graph built in the visual builder",
        responses={400: {"model": ErrorResponse}},
    )
    def submit_from_graph(payload: GraphRunRequest) -> Dict[str, str]:
        """Translate a builder canvas spec into a config and submit it as a job — closing the
        builder→run loop. An invalid graph (unknown node, no dataset, space mismatch) is a 400
        with the translator/validation message, not a 500.
        """
        try:
            # Shared translate + validate path (dataset choice, topology, embedding spaces) — the
            # same `build_validated_run_config` /ui/run-graph uses, so UI and API can't drift.
            config, _graph = build_validated_run_config(
                payload.spec,
                experiment_name=payload.experiment_name,
                auto_devices=payload.auto_devices,
            )
        except (ConfigurationError, ValueError, KeyError, ImportError) as exc:
            # ImportError: a configured vector-db backend (faiss/chromadb/qdrant) isn't
            # installed — a user-fixable config problem, not a server fault.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        job = jobs.submit_evaluation(config)
        return {"job_id": job.job_id}

    return router
