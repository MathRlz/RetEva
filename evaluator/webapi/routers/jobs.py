"""Job-related WebAPI endpoints."""

from typing import Any, Dict, Optional

import json
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from evaluator.webapi.form_builder import prepare_run_config
from evaluator.webapi.jobs import JobManager
from evaluator.webapi.schemas import ErrorResponse, EvaluationJobRequest, JobSubmitResponse, MatrixJobRequest
from evaluator.webapi.utils import artifact_listing


def _require_job(jobs: JobManager, job_id: str):
    """Fetch a job record or raise HTTP 404."""
    try:
        return jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


# Non-terminal/terminal statuses that have no result to return -> (code, detail).
# A None detail means "use the job's own error message".
_NO_RESULT_STATUS = {
    "queued": (409, "Job still running"),
    "running": (409, "Job still running"),
    "failed": (500, None),
    "cancelled": (409, "Job cancelled"),
}


def _reject_non_result_status(job) -> None:
    """Raise the appropriate HTTPException when a job has no usable result."""
    mapped = _NO_RESULT_STATUS.get(job.status)
    if mapped is not None:
        code, detail = mapped
        raise HTTPException(status_code=code, detail=detail or job.error or "Job failed")
    if job.result is None:
        raise HTTPException(status_code=500, detail="Job completed but produced no result")


def build_jobs_router(jobs: JobManager) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/api/jobs/evaluation",
        response_model=JobSubmitResponse,
        summary="Submit evaluation job",
        responses={400: {"model": ErrorResponse}},
    )
    def submit_evaluation(payload: EvaluationJobRequest) -> Dict[str, str]:
        """Submit an async evaluation job. Poll /api/jobs/{job_id} for status."""
        try:
            config = prepare_run_config(payload.config, auto_devices=payload.auto_devices)
            job = jobs.submit_evaluation(config)
            return {"job_id": job.job_id}
        except ImportError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post(
        "/api/jobs/matrix",
        response_model=JobSubmitResponse,
        summary="Submit matrix evaluation job",
        responses={400: {"model": ErrorResponse}},
    )
    def submit_matrix(payload: MatrixJobRequest) -> Dict[str, str]:
        """Submit an async matrix evaluation job with multiple test setups."""
        try:
            config = prepare_run_config(payload.base_config, auto_devices=payload.auto_devices)
            job = jobs.submit_matrix(config, payload.test_setups, baseline_setup_id=payload.baseline_setup_id)
            return {"job_id": job.job_id}
        except ImportError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/jobs", summary="List all jobs")
    def list_jobs() -> Dict[str, Any]:
        """Return all in-memory job records (queued, running, completed, failed)."""
        return {"jobs": jobs.list_jobs()}

    @router.get(
        "/api/jobs/{job_id}",
        summary="Job status",
        responses={404: {"model": ErrorResponse}},
    )
    def job_status(job_id: str) -> Dict[str, Any]:
        """Return current status of a job. Poll this endpoint until status is terminal."""
        return _require_job(jobs, job_id).to_dict()

    @router.post(
        "/api/jobs/{job_id}/cancel",
        summary="Cancel job",
        responses={404: {"model": ErrorResponse}},
    )
    def cancel_job(job_id: str) -> Dict[str, Any]:
        """Request cancellation of a running job."""
        _require_job(jobs, job_id)
        accepted = jobs.request_cancel(job_id)
        return {"job_id": job_id, "accepted": accepted, "status": jobs.get(job_id).status}

    @router.get(
        "/api/jobs/{job_id}/result",
        summary="Job result",
        responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    )
    def job_result(job_id: str) -> Dict[str, Any]:
        """Return evaluation result for a completed job. 409 if still running."""
        job = _require_job(jobs, job_id)
        _reject_non_result_status(job)
        return {"job_id": job_id, "result": job.result}

    @router.get(
        "/api/jobs/{job_id}/metadata",
        summary="Job metadata",
        responses={404: {"model": ErrorResponse}},
    )
    def job_metadata(job_id: str) -> Dict[str, Any]:
        """Return config snapshot and metadata for a job."""
        job = _require_job(jobs, job_id)

        result_metadata = {}
        if isinstance(job.result, dict):
            result_metadata = job.result.get("metadata", {}) if isinstance(job.result.get("metadata"), dict) else {}
        return {
            "job": job.to_dict(),
            "config": job.config_snapshot,
            "metadata": result_metadata,
        }

    @router.get(
        "/api/jobs/{job_id}/log",
        summary="Captured console output for a job",
        responses={404: {"model": ErrorResponse}},
    )
    def job_log(job_id: str, tail: int = 0) -> Dict[str, Any]:
        """Return the subprocess console log (optionally only the last ``tail`` lines)."""
        _require_job(jobs, job_id)
        return {"job_id": job_id, "log": jobs.get_log(job_id, tail=tail or None)}

    @router.get("/api/jobs/{job_id}/artifacts")
    def job_artifacts(job_id: str) -> Dict[str, Any]:
        job = _require_job(jobs, job_id)

        output_dir: Optional[str] = None
        if job.config_snapshot is not None:
            output_dir = job.config_snapshot.get("output_dir")
        artifacts = artifact_listing(output_dir)
        return {"job_id": job_id, "output_dir": output_dir, "artifacts": artifacts}

    @router.get("/api/jobs/{job_id}/progress", summary="SSE progress stream")
    def job_progress_stream(job_id: str, poll_interval: float = 0.5):
        """Stream job progress events as Server-Sent Events until job is terminal."""
        _require_job(jobs, job_id)

        def _generate():
            sent = 0
            while True:
                try:
                    job = jobs.get(job_id)
                except KeyError:
                    break
                events = job.progress
                while sent < len(events):
                    data = json.dumps(events[sent])
                    yield f"data: {data}\n\n"
                    sent += 1
                if job.status in {"completed", "failed", "cancelled"}:
                    yield f"data: {json.dumps({'phase': 'terminal', 'status': job.status})}\n\n"
                    break
                time.sleep(poll_interval)

        return StreamingResponse(_generate(), media_type="text/event-stream")

    return router
