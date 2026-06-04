"""Async job tracking for WebAPI."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from evaluator import EvaluationConfig

from .utils import utc_now


@dataclass
class JobRecord:
    """In-memory job state for async API runs."""

    job_id: str
    job_type: str
    status: str = "queued"
    submitted_at: str = field(default_factory=utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    config_snapshot: Optional[Dict[str, Any]] = None
    cancel_requested: bool = False
    future: Optional[Future] = None
    progress: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe public representation."""
        last_progress = self.progress[-1] if self.progress else None
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cancel_requested": self.cancel_requested,
            "error": self.error,
            "has_result": self.result is not None,
            "has_config": self.config_snapshot is not None,
            "last_progress": last_progress,
        }


class JobManager:
    """Minimal async job manager for backend API."""

    def __init__(
        self,
        *,
        evaluation_runner: Callable[[EvaluationConfig], Any],
        matrix_runner: Callable[..., Dict[str, Any]],
        max_workers: int = 2,
    ) -> None:
        self._evaluation_runner = evaluation_runner
        self._matrix_runner = matrix_runner
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="evaluator-webapi")
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = Lock()

    def push_progress(self, job_id: str, phase: str, current: int, total: int, message: str) -> None:
        event = {
            "phase": phase,
            "current": current,
            "total": total,
            "message": message,
            "ts": utc_now(),
        }
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].progress.append(event)

    def submit_evaluation(self, config: EvaluationConfig) -> JobRecord:
        job_id = str(uuid4())

        def _cb(phase: str, current: int, total: int, message: str) -> None:
            self.push_progress(job_id, phase, current, total, message)

        return self._submit(
            "evaluation",
            lambda: self._evaluation_runner(config, progress_callback=_cb).to_dict(include_config=True),
            config_snapshot=config.to_dict(),
            job_id=job_id,
        )

    def submit_matrix(
        self,
        config: EvaluationConfig,
        test_setups: List[Dict[str, Any]],
        baseline_setup_id: Optional[str] = None,
    ) -> JobRecord:
        return self._submit(
            "matrix",
            lambda: self._matrix_runner(config, test_setups, baseline_setup_id=baseline_setup_id),
            config_snapshot=config.to_dict(),
        )

    def _submit(
        self,
        job_type: str,
        runner: Callable[[], Dict[str, Any]],
        *,
        config_snapshot: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> JobRecord:
        if job_id is None:
            job_id = str(uuid4())
        job = JobRecord(job_id=job_id, job_type=job_type, config_snapshot=config_snapshot)
        with self._lock:
            self._jobs[job_id] = job
        job.future = self._executor.submit(self._run_job, job_id, runner)
        return job

    def _run_job(self, job_id: str, runner: Callable[[], Dict[str, Any]]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_requested:
                job.status = "cancelled"
                job.finished_at = utc_now()
                return
            job.status = "running"
            job.started_at = utc_now()

        try:
            result = runner()
        except Exception as exc:  # surfaced to API clients
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = utc_now()
            return

        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_requested:
                job.status = "cancelled"
                job.finished_at = utc_now()
                return
            job.result = result
            job.status = "completed"
            job.finished_at = utc_now()

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [job.to_dict() for job in self._jobs.values()]

    def request_cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
            job.cancel_requested = True
            if job.status == "queued" and job.future and job.future.cancel():
                job.status = "cancelled"
                job.finished_at = utc_now()
                return True
            return job.status in {"cancelled", "completed", "failed"}
