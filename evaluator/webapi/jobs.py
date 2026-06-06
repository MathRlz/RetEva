"""Async job tracking for WebAPI.

Each real evaluation runs as a **separate OS process**: the job supervisor launches the
CLI (`evaluate.py --config <tmp>.yaml`) per job. The CLI runs the bundle path that works
reliably; full process isolation means a crashing eval is a *failed job*, not a dead
server. The parent keeps a thread pool of lightweight supervisors that only launch and
monitor the subprocess and parse its output — they do no ML work. Injected (test/custom)
runners run in-process.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from evaluator import EvaluationConfig, run_evaluation, run_evaluation_matrix
from evaluator.cli.utils import generate_output_filename
from evaluator.evaluation.results import EvaluationResults
from evaluator.logging_config import get_logger
from evaluator.storage.leaderboard import ExperimentStore

from .utils import utc_now

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVALUATE_PY = _REPO_ROOT / "evaluate.py"
_PROGRESS_MARKERS = ("PHASE ", "STAGE ")
_ERR_TAIL_LINES = 50
_LOG_CAP = 5000  # max console lines retained per job


# Env vars that importing torch injects into the webapi server process; inherited by the
# eval subprocess they perturb its OpenMP init (KMP_INIT_AT_FORK / KMP_DUPLICATE_LIB_OK).
# Stripped so the child starts from a clean baseline.
_TORCH_INJECTED_ENV = ("KMP_DUPLICATE_LIB_OK", "KMP_INIT_AT_FORK", "TORCHINDUCTOR_CACHE_DIR")


def _subprocess_env() -> Dict[str, str]:
    """Environment for the eval subprocess: one OpenMP runtime + faulthandler.

    The webapi server loads torch + MKL-linked numpy/scipy/sklearn, i.e. two OpenMP
    runtimes (``libgomp`` + ``libiomp``); on a duplicate runtime torch's embedding gather
    (an OpenMP ``at::parallel_for``) can SIGSEGV the job (exit -11) — faulthandler traced it
    to ``torch.nn.functional.embedding``. Best practice is to keep ONE runtime, not to
    serialize: ``MKL_THREADING_LAYER=GNU`` makes MKL use ``libgomp`` (multithreading
    preserved). The torch-injected KMP vars are stripped so the child starts clean.
    ``EVALUATOR_SINGLE_THREAD=1`` is an explicit last-resort fallback that pins native ops to
    one thread (correctness over speed) for stacks the GNU layer doesn't reconcile.
    """
    env = dict(os.environ)
    for var in _TORCH_INJECTED_ENV:
        env.pop(var, None)
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    if env.get("EVALUATOR_SINGLE_THREAD", "").strip().lower() in ("1", "true", "yes"):
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
    env["EVALUATOR_FAULTHANDLER"] = "1"
    return env


def _subprocess_python() -> str:
    """Interpreter for the eval subprocess.

    Defaults to ``sys.executable`` (the server's interpreter). ``EVALUATOR_PYTHON`` overrides
    it, so an operator can target a known-good environment (e.g. the conda env where the CLI
    works) even when the server itself runs elsewhere — the subprocess's Python env, not the
    server's, is what loads torch and must be consistent.
    """
    return (os.environ.get("EVALUATOR_PYTHON") or "").strip() or sys.executable


class _JobFailed(Exception):
    """Subprocess produced no usable result (non-zero exit / missing output)."""


class _JobCancelled(Exception):
    """Job was cancelled while its subprocess was running."""


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
    process: Optional[Any] = None  # subprocess.Popen (subprocess mode only)
    progress: List[Dict[str, Any]] = field(default_factory=list)
    log: List[str] = field(default_factory=list)  # captured subprocess console output

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
        # Route over the one execution core. EVALUATOR_JOB_RUNNER selects how a real run
        # executes: "subprocess" (default — CLI subprocess, isolation: a crash = failed
        # job) or "service" (in-process api.run_evaluation — model reuse, no isolation).
        # Injected/custom runners (tests) always run in-process.
        self._real_runners = (
            evaluation_runner is run_evaluation and matrix_runner is run_evaluation_matrix
        )
        route = os.environ.get("EVALUATOR_JOB_RUNNER", "subprocess").strip().lower()
        self._subprocess = self._real_runners and route != "service"
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="evaluator-webapi")
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = Lock()

    def push_progress(self, job_id: str, phase: str, current: int, total: int, message: str) -> None:
        event = {"phase": phase, "current": current, "total": total, "message": message, "ts": utc_now()}
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].progress.append(event)

    def _append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.log.append(line)
            if len(job.log) > _LOG_CAP:
                del job.log[:-_LOG_CAP]

    def get_log(self, job_id: str, tail: Optional[int] = None) -> List[str]:
        """Captured console output for a job (optionally just the last ``tail`` lines)."""
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            lines = self._jobs[job_id].log
            return list(lines[-tail:]) if tail else list(lines)

    # ── submission ─────────────────────────────────────────────────────────────

    def submit_evaluation(self, config: EvaluationConfig) -> JobRecord:
        job_id = str(uuid4())

        def _inproc() -> Dict[str, Any]:
            cb = lambda *a: self.push_progress(job_id, *a)  # noqa: E731
            return self._evaluation_runner(config, progress_callback=cb).to_dict(include_config=True)

        return self._submit(
            "evaluation", job_id=job_id, config_snapshot=config.to_dict(),
            inproc=_inproc, subproc=lambda jid: self._run_eval_subprocess(jid, config),
        )

    def submit_matrix(
        self, config: EvaluationConfig, test_setups: List[Dict[str, Any]],
        baseline_setup_id: Optional[str] = None,
    ) -> JobRecord:
        job_id = str(uuid4())
        return self._submit(
            "matrix", job_id=job_id, config_snapshot=config.to_dict(),
            inproc=lambda: self._matrix_runner(config, test_setups, baseline_setup_id=baseline_setup_id),
            subproc=lambda jid: self._run_matrix_subprocess(jid, config, test_setups, baseline_setup_id),
        )

    def _submit(self, job_type, *, job_id, config_snapshot, inproc, subproc) -> JobRecord:
        job = JobRecord(job_id=job_id, job_type=job_type, config_snapshot=config_snapshot)
        with self._lock:
            self._jobs[job_id] = job
        job.future = self._executor.submit(self._supervise, job_id, inproc, subproc)
        return job

    # ── execution ──────────────────────────────────────────────────────────────

    def _supervise(self, job_id, inproc, subproc) -> None:
        if not self._start_running(job_id):
            return
        try:
            result = subproc(job_id) if self._subprocess else inproc()
        except _JobCancelled:
            self._finish(job_id, status="cancelled")
        except _JobFailed as exc:
            self._finish(job_id, status="failed", error=str(exc))
        except Exception as exc:  # surfaced to API clients
            self._finish(job_id, status="failed", error=str(exc))
        else:
            self._finish(job_id, status="completed", result=result)

    def _run_eval_subprocess(self, job_id: str, config: EvaluationConfig) -> Dict[str, Any]:
        """Run one evaluation via the CLI in an isolated process; return the result dict."""
        jobdir = Path(tempfile.mkdtemp(prefix="evaljob-"))
        try:
            cfg = deepcopy(config)
            cfg.output_dir = str(jobdir)
            cfg_path = jobdir / "config.yaml"
            cfg.to_yaml(str(cfg_path))

            self._launch_cli(job_id, cfg_path)

            result_file = jobdir / generate_output_filename(cfg)
            if not result_file.exists():
                raise _JobFailed("evaluation finished but produced no result file")
            with open(result_file, "r", encoding="utf-8") as handle:
                result = json.load(handle)

            self._ingest_leaderboard(config, cfg, result)
            return result
        finally:
            shutil.rmtree(jobdir, ignore_errors=True)

    def _run_matrix_subprocess(self, job_id, config, test_setups, baseline_setup_id) -> Dict[str, Any]:
        """Run each matrix setup as its own CLI subprocess; assemble the comparison."""
        from evaluator.services.evaluation_service import (
            _apply_setup_overrides, _build_comparison_bundle,
        )
        runs: List[Dict[str, Any]] = []
        for idx, setup in enumerate(test_setups):
            setup_id = setup.get("setup_id") or setup.get("name") or f"setup_{idx + 1:03d}"
            overrides = setup.get("overrides", {}) or {}
            variant = _apply_setup_overrides(deepcopy(config), overrides)
            if "experiment_name" not in overrides:
                variant.experiment_name = f"{config.experiment_name}__{setup_id}"
            result = self._run_eval_subprocess(job_id, variant)
            runs.append({"setup_id": setup_id, "result": result})
            self.push_progress(job_id, f"setup {setup_id} complete", idx + 1, len(test_setups), "")

        comparison = _build_comparison_bundle(runs, baseline_setup_id=baseline_setup_id)
        return {
            "base_experiment_name": config.experiment_name,
            "num_setups": len(test_setups),
            "runs": runs,
            "comparison": comparison,
        }

    def _launch_cli(self, job_id: str, cfg_path: Path) -> None:
        """Launch evaluate.py, stream progress, raise on cancel / non-zero exit."""
        interpreter = _subprocess_python()
        logger.info("job %s: launching eval subprocess with %s", job_id, interpreter)
        proc = subprocess.Popen(
            [interpreter, str(_EVALUATE_PY), "--config", str(cfg_path)],
            cwd=str(_REPO_ROOT),
            env=_subprocess_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        with self._lock:
            self._jobs[job_id].process = proc

        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            self._append_log(job_id, line)
            if any(marker in line for marker in _PROGRESS_MARKERS):
                self.push_progress(job_id, _extract_phase(line), 0, 0, line)
            if self._cancel_requested(job_id):
                _kill_process_group(proc)
                break
        proc.wait()

        if self._cancel_requested(job_id):
            raise _JobCancelled()
        if proc.returncode != 0:
            tail = "\n".join(self.get_log(job_id, tail=_ERR_TAIL_LINES))
            raise _JobFailed(f"worker exited with code {proc.returncode}\n{tail}")

    def _ingest_leaderboard(self, original_config, cfg, result: Dict[str, Any]) -> None:
        """Record the run on the leaderboard (best-effort; never fails the job)."""
        try:
            metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
            store = ExperimentStore(
                db_path=str(Path(original_config.output_dir) / "leaderboard.sqlite")
            )
            store.ingest_result(EvaluationResults(metrics=result, config=cfg, metadata=metadata))
        except Exception as exc:  # noqa: BLE001
            logger.warning("leaderboard ingest failed: %s", exc)

    # ── state transitions ──────────────────────────────────────────────────────

    def _start_running(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_requested:
                job.status = "cancelled"
                job.finished_at = utc_now()
                return False
            job.status = "running"
            job.started_at = utc_now()
            return True

    def _finish(self, job_id: str, *, status: str, result=None, error=None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_requested and status != "cancelled":
                status, result = "cancelled", None
            job.status = status
            job.result = result
            job.error = error
            job.process = None
            job.finished_at = utc_now()

    def _cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.cancel_requested)

    # ── queries / control ──────────────────────────────────────────────────────

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
            proc = job.process
        if proc is not None and proc.poll() is None:
            _kill_process_group(proc)
        with self._lock:
            return self._jobs[job_id].status in {"cancelled", "completed", "failed"}


def _extract_phase(line: str) -> str:
    """Pull a short phase label from a CLI log line for the progress feed."""
    for marker in _PROGRESS_MARKERS:
        if marker in line:
            return line.split(marker, 1)[1].strip() or marker.strip()
    return line.strip()


def _kill_process_group(proc) -> None:
    """Terminate the subprocess and its children (DataLoader workers, etc.)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
