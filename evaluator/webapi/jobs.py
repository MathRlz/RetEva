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
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from evaluator import EvaluationConfig, run_evaluation, run_evaluation_matrix
from evaluator.cli.utils import generate_output_filename
from evaluator.evaluation.results import EvaluationResults
from evaluator.logging_config import get_logger
from evaluator.storage.leaderboard import ExperimentStore

from .utils import utc_now

logger = get_logger(__name__)

# The eval CLI runs as a module (`python -m evaluator.cli`) — the repo-root
# `evaluate.py` wrapper is not shipped in the wheel, so a file-relative path breaks
# for pip-installed servers. cwd stays the server's cwd so relative dataset/config
# paths resolve exactly like the preset loading does.
_CLI_MODULE = "evaluator.cli"


def _cli_command(interpreter: str, cfg_path: Path) -> list:
    """Argv for one eval job (module form — patchable test seam)."""
    return [interpreter, "-m", _CLI_MODULE, "--config", str(cfg_path)]
_PROGRESS_FILENAME = "progress.jsonl"
_ERR_TAIL_LINES = 50
_LOG_CAP = 5000  # max console lines retained per job
_CANCEL_POLL_S = 0.5  # how often to re-check for a cancel while the child runs


# Env vars that importing torch injects into the webapi server process; inherited by the
# eval subprocess they perturb its OpenMP init (KMP_INIT_AT_FORK / KMP_DUPLICATE_LIB_OK).
# Stripped so the child starts from a clean baseline.
_TORCH_INJECTED_ENV = ("KMP_DUPLICATE_LIB_OK", "KMP_INIT_AT_FORK", "TORCHINDUCTOR_CACHE_DIR")


def _subprocess_env(progress_path: Optional[Path] = None) -> Dict[str, str]:
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
    # Structured node-granular progress: the executor's ProgressSink.from_env writes
    # node_start/node_complete JSONL here, which the supervisor tails (no stdout scraping).
    if progress_path is not None:
        env["EVALUATOR_PROGRESS_FILE"] = str(progress_path)
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
        evaluation_runner: Callable[..., Any],
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

            self._launch_cli(job_id, cfg_path, jobdir / _PROGRESS_FILENAME)

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

    def _launch_cli(
        self, job_id: str, cfg_path: Path, progress_path: Optional[Path] = None
    ) -> None:
        """Launch evaluate.py, stream logs + structured progress, raise on cancel/non-zero exit.

        Console output is captured for the log + error tail; *progress* comes from the
        subprocess's ``EVALUATOR_PROGRESS_FILE`` JSONL (node_start/node_complete events),
        tailed incrementally — no brittle parsing of human-readable stdout lines.
        """
        interpreter = _subprocess_python()
        logger.info("job %s: launching eval subprocess with %s", job_id, interpreter)
        proc = subprocess.Popen(
            _cli_command(interpreter, cfg_path),
            cwd=os.getcwd(),
            env=_subprocess_env(progress_path),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        with self._lock:
            self._jobs[job_id].process = proc

        assert proc.stdout is not None
        prog = _ProgressTail(progress_path)

        # Drain stdout in a reader thread so a cancel is honored on a fixed cadence even when
        # the child goes silent — the blocking ``for line in stdout`` could otherwise stall
        # indefinitely and never re-check the cancel flag (F3).
        def _drain() -> None:
            for raw in proc.stdout:
                self._append_log(job_id, raw.rstrip())
                for event in prog.read_new():
                    self._push_progress_event(job_id, event)

        reader = Thread(target=_drain, name=f"job-{job_id}-stdout", daemon=True)
        reader.start()
        while reader.is_alive():
            if self._cancel_requested(job_id):
                _kill_process_group(proc)
                break
            reader.join(timeout=_CANCEL_POLL_S)
        proc.wait()
        reader.join(timeout=2)  # let it drain remaining output + observe EOF
        proc.stdout.close()
        # Final drain: events emitted after the last stdout line we read.
        for event in prog.read_new():
            self._push_progress_event(job_id, event)
        prog.close()

        if self._cancel_requested(job_id):
            raise _JobCancelled()
        if proc.returncode != 0:
            tail = "\n".join(self.get_log(job_id, tail=_ERR_TAIL_LINES))
            raise _JobFailed(f"worker exited with code {proc.returncode}\n{tail}")

    def _push_progress_event(self, job_id: str, record: Dict[str, Any]) -> None:
        """Translate one structured progress record (ProgressSink JSONL) into the job's
        progress feed: node name as the phase, level/total_levels as the progress fraction."""
        event = str(record.get("event", ""))
        node = str(record.get("node", ""))
        level = int(record.get("level", 0) or 0)
        total = int(record.get("total_levels", 0) or 0)
        self.push_progress(job_id, node or event, level, total, f"{event} {node}".strip())

    def _ingest_leaderboard(self, original_config, cfg, result: Dict[str, Any]) -> None:
        """Record the run on the leaderboard (best-effort; never fails the job).

        On failure the run still 'succeeds', but the error is surfaced on the result as
        ``leaderboard_warning`` (not only logged) so a stale leaderboard is visible to the
        caller instead of silent."""
        try:
            raw_metadata = result.get("metadata")
            metadata: Dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
            store = ExperimentStore(
                db_path=str(Path(original_config.output_dir) / "leaderboard.sqlite")
            )
            store.ingest_result(EvaluationResults(metrics=result, config=cfg, metadata=metadata))
        except Exception as exc:  # noqa: BLE001
            logger.warning("leaderboard ingest failed: %s", exc)
            if isinstance(result, dict):
                result["leaderboard_warning"] = f"leaderboard ingest failed: {exc}"

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


class _ProgressTail:
    """Incrementally reads a subprocess's progress JSONL (one event per line).

    Opens the file lazily (the subprocess creates it), yields each newly-appended
    *complete* (newline-terminated) line as a parsed record, and re-seeks past a partial
    trailing write so a half-flushed line is retried on the next poll.
    """

    def __init__(self, path: Optional[Path]) -> None:
        self._path = path
        self._fh = None

    def read_new(self) -> List[Dict[str, Any]]:
        if self._fh is None:
            if self._path is None or not self._path.exists():
                return []
            self._fh = open(self._path, "r", encoding="utf-8")
        records: List[Dict[str, Any]] = []
        while True:
            pos = self._fh.tell()
            line = self._fh.readline()
            if not line.endswith("\n"):  # EOF or partial line — rewind, wait for more
                self._fh.seek(pos)
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue  # skip a malformed line rather than break the feed
        return records

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _kill_process_group(proc) -> None:
    """Terminate the subprocess and its children (DataLoader workers, etc.)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
