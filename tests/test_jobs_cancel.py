"""webapi job cancellation works even for a silent subprocess (audit A3/F3)."""

import threading
import time

from evaluator.webapi import jobs as jobs_mod
from evaluator.webapi.jobs import JobManager, JobRecord, _JobCancelled


def _manager():
    return JobManager(evaluation_runner=lambda *a, **k: {})


def test_cancel_unblocks_a_silent_subprocess(tmp_path, monkeypatch):
    jm = _manager()
    job_id = "j1"
    with jm._lock:
        jm._jobs[job_id] = JobRecord(job_id=job_id, job_type="eval", config_snapshot={})

    # A child that produces NO stdout and runs a long time — the old blocking
    # ``for line in stdout`` loop would never re-check the cancel flag.
    monkeypatch.setattr(
        jobs_mod, "_cli_command",
        lambda interp, cfg: [interp, "-c", "import time; time.sleep(30)"],
    )
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("x: 1\n")

    outcome = {}

    def run():
        try:
            jm._launch_cli(job_id, cfg_path)
        except _JobCancelled:
            outcome["cancelled"] = True
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = repr(exc)

    t = threading.Thread(target=run)
    t.start()
    time.sleep(1.0)  # let the subprocess start + the reader/poll loop spin up
    jm.request_cancel(job_id)
    t.join(timeout=6)

    assert not t.is_alive(), "cancel did not unblock the silent subprocess"
    assert outcome.get("cancelled") is True, outcome
