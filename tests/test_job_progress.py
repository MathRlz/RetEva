"""Phase 3 — job-progress soft-stall hint: a still-running job whose last progress update is old
shows a "no update for Ns" note, so a hung run is visible instead of a frozen progress bar.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from evaluator.webapi.ui.jobs import _seconds_since_progress


def _ts(seconds_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


def test_seconds_since_progress_running():
    secs = _seconds_since_progress({"status": "running", "last_progress": {"ts": _ts(42)}})
    assert secs is not None and secs >= 40


def test_seconds_since_progress_none_when_terminal_or_missing():
    assert _seconds_since_progress(
        {"status": "completed", "last_progress": {"ts": _ts(99)}}) is None
    assert _seconds_since_progress({"status": "running", "last_progress": {}}) is None
    assert _seconds_since_progress({"status": "queued"}) is None
    assert _seconds_since_progress(
        {"status": "running", "last_progress": {"ts": "not-a-date"}}) is None


def _render_status(**ctx):
    pytest.importorskip("jinja2")
    from jinja2 import Environment, FileSystemLoader

    tdir = Path(__file__).resolve().parents[1] / "evaluator/webapi/templates"
    env = Environment(loader=FileSystemLoader(str(tdir)), autoescape=True)
    env.globals["asset_version"] = lambda *a: "1"
    env.globals["headline_metrics"] = []
    return env.get_template("_status.html").render(**ctx)


def _running_job():
    return {"status": "running",
            "last_progress": {"step": "asr", "current": 1, "total": 10, "ts": _ts(0)}}


def test_status_shows_stall_note_when_stale():
    html = _render_status(job=_running_job(), job_id="abcd1234",
                          result=None, log_text="", stale_seconds=42)
    assert "no progress update for 42s" in html


def test_status_hides_stall_note_when_fresh():
    html = _render_status(job=_running_job(), job_id="abcd1234",
                          result=None, log_text="", stale_seconds=3)
    assert "no progress update" not in html
