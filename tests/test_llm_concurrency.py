"""`map_completions`: real overlap, input order, isolated failures.

The LLM loops (answer generation, judging) are network-blocking, so concurrency is pure
wall-clock win — but only if it (a) actually overlaps, (b) never reorders results, since the
report must be identical to a serial run.
"""

import threading

import pytest

from evaluator.llm_client.parallel import map_completions


def test_requests_actually_overlap():
    """A barrier proves concurrency deterministically — no timing thresholds."""
    barrier = threading.Barrier(4, timeout=5)

    def _wait(i):
        barrier.wait()       # only returns if 4 workers are in flight together
        return i

    assert map_completions(range(4), _wait, workers=4, desc="t") == [0, 1, 2, 3]


def test_serial_path_does_not_overlap():
    """workers=1 is the old behaviour: the same barrier can never be satisfied."""
    barrier = threading.Barrier(4, timeout=0.5)

    def _wait(i):
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            return "alone"
        return "overlapped"

    assert map_completions(range(4), _wait, workers=1, desc="t") == ["alone"] * 4


def test_results_keep_input_order_despite_completion_order():
    import time

    def _slow_first(i):
        time.sleep(0.05 if i == 0 else 0.0)   # item 0 finishes last
        return i * 10

    assert map_completions(range(5), _slow_first, workers=5, desc="t") == [0, 10, 20, 30, 40]


def test_failures_are_the_callers_business():
    """`fn` owns its error handling (both call sites record a per-item failure); the map does
    not swallow anything, so a raising fn propagates rather than silently dropping an item."""
    def _boom(i):
        if i == 2:
            raise RuntimeError("item 2")
        return i

    with pytest.raises(RuntimeError, match="item 2"):
        map_completions(range(4), _boom, workers=2, desc="t")

    # the pattern the call sites use: catch inside, return a sentinel
    def _safe(i):
        try:
            return _boom(i)
        except RuntimeError:
            return None

    assert map_completions(range(4), _safe, workers=2, desc="t") == [0, 1, None, 3]


def test_no_pool_for_the_serial_path(monkeypatch):
    import evaluator.llm_client.parallel as mod

    def _explode(*a, **kw):
        raise AssertionError("workers=1 must not create a thread pool")

    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", _explode)
    assert mod.map_completions([1, 2], lambda x: x, workers=1, desc="t") == [1, 2]


def test_empty_input():
    assert map_completions([], lambda x: x, workers=4, desc="t") == []


def test_progress_follows_completion_not_submission_order():
    """A slow FIRST item must not hide the others: with an ordered iterator the display (and any
    heartbeat) freezes on the head of the line while workers keep finishing — which is how a
    stalled sweep looked exactly like a healthy one."""
    import threading

    completed: list = []
    release = threading.Event()

    def _fn(i):
        if i == 0:
            release.wait(timeout=5)        # finishes last
        completed.append(i)
        if len(completed) == 3:            # the other three are through
            release.set()
        return i

    out = map_completions(range(4), _fn, workers=4, desc="t")
    assert out == [0, 1, 2, 3]             # results still in INPUT order
    assert completed[-1] == 0              # but item 0 completed last


def test_heartbeat_reports_progress_when_the_bar_is_suppressed(monkeypatch, caplog):
    """`nohup` runs have no TTY, so bars are disabled and the log is the only signal."""
    import logging

    monkeypatch.setenv("EVALUATOR_NO_PROGRESS", "1")
    logger = logging.getLogger("evaluator.llm_client.parallel")
    logger.addHandler(caplog.handler)
    logger.setLevel(logging.INFO)
    try:
        map_completions(range(10), lambda i: i, workers=2, desc="beat")
    finally:
        logger.removeHandler(caplog.handler)
    assert "/10 done" in caplog.text
