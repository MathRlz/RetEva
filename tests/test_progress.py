"""Threshold-gated progress bar helper (audit C5/F27)."""

import io

from evaluator.utils import progress as progress_mod
from evaluator.utils.progress import (
    PROGRESS_MIN_ITEMS,
    progress_disabled,
    progress_iter,
)


class _FakeStream(io.StringIO):
    """A stderr stand-in with a controllable isatty()."""

    def __init__(self, tty):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def test_short_run_skips_bar_returns_iterable_unchanged():
    items = list(range(5))
    assert progress_iter(items, "x") is items  # below threshold → no bar


def test_env_disables_bar(monkeypatch):
    monkeypatch.setenv("EVALUATOR_NO_PROGRESS", "1")
    items = list(range(1000))
    assert progress_iter(items, "x") is items


def test_long_run_wraps_but_yields_same_items(monkeypatch):
    monkeypatch.delenv("EVALUATOR_NO_PROGRESS", raising=False)
    items = list(range(PROGRESS_MIN_ITEMS + 10))
    wrapped = progress_iter(items, "x")
    assert wrapped is not items  # wrapped in a tqdm bar
    assert list(wrapped) == items  # ... yielding exactly the same items


def test_min_items_override_forces_bar_for_small_slow_loops(monkeypatch):
    monkeypatch.delenv("EVALUATOR_NO_PROGRESS", raising=False)
    items = list(range(3))
    assert progress_iter(items, "x") is items  # default threshold: no bar
    assert list(progress_iter(items, "x", min_items=1)) == items  # forced bar still iterates


def test_total_for_lenless_iterable(monkeypatch):
    monkeypatch.delenv("EVALUATOR_NO_PROGRESS", raising=False)
    gen = (i for i in range(PROGRESS_MIN_ITEMS + 5))
    assert list(progress_iter(gen, "x", total=PROGRESS_MIN_ITEMS + 5)) == list(
        range(PROGRESS_MIN_ITEMS + 5)
    )


def test_bar_is_enabled_by_default_in_a_tty(monkeypatch):
    # Default behaviour: an interactive terminal shows the bar.
    monkeypatch.delenv("EVALUATOR_NO_PROGRESS", raising=False)
    monkeypatch.setattr(progress_mod.sys, "stderr", _FakeStream(tty=True))
    assert progress_mod._interactive() is True
    bar = progress_iter(list(range(PROGRESS_MIN_ITEMS + 10)), "x")
    assert bar.disable is False  # bar actually drawing


def test_bar_is_quiet_when_stderr_is_not_a_tty(monkeypatch):
    # Piped/redirected output (logs, CI, captured streams): bar suppressed.
    monkeypatch.delenv("EVALUATOR_NO_PROGRESS", raising=False)
    monkeypatch.delenv("EVALUATOR_FORCE_PROGRESS", raising=False)
    monkeypatch.setattr(progress_mod.sys, "stderr", _FakeStream(tty=False))
    assert progress_mod._interactive() is False
    bar = progress_iter(list(range(PROGRESS_MIN_ITEMS + 10)), "x")
    assert bar.disable is True  # constructed but not drawing


def test_progress_disabled_precedence(monkeypatch):
    # NO_PROGRESS wins over everything; then FORCE; then the TTY check.
    monkeypatch.setattr(progress_mod.sys, "stderr", _FakeStream(tty=False))  # piped
    monkeypatch.delenv("EVALUATOR_NO_PROGRESS", raising=False)
    monkeypatch.delenv("EVALUATOR_FORCE_PROGRESS", raising=False)
    assert progress_disabled() is True  # piped, no overrides → off

    monkeypatch.setenv("EVALUATOR_FORCE_PROGRESS", "1")
    assert progress_disabled() is False  # forced on despite non-TTY (the webapi case)

    monkeypatch.setenv("EVALUATOR_NO_PROGRESS", "1")
    assert progress_disabled() is True  # kill switch beats force


def test_force_progress_enables_bar_when_piped(monkeypatch):
    # The webapi subprocess case: stderr is a pipe but EVALUATOR_FORCE_PROGRESS is set,
    # so the bar must be enabled (drawing) in the streamed job console.
    monkeypatch.delenv("EVALUATOR_NO_PROGRESS", raising=False)
    monkeypatch.setenv("EVALUATOR_FORCE_PROGRESS", "1")
    monkeypatch.setattr(progress_mod.sys, "stderr", _FakeStream(tty=False))
    bar = progress_iter(list(range(PROGRESS_MIN_ITEMS + 10)), "x")
    assert bar.disable is False  # forced on
