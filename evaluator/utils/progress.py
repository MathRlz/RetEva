"""Consistent, threshold-gated tqdm progress bars for long data loops (audit C5/F27).

One helper so every data-heavy node shows the same bar style, and short runs / tests /
non-interactive output stay quiet automatically.

Visibility precedence (``progress_disabled``), highest first:

1. ``EVALUATOR_NO_PROGRESS`` set            → off (kill switch).
2. ``EVALUATOR_FORCE_PROGRESS`` set         → on, even when stderr is not a TTY. The webapi
   sets this for the eval subprocess (its stderr is a pipe the supervisor streams to the
   job console), so the bar still shows there.
3. otherwise                                → on for an interactive TTY, off when piped.

Every tqdm bar in the codebase routes its ``disable`` through :func:`progress_disabled` so
all three paths (terminal CLI, piped CLI, webapi job) behave consistently.
"""
import os
import sys

try:  # tqdm is a declared dependency, but never let a bar break a run
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# Below this many items a progress bar is just noise — iterate plainly.
PROGRESS_MIN_ITEMS = 50


def _interactive() -> bool:
    """True when stderr is an interactive terminal — a live bar only makes sense there.

    Bars draw to stderr (tqdm's default stream), so the TTY check tracks stderr. When the
    output is piped/redirected (logs, CI, a captured test stream), this is False.
    """
    stream = sys.stderr
    try:
        return bool(stream is not None and stream.isatty())
    except Exception:  # a replaced stream without isatty() — treat as non-interactive
        return False


def progress_disabled() -> bool:
    """Whether tqdm bars should be suppressed (the shared ``disable=`` decision).

    See the module docstring for the precedence: ``EVALUATOR_NO_PROGRESS`` forces off,
    ``EVALUATOR_FORCE_PROGRESS`` forces on (used by the webapi subprocess whose stderr is a
    pipe), otherwise show only in an interactive TTY.
    """
    if os.environ.get("EVALUATOR_NO_PROGRESS"):
        return True
    if os.environ.get("EVALUATOR_FORCE_PROGRESS"):
        return False
    return not _interactive()


def progress_iter(iterable, desc, *, total=None, unit="it", min_items=PROGRESS_MIN_ITEMS):
    """Wrap a long data loop in a uniform tqdm bar.

    The bar shows by default in an interactive terminal (and when forced via
    ``EVALUATOR_FORCE_PROGRESS`` — e.g. the webapi job subprocess). It is suppressed (the
    iterable returned unchanged) when the work is short (``total < min_items``), when
    ``EVALUATOR_NO_PROGRESS`` is set, or when tqdm is unavailable; when stderr is a
    non-forced pipe the bar is constructed disabled (a no-op passthrough).

    Args:
        iterable: the thing to iterate.
        desc: bar label.
        total: item count (inferred from ``len(iterable)`` when possible).
        unit: per-item unit label (e.g. "doc", "query", "clip").
        min_items: skip the bar below this many items.
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    if tqdm is None or os.environ.get("EVALUATOR_NO_PROGRESS"):
        return iterable
    if total is not None and total < min_items:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit, disable=progress_disabled())
