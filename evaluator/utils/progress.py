"""Consistent, threshold-gated tqdm progress bars for long data loops.

One helper so every data-heavy node shares the same bar style and short/test/non-interactive
runs stay quiet automatically.
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
    """True when stderr is an interactive terminal (bars draw to stderr, so a live bar only
    makes sense when that's a TTY, not piped/redirected)."""
    stream = sys.stderr
    try:
        return bool(stream is not None and stream.isatty())
    except Exception:  # a replaced stream without isatty() — treat as non-interactive
        return False


def progress_disabled() -> bool:
    """Whether tqdm bars should be suppressed (the shared ``disable=`` decision).

    Precedence: ``EVALUATOR_NO_PROGRESS`` forces off; ``EVALUATOR_FORCE_PROGRESS`` forces on
    (the webapi subprocess sets this since its stderr is a pipe, not a TTY); otherwise on
    only for an interactive TTY.
    """
    if os.environ.get("EVALUATOR_NO_PROGRESS"):
        return True
    if os.environ.get("EVALUATOR_FORCE_PROGRESS"):
        return False
    return not _interactive()


def progress_iter(iterable, desc, *, total=None, unit="it", min_items=PROGRESS_MIN_ITEMS):
    """Wrap a long data loop in a uniform tqdm bar.

    Suppressed (returns ``iterable`` unchanged) when the work is short (``total <
    min_items``), ``EVALUATOR_NO_PROGRESS`` is set, or tqdm is unavailable. Otherwise shows
    for an interactive TTY or when ``EVALUATOR_FORCE_PROGRESS`` is set (e.g. the webapi job
    subprocess); a non-forced piped run still builds the bar, just disabled (a no-op
    passthrough).
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
