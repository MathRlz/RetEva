"""Consistent, threshold-gated tqdm progress bars for long data loops (audit C5/F27).

One helper so every data-heavy node shows the same bar style, and short runs / tests /
non-interactive output stay quiet automatically.
"""
import os

try:  # tqdm is a declared dependency, but never let a bar break a run
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# Below this many items a progress bar is just noise — iterate plainly.
PROGRESS_MIN_ITEMS = 50


def progress_iter(iterable, desc, *, total=None, unit="it", min_items=PROGRESS_MIN_ITEMS):
    """Wrap a long data loop in a uniform tqdm bar.

    Returns the iterable unchanged (no bar) when the work is short (``total < min_items``),
    when ``EVALUATOR_NO_PROGRESS`` is set, or when tqdm is unavailable. Non-interactive
    output (no TTY) is handled by tqdm's own ``disable=None``.

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
    return tqdm(iterable, desc=desc, total=total, unit=unit, disable=None)
