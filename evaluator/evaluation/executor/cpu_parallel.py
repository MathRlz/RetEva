"""Order-preserving parallel map for CPU-bound per-item stage work: a per-item unit is safe to
run this way only if order-preserving and determinism-neutral (depends only on its own item,
via per-item seeding). Default backend is ``sync``, so this is inert until a stage opts in."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, List, Sequence, Tuple

BACKENDS = ("sync", "thread", "process")


def _worker_count(workers: int, n_items: int) -> int:
    auto = os.cpu_count() or 1
    requested = workers if workers and workers > 0 else auto
    return max(1, min(requested, n_items))


def parallel_map(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    *,
    backend: str = "sync",
    workers: int = 0,
) -> List[Any]:
    """Map ``fn`` over ``items``, returning results in input order. ``backend="process"``
    requires ``fn`` + items to be picklable. The first worker exception propagates; an empty
    input returns ``[]``."""
    if backend not in BACKENDS:
        raise ValueError(f"cpu_stage_executor must be one of {BACKENDS}, got {backend!r}")
    items = list(items)
    if not items or backend == "sync":
        return [fn(x) for x in items]
    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    with executor_cls(max_workers=_worker_count(workers, len(items))) as ex:
        return list(ex.map(fn, items))  # map preserves input order


def resolve_cpu_backend(config: Any) -> Tuple[str, int]:
    """``(backend, workers)`` from a config's ``cpu_stage_executor`` / ``cpu_stage_workers``
    (validated). Absent config → the inert ``("sync", 0)`` default."""
    backend = str(getattr(config, "cpu_stage_executor", "sync") or "sync")
    if backend not in BACKENDS:
        raise ValueError(f"cpu_stage_executor must be one of {BACKENDS}, got {backend!r}")
    return backend, int(getattr(config, "cpu_stage_workers", 0) or 0)


def run_per_item(s: Any, fn: Callable, items: Sequence[Any], **bound: Any) -> List[Any]:
    """Map a picklable per-item ``fn`` over ``items`` via :func:`parallel_map`, resolving the
    backend from the run config (``s.config``); ``**bound`` are bound onto ``fn`` via
    ``functools.partial``."""
    import functools

    backend, workers = resolve_cpu_backend(s.config)
    f = functools.partial(fn, **bound) if bound else fn
    return parallel_map(f, items, backend=backend, workers=workers)
