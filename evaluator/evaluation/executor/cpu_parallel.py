"""Order-preserving parallel map for CPU-bound per-item stage work (Roadmap 4b).

Today's intra-level concurrency (``executor/parallel.py``) is thread-based and exchanges
artifacts through a shared in-process ``ctx`` — great for GPU/IO-bound *nodes*, but GIL-bound
for CPU stages (correction / augmentation / metrics), and a process pool can't share that ctx.
This is the orthogonal primitive: a parallel map over the per-item work *inside* one stage,
which a ProcessPool can speed up because each item is independent.

Two invariants make it safe to drop in for a stage's per-item loop:

- **order-preserving** — results come back in input order regardless of backend (``map`` keeps
  submission order), so downstream id↔value alignment is unchanged;
- **determinism-neutral** — the per-item function must depend only on its own item (the engine's
  per-item seeding, ``item_seed(seed, query_id, …)``, already guarantees this), so "sync",
  "thread", and "process" produce identical results.

Default backend is ``sync`` (today's in-line map), so this is inert until a stage opts in.
"""

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
    """Map ``fn`` over ``items``, returning results **in input order**.

    ``backend``: ``"sync"`` (in-line list), ``"thread"`` (:class:`ThreadPoolExecutor`), or
    ``"process"`` (:class:`ProcessPoolExecutor` — ``fn`` + items must be picklable). The first
    worker exception propagates. An empty input returns ``[]``."""
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
    backend from the run config (``s.config``). ``**bound`` are bound onto ``fn`` with
    ``functools.partial``. The single shape behind the 4b per-item stages (WER/CER, augment,
    correct, optimize, augment_audio)."""
    import functools

    backend, workers = resolve_cpu_backend(s.config)
    f = functools.partial(fn, **bound) if bound else fn
    return parallel_map(f, items, backend=backend, workers=workers)
