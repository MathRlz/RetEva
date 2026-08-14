"""Run LLM calls with several requests in flight.

The work is network-blocking (one synchronous POST per item) and the server batches concurrent
requests into shared forward passes, so a serial loop leaves the GPU idle between calls. Threads
are the right tool here: ``LLMClient`` holds no mutable state, ``CostTracker.record`` is
lock-guarded, and the response cache is only touched under ``use_cache``.

Results come back in INPUT ORDER — a concurrent run assigns every answer to the query that asked
for it, which a completion-order collect would not.

That is ordering, not bit-reproducibility: the SERVER batches concurrent requests together, and a
different batch composition changes the numerics enough to flip near-tie tokens. Measured on 12
questions (mistral, seed 42, temp 0): serial vs serial is 12/12 identical, serial vs 4-way
concurrent is 5/12 — same decisions, different wording. `concurrency: 1` (the default) is the
reproducible path.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, List

from ..logging_config import get_logger

logger = get_logger(__name__)


def map_completions(
    items: Iterable[Any],
    fn: Callable[[Any], Any],
    *,
    workers: int = 1,
    desc: str = "LLM",
    unit: str = "case",
) -> List[Any]:
    """Apply ``fn`` to every item, up to ``workers`` in flight, preserving input order.

    ``workers <= 1`` runs inline — the serial path stays exactly what it was, with no pool
    created. ``fn`` owns its error handling (both call sites already record a failure per item);
    nothing is swallowed here.
    """
    from tqdm import tqdm

    from ..utils.progress import progress_disabled

    materialized = list(items)
    total = len(materialized)
    if total == 0:
        return []

    if workers <= 1:
        return [
            fn(item)
            for item in tqdm(materialized, desc=desc, unit=unit,
                             disable=progress_disabled())
        ]

    from concurrent.futures import ThreadPoolExecutor

    workers = min(int(workers), total)
    logger.info("%s: %d item(s), %d concurrent request(s)", desc, total, workers)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm") as pool:
        # executor.map yields in submission order → the assembled list is order-stable even
        # though completion order is not.
        return list(
            tqdm(pool.map(fn, materialized), total=total, desc=desc, unit=unit,
                 disable=progress_disabled())
        )
