"""Run LLM calls with several requests in flight.

Threads, not async/multiprocessing: the work is network-blocking and the server batches
concurrent requests into one forward pass, so a serial loop leaves the GPU idle. Results come
back in INPUT order. Concurrency isn't bit-reproducible — a different batch composition flips
near-tie tokens (measured 5/12 differing, 4-way vs serial, same seed) — so `concurrency: 1`
is the reproducible default.

Thread-safe because LLMClient holds no mutable state, CostTracker.record is lock-guarded, and
the response cache is only touched under use_cache.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, List

from ..logging_config import get_logger
from ..utils.progress import progress_disabled

logger = get_logger(__name__)


def map_completions(
    items: Iterable[Any],
    fn: Callable[[Any], Any],
    *,
    workers: int = 1,
    desc: str = "LLM",
    unit: str = "case",
    heartbeat_s: float = 60.0,
) -> List[Any]:
    """Apply ``fn`` to every item, up to ``workers`` in flight, preserving input order.

    ``workers <= 1`` runs inline (no pool created). ``fn`` owns its error handling; nothing
    is swallowed here.
    """
    from tqdm import tqdm

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

    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    workers = min(int(workers), total)
    logger.info("%s: %d item(s), %d concurrent request(s)", desc, total, workers)

    results: List[Any] = [None] * total
    started = last_event = time.monotonic()
    done_count = 0
    next_beat = max(1, total // 10)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm") as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(materialized)}
        # Progress follows COMPLETION order (an ordered iterator would hide a stalled worker
        # behind a slow one — indistinguishable from healthy for hours); results still
        # assemble by index.
        pending = tqdm(as_completed(futures), total=total, desc=desc, unit=unit,
                       disable=progress_disabled())
        for future in pending:
            results[futures[future]] = future.result()
            done_count += 1
            now = time.monotonic()
            # Progress bars are suppressed off-TTY (every `nohup` run), hence this coarse
            # log heartbeat.
            if done_count % next_beat == 0 or now - last_event > heartbeat_s:
                logger.info(
                    "%s: %d/%d done (%.0fs elapsed, %d in flight)",
                    desc, done_count, total, now - started, total - done_count,
                )
            last_event = now
    return results
