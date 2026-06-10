"""Per-item failure isolation: drop-and-log, not abort-the-run (architecture §3, task T1).

The design contract is that one bad item must not sink a whole evaluation — a node that throws
on a single query drops *that id* (logged + recorded) and the run continues, the id simply absent
from the downstream measurement. Until now only ASR honored this; every other node aborted.

Two helpers cover the two execution shapes in the executor:

- ``isolate_items`` — for a genuine per-item loop (e.g. an LLM call per query). Runs ``fn(id, x)``
  per item, catching exceptions, and returns only the survivors plus the dropped ids.
- ``isolate_batch`` — for a *batched* op (embedding, vector search) feeding the executor's
  positional ``all_*`` lists. A physical drop there would desync those parallel lists, so on a
  batch failure it retries per item and substitutes a caller-supplied ``placeholder`` for a
  failing item (keeping alignment) while still recording the dropped id. The keyed ``report``
  path (the source of truth) then excludes those ids, so a placeholder never reaches a metric.

Both funnel dropped ids into a ``DropSink`` so provenance can surface ``dropped_by_node`` (shares
the S2 plumbing).
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..logging_config import get_logger

logger = get_logger(__name__)


class DropSink:
    """Accumulates per-node dropped item ids across the run (node_id → [query_id…]).

    Thread-safe (T5): branch handlers running in parallel may each drop items concurrently.
    """

    def __init__(self) -> None:
        self.by_node: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def record(self, node_id: str, item_id: str, exc: BaseException) -> None:
        with self._lock:
            self.by_node.setdefault(node_id, []).append(str(item_id))
        logger.warning(
            "dropped item '%s' at node '%s': %s: %s",
            item_id,
            node_id,
            type(exc).__name__,
            exc,
        )

    def all_dropped_ids(self) -> set:
        return {i for ids in self.by_node.values() for i in ids}


def isolate_items(
    ids: Sequence[str],
    values: Sequence[Any],
    fn: Callable[[str, Any], Any],
    *,
    node_id: str,
    sink: Optional[DropSink] = None,
) -> Tuple[List[str], List[Any]]:
    """Apply ``fn(id, value)`` per item, dropping (logging + recording) any that raise.

    Returns ``(kept_ids, results)`` aligned to each other — the failing items are simply absent.
    Use for terminal/per-item work where a physical drop does not desync positional state.
    """
    kept_ids: List[str] = []
    results: List[Any] = []
    for item_id, value in zip(ids, values):
        try:
            results.append(fn(item_id, value))
            kept_ids.append(item_id)
        except Exception as exc:  # noqa: BLE001 - drop-and-log is the contract
            if sink is not None:
                sink.record(node_id, item_id, exc)
    return kept_ids, results


def isolate_batch(
    ids: Sequence[str],
    values: Sequence[Any],
    batch_fn: Callable[[List[Any]], Sequence[Any]],
    item_fn: Callable[[Any], Any],
    *,
    node_id: str,
    placeholder: Any,
    sink: Optional[DropSink] = None,
) -> List[Any]:
    """Run ``batch_fn(values)``; on failure, retry per item with ``item_fn``.

    A per-item failure is recorded and its slot filled with ``placeholder`` so the result stays
    positionally aligned to ``values`` (no desync of the executor's parallel ``all_*`` lists).
    The keyed ``report`` excludes recorded ids, so the placeholder never reaches a metric.
    Returns a list aligned 1:1 with ``values``."""
    try:
        return list(batch_fn(list(values)))
    except Exception as exc:  # noqa: BLE001 - fall back to isolated per-item
        logger.warning(
            "batch op at node '%s' failed (%s: %s); retrying per item",
            node_id,
            type(exc).__name__,
            exc,
        )
    out: List[Any] = []
    for item_id, value in zip(ids, values):
        try:
            out.append(item_fn(value))
        except Exception as exc:  # noqa: BLE001 - drop-and-log
            if sink is not None:
                sink.record(node_id, item_id, exc)
            out.append(placeholder)
    return out
