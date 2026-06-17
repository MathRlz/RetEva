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
        # Per-drop attribution (R7): the failing node, item, and the exception that caused it,
        # so a shrinking sample becomes "10% failed embedding (RuntimeError: clip too long)"
        # instead of an opaque count.
        self.details: List[Dict[str, str]] = []
        self._lock = threading.Lock()

    def record(self, node_id: str, item_id: str, exc: BaseException) -> None:
        with self._lock:
            self.by_node.setdefault(node_id, []).append(str(item_id))
            self.details.append({
                "node": node_id,
                "item_id": str(item_id),
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
            })
        logger.warning(
            "dropped item '%s' at node '%s': %s: %s",
            item_id,
            node_id,
            type(exc).__name__,
            exc,
        )

    def all_dropped_ids(self) -> set:
        return {i for ids in self.by_node.values() for i in ids}

    def failure_summary(self) -> Dict[str, Any]:
        """Aggregate the drops for a report ``failure_analysis`` section (R7): total dropped,
        per-node counts, the top error types, and a few examples for debugging. Empty dict
        when nothing was dropped (so a clean run adds nothing to the report)."""
        from collections import Counter

        with self._lock:
            if not self.details:
                return {}
            return {
                "total_dropped": len(self.details),
                "by_node": {n: len(ids) for n, ids in self.by_node.items()},
                "top_error_types": Counter(
                    d["error_type"] for d in self.details
                ).most_common(10),
                "examples": [dict(d) for d in self.details[:20]],
            }


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
