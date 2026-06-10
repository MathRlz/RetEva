"""RunContext: the per-node artifact blackboard threaded through DAG handlers.

Artifacts are keyed by ``(producer_node_id, output_name)`` so the same artifact *type*
can be produced by multiple node instances (e.g. two ``rerank`` nodes both produce
``retrieved``) and each consumer reads a *specific* producer's output. A consumer
resolves its inputs to concrete ``(producer_id, output_name)`` slots via the graph's
edge bindings (see ``pipeline/stage_graph.py``); this class is just the typed store.

For single-instance graphs (``id == type``) the binding points at the unique producer,
reproducing today's "shared artifact" semantics without special-casing.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterator, Tuple

_Slot = Tuple[str, str]  # (producer_node_id, output_name)


class RunContext:
    """Typed artifact store keyed by ``(producer_node_id, output_name)``.

    Thread-safe: under intra-level parallel execution (T5) several branch handlers write
    their own ``(node_id, name)`` slots concurrently. A lock guards the dict so those
    concurrent puts/reads are safe; slots are disjoint per producer, so there is no
    cross-branch contention beyond the dict itself."""

    def __init__(self) -> None:
        self._store: Dict[_Slot, Any] = {}
        self._lock = threading.RLock()

    def put(self, node_id: str, name: str, value: Any) -> None:
        """Record ``name`` produced by node ``node_id``."""
        with self._lock:
            self._store[(node_id, name)] = value

    def has(self, node_id: str, name: str) -> bool:
        with self._lock:
            return (node_id, name) in self._store

    def get(self, node_id: str, name: str) -> Any:
        """Return the artifact ``node_id.name`` or raise ``KeyError`` if absent."""
        with self._lock:
            try:
                return self._store[(node_id, name)]
            except KeyError:
                raise KeyError(
                    f"artifact '{name}' from node '{node_id}' not in RunContext"
                ) from None

    def get_opt(self, node_id: str, name: str, default: Any = None) -> Any:
        """Return the artifact or ``default`` when absent (no raise)."""
        with self._lock:
            return self._store.get((node_id, name), default)

    def outputs_of(self, node_id: str) -> Dict[str, Any]:
        """All artifacts produced by ``node_id`` as ``{output_name: value}``."""
        with self._lock:
            return {
                name: value
                for (nid, name), value in self._store.items()
                if nid == node_id
            }

    def slots(self) -> Iterator[_Slot]:
        """Iterate a snapshot of the ``(producer_id, output_name)`` slots stored."""
        with self._lock:
            return iter(list(self._store.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, slot: _Slot) -> bool:
        with self._lock:
            return slot in self._store
