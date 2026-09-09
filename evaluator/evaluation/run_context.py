"""RunContext: the per-node artifact blackboard threaded through DAG handlers, keyed by
``(producer_node_id, output_name)`` so multiple instances of the same node kind (e.g. two
``rerank`` nodes) resolve to distinct producers via the graph's edge bindings (see
``pipeline/graph/wiring.py``).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterator, Tuple

_Slot = Tuple[str, str]  # (producer_node_id, output_name)


class RunContext:
    """Thread-safe: a lock guards concurrent puts/reads from parallel branch handlers; slots
    are disjoint per producer, so there's no cross-branch contention beyond the dict itself."""

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

    def slots(self) -> Iterator[_Slot]:
        """Iterate a snapshot of the ``(producer_id, output_name)`` slots stored."""
        with self._lock:
            return iter(list(self._store.keys()))
