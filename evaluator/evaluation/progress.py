"""Live, node-granular run observability (task T10).

Before this, progress was phase-granular and per-query traces only materialised at run end, so a
long run was opaque while it executed. ``ProgressSink`` emits a structured event at each node's
start and completion (with duration + cumulative stage timing) to:

- the existing ``progress_callback`` (so the in-process webapi job log updates live), and
- an optional JSONL file (``EVALUATOR_PROGRESS_FILE``) that *any* observer — the webapi polling a
  subprocess, a CLI ``tail -f`` — can read incrementally to watch the DAG advance node by node.

Thread-safe so T5's parallel branch workers can emit concurrently.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProgressEvent:
    """The typed, stable progress contract (R7) an external dashboard can consume.

    A node-lifecycle event with the DAG position; ``extra`` carries event-specific fields
    (e.g. ``duration_s`` on ``node_complete``). ``to_record`` is the JSONL wire form — the
    same flat dict the file/callback already emitted, so this typing is additive.
    """

    ts: float
    event: str  # "node_start" | "node_complete" | …
    node: str
    level: int
    total_levels: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "event": self.event,
            "node": self.node,
            "level": self.level,
            "total_levels": self.total_levels,
            **self.extra,
        }


class ProgressSink:
    """Fan-out of node-lifecycle events to a progress callback and/or a JSONL file."""

    def __init__(
        self,
        *,
        path: Optional[str] = None,
        callback: Optional[Callable[[str, int, int, str], None]] = None,
        total_levels: int = 0,
    ) -> None:
        self._path = path or None
        self._callback = callback
        self._total_levels = total_levels
        self._lock = threading.Lock()

    @classmethod
    def from_env(
        cls,
        callback: Optional[Callable[[str, int, int, str], None]] = None,
        total_levels: int = 0,
    ) -> "ProgressSink":
        return cls(
            path=os.environ.get("EVALUATOR_PROGRESS_FILE"),
            callback=callback,
            total_levels=total_levels,
        )

    def emit(self, event: str, *, node: str = "", level: int = 0, **extra: Any) -> None:
        """Record one event (``node_start`` / ``node_complete`` / …). Best-effort: an I/O or
        callback error is logged, never raised — observability must not break the run.
        """
        evt = ProgressEvent(
            ts=round(time.time(), 3),
            event=event,
            node=node,
            level=level,
            total_levels=self._total_levels,
            extra=dict(extra),
        )
        record = evt.to_record()
        message = f"{event} {node}".strip()
        with self._lock:
            if self._path:
                try:
                    with open(self._path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")
                        fh.flush()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("progress sink write failed: %s", exc)
            if self._callback is not None:
                try:
                    self._callback(event, level, self._total_levels, message)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("progress callback failed: %s", exc)
