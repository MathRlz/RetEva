"""Opt-in mid-run artifact inspection (architecture-improvements §7).

Set ``EVALUATOR_DUMP_ARTIFACTS=node_a,node_b`` to have the executor write each named node's
published artifacts to ``<EVALUATOR_DUMP_DIR or 'artifact_dumps'>/<node>.<artifact>.jsonl``
right after the node runs — the debugging hook researchers otherwise hand-roll, with no code
change. Best-effort: a dump failure is logged, never raised.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..logging_config import get_logger

logger = get_logger(__name__)


def _dump_targets() -> set:
    raw = os.environ.get("EVALUATOR_DUMP_ARTIFACTS", "")
    return {t.strip() for t in raw.split(",") if t.strip()}


def _rows(value: Any):
    """Best-effort (id, value) rows for an artifact. ItemSets become per-item rows; anything
    else is wrapped as a single ``{"value": …}`` row."""
    items = getattr(value, "items", None)
    if callable(items):
        try:
            for item_id, val in value.items():
                yield {"id": str(item_id), "value": _coerce(val)}
            return
        except Exception as exc:  # noqa: BLE001 - fall through to the scalar form
            logger.debug("artifact dump: items() iteration failed (%s); scalar form", exc)
    yield {"value": _coerce(value)}


def _coerce(val: Any) -> Any:
    """JSON-friendly form: ndarrays → list (truncated), dataclasses/objects → str fallback."""
    tolist = getattr(val, "tolist", None)
    if callable(tolist):
        try:
            out = val.tolist()
            return out[:64] if isinstance(out, list) else out
        except Exception as exc:  # noqa: BLE001
            logger.debug("artifact dump: tolist() failed (%s); using str()", exc)
            return str(val)
    return val


def maybe_dump_node_artifacts(state: Any, node: Any) -> None:
    """If ``node`` is a dump target, write its published artifacts to JSONL (env-gated)."""
    targets = _dump_targets()
    if not targets or getattr(node, "id", None) not in targets:
        return
    ctx = getattr(state, "ctx", None)
    if ctx is None:
        return
    out_dir = os.environ.get("EVALUATOR_DUMP_DIR", "artifact_dumps")
    try:
        os.makedirs(out_dir, exist_ok=True)
        for producer, name in list(ctx.slots()):
            if producer != node.id:
                continue
            value = ctx.get_opt(producer, name, None)
            path = os.path.join(out_dir, f"{node.id}.{name}.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                for row in _rows(value):
                    fh.write(json.dumps(row, default=str) + "\n")
            logger.info("artifact dump: %s.%s → %s", node.id, name, path)
    except OSError as exc:
        logger.warning("artifact dump for node '%s' failed: %s", node.id, exc)
