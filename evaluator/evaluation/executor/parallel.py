"""Intra-level parallel execution + device serialization.

Runs a topological level's independent nodes concurrently on threads, serialized per device
so two nodes never contend for one GPU. The shared stage-times lock lives in
``engine`` (``_run_one_node`` owns the accumulation).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

from .state import RunState

# The shared stage-times lock lives in ``engine`` (``_run_one_node`` owns the
# accumulation); re-exported here so ``phased`` and tests can reach it via either module.
from .engine import _run_one_node, _STAGE_TIMES_LOCK  # noqa: F401

# Device-config field per model node (which `config.model.*_device` it runs on), from the
# shared stage-metadata table (one source for this + the offload planner's pipeline map). This
# MUST cover every node declared `category="model"` — a model node missing here would run
# unplaced and can SIGSEGV on a heterogeneous multi-GPU box. The self-check below fails loudly
# at import if a new model node forgets its placement (instead of at runtime).
from .stage_meta import DEVICE_ATTR as _STAGE_DEVICE_ATTR


def _assert_model_nodes_are_device_managed() -> None:
    from ...pipeline.graph.registry import model_node_kinds

    # model_node_kinds() resolves operators to their model sub-kinds (legacy names), so the
    # device map is keyed/checked by the same legacy kind _node_device looks up.
    missing = model_node_kinds() - set(_STAGE_DEVICE_ATTR)
    if missing:
        raise RuntimeError(
            f"category='model' nodes missing a device mapping in _STAGE_DEVICE_ATTR: "
            f"{sorted(missing)} — add their config.model.*_device field (else they run "
            f"unplaced and can SIGSEGV on heterogeneous multi-GPU)."
        )


_assert_model_nodes_are_device_managed()


def _node_device(node: Any, state: "RunState") -> str:
    """The device a node runs on, for serializing same-device work (T5). A per-node
    ``params.device`` (branch override) wins, else the stage's global device, else ``cpu``.
    """
    params = getattr(node, "params", None) or {}
    if params.get("device"):
        return str(params["device"])
    cm = getattr(state.config, "model", None) if state.config is not None else None
    from ...pipeline.graph.operators import node_kind

    attr = _STAGE_DEVICE_ATTR.get(node_kind(node.stage, params))
    dev = getattr(cm, attr, None) if (cm is not None and attr) else None
    return str(dev) if dev else "cpu"


def _run_level_parallel(
    state: "RunState", level, ctx_for, sink: Any = None, level_idx: int = 0
) -> None:
    """Run a level's nodes concurrently on their per-branch views, serialized per device.

    Threads (not processes): the heavy work is GPU/LLM/IO-bound and releases the GIL, and a
    shared in-process ``ctx`` is how branches exchange artifacts. A per-device lock means
    same-device nodes still run one-at-a-time (no single-GPU contention / OOM), while nodes on
    distinct devices overlap. Exceptions propagate (drop-and-log already handles per-item).
    """
    device_locks: Dict[str, threading.Lock] = {
        _node_device(n, state): threading.Lock() for n in level
    }
    max_workers = min(len(level), len(device_locks)) or 1

    def _worker(node: Any) -> None:
        with device_locks[_node_device(node, state)]:
            _run_one_node(ctx_for(node), node, sink, level_idx)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, node) for node in level]
        for fut in as_completed(futures):
            fut.result()  # re-raise the first worker exception
