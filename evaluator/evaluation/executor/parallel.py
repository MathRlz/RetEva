"""Device placement for intra-level parallel execution.

Which device a node runs on, so ``engine`` can serialize same-device work and let nodes on
distinct devices overlap, plus the registry self-check that every model node has a placement.
"""

from __future__ import annotations

from typing import Any

from .state import RunState

# Device-config field per model node (which `config.model.*_device` it runs on), from the
# shared stage-metadata table (one source for this + the offload planner's pipeline map). This
# MUST cover every node declared `category="model"` — a model node missing here would run
# unplaced and can SIGSEGV on a heterogeneous multi-GPU box. The self-check below fails loudly
# at import if a new model node forgets its placement (instead of at runtime).
from .stage_meta import DEVICE_ATTR as _STAGE_DEVICE_ATTR
from ...pipeline.graph.registry import model_node_kinds
from ...pipeline.graph.operators import node_kind


def assert_model_nodes_are_device_managed() -> None:
    """Registry sanity: every category='model' node kind must have a device mapping. Called
    from ``run_graph``'s pre-flight so a mismatch fails a *run* with context, not
    ``import evaluator``."""

    # model_node_kinds() resolves operators to their model sub-kinds (legacy names), so the
    # device map is keyed/checked by the same legacy kind _node_device looks up.
    missing = model_node_kinds() - set(_STAGE_DEVICE_ATTR)
    if missing:
        raise RuntimeError(
            f"category='model' nodes missing a device mapping in _STAGE_DEVICE_ATTR: "
            f"{sorted(missing)} — add their config.model.*_device field (else they run "
            f"unplaced and can SIGSEGV on heterogeneous multi-GPU)."
        )


def _node_device(node: Any, state: "RunState") -> str:
    """The device a node runs on, for serializing same-device work. A per-node
    ``params.device`` (branch override) wins, else the stage's global device, else ``cpu``.
    """
    params = getattr(node, "params", None) or {}
    if params.get("device"):
        return str(params["device"])
    cm = getattr(state.config, "model", None) if state.config is not None else None

    attr = _STAGE_DEVICE_ATTR.get(node_kind(node.stage, params))
    dev = getattr(cm, attr, None) if (cm is not None and attr) else None
    return str(dev) if dev else "cpu"
