"""Registry of executable pipeline stages (DAG node handlers).

Mirrors the model-family registries: a stage handler registers itself with the
``@register_stage_handler`` decorator and the executor discovers it by name, so
adding a stage no longer means editing a central dispatch dict. Each spec also
carries the executor's timing policy (whether the handler times itself, and which
``stage_times`` bucket the executor accumulates its runtime into).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

StageFn = Callable[[Any], None]


@dataclass(frozen=True)
class StageSpec:
    """A registered stage handler plus the executor's timing policy for it."""

    stage: str
    fn: StageFn
    self_timed: bool = False  # handler records its own stage_times entry
    time_key: Optional[str] = None  # bucket the executor accumulates runtime into


_STAGE_REGISTRY: Dict[str, StageSpec] = {}


def register_stage_handler(
    stage: str, *, self_timed: bool = False, time_key: Optional[str] = None
) -> Callable[[StageFn], StageFn]:
    """Register a stage handler under ``stage`` (decorator)."""

    def decorator(fn: StageFn) -> StageFn:
        if stage in _STAGE_REGISTRY:
            raise ValueError(f"Stage handler already registered: {stage}")
        _STAGE_REGISTRY[stage] = StageSpec(stage, fn, self_timed, time_key)
        return fn

    return decorator


def get_stage_spec(stage: str) -> StageSpec:
    """Return the spec for ``stage`` or raise with the registered names."""
    try:
        return _STAGE_REGISTRY[stage]
    except KeyError:
        registered = ", ".join(sorted(_STAGE_REGISTRY))
        raise KeyError(
            f"No stage handler registered: {stage}. Registered: {registered}"
        ) from None


def stage_registry() -> Dict[str, StageSpec]:
    """Return a copy of the full stage registry."""
    return dict(_STAGE_REGISTRY)


def validate_node_handler_consistency(*, strict: bool = False):
    """Cross-check the node-type registry against the handler registry (extensibility §5).

    A node type is declared in two places that must agree: ``register_stage_node`` (the type +
    I/O contract, in ``pipeline/graph/operators_catalog.py``) and ``@register_stage_handler``
    (the executable, here). Adding one without the other is a drift bug that today only
    surfaces at run time. This is the global guard: it returns
    ``(node_types_without_handler, handlers_without_node_type)`` after triggering both
    registries' lazy population. The first set is a hard inconsistency — a node that can be
    wired into a graph but cannot execute; the second is usually benign. ``strict=True`` raises
    on the first.
    """
    import evaluator.evaluation.handlers  # noqa: F401 - registers all stage handlers
    from ..pipeline.graph.registry import registered_stage_names

    handlers = set(stage_registry().keys())
    nodes = set(registered_stage_names())
    missing = nodes - handlers
    orphan = handlers - nodes
    if strict and missing:
        raise ValueError(
            "node types registered without an executable handler "
            f"(add @register_stage_handler): {sorted(missing)}"
        )
    return missing, orphan


def validate_graph_handlers(graph: Any) -> None:
    """Pre-flight (audit M3): every node in ``graph`` must have a registered handler.

    Called before dispatch so a typo'd / unregistered stage fails immediately with a
    clear message, instead of crashing mid-run after heavy setup (model loads, TTS)."""
    missing = sorted(
        {node.stage for node in graph.nodes if node.stage not in _STAGE_REGISTRY}
    )
    if missing:
        registered = ", ".join(sorted(_STAGE_REGISTRY))
        raise ValueError(
            f"No stage handler registered for node type(s): {', '.join(missing)}. "
            f"Registered handlers: {registered}"
        )
