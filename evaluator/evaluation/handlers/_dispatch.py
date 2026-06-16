"""Shared operator → legacy-body dispatch for the collapsed handlers.

Every collapsed operator handler (``embed``, ``transform``, ``search``, ``measure``, …) routes
to one of the old per-node bodies by the node's *legacy kind*
(``node_kind(operator, params)``). This centralizes the resolve-then-dispatch boilerplate they
all repeated, and raises a clear error for a kind with no registered body instead of a bare
``KeyError`` (dict form) or a silent default fall-through (if/else form).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

if TYPE_CHECKING:  # pragma: no cover
    from ..executor.state import RunState


def dispatch_operator(
    operator: str, bodies: "Dict[str, Callable[['RunState'], None]]", s: "RunState"
) -> None:
    """Route the collapsed ``operator`` handler to the body for the node's legacy kind.

    ``bodies`` maps ``node_kind(operator, s.node_params)`` → a ``def body(s)``. The map must
    cover every legacy kind the operator can resolve to (verified by the handler-routing tests).
    """
    from ...pipeline.graph.operators import node_kind

    kind = node_kind(operator, s.node_params)
    body = bodies.get(kind)
    if body is None:
        raise KeyError(
            f"{operator!r} operator has no handler body for legacy kind {kind!r}"
        )
    return body(s)
