"""Shared operator → legacy-body dispatch for the collapsed handlers.

Routes each collapsed operator handler (``embed``, ``transform``, ``search``, ``measure``, …)
to a per-node body by legacy kind, raising a clear error for an unregistered kind instead of
a bare ``KeyError`` or silent fall-through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict
from ...pipeline.graph.operators import node_kind

if TYPE_CHECKING:  # pragma: no cover
    from ..executor.state import RunState


def dispatch_operator(
    operator: str, bodies: "Dict[str, Callable[['RunState'], None]]", s: "RunState"
) -> None:
    """Route the collapsed ``operator`` handler to the body for the node's legacy kind.

    ``bodies`` maps ``node_kind(operator, s.node_params)`` → a ``def body(s)``. The map must
    cover every legacy kind the operator can resolve to (verified by the handler-routing tests).
    """

    kind = node_kind(operator, s.node_params)
    body = bodies.get(kind)
    if body is None:
        raise KeyError(
            f"{operator!r} operator has no handler body for legacy kind {kind!r}"
        )
    return body(s)
