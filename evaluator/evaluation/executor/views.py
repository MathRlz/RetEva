"""Per-worker state view for intra-level parallel execution (T5).

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X3). ``_NodeView``
isolates the swap-sensitive / per-query attrs (``_VIEW_LOCAL_ATTRS``) per branch while
delegating everything else to the shared base ``RunState``.
"""

from __future__ import annotations

import functools
from typing import Any, Dict

from .state import RunState, per_branch_field_names

# Fields a parallel worker keeps *private* (per-node) so concurrent branch handlers don't
# clobber each other (T5): the node it runs, the swap-sensitive pipeline attrs (`_node_pipeline`
# / corpus_index rebind these transiently), and the per-query scratch accumulators. Everything
# else (ctx, drop_sink, config, cache, service_provider, results, stage_times) is shared — those
# are either thread-safe (ctx/drop_sink) or only written by serial/solo nodes. Cross-node data
# flows via the keyed `ctx`, not these accumulators (Phase R), so isolating them is safe.
#
# Derived from the RunState field `scope` markers (M1b) — not hand-maintained. Adding a
# RunState field without a scope marker fails the classification test; marking it
# `scope: node` automatically isolates it here.
_VIEW_LOCAL_ATTRS = per_branch_field_names()


class _NodeView:
    """Per-worker view of ``RunState`` for intra-level parallel execution (T5).

    Reads/writes of the private attrs (see ``_VIEW_LOCAL_ATTRS``) are isolated per worker —
    initialised from the base's current values, then diverging locally — so two branch handlers
    running concurrently never stomp each other's ``current_node`` or transiently-swapped
    pipeline. Everything else delegates to the shared base; method calls (``put_artifact`` …)
    are re-bound to the view so they use the view's ``current_node`` over the shared, thread-safe
    ``ctx``."""

    def __init__(self, base: "RunState", node: Any) -> None:
        object.__setattr__(self, "_base", base)
        # Seed each private attr from the base's *current* value (so the branch inherits the
        # shared-prefix state, e.g. a transiently-swapped pipeline) but copy mutable
        # containers so an in-place mutation in one branch can't leak into another.
        local: Dict[str, Any] = {}
        for a in _VIEW_LOCAL_ATTRS:
            if not hasattr(base, a):
                continue
            v = getattr(base, a)
            local[a] = (
                list(v)
                if isinstance(v, list)
                else (dict(v) if isinstance(v, dict) else v)
            )
        local["current_node"] = node
        object.__setattr__(self, "_local", local)

    def __getattr__(self, name: str) -> Any:
        local = object.__getattribute__(self, "_local")
        if name in local:
            return local[name]
        base = object.__getattribute__(self, "_base")
        cls_attr = getattr(type(base), name, None)
        if callable(cls_attr) and not isinstance(cls_attr, property):
            return functools.partial(cls_attr, self)  # bind method to the view
        return getattr(base, name)

    def __setattr__(self, name: str, value: Any) -> None:
        local = object.__getattribute__(self, "_local")
        if name in _VIEW_LOCAL_ATTRS or name in local:
            local[name] = value
        else:
            setattr(object.__getattribute__(self, "_base"), name, value)
