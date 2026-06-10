"""Common-subexpression elimination (in-graph branching, A8).

When branches are expanded into namespaced node copies (``asr@ref``, ``asr@corr``, …),
structurally-identical nodes should run *once*. Two nodes collapse iff a canonical key
matches: ``(type, normalized params, sorted input bindings, sorted deps)`` — computed
bottom-up so a node's input-producer ids are already canonical when its key is taken.
Conservative by construction: it never over-shares (distinct params/inputs ⇒ distinct
keys); under-sharing (e.g. an explicit default vs an omitted one) only costs reuse, never
correctness. See ``evaluator-architecture.md`` §8.
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .registry import _NODE_REGISTRY, StageNode


def _freeze_params(params: Optional[dict], stage: Optional[str] = None) -> Any:
    """Canonical, hashable form of a params dict (recursively key-sorted).

    Defaults are resolved against the node contract before hashing (S7): a key whose value
    is ``None`` or equal to the registered ``param_defaults`` for ``stage`` is dropped, so
    ``{model: <default>}`` and an omitted ``model`` produce the same key and CSE collapses the
    explicit-vs-default twin into a single run (instead of running it twice)."""
    defaults: Mapping[str, Any] = {}
    if stage is not None and stage in _NODE_REGISTRY:
        defaults = _NODE_REGISTRY[stage].param_defaults

    def freeze(value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(sorted((k, freeze(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(freeze(v) for v in value)
        return value

    normalized = {
        k: v
        for k, v in (params or {}).items()
        if v is not None and v != defaults.get(k)
    }
    return freeze(normalized)


def _topo_order(nodes: Tuple[StageNode, ...]) -> List[str]:
    """Kahn topological order over ``depends_on`` (id list); raises on a cycle."""
    by_id = {n.id: n for n in nodes}
    indeg = {n.id: 0 for n in nodes}
    dependents: Dict[str, List[str]] = {n.id: [] for n in nodes}
    for n in nodes:
        for dep in n.depends_on:
            if dep in by_id:
                indeg[n.id] += 1
                dependents[dep].append(n.id)
    ready = sorted([nid for nid, d in indeg.items() if d == 0])
    order: List[str] = []
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for child in dependents[nid]:
            indeg[child] -= 1
            if indeg[child] == 0:
                ready.append(child)
        ready.sort()
    if len(order) != len(nodes):
        raise ValueError("Cycle detected during CSE topological ordering")
    return order


def collapse_common_subexpressions(
    nodes: Tuple[StageNode, ...],
) -> Tuple[StageNode, ...]:
    """Collapse structurally-identical nodes to one, rewiring references to the survivor.

    Returns a new node tuple (in topological order) where duplicate sub-expressions are
    deduplicated. The first occurrence of each canonical key survives; its ``depends_on`` /
    ``bindings`` are rewritten to canonical producer ids.
    """
    by_id = {n.id: n for n in nodes}
    canon: Dict[str, str] = {}  # original id -> surviving canonical id
    key_to_canon: Dict[Any, str] = {}
    kept: Dict[str, StageNode] = {}

    for nid in _topo_order(nodes):
        n = by_id[nid]
        deps = tuple(sorted(canon[d] for d in n.depends_on if d in canon))
        binds = tuple(sorted((art, canon[pid]) for art, pid in n.bindings))
        key = (n.stage, _freeze_params(n.params, n.stage), binds, deps)
        if key in key_to_canon:
            canon[nid] = key_to_canon[key]  # collapse into the earlier twin
            continue
        canon[nid] = nid
        key_to_canon[key] = nid
        kept[nid] = StageNode(
            id=nid,
            stage=n.stage,
            depends_on=deps,
            bindings=binds,
            params=n.params,
        )

    order = _topo_order(nodes)
    return tuple(kept[nid] for nid in order if canon[nid] == nid)
