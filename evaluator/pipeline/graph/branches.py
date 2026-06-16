"""Branch expansion (in-graph branching, A8).

Expand a per-branch template into namespaced node copies, wire each branch in isolation,
then CSE-collapse the shared prefix so identical work runs once while divergent nodes keep
their ``@branch`` provenance. A terminal ``aggregate`` node fans in over every branch.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cse import collapse_common_subexpressions
from .registry import StageGraph, StageNode
from .wiring import build_graph_from_spec


def expand_branches(
    base_node_ids: Sequence[str],
    branches: Sequence[Dict[str, Any]],
    *,
    edges: Optional[Dict[str, Sequence[str]]] = None,
) -> Tuple[List[dict], Dict[str, Sequence[str]]]:
    """Expand a per-branch template into one namespaced node-spec list (pre-CSE).

    ``base_node_ids`` is the shared ordered node-type template; ``branches`` is a list of
    ``{id, <node_type>: <model-str | params-dict>}`` — each entry overrides specific nodes'
    params for that branch. Every base node is copied as ``<type>@<branch>`` with the branch's
    override (if any). Feed the result to :func:`build_graph_from_spec` then
    :func:`collapse_common_subexpressions`: identical (un-overridden) prefix nodes collapse to
    one, so shared work runs once while divergent nodes keep their ``@branch`` provenance.
    """
    specs: List[dict] = []
    new_edges: Dict[str, Sequence[str]] = {}
    for branch in branches:
        bid = str(branch["id"])
        overrides = {k: v for k, v in branch.items() if k != "id"}
        idmap: Dict[str, str] = {}
        for entry in base_node_ids:
            # Template entries may be node-type strings or {id, type, params} dicts
            # (e.g. dataset_source carrying its injected `fields` column schema).
            if isinstance(entry, dict):
                ntype = entry.get("type") or entry["id"]
                # The branch id stems on the node's *id* (the pinned legacy name), not its
                # type — so an operator template (id="asr", type="convert") still yields
                # `asr@ref` and matches the user's `{asr: …}` override by legacy name.
                stem = entry.get("id") or ntype
                base_params: Optional[dict] = dict(entry.get("params") or {}) or None
            else:
                ntype = stem = entry
                base_params = None
            nid = f"{stem}@{bid}"
            idmap[stem] = nid
            override = overrides.get(stem)
            if override is None:
                params = base_params
            elif isinstance(override, dict):
                params = {**(base_params or {}), **override}
            else:
                params = {**(base_params or {}), "model": override}
            specs.append({"id": nid, "type": ntype, "params": params})
        for to_t, froms in (edges or {}).items():
            new_edges[idmap.get(to_t, to_t)] = [idmap.get(f, f) for f in froms]
    return specs, new_edges


def build_branched_graph(
    base_node_ids: Sequence[str],
    branches: Sequence[Dict[str, Any]],
    *,
    mode: str = "custom",
    edges: Optional[Dict[str, Sequence[str]]] = None,
) -> StageGraph:
    """Expand ``branches`` over the template and CSE-collapse shared nodes.

    Each branch is wired **in isolation** (so auto-wiring binds a node only to its own
    branch's producers — no cross-branch leakage), then identical nodes are collapsed across
    branches by CSE so the shared prefix runs once.
    """
    all_nodes: List[StageNode] = []
    for branch in branches:
        specs, branch_edges = expand_branches(base_node_ids, [branch], edges=edges)
        graph = build_graph_from_spec(specs, mode=mode, edges=branch_edges)
        all_nodes.extend(graph.nodes)
    collapsed = collapse_common_subexpressions(tuple(all_nodes))
    # One terminal aggregate node depends on every branch's metrics → builds the
    # per-branch report + cross-branch deltas (scans the ctx, W6/A9). Resolve via
    # node_kind so the predicate matches ONLY the report node (`metrics`) + retrieval —
    # not every `measure`/`search` operator node (which would shift aggregate's deps).
    from .operators import expand_alias, node_kind

    ordering = tuple(
        sorted(
            n.id
            for n in collapsed
            if node_kind(n.stage, n.params) in ("metrics", "retrieval")
        )
    )
    if ordering:
        # The aggregate node is the `sink` operator (target:aggregate) — create it directly
        # in operator form (it bypasses _normalize_spec_item) so it has a registered handler,
        # keeping the pinned id "aggregate".
        agg_op, agg_params = expand_alias("aggregate", None)
        collapsed = collapsed + (
            StageNode(
                id="aggregate", stage=agg_op, depends_on=ordering, params=agg_params
            ),
        )
    return StageGraph(mode=mode, nodes=collapsed)
