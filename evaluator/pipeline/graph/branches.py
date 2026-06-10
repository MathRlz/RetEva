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
        for ntype in base_node_ids:
            nid = f"{ntype}@{bid}"
            idmap[ntype] = nid
            override = overrides.get(ntype)
            if override is None:
                params: Optional[dict] = None
            elif isinstance(override, dict):
                params = dict(override)
            else:
                params = {"model": override}
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
    # per-branch report + cross-branch deltas (scans the ctx, W6/A9).
    ordering = tuple(
        sorted(n.id for n in collapsed if n.stage in ("metrics", "retrieval"))
    )
    if ordering:
        collapsed = collapsed + (
            StageNode(id="aggregate", stage="aggregate", depends_on=ordering),
        )
    return StageGraph(mode=mode, nodes=collapsed)
