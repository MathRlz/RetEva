"""Graph creation: artifact auto-wiring.

The single way a graph is built. Given an ordered list of node ids, each node depends
on every earlier node that produces one of its required or (present) optional input
artifacts. Parallelism and branch merges (e.g. fusion's two embedders, the corpus
branch) fall out of the data dependencies — no hand-wired edges. ``edges`` adds extra
ordering dependencies not implied by data. This is shared by the named-mode presets and
the explicit ``graph:`` config block.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .registry import (
    StageGraph,
    StageNode,
    _effective_outputs,
    get_stage_node_def,
    validate_graph_artifacts,
)


def _normalize_spec_item(item: Any) -> Tuple[str, str, Optional[dict]]:
    """A spec item is a node-type string (``"rerank"`` → id==type) or a dict
    ``{id, type, params}`` for a distinct instance (e.g. ``{id: "rerank_b",
    type: "rerank", params: {...}}``)."""
    if isinstance(item, str):
        return item, item, None
    if isinstance(item, dict):
        type_ = item.get("type") or item.get("id")
        node_id = item.get("id") or type_
        if not type_:
            raise ValueError(f"graph node spec needs 'type' (or 'id'): {item}")
        return str(node_id), str(type_), item.get("params")
    raise ValueError(f"invalid graph node spec: {item!r}")


def _wire_nodes(
    node_ids: Sequence[Any],
    edges: Optional[Dict[str, Sequence[str]]] = None,
) -> Tuple[StageNode, ...]:
    edges = edges or {}
    items = [_normalize_spec_item(it) for it in node_ids]  # (id, type, params)
    type_by_id = {nid: ntype for nid, ntype, _ in items}
    params_by_id = {nid: params for nid, _, params in items}
    produced_anywhere: set[str] = set()
    for _, ntype, params in items:
        produced_anywhere.update(_effective_outputs(ntype, params))

    nodes: List[StageNode] = []
    for i, (nid, ntype, params) in enumerate(items):
        ndef = get_stage_node_def(ntype)
        prior = [pid for pid, _, _ in items[:i]]
        wanted = list(ndef.inputs) + [
            art for art in ndef.optional_inputs if art in produced_anywhere
        ]
        deps = set(edges.get(nid, ()))
        bindings: List[Tuple[str, str]] = []
        for art in wanted:
            for pid in prior:
                if art in _effective_outputs(type_by_id[pid], params_by_id[pid]):
                    deps.add(pid)
                    bindings.append((art, pid))
        nodes.append(
            StageNode(
                id=nid,
                stage=ntype,
                depends_on=tuple(sorted(deps)),
                bindings=tuple(bindings),
                params=params,
            )
        )
    return tuple(nodes)


def build_graph_from_spec(
    node_ids: Sequence[Any],
    *,
    mode: str = "custom",
    edges: Optional[Dict[str, Sequence[str]]] = None,
) -> StageGraph:
    """Build a validated DAG from an ordered node-id list via artifact auto-wiring.

    ``edges`` (node id -> extra dependency ids) augments the auto-wired data
    dependencies for ordering not implied by artifacts. Validated for satisfiable
    required inputs + no cycles.
    """
    graph = StageGraph(mode=mode, nodes=_wire_nodes(node_ids, edges))
    validate_graph_artifacts(graph)
    return graph
