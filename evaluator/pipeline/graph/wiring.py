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
    OneOf,
    StageGraph,
    StageNode,
    _effective_outputs,
    _resolve,
    get_stage_node_def,
    validate_graph_artifacts,
)


def _normalize_spec_item(item: Any) -> Tuple[str, str, Optional[dict]]:
    """A spec item is a node-type string (``"rerank"`` → id==type) or a dict
    ``{id, type, params}`` for a distinct instance (e.g. ``{id: "rerank_b",
    type: "rerank", params: {...}}``).

    Operator-abstraction: this is the single chokepoint that expands a legacy node-type
    name (or an operator-authored ``{op}`` spec) into ``(operator, fields)`` — the node
    ``id`` is left UNCHANGED (so a legacy ``"corpus_embedding"`` becomes
    ``id="corpus_embedding", type="embed", params={axis:corpus}``, keeping keyed-artifact +
    branch ids stable for parity)."""
    from .operators import expand_alias

    if isinstance(item, str):
        node_id, type_, params = item, item, None
    elif isinstance(item, dict):
        type_ = item.get("type") or item.get("id")
        node_id = item.get("id") or type_
        if not type_:
            raise ValueError(f"graph node spec needs 'type' (or 'id'): {item}")
        params = item.get("params")
    else:
        raise ValueError(f"invalid graph node spec: {item!r}")
    type_, params = expand_alias(str(type_), params)
    return str(node_id), str(type_), params


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

    def _present(art: Any) -> bool:
        """Is an optional input available anywhere in the graph (OneOf = any alternative)?"""
        if isinstance(art, OneOf):
            return any(a in produced_anywhere for a in art)
        return art in produced_anywhere

    nodes: List[StageNode] = []
    for i, (nid, ntype, params) in enumerate(items):
        ndef = get_stage_node_def(ntype)
        prior = [pid for pid, _, _ in items[:i]]

        def _producers_of(art: str) -> List[str]:
            return [
                pid
                for pid in prior
                if art in _effective_outputs(type_by_id[pid], params_by_id[pid])
            ]

        # Resolve callable (operator) ports against this instance's params; keep OneOf
        # objects intact (don't flatten — the wiring below binds each alternative).
        node_inputs = _resolve(ndef.inputs, params)
        node_optional = _resolve(ndef.optional_inputs, params)
        wanted = list(node_inputs) + [
            art for art in node_optional if _present(art)
        ]
        deps = set(edges.get(nid, ()))
        bindings: List[Tuple[str, str]] = []
        aliases: List[Tuple[str, str]] = []
        for art in wanted:
            if isinstance(art, OneOf):
                # Bind EVERY alternative that has an upstream producer (priority order) and
                # record the ordered candidate list under the canonical key. ``s.input(key)``
                # then reads the highest-priority alternative that actually *published* at
                # run time — explicit priority (not execution order) WITH runtime fallback
                # (e.g. fusion bails → retrieval falls back to the audio vectors).
                cands: List[str] = []
                for alt in art:
                    producers = _producers_of(alt)
                    if producers:
                        for pid in producers:
                            deps.add(pid)
                            bindings.append((alt, pid))
                        cands.append(alt)
                if art.key is not None and cands:
                    aliases.append((art.key, tuple(cands)))
            else:
                for pid in _producers_of(art):
                    deps.add(pid)
                    bindings.append((art, pid))
        nodes.append(
            StageNode(
                id=nid,
                stage=ntype,
                depends_on=tuple(sorted(deps)),
                bindings=tuple(bindings),
                input_aliases=tuple(aliases),
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
