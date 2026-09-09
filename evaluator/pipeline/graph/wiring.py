"""Graph creation: explicit port-level edge binding + the auto-wiring authoring assistant.

The auto-wirer (``_wire_nodes``) is an AUTHORING assistant only; no config reaches it at
load time — ``bind_explicit_edges`` reconstructs bindings from the config's edge list."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .registry import (
    OneOf,
    StageGraph,
    StageNode,
    effective_outputs,
    _resolve,
    get_stage_node_def,
    validate_graph_artifacts,
)
from .operators import expand_alias
from ..artifacts import artifact_modality, is_registered


def _normalize_spec_item(item: Any) -> Tuple[str, str, Optional[dict]]:
    """A spec item is a node-type string (id==type) or a ``{id, type, params}`` dict for a
    distinct instance. The single chokepoint expanding a legacy/operator-authored type into
    ``(operator, fields)`` while leaving ``id`` UNCHANGED, keeping keyed-artifact + branch ids
    stable for parity (e.g. legacy ``corpus_embedding`` -> type ``embed``, same id)."""

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
        produced_anywhere.update(effective_outputs(ntype, params))

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
                if art in effective_outputs(type_by_id[pid], params_by_id[pid])
            ]

        # Keep OneOf objects intact (don't flatten) — the wiring below binds each alternative.
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
                # Bind every alternative with an upstream producer (priority order); `s.input`
                # then reads the highest-priority one actually published at run time — explicit
                # priority with runtime fallback (e.g. fusion bails -> falls back to audio vectors).
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


def _node_ports(ntype: str, params: Optional[dict]) -> List[Tuple[str, Tuple[str, ...], bool]]:
    """A node instance's input PORTS in declaration order: ``(key, accepted_names, required)``.
    A OneOf port is keyed by its canonical key and accepts every alternative (priority order);
    a plain port is keyed by (and accepts only) its artifact name."""
    ndef = get_stage_node_def(ntype)
    ports: List[Tuple[str, Tuple[str, ...], bool]] = []
    for art, required in [(a, True) for a in _resolve(ndef.inputs, params)] + [
        (a, False) for a in _resolve(ndef.optional_inputs, params)
    ]:
        if isinstance(art, OneOf):
            ports.append((art.key or art[-1], tuple(art), required))
        else:
            ports.append((art, (art,), required))
    return ports


def port_modality(names: Tuple[str, ...]):
    """The homogeneous modality of a port's declared names, or ``None`` when mixed or
    unregistered — ``None`` keeps the port's strict closed-name gate (self-guarding without
    a registration-time assert)."""

    mods = {artifact_modality(n) for n in names if is_registered(n)}
    return mods.pop() if len(mods) == 1 else None


def bind_explicit_edges(
    node_ids: Sequence[Any], edges: Sequence[dict]
) -> Tuple[StageNode, ...]:
    """Explicit port-level wiring: bindings come from the config's ``edges``, not from
    artifact-name derivation. Reconstruction is deterministic regardless of YAML edge order
    (ports in declaration order, OneOf alternatives in priority order, same-artifact producers
    in ``graph.nodes`` order), reproducing ``_wire_nodes``'s graph bit-for-bit.

    Any port (OneOf or plain) also accepts a same-modality artifact via an explicit edge — the
    declared name(s) are a default/tiebreak, not a closed gate. Bound candidates are ranked by
    **derivation**: a candidate produced from another bound candidate supersedes it (scoped to
    real producer-ancestor chains, not global depth, so parallel streams like audio/text vectors
    aren't wrongly reordered); ties keep declared-chain order. Canonical-order configs reproduce
    the declared ranking bit-for-bit (golden-locked)."""
    items = [_normalize_spec_item(it) for it in node_ids]
    seen: Dict[str, int] = {}
    for i, (nid, _, _) in enumerate(items):
        if nid in seen:
            raise ValueError(
                f"duplicate node id {nid!r} in graph.nodes (positions {seen[nid]} and {i}) — "
                "every node id must be unique."
            )
        seen[nid] = i
    index_of = {nid: i for i, (nid, _, _) in enumerate(items)}
    port_edges, ordering = _parse_edges(edges, items, index_of)
    ancestors = _ancestor_sets(items, port_edges)

    by_consumer: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for src, out, dst, inp in port_edges:
        by_consumer.setdefault((dst, inp), []).append((src, out))

    produced_anywhere: set = set()
    for _, ntype, params in items:
        produced_anywhere.update(effective_outputs(ntype, params))

    nodes: List[StageNode] = []
    for nid, ntype, params in items:
        ports = _node_ports(ntype, params)
        port_keys = {k for k, _, _ in ports}
        deps = set(ordering.get(nid, ()))
        bindings: List[Tuple[str, str]] = []
        aliases: List[Tuple[str, Tuple[str, ...]]] = []
        for key, names, required in ports:
            incoming = by_consumer.get((nid, key), [])
            # Any port accepts same-modality routed extras (see get_artifact/input).
            routed_extras = _routed_extras(nid, key, names, incoming)
            producers_by_name: Dict[str, List[str]] = {}
            for src, out in incoming:
                producers_by_name.setdefault(out, []).append(src)
            ranked = _rank_port_candidates(
                names, routed_extras, producers_by_name, ancestors, index_of
            )
            cands: List[str] = []
            for alt in ranked:
                # node-list order — the runtime's newest-first read stays meaningful
                producers = sorted(producers_by_name[alt], key=index_of.__getitem__)
                for src in producers:
                    deps.add(src)
                    bindings.append((alt, src))
                cands.append(alt)
            if cands and (len(names) > 1 or routed_extras):
                # A plain port with no routed extras stays alias-free (parity with
                # `_wire_nodes`); recorded only for a real OneOf chain or a routed extra.
                aliases.append((key, tuple(cands)))
            if required and not incoming and any(a in produced_anywhere for a in names):
                raise ValueError(
                    f"node '{nid}' required input '{key}' has no edge, but the graph "
                    f"produces {[a for a in names if a in produced_anywhere]} — add "
                    f"{{from: <producer>, output: <artifact>, to: {nid}, input: {key}}}."
                )
        # an edge naming a port this node doesn't have (params-resolved) is a mistake
        for (dst, inp), routed in by_consumer.items():
            if dst == nid and inp not in port_keys:
                raise ValueError(
                    f"node '{nid}' has no input port '{inp}' "
                    f"(ports: {sorted(port_keys)}; edge from {routed[0][0]})."
                )
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


def is_port_edge(e: Any) -> bool:
    """A port edge carries at least one port name; ``{from, to}`` alone is ordering-only."""
    return isinstance(e, dict) and ("output" in e or "input" in e)


def normalize_port_edge(e: dict) -> dict:
    """Fill in the omitted ``output`` of the shorthand form: most port edges connect a
    same-named artifact->port, so ``{from, to, input: X}`` means ``output == input``; it's
    spelled out only for a rename (OneOf alias chains). An ``output`` without an ``input``
    stays an error: which port it feeds is unknowable."""
    if "input" not in e:
        raise ValueError(
            f"a port edge needs 'input' (the consumer's port); add it, or drop 'output' "
            f"for an ordering-only edge: {e!r}"
        )
    return e if "output" in e else {**e, "output": e["input"]}


def _parse_edges(
    edges: Sequence[dict], items: list, index_of: Dict[str, int]
) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, set]]:
    """Validate the raw edge list → (port_edges, ordering-only deps per consumer)."""
    type_by_id = {nid: ntype for nid, ntype, _ in items}
    # Ignore `suppress_outputs` for edge validation — it's display-only metadata and must
    # never invalidate a run-shape edge.
    params_by_id = {
        nid: ({k: v for k, v in params.items() if k != "suppress_outputs"} or None)
        if isinstance(params, dict) else params
        for nid, _, params in items
    }

    port_edges: List[Tuple[str, str, str, str]] = []  # (from, output, to, input)
    ordering: Dict[str, set] = {}
    seen: set = set()
    for e in edges or ():
        if not isinstance(e, dict) or "from" not in e or "to" not in e:
            raise ValueError(f"each graph.edges item needs 'from' and 'to': {e!r}")
        src, dst = str(e["from"]), str(e["to"])
        for endpoint in (src, dst):
            if endpoint not in index_of:
                raise ValueError(
                    f"graph.edges names unknown node '{endpoint}' "
                    f"(known: {sorted(index_of)})."
                )
        if not is_port_edge(e):
            ordering.setdefault(dst, set()).add(src)
            continue
        e = normalize_port_edge(e)
        out, inp = str(e["output"]), str(e["input"])
        if index_of[src] >= index_of[dst]:
            raise ValueError(
                f"edge {src}.{out} → {dst}.{inp}: the producer must come before the "
                f"consumer in graph.nodes."
            )
        if out not in effective_outputs(type_by_id[src], params_by_id[src]):
            raise ValueError(
                f"edge {src}.{out} → {dst}.{inp}: '{src}' does not produce '{out}' "
                f"(outputs: {list(effective_outputs(type_by_id[src], params_by_id[src]))})."
            )
        key = (src, out, dst, inp)
        if key in seen:
            raise ValueError(f"duplicate edge {src}.{out} → {dst}.{inp}.")
        seen.add(key)
        port_edges.append(key)

    return port_edges, ordering


def _ancestor_sets(items: list, port_edges: list) -> "Dict[str, frozenset]":
    """Data-flow ancestor sets per node (transitive, over PORT edges only). Drives candidate
    ranking: a candidate DERIVED from another bound candidate supersedes it — global depth is
    the wrong heuristic, since parallel streams like audio vs text vectors differ in depth
    without either being "more processed"."""
    data_in: Dict[str, set] = {}
    for src, _out, dst, _inp in port_edges:
        data_in.setdefault(dst, set()).add(src)
    ancestors: Dict[str, frozenset] = {}
    # graph.nodes order is topologically valid for port edges (checked in _parse_edges): one
    # forward pass suffices
    for nid, _, _ in items:
        anc: set = set()
        for src in data_in.get(nid, ()):
            anc.add(src)
            anc.update(ancestors.get(src, ()))
        ancestors[nid] = frozenset(anc)
    return ancestors


def _routed_extras(
    nid: str, key: str, names: Tuple[str, ...], incoming: list
) -> List[str]:
    """Same-modality artifacts routed beyond a port's declared names (OneOf or plain): an
    explicit edge naming one is accepted and read through the same alias-resolution
    ``get_artifact``/``input`` share; a mismatched modality is still an error."""
    port_mod = port_modality(names)
    extras: List[str] = []
    for src, out in incoming:
        if out in names or out in extras:
            continue
        out_mod = port_modality((out,))
        if port_mod is None or out_mod != port_mod:
            raise ValueError(
                f"edge {src}.{out} → {nid}.{key}: port '{key}' accepts "
                f"{list(names)}"
                + (
                    f" or any {port_mod.value}-modality artifact"
                    if port_mod is not None else ""
                )
                + f", not '{out}'."
            )
        extras.append(out)
    return extras


def _rank_port_candidates(
    names: Tuple[str, ...],
    routed_extras: List[str],
    producers_by_name: "Dict[str, List[str]]",
    ancestors: "Dict[str, frozenset]",
    index_of: "Dict[str, int]",
) -> List[str]:
    """Rank a port's bound candidates: one produced FROM another bound candidate supersedes it
    (scoped to real derivation chains, not just depth); unrelated candidates keep declared-chain
    order (routed extras first, then producer node-list index). Canonical-order configs
    reproduce the declared order bit-for-bit (golden-locked)."""
    bound = [
        a for a in tuple(names) + tuple(routed_extras)
        if a in producers_by_name
    ]
    chain_pos = {a: i for i, a in enumerate(names)}
    order_pos = {
        a: (chain_pos.get(a, -1),
            min(index_of[s] for s in producers_by_name[a]))
        for a in bound
    }

    def _descends(a: str, b: str) -> bool:
        """Some producer of ``a`` derives (transitively) from some producer of ``b``."""
        return any(
            pb in ancestors.get(pa, ())
            for pa in producers_by_name[a]
            for pb in producers_by_name[b]
        )

    must_precede = {
        a: {
            b for b in bound
            if b != a and _descends(a, b) and not _descends(b, a)
        }
        for a in bound
    }
    ranked: List[str] = []
    remaining = list(bound)
    while remaining:
        ready = [
            a for a in remaining
            if not any(a in must_precede[b] for b in remaining)
        ]
        # `ready` is empty only on mutual descent (degenerate) — declared order decides
        nxt = min(ready or remaining, key=order_pos.__getitem__)
        ranked.append(nxt)
        remaining.remove(nxt)
    return ranked


def edges_for_graph(nodes: Sequence[StageNode]) -> List[dict]:
    """The canonical port-level edge list of an already-wired graph (see :func:`emit_edges`)."""
    edges: List[dict] = []
    seen: set = set()
    for n in nodes:
        producers_of: Dict[str, List[str]] = {}
        for art, prod in n.bindings:
            producers_of.setdefault(art, [])
            if prod not in producers_of[art]:
                producers_of[art].append(prod)
        alias_by_key = dict(getattr(n, "input_aliases", ()) or ())
        for key, names, _required in _node_ports(n.stage, n.params):
            # Overlay routed extras (they live only in the instance aliases) so the emit
            # round-trip keeps a routed binding mapped to its port.
            extras = tuple(
                a for a in alias_by_key.get(key, ()) if a not in names
            )
            for alt in tuple(names) + extras:
                for prod in producers_of.get(alt, ()):
                    tup = (prod, alt, n.id, key)
                    if tup in seen:
                        continue
                    seen.add(tup)
                    # `output` is spelled out only for a rename (a OneOf alternative).
                    edge = {"from": prod, "to": n.id, "input": key}
                    if alt != key:
                        edge = {"from": prod, "output": alt, "to": n.id, "input": key}
                    edges.append(edge)
    return edges


def emit_edges(node_ids: Sequence[Any]) -> List[dict]:
    """Authoring assistant: the canonical port-level edge list making the auto-wired graph
    explicit — ``bind_explicit_edges(nodes, emit_edges(nodes))`` reproduces ``_wire_nodes(nodes)``
    bit-for-bit (consumer/port/OneOf-alternative/producer all in canonical order)."""
    return edges_for_graph(_wire_nodes(node_ids))


def has_port_edges(edges: Any) -> bool:
    return isinstance(edges, (list, tuple)) and any(is_port_edge(e) for e in edges)


def build_graph_from_spec(
    node_ids: Sequence[Any],
    *,
    mode: str = "custom",
    edges: Any = None,
) -> StageGraph:
    """Build a validated DAG from an ordered node-id list. A LIST with port-level entries
    (``{from, output, to, input}``) is explicit wiring (edges define the bindings); a LIST of
    only ``{from, to}`` entries, or the legacy ``{to: [from, ...]}`` dict, layers ordering deps
    on top of artifact auto-wiring. Validated for satisfiable inputs + no cycles either way."""
    if has_port_edges(edges):
        nodes = bind_explicit_edges(node_ids, edges)
    else:
        if isinstance(edges, (list, tuple)):  # ordering-only list → legacy dict form
            as_dict: Dict[str, List[str]] = {}
            for e in edges:
                if not isinstance(e, dict) or "from" not in e or "to" not in e:
                    raise ValueError(f"each graph.edges item needs 'from' and 'to': {e!r}")
                as_dict.setdefault(str(e["to"]), []).append(str(e["from"]))
            edges = as_dict
        nodes = _wire_nodes(node_ids, edges)
    graph = StageGraph(mode=mode, nodes=nodes)
    validate_graph_artifacts(graph)
    return graph
