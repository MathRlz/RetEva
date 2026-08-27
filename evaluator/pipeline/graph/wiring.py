"""Graph creation: explicit port-level edge binding + the auto-wiring authoring assistant.

A CONFIG graph is explicit — ``bind_explicit_edges`` reconstructs bindings from the config's
``{from, output, to, input}`` edge list (the loader rejects a nodes-without-edges config; E6).
The artifact auto-wirer (``_wire_nodes``) survives as the AUTHORING assistant only: it powers
``emit_edges`` (the ``evaluator graph --emit-edges`` generator), the builder's plumbing derive,
the template skeletons, and the legacy ``model.pipeline_mode`` back-compat path — no config
reaches it at load time.
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


def _port_modality(names: Tuple[str, ...]):
    """The homogeneous modality of a port's declared names, or ``None`` when mixed or
    unregistered — ``None`` keeps the port's strict closed-name gate (self-guarding without
    a registration-time assert)."""
    from ..artifacts import artifact_modality, is_registered  # lazy: import cycle

    mods = {artifact_modality(n) for n in names if is_registered(n)}
    return mods.pop() if len(mods) == 1 else None


def bind_explicit_edges(
    node_ids: Sequence[Any], edges: Sequence[dict]
) -> Tuple[StageNode, ...]:
    """Explicit port-level wiring (E2): bindings come from the config's ``edges``, not from
    artifact-name derivation. An edge ``{from, output, to, input}`` binds the producer's
    ``output`` artifact into the consumer's ``input`` port; a portless ``{from, to}`` adds an
    ordering dependency only. Reconstruction is deterministic and independent of YAML edge
    order: ports are walked in declaration order, OneOf alternatives in priority order, and
    same-artifact producers in ``graph.nodes`` order (the runtime's newest-first read keeps
    its meaning) — so a generated edge set reproduces ``_wire_nodes``'s graph bit-for-bit.

    **Type-open ports (B2).** Any port — a multi-name (OneOf) port or a plain (single-name)
    one — additionally accepts any registered artifact of the SAME modality via an explicit
    edge; the declared name(s) are a default and a tiebreak, not a closed gate. So e.g.
    ``optimized_query_text`` can be routed into ``query_correction``'s ``query_text`` port
    (correction-after-optimization), and ``dataset_source``'s ``query_text`` can be routed
    into ``asr``'s plain ``reference_transcription`` port — a new text-transform stage or
    an unconventional GT source needs no per-consumer registration edit. A port's bound
    candidates are ranked by **derivation**: a candidate produced from another bound
    candidate supersedes it (the most-processed variant, scoped to real producer-ancestor
    chains — global depth would wrongly reorder parallel streams like audio vs text
    vectors); unrelated candidates keep declared-chain order, with routed extras first and
    producer node-list index as the final tiebreak. Canonical-order configs reproduce the
    declared ranking bit-for-bit (golden-locked). Both port kinds resolve through the same
    alias candidate list at run time — ``RunState.input``/``get_artifact`` both consult
    ``input_aliases`` (``state.py:_input_candidates``), so a plain port's literal-name
    readers transparently pick up a routed extra without a handler code change."""
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
        produced_anywhere.update(_effective_outputs(ntype, params))

    nodes: List[StageNode] = []
    for nid, ntype, params in items:
        ports = _node_ports(ntype, params)
        port_keys = {k for k, _, _ in ports}
        deps = set(ordering.get(nid, ()))
        bindings: List[Tuple[str, str]] = []
        aliases: List[Tuple[str, Tuple[str, ...]]] = []
        for key, names, required in ports:
            incoming = by_consumer.get((nid, key), [])
            # B2: any port accepts same-modality routed extras — both OneOf and plain
            # ports resolve through the alias candidate list (see get_artifact/input).
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
                # `_wire_nodes`'s auto-wiring — `_input_candidates` falls back to `(key,)`
                # regardless); recorded only when there's a real OneOf chain or a B2 extra
                # so `get_artifact`/`input` see the routed artifact's real name.
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
    """Fill in the omitted ``output`` of the shorthand form.

    Most port edges connect an artifact to a same-named port (`corpus` → `corpus`), so
    ``{from, to, input: X}`` means ``output == input``. ``output`` is spelled out only where
    the names differ — the OneOf alias chains (``text_query_vectors`` → ``query_vectors``).
    An ``output`` without an ``input`` stays an error: which port it feeds is unknowable.
    """
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
    # For edge validation, ignore `suppress_outputs` — it is display-only metadata (the canvas
    # hides ports a later transform supersedes) and must never invalidate a run-shape edge.
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
        if out not in _effective_outputs(type_by_id[src], params_by_id[src]):
            raise ValueError(
                f"edge {src}.{out} → {dst}.{inp}: '{src}' does not produce '{out}' "
                f"(outputs: {list(_effective_outputs(type_by_id[src], params_by_id[src]))})."
            )
        key = (src, out, dst, inp)
        if key in seen:
            raise ValueError(f"duplicate edge {src}.{out} → {dst}.{inp}.")
        seen.add(key)
        port_edges.append(key)

    return port_edges, ordering


def _ancestor_sets(items: list, port_edges: list) -> "Dict[str, frozenset]":
    """Data-flow ancestor sets per node (transitive, over PORT edges only — ordering edges
    are not data flow). graph.nodes order is topologically valid for port edges
    (validated in _parse_edges), so one forward pass suffices. Drives the B2 candidate
    ranking: a candidate DERIVED from another bound candidate supersedes it. (Global depth
    is the wrong heuristic — parallel streams like audio vs text vectors differ in depth
    without one being "more processed"; the deliberate audio-first fallback must survive.)
    """
    data_in: Dict[str, set] = {}
    for src, _out, dst, _inp in port_edges:
        data_in.setdefault(dst, set()).add(src)
    ancestors: Dict[str, frozenset] = {}
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
    """Same-modality artifacts routed beyond a port's declared names (B2).

    Applies to both OneOf ports (multiple declared names) and plain (single-name) ports:
    an explicit edge naming a same-modality artifact is accepted and read through the
    same alias-resolution ``get_artifact``/``input`` share — a mismatched modality is
    still an error.
    """
    port_mod = _port_modality(names)
    extras: List[str] = []
    for src, out in incoming:
        if out in names or out in extras:
            continue
        out_mod = _port_modality((out,))
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
    """Rank a port's BOUND candidates: a candidate produced FROM another bound candidate
    supersedes it (the most-processed variant, scoped to real derivation chains);
    unrelated candidates keep declared-chain order (routed extras first, then producer
    node-list index). Canonical-order configs reproduce the declared order bit-for-bit
    (golden-locked); an inverted topology (optimize→correct) correctly ranks corrected
    above optimized."""
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
            # B2: routed extras live only in the instance aliases — overlay them so a
            # routed binding still maps to its port and the emit round-trip keeps it.
            extras = tuple(
                a for a in alias_by_key.get(key, ()) if a not in names
            )
            for alt in tuple(names) + extras:
                for prod in producers_of.get(alt, ()):
                    tup = (prod, alt, n.id, key)
                    if tup in seen:
                        continue
                    seen.add(tup)
                    # Shorthand when the artifact and the port share a name; `output` is
                    # spelled out only for a rename (a OneOf alternative).
                    edge = {"from": prod, "to": n.id, "input": key}
                    if alt != key:
                        edge = {"from": prod, "output": alt, "to": n.id, "input": key}
                    edges.append(edge)
    return edges


def emit_edges(node_ids: Sequence[Any]) -> List[dict]:
    """Authoring assistant (E3): the canonical port-level edge list that makes the auto-wired
    graph explicit — ``bind_explicit_edges(nodes, emit_edges(nodes))`` reproduces
    ``_wire_nodes(nodes)`` bit-for-bit. Canonical order: consumer in node-list order, ports in
    declaration order, OneOf alternatives in priority order, producers in node-list order.
    The docs-axis double-port (same artifact, same key on two ports) dedups to one edge —
    reconstruction re-binds it through both ports."""
    return edges_for_graph(_wire_nodes(node_ids))


def _has_port_edges(edges: Any) -> bool:
    return isinstance(edges, (list, tuple)) and any(is_port_edge(e) for e in edges)


def build_graph_from_spec(
    node_ids: Sequence[Any],
    *,
    mode: str = "custom",
    edges: Any = None,
) -> StageGraph:
    """Build a validated DAG from an ordered node-id list.

    ``edges`` forms:
    - a LIST with port-level entries (``{from, output, to, input}``, plus optional
      ordering-only ``{from, to}`` entries) → **explicit wiring**: the edges define the
      bindings (E2; the target state — auto-derivation is the authoring assistant).
    - a LIST of only ``{from, to}`` entries, or the legacy ``{to: [from, ...]}`` dict →
      ordering deps on top of artifact auto-wiring (transitional).
    Validated for satisfiable required inputs + no cycles either way.
    """
    if _has_port_edges(edges):
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
