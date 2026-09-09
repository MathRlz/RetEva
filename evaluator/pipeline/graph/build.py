"""Build the execution DAG from a config's explicit ``graph.nodes``.

A config/canvas author writes only the *meaningful* nodes; the metric-comparison / report /
trace / finalize plumbing is derived here (:func:`complete_structural_plumbing`) before wiring.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .registry import (
    ARTIFACT_REFERENCE_TEXT,
    ARTIFACT_REFERENCE_TRANSCRIPTION,
    StageGraph,
)
from .wiring import has_port_edges, build_graph_from_spec, emit_edges
from .operators import node_kind
from ...config.dataset_descriptors import resolve_dataset_descriptor

logger = logging.getLogger(__name__)


def _spec_type(spec: Any) -> str:
    """A graph-spec item is a node-type string OR a ``{id, type, params}`` dict."""
    return spec if isinstance(spec, str) else spec.get("type")


def _spec_params(spec: Any) -> dict:
    return {} if isinstance(spec, str) else (spec.get("params") or {})


def _spec_id(spec: Any) -> str:
    """A node-list entry's id: the explicit ``id``, else its type (bare-string specs and dict
    specs without an id both use the type as their id, matching how the executor names them)."""
    return spec if isinstance(spec, str) else (spec.get("id") or _spec_type(spec))


def _edge_key(e: Dict[str, Any]) -> tuple:
    """``output`` may be omitted when it equals ``input`` — compare on the resolved pair so
    the short and long spellings of one edge dedup against each other."""
    return (e.get("from"), e.get("output", e.get("input")), e.get("to"), e.get("input"))


def graph_label(graph_override: Optional[dict]) -> Optional[str]:
    """The run/leaderboard label for an explicit graph, derived from its node kinds — the graph
    is the spec. ``None`` when the graph has no query head (a custom shape)."""

    kinds = set()
    for n in (graph_override or {}).get("nodes") or []:
        if isinstance(n, str):
            t, p = n, {}
        else:
            t, p = n.get("type") or n.get("id"), n.get("params") or {}
        kinds.add(node_kind(t, p))
    # audio_embedding + text_embedding is cross-modal fusion, audio_embedding alone is not —
    # the two must stay distinguishable.
    if "audio_embedding" in kinds:
        return "audio_text_retrieval" if "text_embedding" in kinds else "audio_emb_retrieval"
    if "asr" in kinds:
        return "asr_text_retrieval" if "retrieval" in kinds else "asr_only"
    return None


def run_label(graph_override: Optional[dict]) -> str:
    """The run's mode label: the query-head label, else text_retrieval for a text-query graph
    (question text straight into retrieval, no voice front-end), else custom."""

    label = graph_label(graph_override)
    if label:
        return label
    kinds = {
        node_kind(_spec_type(n), _spec_params(n))
        for n in (graph_override or {}).get("nodes") or []
    }
    return "text_retrieval" if {"text_embedding", "retrieval"} <= kinds else "custom"


def structural_specs(nodes: list) -> list:
    """The plumbing nodes an explicit graph needs, in execution-slot order."""

    kinds = {node_kind(_spec_type(n), _spec_params(n)) for n in nodes}
    label = graph_label({"nodes": nodes})
    # A pure TEXT retrieval graph (no asr/audio query head) gets no label but still needs the
    # full retrieval chain; the chain doesn't depend on the query head's modality.
    if label is None and "retrieval" in kinds:
        label = "asr_text_retrieval"
    if label is None:
        return []
    if label == "asr_only":
        return ["transcription_metrics", "metrics", "finalize"]
    out: list = []
    if label == "asr_text_retrieval" and "asr" in kinds:
        out.append("transcription_metrics")
    if label == "audio_text_retrieval" and ("fusion" in kinds or "result_fusion" in kinds):
        out.append("embedding_alignment_metrics")
    out.append("retrieval_metrics")
    out.append("metrics")
    if "answer_gen" in kinds:
        out.append("answer_metrics")
    if "answer_judge" in kinds:
        out.append("build_query_traces")
    out.append("finalize")
    return out


def _topo_sort_nodes(nodes: list, edges: list) -> list:
    """Stable topo sort (Kahn's algorithm, ties broken by original position): the loader
    requires a producer before its consumer in ``graph.nodes``."""
    import heapq

    index = {_spec_id(n): i for i, n in enumerate(nodes)}
    deps: Dict[str, set] = {nid: set() for nid in index}
    for e in edges:
        f, t = e.get("from"), e.get("to")
        if f in index and t in index and f != t:
            deps[t].add(f)

    ready = [(i, nid) for nid, i in index.items() if not deps[nid]]
    heapq.heapify(ready)
    placed: List[str] = []
    remaining = set(index)
    while ready:
        _, nid = heapq.heappop(ready)
        if nid not in remaining:
            continue
        remaining.discard(nid)
        placed.append(nid)
        for other in remaining:
            deps[other].discard(nid)
        for other in list(remaining):
            if not deps[other]:
                heapq.heappush(ready, (index[other], other))
    # A cycle (shouldn't happen for a valid DAG) leaves leftovers — keep their relative order,
    # appended at the end, rather than dropping them.
    placed.extend(sorted(remaining, key=index.get))

    by_id = {_spec_id(n): n for n in nodes}
    return [by_id[nid] for nid in placed]


#: node_kind()s that can each anchor their own per-variant structural chain (metrics/finalize/
#: traces) when more than one is present in an explicit graph. Kept as an explicit, named, tested
#: set rather than one hardcoded kind check: promoting a new kind to "variant root" is a conscious
#: decision (comparing N of them needs its own chain), not something that should silently
#: start/stop working as the registry changes. `retrieval`: N ASR/encoder/retrieval-strategy
#: variants. `answer_gen`: comparing N LLMs on one shared retrieval — the documented way LLM
#: comparison is modeled (two independent `answer_gen` nodes, not a param list on one node).
#: `tests/test_webapi_registry_contract.py::test_variant_root_kinds_guard_...` fails loudly if a
#: new registry kind looks like a plausible third root (its output consumed by a structural node)
#: and isn't accounted for here or on that test's exemption list — see that test before editing.
VARIANT_ROOT_KINDS = frozenset({"retrieval", "answer_gen"})


def _adjacency(edges: list) -> Tuple[Dict[str, set], Dict[str, set]]:
    """Forward (``from`` -> ``to``) and backward (``to`` -> ``from``) maps of the raw edge list.
    Ordering-only edges missing either end are skipped."""
    forward: Dict[str, set] = {}
    backward: Dict[str, set] = {}
    for e in edges:
        f, t = e.get("from"), e.get("to")
        if f and t:
            forward.setdefault(f, set()).add(t)
            backward.setdefault(t, set()).add(f)
    return forward, backward


def _reachable(start: str, adjacency: Dict[str, set]) -> set:
    """Every node reachable from ``start`` over ``adjacency``, ``start`` included."""
    stack, seen = [start], set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adjacency.get(cur, ()))
    return seen


def _terminal_variant_ids(variant_ids: List[str], edges: list) -> List[str]:
    """``variant_ids`` with no other variant-root node reachable downstream — an upstream
    hop feeding into another variant-root node is an earlier stage of the same pipeline,
    not an independent variant."""
    forward, _ = _adjacency(edges)
    variant_set = set(variant_ids)
    terminal = []
    for rid in variant_ids:
        stack, seen = list(forward.get(rid, ())), set()
        reaches_another = False
        while stack and not reaches_another:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur in variant_set:
                reaches_another = True
                break
            stack.extend(forward.get(cur, ()))
        if not reaches_another:
            terminal.append(rid)
    return terminal


def _variant_roots_diverge(variant_ids: List[str], edges: list) -> bool:
    """Whether ``variant_ids`` are independent comparisons (disjoint downstream reachable
    sets), vs. parallel sources that converge into one shared pipeline (e.g. two retrieval
    variants feeding one `result_fusion`) — which needs a single shared chain, not
    per-root ones."""
    if len(variant_ids) < 2:
        return True
    forward, _ = _adjacency(edges)
    reach: Dict[str, set] = {}
    for rid in variant_ids:
        for cur in _reachable(rid, forward):
            reach.setdefault(cur, set()).add(rid)
    return all(len(rids) <= 1 for rids in reach.values())


def complete_structural_plumbing(graph: Dict[str, Any]) -> None:
    """Append the structural plumbing (:func:`structural_specs`) an explicit graph needs to run
    as a full DAG, plus the edges wiring it in. Mutates ``graph`` — the RAW pre-translation
    ``{"nodes": [...], "edges": [...]}`` dict, before edge validation."""
    # Must run before edge validation (a node's own binding can name a structural producer
    # absent from the submitted graph); each VARIANT_ROOT_KINDS node gets its own chain so
    # `emit_edges` doesn't silently merge every variant's metrics into one shared chain.
    override = graph
    if not (isinstance(override, dict) and override.get("nodes")):
        return

    # Reject a duplicate id here, early: later this function's own id-keyed logic would
    # mis-derive a confusing downstream error instead of naming the real cause.
    seen_ids: Dict[str, int] = {}
    for i, n in enumerate(override["nodes"]):
        nid = _spec_id(n)
        if nid in seen_ids:
            raise ValueError(
                f"duplicate node id {nid!r} in graph.nodes (positions {seen_ids[nid]} and {i}) "
                "— every node id must be unique."
            )
        seen_ids[nid] = i

    nodes = override["nodes"]
    structural_template = structural_specs(nodes)
    if not structural_template:
        return
    edges = override.get("edges") or []
    port_edges = has_port_edges(edges)
    raw_variant_ids = [
        _spec_id(n) for n in nodes
        if node_kind(_spec_type(n), _spec_params(n)) in VARIANT_ROOT_KINDS
    ]
    # A variant-root feeding into another variant-root (query-refine's 2-hop pipeline) is an
    # earlier stage of one pipeline, not a sibling variant — only the terminal hop's chain matters.
    variant_ids = _terminal_variant_ids(raw_variant_ids, edges)

    if len(variant_ids) > 1 and port_edges and _variant_roots_diverge(variant_ids, edges):
        override["edges"] = list(edges) + _complete_with_plumbing_per_variant(
            nodes, edges, structural_template, variant_ids
        )
    else:
        # Single variant-root node (the common case) or a non-port-edge graph: one shared chain.
        have = {node_kind(_spec_type(n), _spec_params(n)) for n in nodes}
        by_id = {_spec_id(n): n for n in nodes}
        appended = []
        for spec in structural_template:
            k = node_kind(_spec_type(spec), _spec_params(spec))
            if k in have:
                continue
            candidate_id = _spec_id(spec)
            clash = by_id.get(candidate_id)
            if clash is not None:
                # A renamed meaningful node already occupies the id a structural node needs;
                # appending anyway would leave two nodes sharing one id and silently collapse
                # one of them downstream instead of erroring here where the cause is clear.
                raise ValueError(
                    f"cannot auto-derive the structural node {candidate_id!r} (needed for "
                    f"{k!r}): that id is already used by a {node_kind(_spec_type(clash), _spec_params(clash))!r} "
                    f"node — rename it. Reserved structural ids for this graph: "
                    f"{sorted({_spec_id(s) for s in structural_template})}."
                )
            nodes.append(spec)
            appended.append(candidate_id)
            have.add(k)
            by_id[candidate_id] = spec
        # Port-wired plumbing needs its own edges too, derived over the full node list.
        if appended and port_edges:
            app_ids = set(appended)
            existing = {_edge_key(e) for e in edges}
            derived = [
                e for e in emit_edges(nodes)
                if (e["to"] in app_ids or e["from"] in app_ids) and _edge_key(e) not in existing
            ]
            override["edges"] = list(edges) + derived

    # A node appended above can land after a meaningful node that consumes it; re-sort now that
    # every edge is known (a no-op when nothing actually needs moving).
    if override.get("edges"):
        override["nodes"][:] = _topo_sort_nodes(override["nodes"], override["edges"])


def _complete_with_plumbing_per_variant(
    nodes: list, edges: list, structural_template: list, variant_ids: List[str],
) -> list:
    """Append one private structural chain per variant-root node (mutates ``nodes``), each wired via
    `emit_edges` scoped to its own ancestors plus what's downstream and not exclusively owned by a
    different variant-root, so one variant's output can never bind into another's chain."""
    forward, backward = _adjacency(edges)
    # Which variant root(s) can reach each node over the meaningful edges only (the structural
    # chains don't exist yet) — reachable from exactly one variant id means exclusive to it.
    reach: Dict[str, set] = {}
    forward_reach: Dict[str, set] = {}
    for rid in variant_ids:
        seen = _reachable(rid, forward)
        for cur in seen:
            reach.setdefault(cur, set()).add(rid)
        forward_reach[rid] = seen
    owned = {
        rid: {n for n, rids in reach.items() if rids == {rid}}
        for rid in variant_ids
    }

    def _ancestors(rid: str) -> set:
        """``rid``'s upstream lineage; variant roots can legitimately share a subset of it, which is
        why "not exclusively owned by someone else" is too broad a scope for this purpose."""
        seen = _reachable(rid, backward)
        seen.discard(rid)
        return seen

    id_to_spec = {_spec_id(n): n for n in nodes}
    all_ids = set(id_to_spec)
    existing = {_edge_key(e) for e in edges}
    derived_edges: list = []
    for rid in variant_ids:
        other_owned = set().union(*(owned[o] for o in variant_ids if o != rid))
        # rid's lineage + itself + whatever downstream of it isn't exclusively another variant's
        # (the query-refine 2-hop case).
        scope_ids = _ancestors(rid) | {rid} | (forward_reach.get(rid, set()) - other_owned)
        # Reuse a pre-existing dangling id (e.g. `build_query_traces_dense`) an author already
        # bound to, rather than inventing a new one that leaves that edge dangling.
        dangling = {
            e["from"] for e in edges
            if e.get("to") in owned[rid] and e.get("from") and e["from"] not in all_ids
        }
        root_kind = node_kind(_spec_type(id_to_spec[rid]), _spec_params(id_to_spec[rid]))
        root_prefix = f"{root_kind}_"
        suffix = (
            rid[len(root_prefix):] if rid.startswith(root_prefix) and len(rid) > len(root_prefix)
            else rid
        )
        chain_nodes = []
        for spec in structural_template:
            base_type = _spec_type(spec)
            reused = next(
                (d for d in dangling if d == base_type or d.startswith(base_type + "_")), None
            )
            new_id = reused or f"{base_type}_{suffix}"
            if new_id in all_ids:
                clash = id_to_spec[new_id]
                clash_kind = node_kind(_spec_type(clash), _spec_params(clash))
                expected_kind = node_kind(base_type, _spec_params(spec))
                if clash_kind != expected_kind:
                    # A renamed meaningful node occupies the id this variant's chain needs;
                    # treating it as "already present" would silently misscope it instead.
                    raise ValueError(
                        f"cannot auto-derive the structural node {new_id!r} (needed for "
                        f"{expected_kind!r} in variant {rid!r}): that id is already used by a "
                        f"{clash_kind!r} node — rename it."
                    )
                continue  # already present (a full-DAG spec, not meaningful-only) — the
                          # existing node is already in scope_nodes below, nothing to add
            new_spec = dict(spec) if isinstance(spec, dict) else {"type": spec}
            new_spec.setdefault("type", base_type)
            new_spec["id"] = new_id
            chain_nodes.append(new_spec)
        scope_nodes = [n for n in nodes if _spec_id(n) in scope_ids] + chain_nodes
        chain_ids = {_spec_id(n) for n in chain_nodes}
        derived_edges.extend(
            e for e in emit_edges(scope_nodes)
            if (e["to"] in chain_ids or e["from"] in chain_ids) and _edge_key(e) not in existing
        )
        nodes.extend(chain_nodes)
    return derived_edges


def _dataset_fields_for(
    config: Any, params: Optional[dict], *, has_tts: bool = False
) -> Optional[dict]:
    """The declared column schema for a dataset_source node instance; multi-source nodes overlay
    their `datasets:` entry on ``config.data``. Unresolvable → None (node keeps static outputs)."""
    from dataclasses import replace

    data = getattr(config, "data", None)
    if data is None:
        return None
    try:
        ds_id = (params or {}).get("dataset")
        if ds_id:
            entry = (getattr(data, "datasets", None) or {}).get(str(ds_id)) or {}
            overlay = {k: v for k, v in entry.items() if k not in ("role", "datasets")}
            data = replace(data, datasets=None, **overlay)
        descriptor = resolve_dataset_descriptor(data)
    except Exception as exc:  # noqa: BLE001 - preview-only; the run path re-raises its own
        logger.debug(
            "dataset descriptor unresolvable for %r (node keeps static outputs): %s",
            (params or {}).get("dataset"), exc,
        )
        return None
    if not descriptor.fields:
        return None
    out = {"fields": dict(descriptor.fields)}
    if descriptor.embedding_space:
        out["embedding_space"] = descriptor.embedding_space
    extra = _derived_source_outputs(descriptor, config, has_tts)
    if extra:
        out["extra_outputs"] = extra
    return out


def _derived_source_outputs(descriptor: Any, config: Any, has_tts: bool) -> Tuple[str, ...]:
    """Artifacts the runtime source publishes that aren't literal dataset columns (self-retrieval
    corpus, republished reference_text, TTS-bridge reference_transcription); missing here,
    ``effective_outputs`` narrows them away though the handler publishes them, so the emitted
    edge list fails to re-bind consumers holding those bindings."""
    # TTS-bridge reference is graph-gated (a tts node exists) — the graph is the spec, not a
    # config.audio_synthesis flag.
    extra: List[str] = list(descriptor.derived_outputs)
    if ARTIFACT_REFERENCE_TEXT not in set(descriptor.fields.values()):
        extra.append(ARTIFACT_REFERENCE_TEXT)
    if (
        ARTIFACT_REFERENCE_TRANSCRIPTION not in set(descriptor.fields.values())
        and ARTIFACT_REFERENCE_TRANSCRIPTION not in extra
        and descriptor.supports_generation
        and has_tts
    ):
        extra.append(ARTIFACT_REFERENCE_TRANSCRIPTION)
    return tuple(extra)


def _attach_dataset_fields(node_spec: Any, config: Any) -> list:
    """Inject ``params.fields`` (the column schema) into dataset_source spec entries and mark
    superseded query outputs, before wiring, so ``effective_outputs`` (and every DAG display
    surface) sees per-dataset columns. Entries may be node-type strings or {id, type, params}."""

    def _kind(e):
        return node_kind(
            e if isinstance(e, str) else (e.get("type") or e.get("id")),
            None if isinstance(e, str) else e.get("params"),
        )

    kinds = {_kind(e) for e in node_spec}
    # A transform that re-produces a query artifact is its real producer, so dataset_source must
    # not also advertise it: tts owns query_audio; asr owns query_text unless a tts node bridges it.
    superseded = []
    if "tts" in kinds:
        superseded.append("query_audio")
    if "asr" in kinds and "tts" not in kinds:
        superseded.append("query_text")
    out = []
    for entry in node_spec:
        if isinstance(entry, str):
            stage, spec = entry, {"id": entry, "type": entry}
        else:
            spec = dict(entry)
            stage = spec.get("type") or spec.get("id")
        if stage != "dataset_source":
            out.append(entry)
            continue
        params = dict(spec.get("params") or {})
        changed = False
        injected = _dataset_fields_for(config, params, has_tts="tts" in kinds)
        if injected and "fields" not in params:
            params.update(injected)
            changed = True
        existing = tuple(params.get("suppress_outputs") or ())
        new_suppress = tuple(a for a in superseded if a not in existing)
        if new_suppress:
            params["suppress_outputs"] = (*existing, *new_suppress)
            changed = True
        if changed:
            spec["params"] = params
            out.append(spec)
        else:
            out.append(entry)
    return out


def _attach_display_fields(graph: StageGraph, config: Any) -> StageGraph:
    """POST-wiring display attach for port-wired graphs: the same dataset column schema +
    superseded-output marks as :func:`_attach_dataset_fields`, but injected after the edges
    defined the bindings — pure display metadata, zero wiring effect."""
    from dataclasses import replace

    kinds = {node_kind(n.stage, n.params) for n in graph.nodes}
    superseded = []
    if "tts" in kinds:
        superseded.append("query_audio")
    if "asr" in kinds and "tts" not in kinds:
        superseded.append("query_text")
    nodes = []
    for n in graph.nodes:
        if node_kind(n.stage, n.params) != "dataset_source":
            nodes.append(n)
            continue
        params = dict(n.params or {})
        injected = _dataset_fields_for(config, params, has_tts="tts" in kinds)
        if injected and "fields" not in params:
            params.update(injected)
        existing = tuple(params.get("suppress_outputs") or ())
        new_suppress = tuple(a for a in superseded if a not in existing)
        if new_suppress:
            params["suppress_outputs"] = (*existing, *new_suppress)
        nodes.append(replace(n, params=params))
    return StageGraph(mode=graph.mode, nodes=tuple(nodes))


def _wire_explicit_graph(graph_override, *, config: Any, attach_fields: bool) -> StageGraph:
    """Wire a config's explicit graph. ``attach_fields`` injects the dataset column schema into
    ``dataset_source`` nodes for the config/preview path (so the DAG display shows real columns)
    but not for the run; with port-level edges it is display-only, applied POST-wiring."""
    override = graph_override or {}
    if not override.get("nodes"):
        raise ValueError("a config needs an explicit graph.nodes to build from")
    nodes = override["nodes"]
    edges = override.get("edges")
    label = graph_label(override) or "custom"
    if has_port_edges(edges):
        # Bindings come from the edges; preview == run. Wire the RAW nodes (no suppress/narrow
        # interference), then attach the display schema.
        graph = build_graph_from_spec(nodes, mode=label, edges=edges)
        return _attach_display_fields(graph, config) if attach_fields else graph
    if attach_fields:
        nodes = _attach_dataset_fields(nodes, config)
    return build_graph_from_spec(nodes, mode=label, edges=edges)


def build_graph_for_config(config: Any) -> StageGraph:
    """The execution DAG for a config — the single source for preview + CLI + the factory's
    build plan. Duck-typed (no config import)."""
    return _wire_explicit_graph(
        getattr(config, "graph_override", None), config=config, attach_fields=True
    )


def build_run_graph(graph_override, *, eval_config) -> StageGraph:
    """The execution DAG for a run: the same graph as :func:`build_graph_for_config` without the
    display-only dataset-field attach, so the run wires from the static source outputs."""
    return _wire_explicit_graph(graph_override, config=eval_config, attach_fields=False)
