"""Named pipeline modes + their derivation into node-id specs.

A named mode (``GRAPH_TEMPLATE_SPECS``) is just an ordered node-id list fed to the
auto-wiring engine. The list is assembled declaratively from a :class:`FeatureSet`
(`graph/assembly.py`); ``build_stage_graph`` / ``build_graph_for_config`` are the public
entry points (the former for tests, the latter the single config chokepoint).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .assembly import FeatureSet, assemble_specs
from .registry import (
    ARTIFACT_REFERENCE_TEXT,
    ARTIFACT_REFERENCE_TRANSCRIPTION,
    StageGraph,
    is_structural,
    node_model_field,
    validate_graph_artifacts,
)
from .wiring import (
    _has_port_edges,
    _normalize_spec_item,
    _wire_nodes,
    build_graph_from_spec,
    emit_edges,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphTemplateSpec:
    name: str
    required_model_fields: Tuple[str, ...]


def _required_model_fields(name: str) -> Tuple[str, ...]:
    """Union of model fields over a template's nodes (maximal feature set)."""
    fields: List[str] = []
    for spec in assemble_specs(name, FeatureSet.maximal()):
        _id, ntype, _params = _normalize_spec_item(spec)
        field = node_model_field(ntype, _params)  # resolves a callable model_field
        if field and field not in fields:
            fields.append(field)
    return tuple(fields)


def _make_template_spec(name: str) -> GraphTemplateSpec:
    return GraphTemplateSpec(
        name=name,
        required_model_fields=_required_model_fields(name),
    )


GRAPH_TEMPLATE_SPECS: Dict[str, GraphTemplateSpec] = {
    name: _make_template_spec(name)
    for name in (
        "asr_only",
        "asr_text_retrieval",
        "audio_emb_retrieval",
        "audio_text_retrieval",
    )
}


def resolve_graph_template(name: str) -> GraphTemplateSpec:
    if name not in GRAPH_TEMPLATE_SPECS:
        available = ", ".join(sorted(GRAPH_TEMPLATE_SPECS.keys()))
        raise ValueError(f"Unknown graph template: {name}. Available templates: {available}")
    return GRAPH_TEMPLATE_SPECS[name]


def build_stage_graph(mode: str, **features: Any) -> StageGraph:
    """Build the execution DAG for a pipeline mode; kwargs are ``FeatureSet`` fields
    (``rerank_enabled=True``, ``refine_ops=(…)``, …) — an unknown flag fails loudly."""
    resolve_graph_template(mode)  # validates the mode
    if "refine_ops" in features:
        features["refine_ops"] = tuple(features["refine_ops"])
    graph = StageGraph(
        mode=mode, nodes=_wire_nodes(assemble_specs(mode, FeatureSet(**features)))
    )
    validate_graph_artifacts(graph)
    return graph


def _dataset_fields_for(
    config: Any, params: Optional[dict], *, has_tts: bool = False
) -> Optional[dict]:
    """The declared column schema for a dataset_source node instance.

    Multi-source nodes (``params.dataset``) resolve their `datasets:` entry overlaid on
    ``config.data`` (same overlay as ``load_runtime_datasets``); single-source resolves
    the global ``config.data``. Unresolvable → None (the node keeps the static outputs).
    """
    from dataclasses import replace

    from ...datasets.descriptor import resolve_dataset_descriptor

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
    """Artifacts the runtime source publishes that aren't literal dataset columns, so the
    preview wires the nodes the run wires from the static source outputs:
    - a self-retrieval ``corpus`` (audio datasets retrieve against their own items — no corpus
      column exists),
    - the retrieval-side ``reference_text`` (the question text republished under its own name —
      ``handlers/source.py`` always publishes it), and
    - the TTS-bridge ASR ``reference_transcription`` (the synthesized speech's reference IS the
      question text; published whenever a tts node bridges text→audio).

    A source output missing here is narrowed away by ``_effective_outputs`` even though the
    handler publishes it, which leaves consumers holding bindings the graph says are
    impossible — the emitted edge list then fails to re-bind.
    """
    # Self-retrieval corpus is descriptor-level (shared with the builder picker via
    # descriptor.derived_outputs); the TTS-bridge reference is graph-gated (a tts node exists —
    # the graph is the spec, not a config.audio_synthesis flag).
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
    """Inject ``params.fields`` (the column schema) into dataset_source spec entries, and mark
    ``query_audio`` suppressed when a tts node in the same graph is its real producer.

    Runs before wiring so ``_effective_outputs`` (and every DAG display surface) sees
    per-dataset columns. Entries may be node-type strings or {id, type, params} dicts.
    """
    from .operators import node_kind

    def _kind(e):
        return node_kind(
            e if isinstance(e, str) else (e.get("type") or e.get("id")),
            None if isinstance(e, str) else e.get("params"),
        )

    kinds = {_kind(e) for e in node_spec}
    # A transform that RE-PRODUCES a query artifact is its real producer, so the dataset_source
    # must not also advertise it (its port would dangle — the consumer's edge resolves to the
    # transform — and it would show an input the dataset lacks). tts owns query_audio (synthesized
    # from the question). asr owns query_text (the hypothesis) UNLESS a tts node consumes the
    # source's query_text first (the TTS bridge, where the question text feeds tts).
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
    """POST-wiring display attach for explicitly-wired graphs (E4): inject the dataset column
    schema + superseded-output marks onto ``dataset_source`` nodes AFTER the edges defined the
    bindings — pure display metadata (canvas columns, narrowed preview ports), zero wiring
    effect. Mirrors ``_attach_dataset_fields``'s injected params."""
    from dataclasses import replace

    from .operators import node_kind

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


def _wire_mode_graph(
    mode: Optional[str],
    features: "FeatureSet",
    *,
    graph_override: Optional[dict],
    config: Any,
    attach_fields: bool,
) -> StageGraph:
    """Shared assembly tail for both graph builders: an explicit ``graph_override`` if given,
    else ``assemble_specs(mode, features)``; wire it. ``attach_fields`` injects the dataset
    column schema into ``dataset_source`` nodes for the config/preview path (so the DAG
    display shows real columns) but NOT for the run. With explicit PORT-LEVEL edges (E4) the
    attach is display-only and happens POST-wiring — the edges define the bindings for preview
    and run alike, so the two builders produce ONE graph; on the legacy auto-wire path the attach
    still narrows wiring (the historical preview/run divergence). A mode-less explicit DAG labels
    "custom".

    No graph.branches: comparing N variants of a node is just N distinctly-named entries in
    the one node list, wired from whatever they share via ordinary edges — nothing to expand
    or CSE-collapse, the graph already says exactly what runs."""
    from .wiring import _has_port_edges

    override = graph_override or {}
    if override.get("nodes"):
        nodes = override["nodes"]
        edges = override.get("edges")
        if _has_port_edges(edges):
            # Explicit wiring: bindings come from the edges; preview == run. Wire the RAW
            # nodes (no suppress/narrow interference), then attach the display schema.
            graph = build_graph_from_spec(nodes, mode=mode or "custom", edges=edges)
            return _attach_display_fields(graph, config) if attach_fields else graph
        if attach_fields:
            nodes = _attach_dataset_fields(nodes, config)
        return build_graph_from_spec(nodes, mode=mode or "custom", edges=edges)
    if mode is None:
        raise ValueError(
            "a config needs an explicit graph.nodes to build from (or a builder template name)."
        )
    base = assemble_specs(mode, features)
    if attach_fields:
        base = _attach_dataset_fields(base, config)
    return build_graph_from_spec(base, mode=mode)
    return build_graph_from_spec(base, mode=mode)


def _config_template(config: Any) -> Optional[str]:
    """The graph template a config selects (``graph_override['template']``, back-compat
    ``model.pipeline_mode`` / ConfigTemplates only). ``None`` for every explicit-graph config —
    the run derives the label from the node kinds instead."""
    override = getattr(config, "graph_override", None) or {}
    template = override.get("template")
    return str(template) if template else None


def resolve_template_label(config: Any) -> Optional[str]:
    """THE template-label resolver: an explicit back-compat template reference wins, else
    the label derives from the explicit graph's node kinds (``label_from_graph``); ``None``
    for a custom shape. ``EvaluationConfig.graph_template``, ``build_graph_for_config`` and
    the executor's mode detection all resolve through here."""
    return _config_template(config) or label_from_graph(
        getattr(config, "graph_override", None)
    )


def label_from_graph(graph_override: Optional[dict]) -> Optional[str]:
    """Derive the run/leaderboard mode LABEL from an explicit graph's node kinds — the graph is
    the spec (no ``graph.mode``). Mirrors ``detect_graph_template``'s pipeline logic on nodes so
    audio_emb stays distinguishable from audio_text fusion: audio_embedding + text_embedding ⇒
    ``audio_text_retrieval``; audio_embedding alone ⇒ ``audio_emb_retrieval``; asr + retrieval ⇒
    ``asr_text_retrieval``; asr alone ⇒ ``asr_only``. ``None`` when no query head is present (a
    custom graph — the run then derives a 'custom' label)."""
    from .operators import node_kind

    kinds = set()
    for n in (graph_override or {}).get("nodes") or []:
        if isinstance(n, str):
            t, p = n, {}
        else:
            t, p = n.get("type") or n.get("id"), n.get("params") or {}
        kinds.add(node_kind(t, p))
    if "audio_embedding" in kinds:
        return "audio_text_retrieval" if "text_embedding" in kinds else "audio_emb_retrieval"
    if "asr" in kinds:
        return "asr_text_retrieval" if "retrieval" in kinds else "asr_only"
    return None


# ---------------------------------------------------------------------------------------------
# Structural-plumbing auto-derivation for an EXPLICIT graph (Approach B: UI-graph ↔
# execution-DAG split, evaluator-architecture.md §12.1/§12.2). A config/canvas author writes
# only the *meaningful* nodes; the metric-comparison/report/trace/finalize plumbing
# (`is_structural`) is derived here — for every explicit-graph entry point (CLI YAML via
# `config/graph_config.py:build_evaluation_config_kwargs`, and the webapi builder/Config&Run,
# which both funnel through that same function). One implementation, not two staying in sync.
# ---------------------------------------------------------------------------------------------

def _spec_type(spec: Any) -> str:
    """A graph-spec item is a node-type string OR a ``{id, type, params}`` dict."""
    return spec if isinstance(spec, str) else spec.get("type")


def _spec_params(spec: Any) -> dict:
    return {} if isinstance(spec, str) else (spec.get("params") or {})


def _spec_id(spec: Any) -> str:
    """A node-list entry's id — the explicit ``id``, else its type (bare-string specs and
    dict specs with no id both use the type as their id, matching how the executor names
    them)."""
    return spec if isinstance(spec, str) else (spec.get("id") or _spec_type(spec))


def _edge_key(e: Dict[str, Any]) -> tuple:
    """``output`` may be omitted when it equals ``input`` — compare on the resolved pair so
    the short and long spellings of one edge dedup against each other."""
    return (e.get("from"), e.get("output", e.get("input")), e.get("to"), e.get("input"))


#: FeatureSet fields deliberately NOT set by `_infer_features` — grouped by why, so
#: `tests/test_webapi_registry_contract.py::test_infer_features_covers_every_feature_flag`
#: can check every field is either inferred below or accounted for here, instead of a new
#: FeatureSet flag silently going ungoverned (drifting whichever way its dataclass default
#: happens to point, with no plumbing decision ever made for it).
_INFER_FEATURES_STRUCTURAL_GATING_EXEMPT = frozenset({
    "sink_enabled",   # sinks ARE structural/derived, never drawn on the canvas at all
    "trace_enabled",  # traces ride the judge — `judge_enabled` (inferred) already covers it
})
#: Meaningful-*shape* flags `assemble_specs` also reads, but which only reshape nodes ALREADY
#: kept by another inferred flag (they never add/remove a structural node on their own) — no
#: separate re-inference needed. If one of these ever starts gating a structural node's
#: presence/absence, move it out of this set and add its inference above.
_INFER_FEATURES_SHAPE_ONLY_EXEMPT = frozenset({
    "hybrid_retrieval", "rag_rounds", "refine_method", "refine_context_top_k", "refine_ops",
})


def _infer_features(nodes: list) -> Any:
    """Derive a ``FeatureSet`` from which meaningful nodes are present — drives which
    *structural* plumbing (which metric comparisons, traces, …) `complete_structural_plumbing`
    appends.

    Only flags reachable from a meaningful node are inferred; every other ``FeatureSet`` field
    must be on one of the two exemption sets above (checked by a contract test) — never silently
    ungoverned. A future structural node gated on a NEW flag must be added here."""
    from .operators import node_kind

    by_kind: Dict[str, dict] = {}
    for n in nodes:
        by_kind.setdefault(node_kind(_spec_type(n), _spec_params(n)), _spec_params(n))
    has = by_kind.__contains__
    return FeatureSet(
        correction_enabled=has("query_correction"),
        query_opt_enabled=has("query_optimization") or has("multi_query_retrieval"),
        query_opt_method=(by_kind.get("query_optimization", {}).get("method")
                          or by_kind.get("multi_query_retrieval", {}).get("method") or "rewrite"),
        rerank_enabled=has("rerank"),
        mmr_enabled=has("mmr"),
        threshold_enabled=has("threshold"),
        answer_gen_enabled=has("answer_gen"),
        judge_enabled=has("answer_judge"),  # assemble_specs adds build_query_traces for the judge
        embedding_fusion_enabled=has("fusion") or has("result_fusion"),
        result_fusion_enabled=has("result_fusion"),
        audio_synthesis_enabled=has("tts"),
    )


def _topo_sort_nodes(nodes: list, edges: list) -> list:
    """Stable topological sort of ``nodes`` by ``edges``' from→to precedence (the graph loader
    requires a producer to be listed before its consumer in ``graph.nodes``). Kahn's algorithm
    with ties broken by original position, so a graph with no ordering problem keeps its
    authored order untouched — this only moves what actually needs moving (e.g. a `metrics`/
    `build_query_traces` node appended after `complete_structural_plumbing` can land after a
    meaningful node that consumes it, like `answer_judge`, which the loader rejects)."""
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
#: traces) when more than one is present in an explicit graph — the R5 multi-variant fix. Kept
#: as an explicit, named, tested set rather than one hardcoded kind check: promoting a new kind
#: to "variant root" is a conscious decision (comparing N of them needs its own chain, the exact
#: bug class R5 fixed for `retrieval`), not something that should silently start/stop working as
#: the registry changes. `retrieval`: the original case (N ASR/encoder/retrieval-strategy
#: variants). `answer_gen`: comparing N LLMs on one shared retrieval (not yet exercised by a real
#: config, but the documented design decision — see evaluator-architecture.md/TODO.md — for how
#: LLM comparison is modeled: two independent `answer_gen` nodes, not a param list on one node).
#: `tests/test_webapi_registry_contract.py::test_variant_root_kinds_guard_...` fails loudly if a
#: new registry kind looks like a plausible third root (its output consumed by a structural node)
#: and isn't accounted for here or on that test's exemption list — see that test before editing.
VARIANT_ROOT_KINDS = frozenset({"retrieval", "answer_gen"})


def _terminal_variant_ids(variant_ids: List[str], edges: list) -> List[str]:
    """``variant_ids`` (nodes of a :data:`VARIANT_ROOT_KINDS` kind) with no OTHER variant-root
    node reachable downstream of them — the "final hop" candidates for their own structural
    chain. An upstream hop that feeds INTO another variant-root node (query-refine's 2-hop
    retrieval pipeline) is an earlier stage of the SAME pipeline, not an independent variant;
    only the terminal hop's results ever reach a judge/metrics node."""
    forward: Dict[str, set] = {}
    for e in edges:
        f, t = e.get("from"), e.get("to")
        if f and t:
            forward.setdefault(f, set()).add(t)
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
    """Whether ``variant_ids`` are genuinely independent comparisons (their downstream
    reachable sets never overlap) rather than parallel sources that CONVERGE back into one
    combined pipeline (e.g. `hybrid_retrieval.yaml`'s `retrieval` + `retrieval_sparse`, both
    feeding one shared `result_fusion` before a single `metrics`/`finalize`). Two terminal
    variant-root candidates can both be "terminal" per `_terminal_variant_ids` (neither reaches
    the OTHER root) while still sharing everything downstream of a fusion/combine node — that's
    ONE variant with two sources, not two variants to compare separately, and must get the
    single shared structural chain, not a per-root suffixed one."""
    if len(variant_ids) < 2:
        return True
    forward: Dict[str, set] = {}
    for e in edges:
        f, t = e.get("from"), e.get("to")
        if f and t:
            forward.setdefault(f, set()).add(t)
    reach: Dict[str, set] = {}
    for rid in variant_ids:
        stack, seen = [rid], set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            reach.setdefault(cur, set()).add(rid)
            stack.extend(forward.get(cur, ()))
    return all(len(rids) <= 1 for rids in reach.values())


def complete_structural_plumbing(graph: Dict[str, Any]) -> None:
    """Approach B (UI-graph ↔ execution-DAG split): a config/canvas author writes only the
    *meaningful* operations; here we append the *structural* plumbing the template would derive
    (the metric comparisons + report + traces + finalize) so an explicit graph runs as the full
    DAG. On the explicit-wiring path the appended nodes' edges are derived here via the
    authoring wirer (E5); on the legacy path they carry no edges and auto-wire. The caller's
    meaningful nodes (incl. per-node models) are untouched, so edit-time checks stay exact.
    No-op on the template-only path (no explicit nodes): `assemble_specs` already adds the
    plumbing there.

    Mutates ``graph`` (the RAW pre-translation ``{"nodes": [...], "edges": [...]}`` dict) — must
    be called before edge validation, since a meaningful node's own binding can name a
    structural producer (e.g. `answer_judge` → `build_query_traces`) that doesn't exist in the
    submitted graph yet. The plumbing template's mode is derived internally (via
    `label_from_graph`, with a generic fallback for a pure-text-retrieval graph that has no
    asr/audio query head) — callers never need to compute or pass one.

    R5/multi-variant: a graph comparing several paths (multiple ASR models, audio encoders,
    retrieval strategies, LLMs, ...) has several meaningful nodes of a :data:`VARIANT_ROOT_KINDS`
    kind. Appending only ONE shared structural chain (the pre-fix behavior) let `emit_edges` bind
    every variant's measured output into that single chain — silently merging all variants'
    metrics into one. Each variant-root node gets its own chain instead, suffixed by its id and
    wired only to what's exclusively downstream of it (`_complete_with_plumbing_per_variant`)."""
    override = graph
    if not (isinstance(override, dict) and override.get("nodes")):
        return
    from .operators import node_kind

    # Reject a duplicate id in the CALLER'S OWN node list immediately — the same guard
    # `wiring.py:bind_explicit_edges` applies later is too late here: this function's own
    # id-keyed logic (the per-variant suffix computation, `emit_edges`'s auto-wiring) can
    # mis-derive a confusing, unrelated-looking downstream error (e.g. a bogus "duplicate edge"
    # from processing the same id twice) instead of naming the real cause.
    seen_ids: Dict[str, int] = {}
    for i, n in enumerate(override["nodes"]):
        nid = _spec_id(n)
        if nid in seen_ids:
            raise ValueError(
                f"duplicate node id {nid!r} in graph.nodes (positions {seen_ids[nid]} and {i}) "
                "— every node id must be unique."
            )
        seen_ids[nid] = i

    mode = label_from_graph(override)
    if mode is None:
        # label_from_graph only recognizes an asr/audio_embedding query head — a pure TEXT
        # retrieval graph (text_embedding + retrieval, no ASR/audio node at all — a real shape,
        # e.g. configs/pubmed_qa_rag_fulltext.yaml) has neither, so it returns None and plumbing
        # completion would silently no-op (no metrics/finalize at all). The structural chain
        # (retrieval_metrics/metrics/build_query_traces/finalize) doesn't depend on the query
        # head's modality, so "asr_text_retrieval"'s template is a safe, generic stand-in here —
        # used ONLY to pick the structural template, not as the graph's real mode label
        # (computed independently downstream, stays "custom").
        kinds = {node_kind(_spec_type(n), _spec_params(n)) for n in override.get("nodes") or []}
        if "retrieval" in kinds:
            mode = "asr_text_retrieval"
    if mode is None:
        return

    nodes = override["nodes"]
    edges = override.get("edges") or []
    port_edges = _has_port_edges(edges)
    kinds_present = {node_kind(_spec_type(n), _spec_params(n)) for n in nodes}
    structural_template = [
        spec for spec in assemble_specs(mode, _infer_features(nodes))
        if is_structural(_spec_type(spec), _spec_params(spec))
        # `transcription_metrics` (an ASR-specific measure) declares no formal input port — it
        # can't be filtered by input-satisfiability like `answer_metrics`/
        # `embedding_alignment_metrics` are (via `_infer_features`/assemble_specs' own gating).
        # It's unconditionally part of every asr-inclusive mode's template, which is correct for
        # a REAL asr_text_retrieval/asr_only graph but wrong when that mode was only borrowed
        # above as a generic stand-in for a graph with no asr node at all (pure text retrieval).
        and not (node_kind(_spec_type(spec), _spec_params(spec)) == "transcription_metrics"
                 and "asr" not in kinds_present)
    ]
    raw_variant_ids = [
        _spec_id(n) for n in nodes
        if node_kind(_spec_type(n), _spec_params(n)) in VARIANT_ROOT_KINDS
    ]
    # A variant-root node feeding INTO another variant-root node (query-refine's 2-hop pipeline:
    # retrieval -> query_refine -> retrieval_refined) is an earlier stage of ONE pipeline, not
    # a sibling variant to compare — only the terminal hop's chain matters (the original hop's
    # results never reach the judge/metrics directly). Only nodes with no OTHER variant-root node
    # downstream of them are candidates for their own chain.
    variant_ids = _terminal_variant_ids(raw_variant_ids, edges)

    if len(variant_ids) > 1 and port_edges and _variant_roots_diverge(variant_ids, edges):
        override["edges"] = list(edges) + _complete_with_plumbing_per_variant(
            nodes, edges, structural_template, variant_ids
        )
    else:
        # Single variant-root node (the common case) or a legacy non-port-edge graph: one shared
        # structural chain, exactly as before.
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
                # A meaningful node (typically user-renamed) already occupies the id a
                # structural node needs — e.g. a `retrieval` node renamed to "metrics"
                # collides with the auto-derived report node of that same name. Silently
                # appending anyway would leave two nodes sharing one id; `_topo_sort_nodes`'s
                # own id-keyed dicts would then silently collapse one of them (dropping its
                # edges onto the wrong node) instead of erroring here where the cause is clear.
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
        # E5: with explicit port wiring the appended plumbing needs its edges too — derive
        # them via the authoring wirer over the FULL list and keep those touching an appended
        # node (the caller's drawn edges stay authoritative for the meaningful graph).
        if appended and port_edges:
            app_ids = set(appended)
            existing = {_edge_key(e) for e in edges}
            derived = [
                e for e in emit_edges(nodes)
                if (e["to"] in app_ids or e["from"] in app_ids) and _edge_key(e) not in existing
            ]
            override["edges"] = list(edges) + derived

    # A node appended above can land after a meaningful node that consumes it (e.g.
    # `build_query_traces` appended after an already-present `answer_judge`) — the loader
    # requires a producer to precede its consumer in `graph.nodes`. Fix up the order using
    # every edge now known; a no-op when nothing actually needs moving.
    if override.get("edges"):
        override["nodes"][:] = _topo_sort_nodes(override["nodes"], override["edges"])


def _complete_with_plumbing_per_variant(
    nodes: list, edges: list, structural_template: list, variant_ids: List[str],
) -> list:
    """Append one private structural chain per variant-root node (mutates ``nodes``); returns
    the derived edges for all of them. A chain's nodes are wired via `emit_edges` scoped to
    ``rid``'s own ancestors plus whatever's downstream of it and NOT exclusively owned by a
    DIFFERENT variant-root node — so one variant's measured output can never bind into
    another's chain, AND a structural node with no adjacency of its own (transcription_metrics:
    a plain, non-OneOf ``query_text`` input — see the template-assembly comment above) can't
    silently bind to a SIBLING variant's ancestor of the same kind either."""
    forward: Dict[str, set] = {}
    backward: Dict[str, set] = {}
    for e in edges:
        f, t = e.get("from"), e.get("to")
        if f and t:
            forward.setdefault(f, set()).add(t)
            backward.setdefault(t, set()).add(f)
    # Which variant root(s) can reach each node, walking forward over the MEANINGFUL edges
    # only (the structural chains don't exist yet) — a node reachable from exactly one
    # variant id is that variant's exclusive property (its own rerank/answer_gen/judge/...).
    reach: Dict[str, set] = {}
    forward_reach: Dict[str, set] = {}
    for rid in variant_ids:
        stack, seen = [rid], set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            reach.setdefault(cur, set()).add(rid)
            stack.extend(forward.get(cur, ()))
        forward_reach[rid] = seen
    owned = {
        rid: {n for n, rids in reach.items() if rids == {rid}}
        for rid in variant_ids
    }

    def _ancestors(rid: str) -> set:
        """Nodes that can reach ``rid`` walking BACKWARD over the meaningful edges — its
        true upstream lineage (asr/tts/text_embedding/corpus_embedding/dataset_source/...).
        Two variant roots can legitimately share a subset of these (e.g. two retrieval
        variants over the same asr+tts but different text_embedding) — that sharing is
        exactly why the old "not exclusively owned by someone else" scope was too broad: a
        node shared by 2 of 18 variants was never exclusively owned by any ONE of them,
        so it stayed in EVERY other variant's scope too, not just its real two owners."""
        stack, seen = [rid], set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(backward.get(cur, ()))
        seen.discard(rid)
        return seen

    from .operators import node_kind

    id_to_spec = {_spec_id(n): n for n in nodes}
    all_ids = set(id_to_spec)
    existing = {_edge_key(e) for e in edges}
    derived_edges: list = []
    for rid in variant_ids:
        other_owned = set().union(*(owned[o] for o in variant_ids if o != rid))
        # rid's own upstream lineage (correctly excludes a SIBLING variant's ancestor of
        # the same kind, e.g. a different asr node) + rid itself + whatever's downstream
        # of rid and not exclusively claimed by a DIFFERENT variant (the query-refine
        # 2-hop case this scope was originally built for).
        scope_ids = _ancestors(rid) | {rid} | (forward_reach.get(rid, set()) - other_owned)
        # A per-variant meaningful node the author already wrote (e.g. `answer_judge_dense`) may
        # already have its OWN binding to a structural producer by a SPECIFIC id (e.g.
        # `build_query_traces_dense`) that doesn't exist yet — reuse that exact id instead of
        # inventing a new one, or the existing edge stays dangling (unknown node). Falls back
        # to `<type>_<suffix>` (stripping the ROOT's own kind name off its id — e.g.
        # `retrieval_dense` -> "dense", `answer_gen_mms` -> "mms" — matching the convention real
        # multi-variant configs already use, whichever :data:`VARIANT_ROOT_KINDS` kind the root
        # actually is) when nothing references it yet.
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
                    # A meaningful node (typically user-renamed) occupies the id this
                    # variant's structural chain needs — e.g. a node renamed to "metrics_dense"
                    # collides with the auto-derived per-variant report node of that name.
                    # Blindly treating it as "already present" (the old behavior) would silently
                    # scope a real meaningful node into someone else's structural chain instead
                    # of erroring here where the cause is clear.
                    raise ValueError(
                        f"cannot auto-derive the structural node {new_id!r} (needed for "
                        f"{expected_kind!r} in variant {rid!r}): that id is already used by a "
                        f"{clash_kind!r} node — rename it."
                    )
                continue  # already present (e.g. a full-DAG spec, not meaningful-only) — the
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


def build_graph_for_config(config: Any) -> StageGraph:
    """Build the execution DAG for a config: the explicit ``graph.nodes`` (every config), else a
    back-compat template reference expanded with its feature flags. Single source for
    preview + CLI + the factory's build plan. Duck-typed (no config import). The run uses
    :func:`build_run_graph`, which sources its feature flags from the built pipelines instead."""
    override = getattr(config, "graph_override", None)
    return _wire_mode_graph(
        resolve_template_label(config),
        # explicit graphs ignore feature flags; derive them only for the template path
        FeatureSet() if (override or {}).get("nodes") else _features_from_config(config),
        graph_override=override,
        config=config,
        attach_fields=True,
    )


def build_run_graph(
    mode,
    *,
    graph_override,
    embedding_fusion_config,
    query_opt_config,
    retrieval_pipeline,
    eval_config,
    query_correction_config=None,
    trace_limit=0,
):
    """Build the execution DAG for a run: like :func:`build_graph_for_config` but the
    feature flags are sourced from what actually got built/bound at runtime — fusion /
    query-opt / correction from the run's ``RunFeatures``, rerank / mmr / threshold off the
    built retrieval pipeline's strategy config, ``trace`` from the run's trace limit — so the
    graph reflects reality, not just the config's declared intent. (Moved from the former
    ``pipeline/run_graph.py``; shares ``_wire_mode_graph`` with the config builder.)"""
    from dataclasses import replace

    # An explicit graph ignores feature flags entirely (_wire_mode_graph wires the nodes);
    # only the back-compat template path (model.pipeline_mode / a template reference)
    # derives its node list from them — skip the wasted derivation otherwise.
    if (graph_override or {}).get("nodes"):
        return _wire_mode_graph(
            mode, FeatureSet(), graph_override=graph_override,
            config=eval_config, attach_fields=False,
        )

    def _enabled(cfg):
        return bool(cfg is not None and getattr(cfg, "enabled", False))

    fusion_on = _enabled(embedding_fusion_config)
    # The built retrieval pipeline is the authoritative source for the refine sub-steps
    # (rerank / mmr / threshold) — read them off its strategy config.
    sc = getattr(retrieval_pipeline, "strategy_config", None)
    rerank_on = bool(
        sc
        and (
            str(sc.reranking.mode) != "none"
            or getattr(retrieval_pipeline, "reranker", None) is not None
        )
    )
    mmr_on = bool(sc and sc.post_processing.use_mmr)
    threshold_on = bool(sc and sc.post_processing.min_similarity_threshold is not None)
    features = replace(
        _features_from_config(eval_config),
        embedding_fusion_enabled=fusion_on,
        result_fusion_enabled=fusion_on
        and getattr(embedding_fusion_config, "level", "embedding") == "result",
        query_opt_enabled=_enabled(query_opt_config),
        query_opt_method=str(getattr(query_opt_config, "method", "rewrite")),
        correction_enabled=_enabled(query_correction_config),
        rerank_enabled=rerank_on,
        mmr_enabled=mmr_on,
        threshold_enabled=threshold_on,
        trace_enabled=trace_limit > 0,
    )
    # attach_fields=False — the run wires from the static dataset_source outputs (parity).
    return _wire_mode_graph(
        mode, features, graph_override=graph_override, config=eval_config, attach_fields=False
    )


def _features_from_config(config: Any) -> FeatureSet:
    """The single place config → FeatureSet: each optional capability's `enabled` flag +
    the structural choices (fusion level, query-opt method, rag rounds)."""
    fusion = bool(
        getattr(config, "embedding_fusion", None) and config.embedding_fusion.enabled
    )
    rag = getattr(config, "rag", None)
    _vdb = getattr(config, "vector_db", None)
    return FeatureSet(
        embedding_fusion_enabled=fusion,
        result_fusion_enabled=fusion
        and getattr(config.embedding_fusion, "level", "embedding") == "result",
        query_opt_enabled=bool(
            getattr(config, "query_optimization", None)
            and config.query_optimization.enabled
        ),
        query_opt_method=str(
            getattr(getattr(config, "query_optimization", None), "method", "rewrite")
        ),
        hybrid_retrieval=bool(
            _vdb and str(getattr(_vdb, "retrieval_mode", "dense")) == "hybrid"
        ),
        rerank_enabled=bool(
            _vdb
            and (
                getattr(_vdb, "reranker_enabled", False)
                or str(getattr(_vdb, "reranker_mode", "none")) != "none"
            )
        ),
        mmr_enabled=bool(_vdb and getattr(_vdb, "use_mmr", False)),
        threshold_enabled=bool(
            _vdb and getattr(_vdb, "min_similarity_threshold", None) is not None
        ),
        refine_ops=tuple(getattr(_vdb, "refine_ops", None) or ()),
        sink_enabled=bool(
            getattr(config, "dataset_sink", None) and config.dataset_sink.enabled
        ),
        correction_enabled=bool(
            getattr(config, "query_correction", None)
            and config.query_correction.enabled
        ),
        answer_gen_enabled=bool(
            getattr(config, "answer_generation", None)
            and getattr(config.answer_generation, "enabled", False)
        ),
        judge_enabled=bool(
            getattr(config, "judge", None) and getattr(config.judge, "enabled", False)
        ),
        trace_enabled=int(getattr(getattr(config, "data", None), "trace_limit", 0) or 0) > 0,
        rag_rounds=int(getattr(rag, "rounds", 1) or 1),
        refine_method=str(getattr(rag, "refine_method", "rewrite_with_context")),
        refine_context_top_k=int(getattr(rag, "refine_context_top_k", 3) or 3),
        audio_synthesis_enabled=bool(
            getattr(config, "audio_synthesis", None)
            and getattr(config.audio_synthesis, "enabled", False)
        ),
    )
