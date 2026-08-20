"""Registry/config-driven UI helpers for the WebAPI (form builder).

Builds UI options/forms from the registries and config (``node_catalogue``, ``graph_preview``,
``config_to_canvas_spec``, ``resolve_node_form``).
The form → validated-config half lives in ``form_config.py`` (import its public names
from there directly); ``prepare_run_config`` is re-exported here for one remaining caller.
"""

from __future__ import annotations

from typing import Any, Dict, List

from evaluator import EvaluationConfig
from evaluator.config.model_fields import (
    MODEL_FIELD_FAMILY as _MODEL_FIELD_FAMILY,
)
from evaluator.pipeline import build_graph_for_config
# Public introspection surface (the supported boundary — not registry/stage_graph privates).
from evaluator.pipeline.introspection import (
    dataset_columns,
    display_artifact_names,
    display_label,
    get_stage_node_def,
    node_model_field,
    resolve_field as _resolve,
    effective_inputs as _effective_inputs,
    effective_outputs as _effective_outputs,
)
from evaluator.webapi.field_help import FIELD_HELP

# Form → validated-config half (moved to form_config.py); ``prepare_run_config`` is still used
# directly here / by routers.config, so keep that one importable from this module.
from evaluator.webapi.form_config import (  # noqa: F401  (re-export surface)
    prepare_run_config,
)


def nested_config(config: EvaluationConfig) -> Dict[str, Any]:
    """Return nested config shape for WebUI config creator."""
    return {
        "experiment_name": config.experiment_name,
        "output_dir": config.output_dir,
        "runtime": config.to_runtime_dict(),
        "experiment": config.to_experiment_dict(),
    }


def _node_inspect(config: EvaluationConfig, node: Any) -> Dict[str, Any]:
    """What the preview's click-to-inspect panel shows for a node: the configured model
    (resolved from ``config.model`` via the node→config mapping in
    ``graph_config._MODEL_NODE_FIELDS`` — single source, not duplicated), retrieval
    settings, plus any explicit per-node ``params`` (branch/graph overrides) on top.
    Empty values are dropped."""
    from dataclasses import fields as _dc_fields

    from evaluator.config.graph_config import _FEATURE_NODE_CONFIG, _MODEL_NODE_FIELDS
    from evaluator.pipeline.graph.operators import (
        node_kind,
        operator_discriminators,
    )

    # The config-block map is keyed by legacy node name; resolve the operator+fields back.
    kind = node_kind(node.stage, node.params)
    info: Dict[str, Any] = {}
    if kind in _MODEL_NODE_FIELDS:
        for yaml_key, cfg_field in _MODEL_NODE_FIELDS[kind].items():
            val = getattr(config.model, cfg_field, None)
            if val not in (None, "", {}):
                info[yaml_key] = val
    elif kind == "retrieval":
        info["k"] = config.vector_db.k
        info["mode"] = config.vector_db.retrieval_mode
        if getattr(config.vector_db, "reranker_enabled", False):
            # loader-compatible nested form (a bare string round-trips as an unknown key)
            info["reranker"] = {"enabled": True,
                                "mode": config.vector_db.reranker_mode or "token_overlap"}
    elif kind in _FEATURE_NODE_CONFIG:
        # Rehydrate the capability config the loader folded OFF this node (audit 2026-07 #2):
        # the canvas must show the CONFIGURED values, not the form defaults, or a
        # config→builder→config round-trip silently resets the feature to defaults. Only
        # fields differing from the sub-config's defaults ride back onto the node.
        sub = getattr(config, _FEATURE_NODE_CONFIG[kind], None)
        if sub is not None:
            _default = type(sub)()
            for f in _dc_fields(sub):
                val = getattr(sub, f.name)
                if f.name == "enabled":
                    if val is False:  # presence implies True; only an explicit off is signal
                        info["enabled"] = False
                    continue
                if val != getattr(_default, f.name) and val not in (None, "", {}):
                    info[f.name] = val
    # The operator's discriminator fields (axis/modality/op/family/…) are internal selectors
    # already encoded in the node's label/stage — they'd be noise in the inspect panel.
    hidden = operator_discriminators(node.stage) | {"fields"}
    if kind == "dataset_source":
        # Resolve the real dataset name + location (config.data / datasets-map) so an opened
        # config seeds a configured source, not '(from run config)'. The literal `dataset` param
        # is just the map key → drop it (the resolved name replaces it). Keep `fields` (the
        # column→artifact map) so the opened source keeps its columns + narrowed ports.
        info.update(_source_inspect_params(config, node.params or {}))
        if (node.params or {}).get("fields"):
            info["fields"] = node.params["fields"]
        hidden = hidden | {"dataset"}
    for key, val in (node.params or {}).items():
        # `fields` (the dataset column schema) renders as the dedicated columns
        # row in the inspect panel, not as a raw params JSON blob.
        if key not in hidden and val not in (None, "", {}):
            info[key] = val
    return info


def _preview_node(config: EvaluationConfig, node: Any) -> Dict[str, Any]:
    """The rich preview node card the read-only DAG view + the builder seed render from:
    the graph-instance extras (stage/bindings/deps/structural/columns/inspect) plus the SAME
    field-aware form contract ``render_node`` and ``/api/graph/node-form`` use
    (``resolve_node_form`` — one serializer, so the three surfaces cannot drift; a hand-rolled
    copy here once dropped ``input_ports`` and broke the preview's ports)."""
    from ..pipeline.introspection import is_structural

    return {
        "id": node.id,
        "stage": node.stage,
        # structural plumbing (metric comparisons / report / finalize / sinks) → hidden in the
        # simplified view, revealed via "View full DAG"; the builder seed strips it.
        "structural": is_structural(node.stage, dict(node.params or {})),
        "depends_on": list(node.depends_on),
        # resolved data bindings (artifact → producer) — the preview draws
        # these as edges, same shape as /api/graph/template (B2)
        "bindings": [list(b) for b in node.bindings],
        # declared column schema [{name, artifact, type}] — what the diagram
        # shows on dataset nodes so an experiment reads from its DAG
        "columns": dataset_columns(node.params),
        "inspect": _node_inspect(config, node),
        # label / category / domain / inputs / outputs / optional_inputs / input_ports /
        # model_field / family / node_params / default_model — field-resolved per instance.
        **resolve_node_form(node.stage, dict(node.params or {})),
    }


def graph_preview(config: EvaluationConfig) -> Dict[str, Any]:
    from ..datasets.profiles import profile_snapshot

    graph = build_graph_for_config(config)
    payload = graph_render_payload(graph, lambda node: _preview_node(config, node))
    payload["dataset_profile"] = profile_snapshot(config)
    return payload


# Dataset node-keys that *locate* the data — the inverse of _synthesize_dataset_entry pulls these
# off config.data onto the dataset_source node so the canvas carries a runnable source.
_DATASET_LOCATION_KEYS = (
    "questions", "corpus", "prepared_dir", "path", "audio_dir", "transcripts",
    "hf_dataset", "hf_subset", "hf_split", "type", "source",
)
# Per-run knobs — carried through the round-trip only when set to a *non-default* value, so a
# loaded config keeps its trace_limit/batch_size/split without cluttering the canvas with defaults.
_DATASET_RUN_KEYS = ("trace_limit", "batch_size", "split", "max_samples")
_DATASET_NODE_KEYS = (*_DATASET_LOCATION_KEYS, *_DATASET_RUN_KEYS)


def _dataset_source_params(config: EvaluationConfig) -> Dict[str, Any]:
    """The dataset_source node params for a single-source config: ``{dataset, <location>, <run
    knobs that differ from default>}`` read off ``config.data``. Empty for a multi-source
    ``data.datasets`` map (those nodes already reference a map entry by role)."""
    from evaluator.config.graph_config import _DATASET_FIELDS

    data = config.data
    name = getattr(data, "dataset_name", None)
    if not name or getattr(data, "datasets", None):
        return {}
    defaults = type(data)()
    params: Dict[str, Any] = {"dataset": name}
    for node_key in _DATASET_NODE_KEYS:
        cfg_field = _DATASET_FIELDS[node_key]
        value = getattr(data, cfg_field, None)
        if value in (None, "", [], {}):
            continue
        # a run knob at its default carries no information — skip it to keep the canvas clean
        if node_key in _DATASET_RUN_KEYS and value == getattr(defaults, cfg_field, None):
            continue
        params[node_key] = value
    return params


def _source_inspect_params(config: EvaluationConfig, params: Dict[str, Any]) -> Dict[str, Any]:
    """The REAL dataset name + location for a dataset_source node. The name lives in ``config.data``
    (single-source) or ``config.data.datasets[role]`` (a multi-source map keyed by this node) —
    NOT as the node's literal ``dataset`` param, which is only the map key. Without resolving it an
    opened config seeds the blank '(from run config)' instead of the configured dataset."""
    from evaluator.config.graph_config import _DATASET_FIELDS

    dmap = getattr(config.data, "datasets", None)
    entry = dmap.get(params.get("dataset")) if isinstance(dmap, dict) else None
    if entry is None:
        return _dataset_source_params(config)  # single-source: read config.data directly
    get = entry.get if isinstance(entry, dict) else (lambda k: getattr(entry, k, None))
    out: Dict[str, Any] = {}
    if get("dataset_name"):
        out["dataset"] = get("dataset_name")
    for node_key in _DATASET_NODE_KEYS:  # questions/corpus/split/… (run knobs only present if set)
        value = get(_DATASET_FIELDS[node_key])
        if value not in (None, "", [], {}):
            out[node_key] = value
    return out


def lift_single_source_dataset(nodes: list) -> Dict[str, Any]:
    """If a graph has exactly one ``dataset_source`` naming a registered dataset, lift its dataset
    + settings to a node-centric top-level ``dataset:`` block and strip them from the node (leaving
    it structural). Returns the block (``{}`` when not applicable). Mutates ``nodes`` in place — so
    a single-source canvas exports clean and round-trips idempotently (a bare dataset_source wires
    its columns from the top-level block). Multi-source graphs keep their per-node ``dataset``."""
    from evaluator.pipeline.graph.operators import node_kind

    sources = [
        n for n in nodes
        if isinstance(n, dict)
        and node_kind(n.get("type"), n.get("params") or {}) == "dataset_source"
    ]
    if len(sources) != 1:
        return {}
    params = sources[0].get("params") or {}
    ds_id = params.get("dataset")
    if not ds_id:
        return {}
    block = {"id": ds_id}
    for key in _DATASET_NODE_KEYS:
        if params.get(key) not in (None, "", [], {}):
            block[key] = params[key]
    sources[0]["params"] = {
        k: v for k, v in params.items()
        if k not in _DATASET_NODE_KEYS and k not in ("dataset", "fields")
    }
    return block


def config_to_canvas_spec(config: EvaluationConfig) -> Dict[str, Any]:
    """Serialize a config's DAG into the builder canvas seed (the ``/api/graph/template``
    shape: ``{mode, levels, nodes:[{id, type, params, bindings, …form}]}``) — but with each
    node's *configured* model/dataset/settings folded into its params.

    This is the inverse of the builder's ``exportSpec`` → :func:`graph_config.to_legacy_dict`
    fold (which lifts graph-node params into ``config.model`` / ``vector_db`` / ``data``): we
    push the resolved per-node config back onto the node so opening a YAML config shows the real
    experiment, editable, and re-running it from the canvas reproduces the same config.

    A branched config renders its **base** graph (not the CSE-expanded ``@branch`` DAG) and passes
    the branch definitions through, so the Branches panel re-hydrates — the same shape as a saved
    graph — instead of dropping onto the canvas as a tangle of expanded nodes.
    """
    import copy

    from evaluator.pipeline.graph.operators import node_kind

    override = getattr(config, "graph_override", None)
    branches = override.get("branches") if isinstance(override, dict) else None
    # Build the base (un-expanded) DAG for the canvas; keep `config` for the per-node model/
    # dataset fold (its model/data blocks are unchanged by branch expansion).
    base_config = config
    inspect_config = config
    if branches:
        base_config = copy.deepcopy(config)
        base_config.graph_override = {
            k: v for k, v in override.items() if k != "branches"
        }
        # Seed first branch's model-node overrides into inspect_config so _node_inspect
        # can show real values for nodes whose params live only in branches (not in
        # nodes.*). Only fills fields that are currently empty — explicit nodes.* values
        # still win. Result: clicking a node shows first-branch values, not "default".
        from evaluator.config.graph_config import _MODEL_NODE_FIELDS
        inspect_config = copy.deepcopy(config)
        first_branch = branches[0] if branches else {}
        for _node_name, _node_overrides in first_branch.items():
            if _node_name == "id" or not isinstance(_node_overrides, dict):
                continue
            if _node_name not in _MODEL_NODE_FIELDS:
                continue
            for _yaml_key, _cfg_field in _MODEL_NODE_FIELDS[_node_name].items():
                if getattr(inspect_config.model, _cfg_field, None) in (None, "", {}):
                    _val = _node_overrides.get(_yaml_key)
                    if _val not in (None, "", {}):
                        setattr(inspect_config.model, _cfg_field, _val)
    graph = build_graph_for_config(base_config)
    dataset_params = _dataset_source_params(config)

    def _render(node):
        params = dict(node.params or {})
        # Fold the resolved model/retrieval settings (config.model / vector_db) onto the node;
        # setdefault so an explicit per-node graph-override param still wins.
        for key, value in _node_inspect(inspect_config, node).items():
            params.setdefault(key, value)
        # A single-source dataset lives in config.data, not on the node — push it onto the
        # dataset_source so the canvas carries a runnable source.
        if dataset_params and node_kind(node.stage, node.params) == "dataset_source":
            for key, value in dataset_params.items():
                params.setdefault(key, value)
        return render_node(node, params)

    payload = graph_render_payload(graph, _render)
    if branches:
        payload["branches"] = branches
    return payload


def graph_render_payload(graph: Any, node_fn) -> Dict[str, Any]:
    """The common ``{mode, levels, nodes}`` envelope every graph render/preview payload shares:
    the graph mode + topological levels (layout) + one ``node_fn(node)`` per node. Callers supply
    the node shape they need (rich preview / canvas render / minimal build view) and tack any
    extras (dataset_profile, branches, llm, warnings) onto the result."""
    return {
        "mode": graph.mode,
        "levels": [[n.id for n in level] for level in graph.topological_levels()],
        "nodes": [node_fn(node) for node in graph.nodes],
    }


def build_canvas_graph(payload: Dict[str, Any]) -> Any:
    """Build the DAG from a builder canvas spec (``{mode, nodes, edges, branches?}``) — the
    topology-only build (no run config / dataset needed), so a graph-in-progress validates while
    authoring. Branched specs expand + CSE-collapse to the real run shape. Raises on a build error.
    Shared by ``graph_advice`` callers and ``/ui/validate-builder`` (rendered fragment)."""
    from evaluator.pipeline import build_graph_from_spec

    from evaluator.pipeline.graph.wiring import _has_port_edges

    nodes = payload.get("nodes") or []
    mode = payload.get("mode") or "asr_text_retrieval"
    raw = payload.get("edges") or []
    if _has_port_edges(raw):
        edges = [e for e in raw if e.get("from") and e.get("to")]  # explicit port wiring (E5)
    else:
        edges = {}
        for e in raw:
            if e.get("from") and e.get("to"):
                edges.setdefault(e["to"], []).append(e["from"])
    branches = payload.get("branches") or []
    if branches:
        from evaluator.pipeline.graph.branches import build_branched_graph

        return build_branched_graph(nodes, branches, mode=mode, edges=edges or None)
    return build_graph_from_spec(nodes, mode=mode, edges=edges)


def graph_advice(graph: Any, config: Any) -> "tuple[list, list]":
    """Edit-time advice for a built graph: embedding-space ``warnings`` (the same check the Run path
    enforces, surfaced while editing) + the ``metrics`` that will auto-inject given the produced
    artifacts, plus the ground-truth-missing notice (a measurable output is produced but its GT
    field is absent from the chosen dataset). Returns ``(warnings, metrics)``. Shared by the build
    endpoint + the validation fragment so the two can't drift."""
    from evaluator.evaluation.metric_registry import applicable_metrics, list_metrics
    from evaluator.evaluation.validation import check_embedding_spaces

    warnings = list(check_embedding_spaces(graph, config))
    produced: set = set()
    for n in graph.nodes:
        produced.update(_effective_outputs(n.stage, n.params))
    corrected_on = bool(
        getattr(getattr(config, "query_correction", None), "corrected_metrics", False)
    )
    metrics = [m.name for m in applicable_metrics(produced, include_opt_in=corrected_on)]
    # GT-missing: gated to GT the source CAN declare (relevant_docs); the transcription GT is loaded
    # at run-time, not a declared node output, so its absence here is not reliable. Grouped by GT.
    source_gt = _effective_outputs("source", {})
    blocked: dict = {}
    for m in list_metrics():
        if m.gt in source_gt and m.scored in produced and m.gt not in produced:
            blocked.setdefault(m.gt, set()).add(m.name)
    for gt, names in sorted(blocked.items()):
        warnings.append(
            f"No ground truth ({gt}) in the chosen dataset — "
            f"{', '.join(sorted(names))} won't be computed."
        )
    return warnings, metrics


def render_node(node: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """One node in a canvas render payload: id + type + params + resolved bindings + the
    field-aware form contract (ports/label/family/switches) — the shape ``DagView.drawGraph``
    consumes, identical to ``/api/graph/template``."""
    from ..pipeline.introspection import is_structural

    form = resolve_node_form(node.stage, dict(node.params or {}))
    _overlay_routed_aliases(form, node)
    return {
        "id": node.id,
        "type": node.stage,
        "params": params,
        "bindings": [list(b) for b in node.bindings],
        # mark plumbing so the simplified view hides it by default (power-user "view full DAG").
        "structural": is_structural(node.stage, dict(node.params or {})),
        **form,
    }


def _overlay_routed_aliases(form: Dict[str, Any], node: Any) -> None:
    """Surface a B2 *routed* artifact on the port it was wired into.

    An explicit edge may feed a type-open port an artifact the node def never lists (e.g.
    ``dataset_source.reference_text`` → an embedder's ``query_text`` port, how a text-query
    graph gets its query). That routing lives on the graph instance's ``input_aliases``, not in
    the def ``_input_ports`` reads, so without this the canvas cannot match the edge back to its
    port and a round-trip drops it. ``edges_for_graph`` does the same overlay when emitting."""
    ports = form.get("input_ports") or []
    for canonical, candidates in getattr(node, "input_aliases", ()) or ():
        port = next((p for p in ports if p.get("label") == canonical), None)
        if port is None:
            continue
        for art in candidates:
            if art not in port["names"]:
                port["names"].append(art)


def spec_to_render_payload(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build a canvas render payload (``{mode, levels, nodes}``) from a saved builder spec
    (``{mode, nodes:[{id,type,params}], edges}``) — so a stored graph re-renders onto the
    canvas. The DAG is rebuilt (``build_graph_from_spec``) to resolve bindings + levels, but
    each node keeps the *saved* params (models/dataset the user set)."""
    from evaluator.pipeline import build_graph_from_spec

    from evaluator.pipeline.graph.wiring import _has_port_edges

    spec_nodes = [n for n in (spec.get("nodes") or []) if isinstance(n, dict)]
    raw = spec.get("edges") or []
    if _has_port_edges(raw):
        edges = [e for e in raw if isinstance(e, dict) and e.get("from") and e.get("to")]
    else:
        edges = {}
        for edge in raw:
            if isinstance(edge, dict) and edge.get("from") and edge.get("to"):
                edges.setdefault(edge["to"], []).append(edge["from"])
    graph = build_graph_from_spec(
        spec_nodes, mode=spec.get("mode") or "custom", edges=edges or None
    )
    params_by_id = {str(n.get("id")): (n.get("params") or {}) for n in spec_nodes}
    payload = graph_render_payload(
        graph,
        lambda node: render_node(node, params_by_id.get(node.id, dict(node.params or {}))),
    )
    # Pass the variant set + global LLM through verbatim so a saved graph re-hydrates its
    # Branches / LLM panels on load (the DAG above is the *base*; branches expand at run time).
    if spec.get("branches"):
        payload["branches"] = spec["branches"]
    if spec.get("llm"):
        payload["llm"] = spec["llm"]
    return payload


# model_field config path → registry family the builder fetches model choices from
# (_MODEL_FIELD_FAMILY, from config.model_fields). rerank/corpus_embedding carry no
# model_field (per-instance / shared embedder) but still pick from a registry family.
# Operator/legacy stage → registry family for the builder's model picker (rerank/refine
# carry no model_field — the reranker model is per-instance — but pick from a registry).
_STAGE_FAMILY = {"rerank": "reranker", "refine": "reranker",
                 "corpus_embedding": "text_embedding"}


def _node_family(model_field, node_type, params=None):
    """The model registry the node's picker reads: from its model_field, else a per-instance
    family (rerank/corpus_embedding carry no model_field). Resolved field-aware so an operator
    points at the right registry (e.g. ``embed{modality:audio}`` → audio_embedding)."""
    from evaluator.pipeline.graph.operators import node_kind

    # Key on the resolved *kind* (node_kind), never the operator name: a multi-kind operator like
    # `refine` covers rerank (reranker model) AND mmr/threshold (model-free), so an operator-level
    # fallback would wrongly hand the reranker picker to mmr/threshold. node_kind already maps a
    # bare/aliased node to its kind (refine{} → rerank), so this covers every case.
    return (
        _MODEL_FIELD_FAMILY.get(model_field or "")
        or _STAGE_FAMILY.get(node_kind(node_type, params))
    )


def _input_ports(stage: str, params=None) -> List[Dict[str, Any]]:
    """Canvas input ports for a node instance: ONE port per declared input *slot*. A ``OneOf``
    group (the query-text / query-vector chains the auto-wiring chooses among) collapses to a
    single port that accepts any of its alternatives — so an embedder shows one ``query_text``
    port, not five, and ``search`` shows ``query_vectors`` + ``vector_index``, not eleven ports.

    Required ports first, then optional (``optional: true``). Each port carries ``names`` (the
    artifacts it accepts — used for edge matching), a ``label`` (the chain's base name), and —
    for a OneOf port — its ``modality`` (B2: a multi-name port also accepts any registered
    same-modality artifact via an explicit edge, so the canvas may draw such a connection)."""
    from evaluator.pipeline.graph.registry import (
        ARTIFACT_CORPUS,
        OneOf,
        _DOCS_AXIS_CAPABLE,
    )
    from evaluator.pipeline.graph.wiring import _port_modality

    d = get_stage_node_def(stage)
    p = params or {}
    # docs-axis transforms consume the corpus, not the query chain (mirror _effective_inputs).
    if stage in _DOCS_AXIS_CAPABLE and p.get("axis") == "docs":
        required: Any = (ARTIFACT_CORPUS,)
    else:
        required = _resolve(d.inputs, params)

    ports: List[Dict[str, Any]] = []

    def add(art, optional):
        names = list(art) if isinstance(art, OneOf) else [art]
        port: Dict[str, Any] = {"label": names[-1], "names": names, "optional": optional}
        if len(names) > 1:  # type-open OneOf port (B2) — plain ports stay closed
            mod = _port_modality(tuple(names))
            if mod is not None:
                port["modality"] = mod.value
        ports.append(port)

    for art in required:
        add(art, False)
    for art in _resolve(d.optional_inputs, params):
        add(art, True)
    return ports


def _node_param_specs(node_def, params=None) -> list:
    """The builder's per-node form specs, two sources (§9 / §12.1 of the
    architecture doc): a minimal model section for registry-backed nodes (`model` +
    `device` — everything model-specific appears *after* a model is chosen, declared by
    the model author via the registry `Params` schema), plus the node's registered
    `param_spec` (genuinely node-scoped switches). `param_defaults` leftovers render as
    plain text fields.

    **Field-aware** (operator-abstraction): ``params`` are the node's currently-set
    discriminator fields, so the model section resolves to the right registry family (e.g.
    ``embed{modality:audio}`` → audio_embedding) and a callable ``param_spec`` resolves to
    the chosen op's switches (so ``transform``'s ``method`` carries the right choices)."""
    from evaluator.config.graph_config import _MODEL_NODE_FIELDS
    from evaluator.pipeline.introspection import operator_discriminators

    # A model node shows model (+ device): either a legacy name in the config-block map, or
    # an operator whose model_field (resolved for these fields) selects a registry model.
    model_field = _resolve(node_def.model_field, params)
    is_model_node = node_def.stage in _MODEL_NODE_FIELDS or model_field is not None
    # Discriminator fields re-resolve the field-aware form: when one changes the builder
    # re-fetches /api/graph/node-form (ports / model family / param switches all shift).
    discriminators = operator_discriminators(node_def.stage)
    specs: list = []
    if is_model_node or node_def.stage in _STAGE_FAMILY:
        specs.append({"key": "model", "kind": "model"})
        if is_model_node:
            specs.append({"key": "device", "kind": "device"})
    for key, meta in _resolve(node_def.param_spec, params).items():
        entry = {"key": key, "kind": meta.get("kind", "text")}
        if "choices" in meta:
            entry["choices"] = list(meta["choices"])
        if "default" in meta:
            entry["default"] = meta["default"]
        if "show_if" in meta:
            # {controlling_param: [values]} — builder shows the field only when
            # the controlling param's current value matches (e.g. gpu_id ⇐ store).
            entry["show_if"] = dict(meta["show_if"])
        if key in discriminators:
            entry["rerenders"] = True
        _attach_help(entry, meta.get("help"))
        specs.append(entry)
    seen = {s["key"] for s in specs}
    for key in node_def.param_defaults:
        if key not in seen:
            specs.append(_attach_help({"key": key, "kind": "text"}))
    return specs


def _attach_help(entry: Dict[str, Any], declared: str | None = None) -> Dict[str, Any]:
    """Set a field's ``help`` tooltip: an explicit ``param_spec`` ``help`` wins, else the shared
    ``FIELD_HELP`` glossary (keyed by the param name). No key → no tooltip. (Single help source,
    mirroring the config form — so the builder param panel explains the same fields.)"""
    help_text = declared or FIELD_HELP.get(entry["key"])
    if help_text:
        entry["help"] = help_text
    return entry


# User-facing palette sections (the builder groups by these — coarser than the 12 `domain`s),
# keyed on the node KIND so an operator's default kind + an alias preset both resolve. The
# structural plumbing (measure/sink, via `is_structural`) is filtered out before grouping.
_PALETTE_SECTION = {
    "dataset_source": "Data", "dataset_union": "Data",
    "asr": "Models", "tts": "Models", "text_embedding": "Models",
    "audio_embedding": "Models", "corpus_embedding": "Models",
    "answer_gen": "Models", "answer_judge": "Models",
    "vector_db": "Vector DB", "retrieval": "Vector DB", "multi_query_retrieval": "Vector DB",
    "query_correction": "RAG optimization", "query_optimization": "RAG optimization",
    "query_refine": "RAG optimization", "fusion": "RAG optimization",
    "result_fusion": "RAG optimization", "corpus_merge": "RAG optimization",
    "rerank": "RAG optimization", "mmr": "RAG optimization", "threshold": "RAG optimization",
    "augmenter": "Augmentation", "augment_audio": "Augmentation",
}
#: The fixed display order of the palette sections.
PALETTE_SECTIONS = ["Data", "Models", "Vector DB", "RAG optimization", "Augmentation"]


def palette_section(stage: str, params: Dict[str, Any] | None = None) -> "str | None":
    """The user-facing palette section for a node (by its resolved kind); None = ungrouped."""
    from evaluator.pipeline.graph.operators import node_kind

    return _PALETTE_SECTION.get(node_kind(stage, params or {}))


def node_catalogue() -> Dict[str, Any]:
    """The registered stage-node types + their I/O contract (E2): what the visual builder's
    palette offers and how ports connect. Each entry = ``{type, category, domain, model_field,
    family, inputs, outputs, optional_inputs, param_defaults, node_params}`` — ``category``
    (source/model/transform/metric/sink) + ``domain`` are the node's declared class (the
    palette groups by ``domain``, the DAG colors by ``category``); ports connect when an output
    artifact name matches a consumer's input name (the same rule the auto-wiring uses);
    ``family`` names the model registry the node's model picker reads (null = model-free);
    ``node_params`` are the node-centric YAML keys its param form offers."""
    from ..pipeline.graph import _NODE_REGISTRY
    from ..pipeline.introspection import is_structural

    nodes = []
    for stage, d in sorted(_NODE_REGISTRY.items()):
        # An operator's ports / category / model_field can be callable(params); the palette
        # shows the default-resolved contract (empty params → the registered defaults).
        model_field = _resolve(d.model_field, {})
        nodes.append(
            {
                "type": stage,
                "label": display_label(stage, None),
                "category": _resolve(d.category, {}),
                "domain": _resolve(d.domain, {}),
                # UI-graph ↔ execution-DAG split: structural plumbing is filtered from the palette;
                # the rest is grouped by `section` (the user's mental model).
                "structural": is_structural(stage, {}),
                "section": palette_section(stage, {}),
                "model_field": model_field,
                "family": (
                    _MODEL_FIELD_FAMILY.get(model_field or "")
                    or _STAGE_FAMILY.get(stage)
                ),
                "inputs": list(display_artifact_names(_resolve(d.inputs, {}))),
                "outputs": list(_resolve(d.outputs, {})),
                "optional_inputs": list(
                    display_artifact_names(_resolve(d.optional_inputs, {}))
                ),
                "input_ports": _input_ports(stage, {}),
                "param_defaults": dict(d.param_defaults),
                "node_params": _node_param_specs(d),
            }
        )
    return {"nodes": nodes, "presets": _alias_presets()}


def _alias_presets() -> List[Dict[str, Any]]:
    """Named-node 'presets' for the palette: every legacy alias whose fields differ from its
    operator's default resolution (so the bare operator tile doesn't already cover it) —
    e.g. Dataset union, TTS, the per-family metrics, MMR/Threshold, multi-query retrieval.
    Each carries the field-aware contract + the discriminator ``params`` to seed on drop, so
    the user drops a ready-to-wire node instead of hand-setting fields on a generic operator."""
    from evaluator.pipeline.graph.operators import ALIASES, node_kind
    from evaluator.pipeline.graph.registry import is_structural

    presets: List[Dict[str, Any]] = []
    for alias, (operator, fixed) in ALIASES.items():
        if node_kind(operator, {}) == alias:
            continue  # the operator's default resolution == this alias (already a bare tile)
        form = resolve_node_form(alias, None)
        presets.append({
            **form, "palette_id": alias, "preset": True, "params": dict(fixed),
            "structural": is_structural(alias, dict(fixed)),
            "section": palette_section(alias, dict(fixed)),
        })
    return presets


def _inject_dataset_fields(node_type: str, params: Dict[str, Any]) -> None:
    """A `dataset_source` advertises the CHOSEN dataset's *real* fields, not the union of what all
    datasets can output — so its output ports reflect what that dataset actually publishes (and the
    GT fields the user can wire). Resolve `params.fields` (column → artifact) from the descriptor
    when a dataset is set but none supplied; `_effective_outputs` then narrows the ports."""
    from evaluator.pipeline.graph.operators import node_kind

    if node_kind(node_type, params) != "dataset_source":
        return
    if params.get("fields") or not params.get("dataset"):
        return
    from evaluator.datasets.descriptor import get_descriptor

    desc = get_descriptor(str(params["dataset"]))
    if desc and getattr(desc, "fields", None):
        params["fields"] = dict(desc.fields)


def default_model_for(model_field: Any, config: Any = None) -> Any:
    """The model type a ``model_field`` path resolves to when the node picks none: the value on
    the given config (Config & Run — the loaded experiment's flat default), else the dataclass
    default (builder — no config context). ``model.X`` → ModelConfig, ``vector_db.X`` →
    VectorDBConfig; anything else (model-free node) → None."""
    if not model_field or "." not in str(model_field):
        return None
    section, attr = str(model_field).split(".", 1)
    if config is not None:
        holder = getattr(config, section, None)
        if holder is not None:
            return getattr(holder, attr, None)
    if section == "model":
        from evaluator.config.model import ModelConfig

        return getattr(ModelConfig(), attr, None)
    if section == "vector_db":
        from evaluator.config.vector_db import VectorDBConfig

        return getattr(VectorDBConfig(), attr, None)
    return None


def resolve_node_form(node_type: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Field-aware builder form for one operator instance (E2 follow-up): given the node's
    currently-set discriminator fields, resolve its ports + model family + the op-specific
    param switches. So picking ``transform.op = correct`` re-renders with the corrector
    methods (not optimize's rewrite/HyDE), and ``embed{modality:audio}`` shows the audio
    embedding registry. Powers the builder's ``/api/graph/node-form`` re-render."""
    from evaluator.pipeline.graph.operators import expand_alias
    from evaluator.pipeline.graph.registry import node_model_field

    # Accept an operator OR a legacy alias name (expand it to the operator + fields).
    node_type, params = expand_alias(node_type, dict(params or {}))
    _inject_dataset_fields(node_type, params)
    d = get_stage_node_def(node_type)
    model_field = node_model_field(node_type, params)
    family = _node_family(model_field, node_type, params)
    return {
        "type": node_type,
        "label": display_label(node_type, params),
        "category": _resolve(d.category, params),
        "domain": _resolve(d.domain, params),
        "model_field": model_field,
        "family": family,
        # The model this node runs when none is picked (the flat config default the executor
        # reads) — the form's empty option names it instead of a bare "(default)".
        "default_model": default_model_for(model_field),
        "inputs": list(_effective_inputs(node_type, params)),
        "outputs": list(_effective_outputs(node_type, params)),
        "optional_inputs": list(
            display_artifact_names(_resolve(d.optional_inputs, params))
        ),
        # collapsed canvas ports (OneOf → one port); inputs/optional_inputs above stay flat
        "input_ports": _input_ports(node_type, params),
        "node_params": _node_param_specs(d, params),
    }


def required_model_fields_for(template: str) -> List[str]:
    """The model config-field paths a graph *template* needs — derived from the template's nodes
    (each node's ``model_field``), so this is registry/graph-driven, not a per-mode spec lookup.
    Identical fields to the former ``resolve_graph_template(mode).required_model_fields`` for
    the four built-in templates, but works for any template name."""
    from evaluator.pipeline.graph.templates import graph_template

    try:
        graph = graph_template(template)
    except (ValueError, KeyError):
        return []
    fields: List[str] = []
    for node in graph.nodes:
        field = node_model_field(node.stage, node.params)
        if field and field not in fields:
            fields.append(field)
    return fields
