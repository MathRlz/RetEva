"""Registry/config-driven UI helpers for the WebAPI (form builder).

Builds UI options/forms from the registries and config (``create_config_options``,
``node_catalogue``, ``graph_preview``, ``_model_section``, ``_preset_form_context``).
The form → validated-config half lives in ``form_config.py``; its public names are
re-exported here for back-compat (routers/tests import them from this module).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from evaluator import EvaluationConfig, list_presets
from evaluator.config.model_fields import (
    MODEL_FAMILY_FIELDS,
    MODEL_FIELD_FAMILY as _MODEL_FIELD_FAMILY,
)
from evaluator.config.types import (
    DatasetType,
    VectorDBType,
    RETRIEVAL_MODES,
    RERANKER_MODES,
    SERVICE_STARTUP_MODES,
    SERVICE_OFFLOAD_POLICIES,
    DATASET_SOURCES,
)
from evaluator.models.retrieval import list_fusions
from evaluator.datasets import (
    list_known_dataset_names,
    resolve_dataset_profile,
)
from evaluator.pipeline import build_graph_for_config
# Public introspection surface (the supported boundary — not registry/stage_graph privates).
from evaluator.pipeline.introspection import (
    dataset_columns,
    display_artifact_names,
    display_label,
    get_stage_node_def,
    node_category,
    node_domain,
    node_model_field,
    resolve_field as _resolve,
    effective_inputs as _effective_inputs,
    effective_outputs as _effective_outputs,
)
from evaluator.services import ModelServiceProvider
from evaluator.webapi.field_help import FIELD_HELP
from evaluator.webapi.utils import with_provider

# Form → validated-config half (moved to form_config.py); re-exported for back-compat.
from evaluator.webapi.form_config import (  # noqa: F401  (re-export surface)
    _coerce_param,
    _collect_model_params,
    _deep_merge,
    _form_to_config,
    _prepared_config_or_error,
    graph_spec_to_config_dict,
    load_config,
    prepare_run_config,
)


# One deep-merge implementation: the immutable `_deep_merge` (form_config). Kept under the
# historical name for the routers that import `deep_merge_dict`.
deep_merge_dict = _deep_merge


def nested_config(config: EvaluationConfig) -> Dict[str, Any]:
    """Return nested config shape for WebUI config creator."""
    return {
        "experiment_name": config.experiment_name,
        "output_dir": config.output_dir,
        "runtime": config.to_runtime_dict(),
        "experiment": config.to_experiment_dict(),
    }


def create_config_options(
    provider_factory: Callable[[], ModelServiceProvider],
) -> Dict[str, Any]:
    """Build form options for config creator UI."""
    raw_models = with_provider(provider_factory, lambda p: p.list_available_models())

    def _normalize_model_entries(entries: Any) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        if not isinstance(entries, list):
            return normalized
        for entry in entries:
            if isinstance(entry, dict):
                entry_type = str(entry.get("type", "")).strip()
                entry_name = str(entry.get("name", "")).strip() or entry_type
                if entry_type:
                    normalized.append({"type": entry_type, "name": entry_name})
                continue
            if isinstance(entry, str):
                value = entry.strip()
                if value:
                    normalized.append({"type": value, "name": value})
        return normalized

    normalized_models: Dict[str, List[Dict[str, str]]] = {}
    if isinstance(raw_models, dict):
        for family, entries in raw_models.items():
            normalized_models[str(family)] = _normalize_model_entries(entries)
    for required_family in (
        "asr",
        "text_embedding",
        "audio_embedding",
        "tts",
        "reranker",
    ):
        normalized_models.setdefault(required_family, [])

    from evaluator.pipeline.graph.templates import GRAPH_TEMPLATES, list_graph_templates

    defaults = EvaluationConfig()
    return {
        "presets": list_presets(),
        # The four former pipeline modes are now graph templates (config-creation skeletons).
        # Key kept as ``pipeline_modes`` for the existing UI; values are the template names.
        "pipeline_modes": list(GRAPH_TEMPLATES),
        "graph_templates": list_graph_templates(),
        "dataset_types": [dataset_type.value for dataset_type in DatasetType],
        "dataset_sources": list(DATASET_SOURCES),
        "dataset_names": list_known_dataset_names(),
        "vector_db_types": [db.value for db in VectorDBType],
        "retrieval_modes": list(RETRIEVAL_MODES),
        "hybrid_fusion_methods": list_fusions(),
        "reranker_modes": list(RERANKER_MODES),
        "service_runtime": {
            "startup_mode": list(SERVICE_STARTUP_MODES),
            "offload_policy": list(SERVICE_OFFLOAD_POLICIES),
        },
        "tts_providers": sorted(
            {entry["type"] for entry in normalized_models.get("tts", [])}
        ),
        "field_help": dict(FIELD_HELP),
        "models": normalized_models,
        "defaults": nested_config(defaults),
    }


def _node_inspect(config: EvaluationConfig, node: Any) -> Dict[str, Any]:
    """What the preview's click-to-inspect panel shows for a node: the configured model
    (resolved from ``config.model`` via the node→config mapping in
    ``graph_config._MODEL_NODE_FIELDS`` — single source, not duplicated), retrieval
    settings, plus any explicit per-node ``params`` (branch/graph overrides) on top.
    Empty values are dropped."""
    from evaluator.config.graph_config import _MODEL_NODE_FIELDS
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
            info["reranker"] = config.vector_db.reranker_mode or True
    # The operator's discriminator fields (axis/modality/op/family/…) are internal selectors
    # already encoded in the node's label/stage — they'd be noise in the inspect panel.
    hidden = operator_discriminators(node.stage) | {"fields"}
    for key, val in (node.params or {}).items():
        # `fields` (the dataset column schema) renders as the dedicated columns
        # row in the inspect panel, not as a raw params JSON blob.
        if key not in hidden and val not in (None, "", {}):
            info[key] = val
    return info


def _preview_node(config: EvaluationConfig, node: Any) -> Dict[str, Any]:
    """The rich preview node card (label/category/domain/ports/bindings/inspect) the read-only
    DAG view + the builder seed render from."""
    return {
        "id": node.id,
        "stage": node.stage,
        # friendly operator label for the node card (display.py)
        "label": display_label(node.stage, node.params),
        # declared class — the DAG colors nodes by `category`, groups by `domain`
        # (resolve callable operator category/domain for this instance's fields)
        "category": node_category(node.stage, node.params),
        "domain": node_domain(node.stage, node.params),
        "depends_on": list(node.depends_on),
        # resolved data bindings (artifact → producer) — the preview draws
        # these as edges, same shape as /api/graph/template (B2)
        "bindings": [list(b) for b in node.bindings],
        "inputs": list(_effective_inputs(node.stage, node.params)),
        # per-instance outputs: dataset_source honors role + column schema
        "outputs": list(_effective_outputs(node.stage, node.params)),
        "optional_inputs": list(
            display_artifact_names(
                _resolve(get_stage_node_def(node.stage).optional_inputs, node.params)
            )
        ),
        # field-aware form contract (model family + the op-specific param switches)
        # so the builder seed renders THIS instance, not the operator's default tile
        # (corpus_embedding ≠ text_embedding on the canvas).
        "model_field": node_model_field(node.stage, node.params),
        "family": _node_family(
            node_model_field(node.stage, node.params), node.stage, node.params
        ),
        "input_ports": _input_ports(node.stage, node.params),
        "node_params": _node_param_specs(get_stage_node_def(node.stage), node.params),
        # declared column schema [{name, artifact, type}] — what the diagram
        # shows on dataset nodes so an experiment reads from its DAG
        "columns": dataset_columns(node.params),
        "inspect": _node_inspect(config, node),
    }


def graph_preview(config: EvaluationConfig) -> Dict[str, Any]:
    profile = resolve_dataset_profile(
        config.data.dataset_name, config.data.dataset_type
    )
    graph = build_graph_for_config(config)
    payload = graph_render_payload(graph, lambda node: _preview_node(config, node))
    payload["dataset_profile"] = {
        "name": profile.name,
        "dataset_type": str(profile.dataset_type),
        "requires_audio": profile.requires_audio,
        "requires_text": profile.requires_text,
        "supports_generation": profile.supports_generation,
        "evaluation_mode": profile.evaluation_mode,
        "recommended_pipeline_modes": list(profile.recommended_pipeline_modes),
        "pipeline_mode_supported": profile.supports_pipeline_mode(
            str(config.graph_template)
        ),
    }
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


def minimal_edges(nodes: list, edges: list) -> list:
    """Drop edges the artifact auto-wiring would recreate anyway, keeping only genuinely-extra
    ordering deps. ``build_graph_from_spec`` auto-wires from artifacts and treats ``edges`` as
    *additional* deps, so exporting every rendered binding as an explicit edge makes them
    accumulate across round-trips. Returning only the non-auto-wired edges makes the export
    stable (and minimal). ``edges`` is the ``[{from,to}]`` form."""
    from evaluator.pipeline import build_graph_from_spec

    try:
        auto = build_graph_from_spec(nodes, mode="custom")
    except Exception:  # noqa: BLE001 — if it can't auto-wire, keep edges as-is (caller validates)
        return list(edges)
    auto_deps = {(dep, n.id) for n in auto.nodes for dep in n.depends_on}
    return [
        e for e in edges
        if isinstance(e, dict) and (e.get("from"), e.get("to")) not in auto_deps
    ]


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
    if branches:
        base_config = copy.deepcopy(config)
        base_config.graph_override = {
            k: v for k, v in override.items() if k != "branches"
        }
    graph = build_graph_for_config(base_config)
    dataset_params = _dataset_source_params(config)

    def _render(node):
        params = dict(node.params or {})
        # Fold the resolved model/retrieval settings (config.model / vector_db) onto the node;
        # setdefault so an explicit per-node graph-override param still wins.
        for key, value in _node_inspect(config, node).items():
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


def render_node(node: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """One node in a canvas render payload: id + type + params + resolved bindings + the
    field-aware form contract (ports/label/family/switches) — the shape ``DagView.drawGraph``
    consumes, identical to ``/api/graph/template``."""
    return {
        "id": node.id,
        "type": node.stage,
        "params": params,
        "bindings": [list(b) for b in node.bindings],
        **resolve_node_form(node.stage, dict(node.params or {})),
    }


def spec_to_render_payload(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build a canvas render payload (``{mode, levels, nodes}``) from a saved builder spec
    (``{mode, nodes:[{id,type,params}], edges}``) — so a stored graph re-renders onto the
    canvas. The DAG is rebuilt (``build_graph_from_spec``) to resolve bindings + levels, but
    each node keeps the *saved* params (models/dataset the user set)."""
    from evaluator.pipeline import build_graph_from_spec

    spec_nodes = [n for n in (spec.get("nodes") or []) if isinstance(n, dict)]
    edges: Dict[str, list] = {}
    for edge in spec.get("edges") or []:
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

    return (
        _MODEL_FIELD_FAMILY.get(model_field or "")
        or _STAGE_FAMILY.get(node_kind(node_type, params))
        or _STAGE_FAMILY.get(node_type)
    )


def _input_ports(stage: str, params=None) -> List[Dict[str, Any]]:
    """Canvas input ports for a node instance: ONE port per declared input *slot*. A ``OneOf``
    group (the query-text / query-vector chains the auto-wiring chooses among) collapses to a
    single port that accepts any of its alternatives — so an embedder shows one ``query_text``
    port, not five, and ``search`` shows ``query_vectors`` + ``vector_index``, not eleven ports.

    Required ports first, then optional (``optional: true``). Each port carries ``names`` (the
    artifacts it accepts — used for edge matching) and a ``label`` (the chain's base name)."""
    from evaluator.pipeline.graph.registry import (
        ARTIFACT_CORPUS,
        OneOf,
        _DOCS_AXIS_CAPABLE,
    )

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
        ports.append({"label": names[-1], "names": names, "optional": optional})

    for art in required:
        add(art, False)
    for art in _resolve(d.optional_inputs, params):
        add(art, True)
    return ports


def _node_param_specs(node_def, params=None) -> list:
    """The builder's per-node form specs, two sources (BUILDER_UX / §9 of the
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


def node_catalogue() -> Dict[str, Any]:
    """The registered stage-node types + their I/O contract (E2): what the visual builder's
    palette offers and how ports connect. Each entry = ``{type, category, domain, model_field,
    family, inputs, outputs, optional_inputs, param_defaults, node_params}`` — ``category``
    (source/model/transform/metric/sink) + ``domain`` are the node's declared class (the
    palette groups by ``domain``, the DAG colors by ``category``); ports connect when an output
    artifact name matches a consumer's input name (the same rule the auto-wiring uses);
    ``family`` names the model registry the node's model picker reads (null = model-free);
    ``node_params`` are the node-centric YAML keys its param form offers."""
    from ..pipeline.stage_graph import _NODE_REGISTRY

    from ..pipeline.graph.registry import _resolve

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

    presets: List[Dict[str, Any]] = []
    for alias, (operator, fixed) in ALIASES.items():
        if node_kind(operator, {}) == alias:
            continue  # the operator's default resolution == this alias (already a bare tile)
        form = resolve_node_form(alias, None)
        presets.append({**form, "palette_id": alias, "preset": True, "params": dict(fixed)})
    return presets


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
        "inputs": list(_effective_inputs(node_type, params)),
        "outputs": list(_effective_outputs(node_type, params)),
        "optional_inputs": list(
            display_artifact_names(_resolve(d.optional_inputs, params))
        ),
        # collapsed canvas ports (OneOf → one port); inputs/optional_inputs above stay flat
        "input_ports": _input_ports(node_type, params),
        "node_params": _node_param_specs(d, params),
    }


# --- Form building (formerly ui_helpers) ---

# Pipeline model_field path -> (family, type form-field, size form-field). Derived from
# the single source of truth (config.model_fields) so adding a family is a one-line edit.
_MODEL_FIELDS = {
    fam.model_field_path: (fam.registry_family, fam.type_field, fam.size_field)
    for fam in MODEL_FAMILY_FIELDS
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


def _model_section(
    provider_factory, template: str, current: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Build the registry-driven model-select descriptors for a graph template."""
    models = with_provider(provider_factory, lambda p: p.list_available_models())
    from evaluator.models.registry import FAMILY_REGISTRIES

    sections: List[Dict[str, Any]] = []
    for field in required_model_fields_for(template):
        spec = _MODEL_FIELDS.get(field)
        if not spec:
            continue
        family, type_field, size_field = spec
        reg = FAMILY_REGISTRIES.get(family)
        selected_type = current.get(type_field, "")
        sizes = list(reg.get_sizes(selected_type).keys()) if (reg and selected_type) else []
        prefix = type_field[: -len("_model_type")]  # asr / text_emb / audio_emb
        sections.append({
            "family": family,
            "type_field": type_field,
            "size_field": size_field,
            "options": [{"type": e["type"], "name": e["name"]} for e in models.get(family, [])],
            "selected_type": selected_type,
            "sizes": sizes,
            "selected_size": current.get(size_field, ""),
            "params": _model_param_fields(reg, selected_type, prefix, current),
            "extra_field": f"param__{prefix}__extra",
        })
    return sections


def _model_param_fields(reg, model_type: str, prefix: str, current: Dict[str, str]) -> List[Dict[str, Any]]:
    """Registry-declared optional args (Params dataclass) as renderable form fields.

    ``size`` is excluded (it has its own select). Each field carries the registry default
    so the UI shows it; the user can override or leave blank to keep the default.
    """
    if not (reg and model_type):
        return []
    import dataclasses
    fields: List[Dict[str, Any]] = []
    for name, info in reg.get_params_schema(model_type).items():
        if name == "size":
            continue
        default = info.get("default")
        if default is dataclasses.MISSING:
            default = ""
        field = f"param__{prefix}__{name}"
        fields.append({
            "name": name,
            "field": field,
            "value": current.get(field, ""),
            "default": "" if default is None else default,
            "choices": info.get("choices") or [],
            "is_bool": isinstance(default, bool),
        })
    return fields


def _preset_form_context(
    provider_factory, name: str = "", form: Dict[str, str] | None = None
) -> Dict[str, Any]:
    """Build the config-form template context, prefilled from a preset.

    Shared by ``/ui/config`` (default preset, Run-ready on load) and ``/ui/preset``
    (htmx swap when the user picks a preset). Returns the kwargs both templates expect:
    ``options``, ``mode``, ``preset`` (flat form values), ``model_sections``.

    When ``form`` (the current form values) is given, the preset fills gaps but does
    **not** clobber fields the user already entered that the preset doesn't define
    (e.g. dataset/paths — presets carry no dataset). Preset values win where present.
    """
    from evaluator.config.model_presets import get_preset

    cfg: Dict[str, Any] = {}
    if name:
        try:
            cfg = get_preset(name, auto_devices=False)
        except Exception:
            cfg = {}
    m = cfg.get("model", {})
    data = cfg.get("data", {})
    vdb = cfg.get("vector_db", {})
    asx = cfg.get("audio_synthesis", {})
    mode = m.get("pipeline_mode", "asr_text_retrieval")

    def fv(key: str) -> str:
        """Current user-entered form value for ``key`` (empty if none)."""
        return ((form or {}).get(key) or "").strip()

    def pick(preset_value, key, default):
        """Preset value if it set one, else the user's form value, else default."""
        if preset_value not in (None, ""):
            return preset_value
        return fv(key) or default

    # Models: preset wins; fall back to the user's current selection per field.
    current = {
        "asr_model_type": m.get("asr_model_type") or fv("asr_model_type"),
        "asr_size": m.get("asr_size") or fv("asr_size"),
        "text_emb_model_type": m.get("text_emb_model_type") or fv("text_emb_model_type"),
        "text_emb_size": m.get("text_emb_size") or fv("text_emb_size"),
        "audio_emb_model_type": m.get("audio_emb_model_type") or fv("audio_emb_model_type"),
        "audio_emb_size": m.get("audio_emb_size") or fv("audio_emb_size"),
    }
    return {
        "options": create_config_options(provider_factory),
        "mode": mode,
        "preset": {
            "experiment_name": pick(cfg.get("experiment_name"), "experiment_name", "webui_experiment"),
            "output_dir": pick(cfg.get("output_dir"), "output_dir", "evaluation_results/webui"),
            # Presets never carry a dataset — always keep what the user picked.
            "dataset_name": pick(data.get("dataset_name"), "dataset_name", ""),
            "questions_path": pick(data.get("questions_path"), "questions_path", ""),
            "corpus_path": pick(data.get("corpus_path"), "corpus_path", ""),
            "batch_size": pick(data.get("batch_size"), "batch_size", 8),
            "trace_limit": pick(data.get("trace_limit"), "trace_limit", 0),
            "vector_db_type": pick(vdb.get("type"), "vector_db_type", "inmemory"),
            "k": pick(vdb.get("k"), "k", 10),
            "retrieval_mode": pick(vdb.get("retrieval_mode"), "retrieval_mode", "dense"),
            # TTS audio synthesis — prefill so the preset's setting isn't dropped.
            "audio_synthesis_enabled": "on" if asx.get("enabled") else fv("audio_synthesis_enabled"),
            "tts_provider": pick(asx.get("provider"), "tts_provider", "mms"),
            "tts_language": pick(asx.get("language"), "tts_language", "en"),
            "tts_voice": pick(asx.get("voice"), "tts_voice", ""),
        },
        "model_sections": _model_section(provider_factory, mode, current),
    }
