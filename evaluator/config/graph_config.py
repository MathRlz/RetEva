"""Node-centric YAML config → ``EvaluationConfig``.

Translates the node-centric YAML into the legacy config dict and reuses
``EvaluationConfig.from_dict`` for construction and validation, so there is one runtime
config object regardless of which YAML surface produced it.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ..errors import ConfigurationError
from .dataset_descriptors import get_descriptor, list_registered_datasets


class GraphConfigError(ConfigurationError):
    """Raised when a node-centric config cannot be translated."""


# nodes.<name>.<key> → ModelConfig field. A top-level nodes.<role> block sets the flat
# model.* default; a node's own inline params stay ON the node as a per-node override
# (resolved via resolved_model_config). Also used to invert legacy_yaml_to_graph_yaml.
MODEL_NODE_FIELDS = {
    "asr": {
        "model": "asr_model_type",
        "size": "asr_size",
        "name": "asr_model_name",
        "device": "asr_device",
        "adapter": "asr_adapter_path",
        "params": "asr_params",
        "quantization": "asr_quantization",
    },
    "text_embedding": {
        "model": "text_emb_model_type",
        "size": "text_emb_size",
        "name": "text_emb_model_name",
        "device": "text_emb_device",
        "adapter": "text_emb_adapter_path",
        "model_path": "text_emb_model_path",
        "embedding_space": "text_emb_embedding_space",
        "params": "text_emb_params",
        "quantization": "text_emb_quantization",
    },
    "audio_embedding": {
        "model": "audio_emb_model_type",
        "size": "audio_emb_size",
        "encoder": "audio_emb_encoder_type",
        "encoder-size": "audio_emb_encoder_size",
        "name": "audio_emb_model_name",
        "device": "audio_emb_device",
        "model_path": "audio_emb_model_path",
        "dim": "audio_emb_dim",
        "embedding_space": "audio_emb_embedding_space",
        "params": "audio_emb_params",
        "quantization": "audio_emb_quantization",
    },
}

# A feature node's params fold into its capability sub-config as the flat DEFAULT but also
# stay ON the node (mirrors MODEL_NODE_FIELDS) since resolve_node_config reads a node's own
# params at run time to keep same-kind nodes (two tts/answer_gen/... in one graph) distinct.
FEATURE_NODE_CONFIG = {
    "tts": "audio_synthesis",
    "answer_judge": "judge",
    "answer_gen": "answer_generation",
    "query_correction": "query_correction",
    "query_optimization": "query_optimization",
    "multi_query_retrieval": "query_optimization",
    "fusion": "embedding_fusion",  # embedding-level only; result_fusion is hybrid rank-fusion (vdb)
}

# dataset.<key> → DataConfig field.
DATASET_FIELDS = {
    "id": "dataset_name",
    "questions": "questions_path",
    "corpus": "corpus_path",
    "prepared_dir": "prepared_dataset_dir",
    "path": "data_path",
    "audio_dir": "audio_dir",
    "transcripts": "transcripts_file",
    "hf_dataset": "huggingface_dataset",
    "hf_subset": "huggingface_subset",
    "hf_split": "huggingface_split",
    "split": "split",
    "batch_size": "batch_size",
    "trace_limit": "trace_limit",
    "type": "dataset_type",
    "source": "dataset_source",
    "max_samples": "max_samples",
    "language": "default_language",
}

# DataConfig field → node-centric YAML key (inverse of DATASET_FIELDS).
DATA_FIELD_TO_KEY = {v: k for k, v in DATASET_FIELDS.items()}

# nodes.retrieval scalar keys → VectorDBConfig field.
_RETRIEVAL_SCALARS = {
    "k": "k",
    "mode": "retrieval_mode",
    "distance": "distance_metric",
    "gpu_id": "gpu_id",
}

# nodes.vector_db keys → VectorDBConfig field(s), the canonical home for the store backend
# choice. `collection` fans out to both backends' fields.
_VECTOR_DB_NODE_KEYS = {
    "store": ("type",),
    "gpu_id": ("gpu_id",),
    "path": ("chromadb_path",),
    "url": ("qdrant_url",),
    "collection": ("chromadb_collection_name", "qdrant_collection_name"),
}


def _vector_db_node_to_config(node: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a nodes.vector_db block into vector_db fields (typo-protected)."""
    vdb: Dict[str, Any] = {}
    for key, value in node.items():
        fields = _VECTOR_DB_NODE_KEYS.get(key)
        if fields is None:
            raise GraphConfigError(
                f"Unknown key '{key}' under nodes.vector_db. "
                f"Allowed: {sorted(_VECTOR_DB_NODE_KEYS)}"
            )
        for f in fields:
            vdb[f] = value
    return vdb


def _map_keys(
    src: Dict[str, Any], mapping: Dict[str, str], where: str
) -> Dict[str, Any]:
    """Rename keys via ``mapping``; unknown keys are an error (typo protection)."""
    out: Dict[str, Any] = {}
    for key, value in src.items():
        if key not in mapping:
            raise GraphConfigError(
                f"Unknown key '{key}' under {where}. Allowed: {sorted(mapping)}"
            )
        out[mapping[key]] = value
    return out


# nested retrieval groups → vector_db field names. Explicit (not a passthrough) so a typo in a
# nested key (e.g. retrieval.reranker.enabld) is a hard error, not a silently-dropped vdb key.
_FUSION_KEYS = {
    "method": "hybrid_fusion_method",
    "dense_weight": "hybrid_dense_weight",
    "rrf_k": "rrf_k",
}
_RERANKER_KEYS = {
    k: f"reranker_{k}" for k in ("enabled", "mode", "top_k", "weight", "model", "device")
}
_MMR_KEYS = {"enabled": "use_mmr", "lambda": "mmr_lambda"}


def _retrieval_to_vector_db(node: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a nodes.retrieval block into vector_db fields."""
    vdb: Dict[str, Any] = {}

    def _nested(value: Dict[str, Any], mapping: Dict[str, str], group: str) -> None:
        for k, v in value.items():
            if k not in mapping:
                raise GraphConfigError(
                    f"Unknown key 'retrieval.{group}.{k}'. Allowed: {sorted(mapping)}"
                )
            vdb[mapping[k]] = v

    for key, value in node.items():
        if key in _RETRIEVAL_SCALARS:
            vdb[_RETRIEVAL_SCALARS[key]] = value
        elif key == "fusion" and isinstance(value, dict):
            _nested(value, _FUSION_KEYS, "fusion")
        elif key == "reranker" and isinstance(value, dict):
            _nested(value, _RERANKER_KEYS, "reranker")
        elif key == "mmr" and isinstance(value, dict):
            _nested(value, _MMR_KEYS, "mmr")
        else:
            raise GraphConfigError(f"Unknown key 'retrieval.{key}'")
    return vdb


def _translate_experiment(new: Dict[str, Any], legacy: Dict[str, Any]) -> None:
    """experiment → experiment_name / output_dir (passthrough scalars already understood
    by from_dict)."""
    exp = new.pop("experiment", None) or {}
    if "name" in exp:
        legacy["experiment_name"] = exp.pop("name")
    if "output_dir" in exp:
        legacy["output_dir"] = exp.pop("output_dir")
    legacy.update(exp)


def _translate_runtime(new: Dict[str, Any], legacy: Dict[str, Any]) -> None:
    """runtime → service_runtime rename (cache/tracking/... pass straight through)."""
    runtime = new.pop("runtime", None) or {}
    if "service" in runtime:
        runtime["service_runtime"] = runtime.pop("service")
    legacy.update(runtime)


def _translate_dataset(new: Dict[str, Any], legacy: Dict[str, Any]) -> None:
    """dataset → data (single-dataset block)."""
    dataset = new.pop("dataset", None) or {}
    if dataset:
        legacy["data"] = _map_keys(dataset, DATASET_FIELDS, "dataset")


def _translate_datasets_map(new: Dict[str, Any], legacy: Dict[str, Any]):
    """datasets: map → data.datasets; each entry's `role` carries into its
    dataset_source node (returns role_by_dataset_id for the graph translator's wiring)."""
    role_by_dataset_id: Dict[str, str] = {}
    datasets = new.pop("datasets", None)
    if datasets is not None:
        if not isinstance(datasets, dict):
            raise GraphConfigError("datasets: must be a map of id -> source spec.")
        translated: Dict[str, Any] = {}
        for sid, entry in datasets.items():
            entry = dict(entry or {})
            role = entry.pop("role", None)
            mapped = _map_keys(entry, DATASET_FIELDS, f"datasets.{sid}")
            if role is not None:
                mapped["role"] = role
                role_by_dataset_id[str(sid)] = str(role)
            translated[str(sid)] = mapped
        legacy.setdefault("data", {})["datasets"] = translated
    return role_by_dataset_id


def _validate_graph_block(graph: Dict[str, Any]) -> None:
    """Reject graph.mode (hard cut-over) and unknown graph keys."""
    if "mode" in graph:
        raise GraphConfigError(
            "graph.mode is no longer supported — a config is an explicit graph. Replace "
            "`graph: {mode: <m>}` with `graph: {nodes: [...]}` (see any configs/*.yaml). "
            "Every capability is a node; the run label is derived from the graph."
        )
    # Reject unknown graph keys — a typo would otherwise be silently dropped (quietly
    # wrong experiment). No graph.branches: a config lists every node/edge explicitly;
    # N variants of a node are just N distinctly-named entries in graph.nodes, wired via
    # ordinary edges.
    _GRAPH_KEYS = {"nodes", "edges"}
    _unknown_graph = set(graph) - _GRAPH_KEYS
    if _unknown_graph:
        raise GraphConfigError(
            f"Unknown key(s) under graph: {sorted(_unknown_graph)}. "
            f"Allowed: {sorted(_GRAPH_KEYS)}."
        )


def _resolve_dataset_refs(
    node_ids: list, legacy: Dict[str, Any], role_by_dataset_id: Dict[str, str]
) -> None:
    """dataset_source nodes inherit the datasets-map role into params, or synthesize a
    picker entry via ``_synthesize_dataset_entry`` when the id isn't in the map; an
    explicit node role wins."""
    from ..pipeline.graph.operators import node_kind

    for item in node_ids:
        if not (isinstance(item, dict)
                and node_kind(item.get("type"), item.get("params") or {})
                == "dataset_source"):
            continue
        params = item.get("params") or {}
        ds_id = params.get("dataset")
        if ds_id is None:
            continue
        map_entries = legacy.get("data", {}).get("datasets") or {}
        in_map = str(ds_id) in role_by_dataset_id or str(ds_id) in map_entries
        if not in_map:
            # Not a datasets-map id: must be a registered dataset id; synthesize a
            # per-node entry so the existing multi-source loader runs it unchanged.
            _synthesize_dataset_entry(item, params, str(ds_id), legacy)
            continue
        if "role" not in params and str(ds_id) in role_by_dataset_id:
            params["role"] = role_by_dataset_id[str(ds_id)]
            item["params"] = params


def _fold_node_params(node_ids: list, legacy: Dict[str, Any]) -> None:
    """Fold config-bearing node params into vector_db / feature sub-configs (self-contained
    list form); a model/vector_db/feature node's own params also stay ON the node as a
    per-node override."""
    from ..pipeline.graph.operators import node_kind, operator_discriminators

    graph_vdb: Dict[str, Any] = {}
    for item in node_ids:
        if isinstance(item, dict):
            ntype, nparams = item.get("type"), item.get("params") or {}
        else:
            ntype, nparams = item, {}
        kind = node_kind(ntype, nparams)
        # A feature node's mere presence enables its capability (graph is the spec);
        # dict params fold below add the rest.
        _cap = FEATURE_NODE_CONFIG.get(kind)
        if _cap:
            legacy[_cap] = {"enabled": True, **(legacy.get(_cap) or {})}
        if not isinstance(item, dict) or not nparams:
            continue
        # Discriminator selectors (op/axis/modality/...) identify the kind, not config —
        # split them out so the fold doesn't choke on them and node_kind still resolves
        # them at build time.
        disc = operator_discriminators(ntype)
        selectors = {k: v for k, v in nparams.items() if k in disc}
        fields = {k: v for k, v in nparams.items() if k not in disc}
        if kind in MODEL_NODE_FIELDS:
            # Node's own model params stay ON the node as a per-node override
            # (resolved_model_config) so two same-role nodes can run distinct models;
            # the flat model.* default only comes from a top-level nodes.<role> block.
            item["params"] = {**selectors, **fields}
        elif kind == "vector_db":
            # Keep node's own vdb params ON the node so two vector_db nodes can use
            # distinct backends; also fold into the shared flat default for
            # node-context-free call sites.
            if fields:
                graph_vdb.update(_vector_db_node_to_config(fields))
            item["params"] = {**selectors, **fields}
        elif kind == "retrieval":
            # mode/vectors are per-arm functional params read off the node directly;
            # only the tuning keys (k/distance/gpu_id/fusion/reranker/mmr) fold into the
            # shared vector_db config.
            keep = {k: v for k, v in fields.items() if k in ("mode", "vectors")}
            fold = {k: v for k, v in fields.items() if k not in ("mode", "vectors")}
            if fold:
                graph_vdb.update(_retrieval_to_vector_db(fold))
            item["params"] = {**selectors, **keep}
        elif kind in FEATURE_NODE_CONFIG:
            # Feature node's params fold into the sub-config as the flat default, but
            # node params WIN over the presence-derived enabled:True (an explicit
            # enabled:false on the node holds); also kept ON the node so two same-kind
            # feature nodes stay distinct at run time.
            cap = FEATURE_NODE_CONFIG[kind]
            legacy[cap] = {**(legacy.get(cap) or {}), **fields}
            item["params"] = {**selectors, **fields}
    if graph_vdb:
        legacy["vector_db"] = {**legacy.get("vector_db", {}), **graph_vdb}


def _validate_edges(graph: Dict[str, Any], node_ids: list) -> list:
    """Schema-validates graph.edges (port-level {from,to,output,input} bindings, or
    portless {from,to} ordering-only) so a typo'd edge fails at load, not deep in the
    wiring."""
    from ..pipeline.graph.wiring import is_port_edge, normalize_port_edge

    edge_list = []
    for edge in graph.get("edges", []) or []:
        if not (isinstance(edge, dict) and "from" in edge and "to" in edge):
            raise GraphConfigError("each graph.edges item needs 'from' and 'to'.")
        unknown = set(edge) - {"from", "to", "output", "input"}
        if unknown:
            raise GraphConfigError(
                f"Unknown key(s) {sorted(unknown)} on a graph.edges item. "
                "Allowed: from, to, output, input."
            )
        # {from, to, input: X} is same-name shorthand, normalized here so downstream
        # sees one shape.
        if is_port_edge(edge):
            try:
                edge = normalize_port_edge(dict(edge))
            except ValueError as exc:
                raise GraphConfigError(str(exc)) from exc
        edge_list.append(dict(edge))
    # A config's graph must be explicit: nodes AND port-level edges; the auto-wirer
    # remains only the authoring assistant (builder / --emit-edges).
    if node_ids and not any(is_port_edge(e) for e in edge_list):
        raise GraphConfigError(
            "graph.nodes requires explicit port-level graph.edges "
            "({from, to, input} entries, plus 'output' when it differs). Generate them with: "
            "evaluator graph --config <yaml> --emit-edges, or connect the ports in "
            "the builder."
        )
    return edge_list


def _translate_graph(
    new: Dict[str, Any],
    legacy: Dict[str, Any],
    role_by_dataset_id: Dict[str, str],
) -> None:
    """graph → explicit node/edge override; graph.branches is unsupported (see
    _validate_graph_block)."""
    graph = new.pop("graph", None) or {}
    _validate_graph_block(graph)
    from ..pipeline.graph.build import complete_structural_plumbing

    complete_structural_plumbing(graph)
    if "nodes" in graph:
        node_ids = graph["nodes"]
        # Each item is a node-type string, or a {id, type, params} dict for a distinct
        # instance (two rerankers with different models). `type` may be the builder
        # canvas's *stage* or the YAML's concrete *kind*; node_kind() collapses both.
        if not isinstance(node_ids, list) or not all(
            isinstance(n, (str, dict)) for n in node_ids
        ):
            raise GraphConfigError(
                "graph.nodes must be a list of node-type strings or {id, type, params} dicts."
            )
        _resolve_dataset_refs(node_ids, legacy, role_by_dataset_id)
        _fold_node_params(node_ids, legacy)
        legacy["graph_override"] = {"nodes": node_ids, "edges": _validate_edges(graph, node_ids)}
    elif "edges" in graph:
        raise GraphConfigError("graph.edges requires graph.nodes.")
    else:
        # Fail here, where the config name is still in hand, rather than deep inside
        # create_pipeline_from_config with no idea which file is missing its graph.
        raise GraphConfigError("graph.nodes is required")


# dataset_source params that are NOT data settings (kept on the node). fields/extra_outputs/
# suppress_outputs are transient canvas-preview schema — never persisted back to run config.
_DATASET_NODE_CONTROL_PARAMS = {"dataset", "role", "fields", "extra_outputs", "suppress_outputs"}


def _synthesize_dataset_entry(
    item: Dict[str, Any],
    params: Dict[str, Any],
    dataset_id: str,
    legacy: Dict[str, Any],
) -> None:
    """Builds data.datasets[<node_id>] = {dataset_name: id, **node settings}, rewrites
    params.dataset to the node id, and validates the descriptor's required_data_fields
    are satisfied."""

    node_id = str(item.get("id") or "dataset_source")
    descriptor = get_descriptor(dataset_id)
    if descriptor is None:
        known = ", ".join(list_registered_datasets())
        raise GraphConfigError(
            f"dataset_source '{node_id}' references unknown dataset '{dataset_id}' "
            f"(not a datasets: map id and not a registered dataset). "
            f"Registered: {known}"
        )
    settings = {
        k: v
        for k, v in params.items()
        if k not in _DATASET_NODE_CONTROL_PARAMS and v not in (None, "")
    }
    mapped = _map_keys(
        settings, DATASET_FIELDS, f"graph node '{node_id}' dataset settings"
    )
    entry = {"dataset_name": dataset_id, **mapped}
    # Use the descriptor's own validator (it knows alternatives like pubmed_qa's
    # prepared_dataset_dir) over the node entry overlaid on the global dataset block.
    from types import SimpleNamespace

    global_data = legacy.get("data", {}) or {}
    overlay = SimpleNamespace(**{**global_data, **entry})
    errors = descriptor.validate_data_config(overlay)  # type: ignore[arg-type]
    if errors:
        required = ", ".join(
            DATA_FIELD_TO_KEY.get(f, f) for f in descriptor.required_data_fields
        )
        raise GraphConfigError(
            f"dataset '{dataset_id}' on node '{node_id}' is missing required "
            f"setting(s) (expects: {required}): " + "; ".join(errors)
        )
    legacy.setdefault("data", {}).setdefault("datasets", {})[node_id] = entry
    new_params = {
        k: v for k, v in params.items() if k in _DATASET_NODE_CONTROL_PARAMS
    }
    new_params["dataset"] = node_id
    item["params"] = new_params


# Nodes configured via their own section, not nodes.<name>: an empty block is fine, keys
# are an error. Named set so _translate_nodes stays a flat dispatch.
_NO_CONFIG_NODES = frozenset({
    "fusion", "corpus_merge", "dataset_union", "augment_audio", "metrics", "finalize",
    "dataset_source", "tts", "query_optimization", "query_correction", "augmenter",
    "rerank", "aggregate", "leaderboard_sink", "tracking_sink",
})


def _translate_nodes(
    new: Dict[str, Any], legacy: Dict[str, Any], model: Dict[str, Any]
) -> None:
    """nodes → model / vector_db / answer_generation (+ structural-node validation)."""
    nodes = new.pop("nodes", None) or {}
    vector_db: Dict[str, Any] = {}
    for name, node in nodes.items():
        node = node or {}
        if name in MODEL_NODE_FIELDS:
            model.update(_map_keys(node, MODEL_NODE_FIELDS[name], f"nodes.{name}"))
        elif name == "vector_db":
            vector_db.update(_vector_db_node_to_config(node))
        elif name == "corpus_embedding":
            # Structural in YAML (its model rides graph-override node params).
            if node:
                raise GraphConfigError(
                    f"Unknown keys under nodes.corpus_embedding: {sorted(node)}"
                )
        elif name == "retrieval":
            vector_db.update(_retrieval_to_vector_db(node))
        elif name == "answer_gen":
            legacy["answer_generation"] = dict(node)
        elif name == "dataset_sink":
            legacy["dataset_sink"] = {"enabled": True, **dict(node)}
        elif name in _NO_CONFIG_NODES:
            # Config lives in its own section, not here (see _NO_CONFIG_NODES).
            if node:
                raise GraphConfigError(
                    f"nodes.{name} takes no config (configure it via its section)"
                )
        else:
            raise GraphConfigError(f"Unknown node '{name}' in nodes:")

    if model:
        legacy["model"] = {**legacy.get("model", {}), **model}
    if vector_db:
        legacy["vector_db"] = {**legacy.get("vector_db", {}), **vector_db}


def build_evaluation_config_kwargs(new: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a node-centric config dict into ``EvaluationConfig.from_dict`` kwargs.

    Deep-copies ``new`` up front since node-mutation helpers below change param dicts
    in place and callers may reuse their own dict afterward; this is the single
    chokepoint, so callers don't need their own copy.
    """
    import copy

    new = copy.deepcopy(new)
    legacy: Dict[str, Any] = {}
    model: Dict[str, Any] = {}

    _translate_experiment(new, legacy)
    _translate_runtime(new, legacy)
    _translate_dataset(new, legacy)
    role_by_dataset_id = _translate_datasets_map(new, legacy)
    _translate_graph(new, legacy, role_by_dataset_id)
    _translate_nodes(new, legacy, model)

    # Escape hatch: remaining top-level keys are legacy-style passthrough (partial
    # migration / power users); deep-merge one level so leftovers coexist with the
    # structured blocks built above instead of overwriting them.
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(legacy.get(key), dict):
            legacy[key] = {**value, **legacy[key]}
        else:
            legacy.setdefault(key, value)
    return legacy


def resolved_model_config(config: Any, node: Any) -> Any:
    """The effective ``ModelConfig`` for a graph node: ``config.model`` overlaid with any
    per-node override params.

    Shared by the factory validation and the per-branch pipeline build — one
    implementation of "a node's params win over the flat default" instead of two.
    """
    from dataclasses import replace

    from ..pipeline.graph.operators import node_kind

    kind = node_kind(node.stage, node.params or {})
    node_to_field = MODEL_NODE_FIELDS.get(kind)
    params = node.params or {}
    if not node_to_field:
        return config.model
    overrides = {field: params[key] for key, field in node_to_field.items() if key in params}
    return replace(config.model, **overrides) if overrides else config.model


def _build_vdb_to_retrieval() -> Dict[str, Tuple[str, ...]]:
    """Derive vector_db-field → nodes.retrieval-path from the forward maps (so the two
    can't silently drift): a scalar → (node_key,); a nested key → (group, sub_key)."""
    inv: Dict[str, Tuple[str, ...]] = {}
    for node_key, vdb_field in _RETRIEVAL_SCALARS.items():
        inv[vdb_field] = (node_key,)
    for group, forward in (
        ("fusion", _FUSION_KEYS),
        ("reranker", _RERANKER_KEYS),
        ("mmr", _MMR_KEYS),
    ):
        for sub_key, vdb_field in forward.items():
            inv[vdb_field] = (group, sub_key)
    return inv


# vector_db field → nodes.retrieval location (inverse of the forward maps above).
_VDB_TO_RETRIEVAL = _build_vdb_to_retrieval()


def legacy_yaml_to_graph_yaml(old: Dict[str, Any]) -> Dict[str, Any]:
    """Reorganize a legacy config dict into the node-centric shape (file migration).

    Lossless: keys not explicitly reorganized are left where they are, so the
    ``build_evaluation_config_kwargs`` escape hatch round-trips them.
    """
    old = dict(old)
    new: Dict[str, Any] = {}

    exp = {}
    if "experiment_name" in old:
        exp["name"] = old.pop("experiment_name")
    if "output_dir" in old:
        exp["output_dir"] = old.pop("output_dir")
    if exp:
        new["experiment"] = exp

    # data → dataset
    data = old.pop("data", None)
    if isinstance(data, dict):
        inv = DATA_FIELD_TO_KEY
        ds, leftover = {}, {}
        for k, v in data.items():
            (ds if k in inv else leftover)[inv.get(k, k)] = v
        if ds:
            new["dataset"] = ds
        if leftover:
            new["data"] = leftover  # passthrough for unmapped data fields

    # model → explicit graph nodes + nodes.{asr,text_embedding,audio_embedding}
    model = old.pop("model", None)
    nodes: Dict[str, Any] = {}
    if isinstance(model, dict):
        model = dict(model)
        # graph is the spec now; pipeline_mode has no node list to migrate to, so drop it.
        model.pop("pipeline_mode", None)

    # explicit graph override → graph.{nodes,edges}; drop when absent (no null noise)
    override = old.pop("graph_override", None)
    if isinstance(override, dict) and override.get("nodes"):
        gblock = new.setdefault("graph", {})
        gblock["nodes"] = list(override["nodes"])
        raw_edges = override.get("edges") or []
        edge_list = (
            list(raw_edges) if isinstance(raw_edges, (list, tuple))
            else [{"from": src, "to": dst} for dst, srcs in raw_edges.items() for src in srcs]
        )
        from ..pipeline.graph.wiring import is_port_edge

        if not any(is_port_edge(e) for e in edge_list):
            # legacy override without port edges: generate them (ordering extras kept)
            from ..pipeline.graph.wiring import emit_edges

            edge_list = emit_edges(gblock["nodes"]) + edge_list
        gblock["edges"] = edge_list

    if isinstance(model, dict):
        field_to_node = {
            field: (node, key)
            for node, m in MODEL_NODE_FIELDS.items()
            for key, field in m.items()
        }
        leftover_model = {}
        for k, v in model.items():
            if k in field_to_node:
                node, key = field_to_node[k]
                nodes.setdefault(node, {})[key] = v
            else:
                leftover_model[k] = v
        if leftover_model:
            new["model"] = leftover_model  # passthrough

    # vector_db → nodes.retrieval + nodes.vector_db.store
    vdb = old.pop("vector_db", None)
    if isinstance(vdb, dict):
        vdb = dict(vdb)
        retrieval: Dict[str, Any] = {}
        if "type" in vdb:
            # Canonical home for the store choice is the vector_db node.
            nodes.setdefault("vector_db", {})["store"] = vdb.pop("type")
        leftover_vdb = {}
        for k, v in vdb.items():
            if k in _VDB_TO_RETRIEVAL:
                path = _VDB_TO_RETRIEVAL[k]
                if len(path) == 1:
                    retrieval[path[0]] = v
                else:
                    retrieval.setdefault(path[0], {})[path[1]] = v
            else:
                leftover_vdb[k] = v
        if retrieval:
            nodes["retrieval"] = retrieval
        if leftover_vdb:
            new["vector_db"] = leftover_vdb  # passthrough advanced fields

    # answer_generation → nodes.answer_gen
    ans = old.pop("answer_generation", None)
    if isinstance(ans, dict):
        nodes["answer_gen"] = ans

    if nodes:
        new["nodes"] = nodes

    # runtime grouping (cosmetic; passthrough preserves correctness either way)
    runtime: Dict[str, Any] = {}
    for key in ("cache", "tracking", "logging", "features"):
        if key in old:
            runtime[key] = old.pop(key)
    if "service_runtime" in old:
        runtime["service"] = old.pop("service_runtime")
    if runtime:
        new["runtime"] = runtime

    # everything else (advanced flags, llm, device_pool, …) stays top-level (passthrough)
    new.update(old)
    return new
