"""Node-centric YAML config → ``EvaluationConfig``.

The human-readable config is organized by *node* (mirroring the DAG): a ``graph:``
block picks the pipeline mode (auto-derive) and a ``nodes:`` block carries each node's
model + params. This loader translates that shape into the legacy config dict and reuses
``EvaluationConfig.from_dict`` for construction + validation (so there is one runtime
config object, only the YAML surface changes). See ``evaluator-architecture.md`` §10.

Example::

    experiment: { name: whisper_jina, output_dir: evaluation_results }
    dataset:    { id: pubmed_qa, questions: q.json, corpus: corpus.json }
    graph:      { mode: asr_text_retrieval }
    nodes:
      asr:            { model: whisper, size: large }
      text_embedding: { model: labse }
      retrieval:      { k: 5, mode: hybrid, reranker: { enabled: true } }
      answer_gen:     { enabled: true }
    runtime:    { cache: { enabled: true }, tracking: { backend: mlflow } }
"""

from __future__ import annotations

from typing import Any, Dict

from ..errors import ConfigurationError
from .evaluation import EvaluationConfig


class GraphConfigError(ConfigurationError):
    """Raised when a node-centric config cannot be translated."""


# nodes.<name>.<key> → ModelConfig field. Each model node shares the same param names.
_MODEL_NODE_FIELDS = {
    "asr": {
        "model": "asr_model_type",
        "size": "asr_size",
        "name": "asr_model_name",
        "device": "asr_device",
        "adapter": "asr_adapter_path",
        "params": "asr_params",
    },
    "text_embedding": {
        "model": "text_emb_model_type",
        "size": "text_emb_size",
        "name": "text_emb_model_name",
        "device": "text_emb_device",
        "adapter": "text_emb_adapter_path",
        "params": "text_emb_params",
    },
    "audio_embedding": {
        "model": "audio_emb_model_type",
        "size": "audio_emb_size",
        "name": "audio_emb_model_name",
        "device": "audio_emb_device",
        "model_path": "audio_emb_model_path",
        "dim": "audio_emb_dim",
        "dropout": "audio_emb_dropout",
        "params": "audio_emb_params",
    },
}

# dataset.<key> → DataConfig field.
_DATASET_FIELDS = {
    "id": "dataset_name",
    "questions": "questions_path",
    "corpus": "corpus_path",
    "prepared_dir": "prepared_dataset_dir",
    "audio_dir": "audio_dir",
    "transcripts": "transcripts_file",
    "hf_dataset": "huggingface_dataset",
    "hf_subset": "huggingface_subset",
    "hf_split": "huggingface_split",
    "split": "huggingface_split",
    "batch_size": "batch_size",
    "trace_limit": "trace_limit",
    "type": "dataset_type",
    "source": "dataset_source",
    "max_samples": "max_samples",
    "language": "default_language",
}

# DataConfig field → node-centric YAML key (inverse of _DATASET_FIELDS; the builder's
# required-settings form and the dataset_source node-param translation speak this).
_DATA_FIELD_TO_KEY = {v: k for k, v in _DATASET_FIELDS.items()}

# nodes.retrieval scalar keys → VectorDBConfig field.
_RETRIEVAL_SCALARS = {
    "k": "k",
    "mode": "retrieval_mode",
    "distance": "distance_metric",
    "gpu_id": "gpu_id",
}

# nodes.vector_db keys → VectorDBConfig field(s) — the canonical home for the store
# backend choice (the §4 split node). `collection` fans out to both backends' fields.
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
    """experiment → experiment_name / output_dir (+ passthrough scalars)."""
    # experiment / runtime / features are already understood by from_dict; pass through.
    exp = new.pop("experiment", None) or {}
    if "name" in exp:
        legacy["experiment_name"] = exp.pop("name")
    if "output_dir" in exp:
        legacy["output_dir"] = exp.pop("output_dir")
    legacy.update(exp)  # any remaining experiment-level scalars


def _translate_runtime(new: Dict[str, Any], legacy: Dict[str, Any]) -> None:
    """runtime → service_runtime rename (cache/tracking/... pass straight through)."""
    runtime = new.pop("runtime", None) or {}
    if "service" in runtime:
        runtime["service_runtime"] = runtime.pop("service")
    # cache / tracking / logging / features / parallel_* pass straight through
    legacy.update(runtime)


def _translate_dataset(new: Dict[str, Any], legacy: Dict[str, Any]) -> None:
    """dataset → data (single-dataset block)."""
    dataset = new.pop("dataset", None) or {}
    if dataset:
        legacy["data"] = _map_keys(dataset, _DATASET_FIELDS, "dataset")


def _translate_datasets_map(new: Dict[str, Any], legacy: Dict[str, Any]):
    """datasets: map (B4) → data.datasets (multi-dataset graphs).

    Each entry's keys are translated to DataConfig field names; an optional `role` is
    kept verbatim and later injected into the referencing dataset_source node's params.
    Returns ``(datasets, role_by_dataset_id)`` so the graph translator can validate
    dataset_source references against whether a datasets: block was present at all.
    """
    role_by_dataset_id: Dict[str, str] = {}
    datasets = new.pop("datasets", None)
    if datasets is not None:
        if not isinstance(datasets, dict):
            raise GraphConfigError("datasets: must be a map of id -> source spec.")
        translated: Dict[str, Any] = {}
        for sid, entry in datasets.items():
            entry = dict(entry or {})
            role = entry.pop("role", None)
            mapped = _map_keys(entry, _DATASET_FIELDS, f"datasets.{sid}")
            if role is not None:
                mapped["role"] = role
                role_by_dataset_id[str(sid)] = str(role)
            translated[str(sid)] = mapped
        legacy.setdefault("data", {})["datasets"] = translated
    return datasets, role_by_dataset_id


def _translate_graph(
    new: Dict[str, Any],
    legacy: Dict[str, Any],
    model: Dict[str, Any],
    datasets: Any,
    role_by_dataset_id: Dict[str, str],
) -> None:
    """graph → pipeline_mode (+ optional explicit node/edge override, config C2)."""
    graph = new.pop("graph", None) or {}
    # Reject unknown graph keys (T2): the graph block is parsed selectively, so a typo
    # (e.g. `branchess:`) would otherwise be silently dropped → a quietly wrong experiment.
    _GRAPH_KEYS = {"mode", "nodes", "edges", "branches"}
    _unknown_graph = set(graph) - _GRAPH_KEYS
    if _unknown_graph:
        raise GraphConfigError(
            f"Unknown key(s) under graph: {sorted(_unknown_graph)}. "
            f"Allowed: {sorted(_GRAPH_KEYS)}."
        )
    if "mode" in graph:
        model["pipeline_mode"] = graph["mode"]
    if "nodes" in graph:
        if "mode" not in graph:
            raise GraphConfigError(
                "graph.nodes requires graph.mode too (mode drives handler behavior)."
            )
        node_ids = graph["nodes"]
        # Items are a node-type string OR a dict {id, type, params} for a distinct
        # instance (e.g. two rerankers with different models).
        if not isinstance(node_ids, list) or not all(
            isinstance(n, (str, dict)) for n in node_ids
        ):
            raise GraphConfigError(
                "graph.nodes must be a list of node-type strings or {id, type, params} dicts."
            )
        # A dataset_source node referencing a `datasets:` entry inherits that entry's
        # role into its params (single source of truth = the datasets map), so graph
        # wiring (_effective_outputs) routes downstream nodes to the right source. A
        # role set explicitly on the node wins; an unknown dataset id is an error.
        for item in node_ids:
            if not (isinstance(item, dict) and item.get("type") == "dataset_source"):
                continue
            params = item.get("params") or {}
            ds_id = params.get("dataset")
            if ds_id is None:
                continue
            map_entries = legacy.get("data", {}).get("datasets") or {}
            in_map = str(ds_id) in role_by_dataset_id or str(ds_id) in map_entries
            if not in_map:
                # Not a datasets-map id → must be a REGISTERED dataset id (the
                # builder's picker). Synthesize a per-node datasets entry from the
                # node's data-setting params and rewrite the reference, so the
                # existing multi-source loader (B1/B4) runs it unchanged.
                _synthesize_dataset_entry(item, params, str(ds_id), legacy)
                continue
            if "role" not in params and str(ds_id) in role_by_dataset_id:
                params["role"] = role_by_dataset_id[str(ds_id)]
                item["params"] = params
        # edges: list of {from, to} → {to: [from, ...]} for build_graph_from_spec
        edges: Dict[str, list] = {}
        for edge in graph.get("edges", []) or []:
            if not (isinstance(edge, dict) and "from" in edge and "to" in edge):
                raise GraphConfigError("each graph.edges item needs 'from' and 'to'.")
            edges.setdefault(edge["to"], []).append(edge["from"])
        legacy["graph_override"] = {"nodes": node_ids, "edges": edges}
    elif "edges" in graph:
        raise GraphConfigError("graph.edges requires graph.nodes.")

    # graph.branches: a variant set expanded + CSE-collapsed into one branched DAG (W6/A8).
    # Each entry is {id, <node_type>: <model-str | params>}. Stored on graph_override so the
    # build path (build_graph_for_config / _build_run_graph) calls build_branched_graph.
    if "branches" in graph:
        if "mode" not in graph:
            raise GraphConfigError("graph.branches requires graph.mode too.")
        branches = graph["branches"]
        if not isinstance(branches, list) or not all(
            isinstance(b, dict) and "id" in b for b in branches
        ):
            raise GraphConfigError(
                "graph.branches must be a list of {id, <node>: <override>} dicts."
            )
        legacy.setdefault("graph_override", {"nodes": [], "edges": {}})[
            "branches"
        ] = branches


# dataset_source node params that are NOT data settings (kept on the node).
_DATASET_NODE_CONTROL_PARAMS = {"dataset", "role", "fields"}


def _synthesize_dataset_entry(
    item: Dict[str, Any],
    params: Dict[str, Any],
    dataset_id: str,
    legacy: Dict[str, Any],
) -> None:
    """A dataset_source node naming a REGISTERED dataset (builder picker path).

    Builds ``data.datasets[<node_id>] = {dataset_name: id, **node data settings}``
    and rewrites ``params.dataset`` to the node id; validates the descriptor's
    ``required_data_fields`` are satisfied (node params or global data fields).
    """
    from ..datasets.descriptor import get_descriptor, list_registered_datasets

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
        settings, _DATASET_FIELDS, f"graph node '{node_id}' dataset settings"
    )
    entry = {"dataset_name": dataset_id, **mapped}
    # Validate with the descriptor's own validator (the authority — it knows
    # alternatives like pubmed_qa's prepared_dataset_dir) over node-entry settings
    # overlaid on the global dataset block.
    from types import SimpleNamespace

    global_data = legacy.get("data", {}) or {}
    overlay = SimpleNamespace(**{**global_data, **entry})
    errors = descriptor.validate_data_config(overlay)  # type: ignore[arg-type]
    if errors:
        required = ", ".join(
            _DATA_FIELD_TO_KEY.get(f, f) for f in descriptor.required_data_fields
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


def _translate_nodes(
    new: Dict[str, Any], legacy: Dict[str, Any], model: Dict[str, Any]
) -> None:
    """nodes → model / vector_db / answer_generation (+ structural-node validation)."""
    nodes = new.pop("nodes", None) or {}
    vector_db: Dict[str, Any] = {}
    for name, node in nodes.items():
        node = node or {}
        if name in _MODEL_NODE_FIELDS:
            model.update(_map_keys(node, _MODEL_NODE_FIELDS[name], f"nodes.{name}"))
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
        elif name in (
            "fusion",
            "corpus_merge",
            "dataset_union",
            "augment_audio",
            "metrics",
            "finalize",
            "dataset_source",
            "tts",
            "query_optimization",
            "query_correction",
            "augmenter",
            "rerank",
            "aggregate",
            "leaderboard_sink",
            "tracking_sink",
        ):
            # structural / config-derived nodes — no per-node config block here
            # (tts via audio_synthesis, rerank via vector_db, query_optimization via
            # query_optimization). Listing them empty is allowed; values are an error.
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


def to_legacy_dict(new: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a node-centric config dict into the legacy ``from_dict`` shape."""
    new = dict(new)
    legacy: Dict[str, Any] = {}
    model: Dict[str, Any] = {}

    _translate_experiment(new, legacy)
    _translate_runtime(new, legacy)
    _translate_dataset(new, legacy)
    datasets, role_by_dataset_id = _translate_datasets_map(new, legacy)
    _translate_graph(new, legacy, model, datasets, role_by_dataset_id)
    _translate_nodes(new, legacy, model)

    # Escape hatch: any remaining top-level keys are legacy-style and passed through
    # (supports partial migration / power users). Deep-merge one level so passthrough
    # leftovers (e.g. unmapped data/model/vector_db fields) coexist with the structured
    # blocks built above instead of overwriting them.
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(legacy.get(key), dict):
            legacy[key] = {**value, **legacy[key]}
        else:
            legacy.setdefault(key, value)
    return legacy


def _invert(mapping: Dict[str, str]) -> Dict[str, str]:
    return {v: k for k, v in mapping.items()}


# vector_db field → where it lands in nodes.retrieval (inverse of the C1 mapping).
_VDB_TO_RETRIEVAL = {
    "k": ("k",),
    "retrieval_mode": ("mode",),
    "distance_metric": ("distance",),
    "gpu_id": ("gpu_id",),
    "hybrid_fusion_method": ("fusion", "method"),
    "hybrid_dense_weight": ("fusion", "dense_weight"),
    "rrf_k": ("fusion", "rrf_k"),
    "reranker_enabled": ("reranker", "enabled"),
    "reranker_model": ("reranker", "model"),
    "reranker_mode": ("reranker", "mode"),
    "reranker_weight": ("reranker", "weight"),
    "reranker_top_k": ("reranker", "top_k"),
    "reranker_device": ("reranker", "device"),
    "use_mmr": ("mmr", "enabled"),
    "mmr_lambda": ("mmr", "lambda"),
}


def legacy_yaml_to_graph_yaml(old: Dict[str, Any]) -> Dict[str, Any]:
    """Reorganize a legacy config dict into the node-centric shape (file migration).

    Lossless: keys not explicitly reorganized are left where they are, so the
    ``to_legacy_dict`` escape hatch round-trips them. The result is verified by
    comparing the rebuilt ``EvaluationConfig`` to the original in the migration script.
    """
    old = dict(old)
    new: Dict[str, Any] = {}

    # experiment
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
        inv = _invert(_DATASET_FIELDS)
        ds, leftover = {}, {}
        for k, v in data.items():
            (ds if k in inv else leftover)[inv.get(k, k)] = v
        if ds:
            new["dataset"] = ds
        if leftover:
            new["data"] = leftover  # passthrough for unmapped data fields

    # model → graph.mode + nodes.{asr,text_embedding,audio_embedding}
    model = old.pop("model", None)
    nodes: Dict[str, Any] = {}
    if isinstance(model, dict):
        model = dict(model)
        if "pipeline_mode" in model:
            new["graph"] = {"mode": model.pop("pipeline_mode")}

    # explicit graph override → graph.{nodes,edges}; drop when absent (no null noise)
    override = old.pop("graph_override", None)
    if isinstance(override, dict) and override.get("nodes"):
        gblock = new.setdefault("graph", {})
        gblock["nodes"] = list(override["nodes"])
        edge_list = [
            {"from": src, "to": dst}
            for dst, srcs in (override.get("edges") or {}).items()
            for src in srcs
        ]
        if edge_list:
            gblock["edges"] = edge_list

    if isinstance(model, dict):
        field_to_node = {
            field: (node, key)
            for node, m in _MODEL_NODE_FIELDS.items()
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

    # vector_db → nodes.retrieval + nodes.corpus_index.store
    vdb = old.pop("vector_db", None)
    if isinstance(vdb, dict):
        vdb = dict(vdb)
        retrieval: Dict[str, Any] = {}
        if "type" in vdb:
            # Canonical home for the store choice is the vector_db node (§4 split).
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


def load_graph_config(path: str, *, validate: bool = True) -> EvaluationConfig:
    """Load a node-centric YAML config and build an ``EvaluationConfig``.

    Thin alias for :meth:`EvaluationConfig.from_yaml` — the single load chokepoint that
    already translates the node-centric shape via :func:`to_legacy_dict`.
    """
    return EvaluationConfig.from_yaml(path, validate=validate)
