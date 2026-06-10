"""Shared config helpers for WebAPI endpoints."""

from typing import Any, Callable, Dict, List

from evaluator import EvaluationConfig, list_presets
from evaluator.config.types import DatasetType, PipelineMode, VectorDBType
from evaluator.datasets import (
    list_known_dataset_names,
    resolve_dataset_profile,
    validate_dataset_runtime_config,
)
from evaluator.pipeline import build_graph_for_config, get_stage_node_def
from evaluator.pipeline.factory import check_backend_dependencies
from evaluator.services import ModelServiceProvider
from evaluator.webapi.utils import with_provider


def deep_merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge patch dict into base dict."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


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

    defaults = EvaluationConfig()
    return {
        "presets": list_presets(),
        "pipeline_modes": [mode.value for mode in PipelineMode],
        "dataset_types": [dataset_type.value for dataset_type in DatasetType],
        "dataset_sources": ["local", "huggingface", "custom"],
        "dataset_names": list_known_dataset_names(),
        "vector_db_types": [db.value for db in VectorDBType],
        "retrieval_modes": ["dense", "sparse", "hybrid"],
        "hybrid_fusion_methods": ["weighted", "rrf", "max_score"],
        "reranker_modes": ["none", "token_overlap", "cross_encoder"],
        "service_runtime": {
            "startup_mode": ["lazy", "eager"],
            "offload_policy": ["on_finish", "never"],
        },
        "tts_providers": sorted(
            {entry["type"] for entry in normalized_models.get("tts", [])}
        ),
        "models": normalized_models,
        "defaults": nested_config(defaults),
    }


def graph_preview(config: EvaluationConfig) -> Dict[str, Any]:
    profile = resolve_dataset_profile(
        config.data.dataset_name, config.data.dataset_type
    )
    graph = build_graph_for_config(config)
    return {
        "mode": graph.mode,
        "nodes": [
            {
                "id": node.id,
                "stage": node.stage,
                "depends_on": list(node.depends_on),
                "inputs": list(get_stage_node_def(node.stage).inputs),
                "outputs": list(get_stage_node_def(node.stage).outputs),
            }
            for node in graph.nodes
        ],
        "levels": [[node.id for node in level] for level in graph.topological_levels()],
        "dataset_profile": {
            "name": profile.name,
            "dataset_type": str(profile.dataset_type),
            "requires_audio": profile.requires_audio,
            "requires_text": profile.requires_text,
            "supports_generation": profile.supports_generation,
            "evaluation_mode": profile.evaluation_mode,
            "recommended_pipeline_modes": list(profile.recommended_pipeline_modes),
            "pipeline_mode_supported": profile.supports_pipeline_mode(
                str(config.model.pipeline_mode)
            ),
        },
    }


def node_catalogue() -> Dict[str, Any]:
    """The registered stage-node types + their I/O contract (E2): what the visual builder's
    palette offers and how ports connect. Each entry = ``{type, model_field, inputs, outputs,
    optional_inputs, param_defaults}`` — ports connect when an output artifact name matches a
    consumer's input name (the same rule the auto-wiring uses)."""
    from ..pipeline.stage_graph import _NODE_REGISTRY

    return {
        "nodes": [
            {
                "type": stage,
                "model_field": d.model_field,
                "inputs": list(d.inputs),
                "outputs": list(d.outputs),
                "optional_inputs": list(d.optional_inputs),
                "param_defaults": dict(d.param_defaults),
            }
            for stage, d in sorted(_NODE_REGISTRY.items())
        ]
    }


def load_config(
    payload_config: Dict[str, Any], *, auto_devices: bool
) -> EvaluationConfig:
    config = EvaluationConfig.from_dict(payload_config)
    if auto_devices:
        config = config.with_auto_devices()
    return config


def prepare_run_config(
    payload_config: Dict[str, Any], *, auto_devices: bool
) -> EvaluationConfig:
    config = load_config(payload_config, auto_devices=auto_devices)
    check_backend_dependencies(config.vector_db)
    validate_dataset_runtime_config(
        config,
        retrieval_required=(str(config.model.pipeline_mode) != "asr_only"),
    )
    return config
