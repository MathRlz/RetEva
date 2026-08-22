"""Serialization helpers for :class:`EvaluationConfig`.

Holds the recursive dataclass/enum serialization primitives and the
``to_dict`` / ``to_runtime_dict`` / ``to_experiment_dict`` / ``to_yaml`` method
bodies. :mod:`evaluator.config.evaluation` keeps thin delegators with the
public signatures; the heavy logic lives here.
"""

from dataclasses import fields
from enum import Enum
from typing import Any, Dict

import yaml


def _serialize_value(val: Any) -> Any:
    """Recursively serialize a value for dict output."""
    if isinstance(val, Enum):
        return str(val)
    if hasattr(val, "__dataclass_fields__"):
        return _serialize_dataclass(val)
    if isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    return val


def _serialize_dataclass(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass to dict, handling enums/nested dataclasses."""
    result: Dict[str, Any] = {}
    for f in fields(obj):
        result[f.name] = _serialize_value(getattr(obj, f.name))
    return result


def to_runtime_dict(config: Any) -> Dict[str, Any]:
    """Return runtime execution configuration surface (auto-serialized)."""
    result: Dict[str, Any] = {}
    for name in config._RUNTIME_FIELDS:
        val = getattr(config, name)
        if val is None:
            continue
        result[name] = _serialize_dataclass(val)
    for name in config._RUNTIME_SCALARS:
        result[name] = getattr(config, name)
    return result


def to_experiment_dict(config: Any) -> Dict[str, Any]:
    """Return experiment/reporting configuration surface (auto-serialized)."""
    result: Dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "output_dir": config.output_dir,
    }
    for name in config._EXPERIMENT_SUBCONFIGS:
        result[name] = _serialize_dataclass(getattr(config, name))
    return result


def to_yaml(config: Any, yaml_path: str) -> None:
    """Save configuration to YAML file."""
    config_dict = {
        "experiment_name": config.experiment_name,
        "output_dir": config.output_dir,
        "runtime": config.to_runtime_dict(),
        "experiment": config.to_experiment_dict(),
    }

    with open(yaml_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def _config_template_label(config: Any) -> Any:
    """The config's pipeline-mode label — the report's ``pipeline_mode`` echo. Delegates to the
    ``graph_template`` property (explicit template if any, else the label derived from the
    graph's node kinds); ``None`` only for a query-head-less custom graph."""
    return getattr(config, "graph_template", None)


def _model_summary(model: Any, family: str) -> str:
    """``type:name_or_size_or_default`` summary string for a model family (telemetry)."""
    mtype = getattr(model, f"{family}_model_type")
    name = getattr(model, f"{family}_model_name", None)
    size = getattr(model, f"{family}_size", None)
    return f"{mtype}:{name or size or 'default'}"


_MODEL_SUMMARY_KEYS = {  # flat-dict key -> (family, the node kind that carries it)
    "asr_model": ("asr", "asr"),
    "text_emb_model": ("text_emb", "text_embedding"),
    "audio_emb_model": ("audio_emb", "audio_embedding"),
}


def _active_model_summaries(config: Any) -> Dict[str, str]:
    """Model summary strings for ONLY the families the executed graph actually uses, so a default
    (e.g. ``asr_model: wav2vec2:default``) never leaks for a node the run never had. Resolves each
    present family's ACTUAL model(s) off the graph's own nodes (the flat default overlaid by each
    node's own params — same resolver factory.py/validation.py use), not the flat default blindly
    — a branched config with 2+ distinct models in one family joins them as ``a+b`` instead of
    silently reporting only the (possibly unset) flat default. Best-effort: if the graph can't be
    built, falls back to the flat default for all three (the pre-graph-config-shape case)."""
    from ..pipeline.graph.operators import node_kind
    from .graph_config import resolved_model_config

    graph = None
    try:
        from ..pipeline.graph.modes import build_graph_for_config

        graph = build_graph_for_config(config)
    except Exception as exc:  # noqa: BLE001 - telemetry must not crash
        from ..logging_config import get_logger

        get_logger(__name__).debug("active-model summary fell back (graph build failed): %s", exc)

    out: Dict[str, str] = {}
    for key, (family, kind) in _MODEL_SUMMARY_KEYS.items():
        if graph is None:
            out[key] = _model_summary(config.model, family)
            continue
        nodes = [n for n in graph.nodes if node_kind(n.stage, n.params) == kind]
        if not nodes:
            continue
        summaries: list = []
        for n in nodes:
            summary = _model_summary(resolved_model_config(config, n), family)
            if summary not in summaries:
                summaries.append(summary)
        out[key] = "+".join(summaries)
    return out


def to_dict(config: Any, *, include_config: bool = False) -> Dict[str, Any]:
    """Convert to dictionary.

    Args:
        include_config: If True return full nested dict that can
            round-trip through ``from_dict``.  If False (default)
            return compact flat dict for telemetry/logging.
    """
    if include_config:
        return to_nested_dict(config)

    result = {
        "experiment_name": config.experiment_name,
        "output_dir": config.output_dir,
        "checkpoint_enabled": config.checkpoint_enabled,
        "checkpoint_interval": config.checkpoint_interval,
        "parallel_enabled": config.parallel_enabled,
        "cache_enabled": config.cache.enabled,
        "pipeline_mode": _config_template_label(config),
        # Only families the executed graph uses — no default leak for an absent node.
        **_active_model_summaries(config),
        "dataset": config.data.dataset_name,
        "questions_path": config.data.questions_path,
        "corpus_path": config.data.corpus_path,
        "batch_size": config.data.batch_size,
        "trace_limit": config.data.trace_limit,
        "judge_enabled": config.judge.enabled,
        "tracking_enabled": config.tracking.enabled,
        "tracking_backend": config.tracking.backend,
        "service_startup_mode": config.service_runtime.startup_mode,
        "service_offload_policy": config.service_runtime.offload_policy,
    }
    if config.device_pool is not None:
        result["device_pool_strategy"] = (
            "manual" if config.device_pool.model_device_overrides else "memory_aware"
        )
    return result


def to_nested_dict(config: Any) -> Dict[str, Any]:
    """Full nested dict that round-trips through ``from_dict``."""
    result: Dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "output_dir": config.output_dir,
        "checkpoint_enabled": config.checkpoint_enabled,
        "checkpoint_interval": config.checkpoint_interval,
        "resume_from_checkpoint": config.resume_from_checkpoint,
        "parallel_enabled": config.parallel_enabled,
    }
    from .evaluation import _FEATURE_SUBCONFIGS

    for key in config._PLAIN_SUBCONFIGS:
        result[key] = _serialize_dataclass(getattr(config, key))
    result["llm"] = _serialize_dataclass(config.llm)
    for key in _FEATURE_SUBCONFIGS:  # each capability sub-config is a top-level field now
        result[key] = _serialize_dataclass(getattr(config, key))
    if config.device_pool is not None:
        result["device_pool"] = _serialize_dataclass(config.device_pool)
    # The graph spec (explicit nodes/edges or a template reference) — so an explicit-graph
    # config round-trips through from_dict (it's a plain JSON-able dict).
    if getattr(config, "graph_override", None):
        result["graph_override"] = config.graph_override
    return result
