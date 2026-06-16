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

from ..config.types import enum_to_str


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
        "pipeline_mode": enum_to_str(config.model.pipeline_mode),
        "asr_model": f"{config.model.asr_model_type}:{config.model.asr_model_name or config.model.asr_size or 'default'}",
        "text_emb_model": f"{config.model.text_emb_model_type}:{config.model.text_emb_model_name or config.model.text_emb_size or 'default'}",
        "audio_emb_model": f"{config.model.audio_emb_model_type}:{config.model.audio_emb_model_name or config.model.audio_emb_size or 'default'}",
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
        result["device_pool_strategy"] = enum_to_str(
            config.device_pool.allocation_strategy
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
    for key in config._PLAIN_SUBCONFIGS:
        result[key] = _serialize_dataclass(getattr(config, key))
    result["llm"] = _serialize_dataclass(config.llm)
    result["features"] = _serialize_dataclass(config.features)
    if config.device_pool is not None:
        result["device_pool"] = _serialize_dataclass(config.device_pool)
    return result
