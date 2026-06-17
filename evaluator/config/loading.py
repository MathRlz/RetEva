"""Construction glue for :class:`EvaluationConfig`.

Holds the ``from_dict`` / ``from_yaml`` / ``from_preset`` method bodies and the
section-merge / sub-config helpers. :mod:`evaluator.config.evaluation` keeps
thin classmethod delegators with the public signatures; the heavy logic lives
here.
"""

from dataclasses import fields
from typing import Any, Dict

import yaml

from ..errors import ConfigurationError
from .audio_synthesis import AudioSynthesisConfig
from .audio_augmentation import AudioAugmentationConfig
from .answer_generation import AnswerGenerationConfig
from .llm_backend import LLMConfig
from .judge import JudgeConfig
from .query_optimization import QueryOptimizationConfig
from .query_correction import QueryCorrectionConfig
from .embedding_fusion import EmbeddingFusionConfig
from .rag_flow import RagFlowConfig
from .device_pool import DevicePoolConfig


def _merge_section_into(config_dict: dict, section_dict) -> None:
    """Flatten a nested section (``runtime``/``experiment``) up into config_dict.

    Dict values are deep-merged with any existing same-key dict (section wins);
    scalar values overwrite. No-op when ``section_dict`` isn't a dict.
    """
    if not isinstance(section_dict, dict):
        return
    for key, value in section_dict.items():
        if isinstance(value, dict) and isinstance(config_dict.get(key), dict):
            merged = dict(config_dict[key])
            merged.update(value)
            config_dict[key] = merged
        else:
            config_dict[key] = value


def _construct_subconfig(cls_: Any, data: Dict[str, Any], path: str) -> Any:
    """Build a sub-config, turning an unknown-key ``TypeError`` into a clear, path-named
    ``ConfigurationError`` (T2): a misspelled key (e.g. ``asr_modal_type``) must fail loudly
    naming ``model.asr_modal_type`` — not run a silently wrong experiment or surface a cryptic
    ``__init__() got an unexpected keyword argument`` error."""
    try:
        return cls_(**data)
    except TypeError as exc:
        msg = str(exc)
        if "unexpected keyword argument" in msg and "'" in msg:
            bad = msg.split("'")[1]
            where = f"{path}.{bad}" if path else bad
            allowed = sorted(f.name for f in fields(cls_))
            raise ConfigurationError(
                f"Unknown config key '{where}'. Allowed under "
                f"{path or cls_.__name__}: {allowed}"
            ) from exc
        raise


def build_from_dict(cls, config_dict: Dict[str, Any], validate: bool = True) -> "Any":
    """Create configuration from a dictionary.

    Args:
        config_dict: Dictionary with configuration values. Nested dicts
                    for 'cache', 'logging', 'model', 'data', 'audio_synthesis',
                    'judge', 'vector_db', 'device_pool' are used to construct sub-configs.
        validate: If True, validate configuration after creation. Defaults to True.

    Returns:
        EvaluationConfig instance.

    Raises:
        ConfigurationError: If validate=True and validation fails.
    """
    from .evaluation import FeaturesConfig, _FEATURE_SUBCONFIGS

    # Create a copy to avoid modifying the original
    config_dict = dict(config_dict)

    # The optional `runtime:`/`experiment:` blocks are flattened up into the
    # top level (deep-merging any dict values) before sub-config construction.
    _merge_section_into(config_dict, config_dict.pop("runtime", None))
    _merge_section_into(config_dict, config_dict.pop("experiment", None))

    # Accept the canonical nested `features:` block AND legacy flat feature
    # keys (judge:, embedding_fusion:, ...) by flattening the former into the
    # latter; nested values win over any stray flat key.
    features_dict = config_dict.pop("features", None)
    if isinstance(features_dict, dict):
        for key, value in features_dict.items():
            config_dict[key] = value

    # Build plain sub-configs from registry — no duplication when adding new ones
    sub_configs: Dict[str, Any] = {
        key: _construct_subconfig(cls_, config_dict.get(key) or {}, key)
        for key, cls_ in cls._PLAIN_SUBCONFIGS.items()
    }

    # Shared LLM backend — propagates to judge/answer_generation/query_optimization
    # as defaults; per-component values override.
    llm_config = LLMConfig(**(config_dict.get("llm") or {}))
    _llm_fields = {
        "model",
        "api_base",
        "api_key_env",
        "temperature",
        "timeout_s",
        "use_local_server",
        "local_server_url",
    }
    _llm_base = {k: v for k, v in vars(llm_config).items() if k in _llm_fields}

    def _merge_llm(comp_dict: dict) -> dict:
        merged = dict(_llm_base)
        merged.update(comp_dict)
        return merged

    # Backwards-compat: map old llm_* names in query_optimization dict
    _qo_raw = dict(config_dict.get("query_optimization") or {})
    for old, new in (
        ("llm_model", "model"),
        ("llm_api_base", "api_base"),
        ("llm_api_key_env", "api_key_env"),
        ("llm_temperature", "temperature"),
    ):
        if old in _qo_raw and new not in _qo_raw:
            _qo_raw[new] = _qo_raw.pop(old)
        elif old in _qo_raw:
            _qo_raw.pop(old)

    sub_configs["llm"] = llm_config

    # Assemble the optional features into one FeaturesConfig (#7). The three
    # LLM-derived features inherit the shared backend defaults; the rest are
    # plain sub-configs.
    features = FeaturesConfig(
        audio_synthesis=AudioSynthesisConfig(
            **(config_dict.get("audio_synthesis") or {})
        ),
        augmentation=AudioAugmentationConfig(**(config_dict.get("augmentation") or {})),
        answer_generation=AnswerGenerationConfig(
            **_merge_llm(config_dict.get("answer_generation") or {})
        ),
        judge=JudgeConfig(**_merge_llm(config_dict.get("judge") or {})),
        query_optimization=QueryOptimizationConfig(**_merge_llm(_qo_raw)),
        query_correction=QueryCorrectionConfig(
            **_merge_llm(config_dict.get("query_correction") or {})
        ),
        embedding_fusion=EmbeddingFusionConfig(
            **(config_dict.get("embedding_fusion") or {})
        ),
        rag=RagFlowConfig(**(config_dict.get("rag") or {})),
    )
    sub_configs["features"] = features

    # device_pool is optional
    device_pool_dict = config_dict.get("device_pool")
    sub_configs["device_pool"] = (
        DevicePoolConfig(**device_pool_dict) if device_pool_dict else None
    )

    # All remaining keys are scalar fields on EvaluationConfig itself.
    # Exclude the feature keys consumed above (they're not EvaluationConfig fields).
    _consumed = set(sub_configs) | set(_FEATURE_SUBCONFIGS)
    main_config = {k: v for k, v in config_dict.items() if k not in _consumed}

    config = _construct_subconfig(cls, {**main_config, **sub_configs}, "")

    if validate:
        config.validate()

    return config


def build_from_yaml(cls, yaml_path: str, validate: bool = True) -> "Any":
    """Load configuration from a node-centric YAML file.

    The on-disk schema is the node-centric shape (experiment/dataset/graph/nodes/
    runtime); ``to_legacy_dict`` translates it before construction. Legacy-shape keys
    still pass through (the translator is backward-compatible), so this is the single
    load chokepoint for the CLI, public API, and presets.

    Args:
        yaml_path: Path to YAML configuration file.
        validate: If True, validate configuration after loading. Defaults to True.

    Returns:
        EvaluationConfig instance.

    Raises:
        ConfigurationError: If validate=True and validation fails.
    """
    from .graph_config import to_legacy_dict

    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    if config_dict is None:
        raise AttributeError("Empty configuration file.")
    return cls.from_dict(to_legacy_dict(config_dict), validate=validate)


# Config sections addressable by the legacy underscore override notation. Matched
# longest-prefix-first so "vector_db_k" resolves to the vector_db section (there is no
# standalone "vector" section); a key matching no section is set at the top level.
_OVERRIDE_SECTIONS = (
    "audio_synthesis",
    "service_runtime",
    "device_pool",
    "vector_db",
    "cache",
    "logging",
    "model",
    "data",
    "judge",
    "tracking",
)


def _resolve_underscore_override(key: str):
    """Map a legacy ``section_subkey`` override to ``(section, sub_key)`` (F13).

    Returns ``(None, key)`` for a plain top-level key. Replaces the hand-sliced prefix
    decoder; the dotted-path branch already does the nested assignment generically.
    """
    for section in _OVERRIDE_SECTIONS:
        prefix = section + "_"
        if key.startswith(prefix) and len(key) > len(prefix):
            return section, key[len(prefix):]
    return None, key


def build_from_preset(
    cls, preset_name: str, validate: bool = True, **overrides: Any
) -> "Any":
    """Create configuration from a named preset with optional overrides.

    Args:
        preset_name: Name of the preset (e.g., 'whisper_labse', 'fast_dev')
        validate: If True, validate configuration after creation. Defaults to True.
        **overrides: Key-value pairs to override preset values.
                    Nested values use underscore notation, e.g.,
                    model_asr_device='cpu' overrides model.asr_device.

    Returns:
        EvaluationConfig instance.

    Raises:
        ConfigurationError: If validate=True and validation fails.

    Example:
        config = EvaluationConfig.from_preset(
            'fast_dev',
            experiment_name='my_experiment',
            model_asr_device='cuda:1',
            data_batch_size=16
        )
    """
    from .model_presets import get_preset

    config_dict = get_preset(preset_name)

    # Apply overrides — dotted paths ("model.asr_device") take priority;
    # underscore notation is kept for backward compatibility.
    for key, value in overrides.items():
        if "." in key:
            # Dotted path: navigate / create nested dicts
            segments = key.split(".")
            d = config_dict
            for seg in segments[:-1]:
                if not isinstance(d.get(seg), dict):
                    d[seg] = {}
                d = d[seg]
            d[segments[-1]] = value
            continue

        # Legacy underscore notation → generic section/sub_key resolution (F13).
        section, sub_key = _resolve_underscore_override(key)
        if section is None:
            config_dict[key] = value
        else:
            config_dict.setdefault(section, {})[sub_key] = value

    return cls.from_dict(config_dict, validate=validate)
