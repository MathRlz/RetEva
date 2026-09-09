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
from ..logging_config import get_logger
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
from .subconfigs import FEATURE_SUBCONFIGS
from .graph_config import GraphConfigError, build_evaluation_config_kwargs
from .model_presets import get_preset


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
    ``ConfigurationError``: a misspelled key (e.g. ``asr_modal_type``) must fail loudly
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


def build_from_dict(cls, config_dict: Dict[str, Any], *, validate: bool = True) -> "Any":
    """Create configuration from a dictionary.

    Returns:
        EvaluationConfig instance.

    Raises:
        ConfigurationError: If validate=True and validation fails.
    """

    config_dict = dict(config_dict)

    # Back-compat: model.pipeline_mode no longer exists (the graph is the spec); drop it
    # before ModelConfig construction, which would otherwise reject the unknown key.
    _model = config_dict.get("model")
    if isinstance(_model, dict) and "pipeline_mode" in _model:
        _model = dict(_model)
        _model.pop("pipeline_mode", None)
        config_dict["model"] = _model

    # runtime:/experiment: blocks flatten into the top level before sub-config construction.
    _merge_section_into(config_dict, config_dict.pop("runtime", None))
    _merge_section_into(config_dict, config_dict.pop("experiment", None))

    if "features" in config_dict:
        raise ConfigurationError(
            "The `features:` block is no longer supported — a capability is enabled by its node in "
            "the graph. Put its settings on the node (e.g. `{id: tts, type: tts, params: {...}}`) "
            "or as a top-level block (audio_synthesis:, judge:, …). See any configs/*.yaml."
        )

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
        "seed",
        "concurrency",
    }
    # A component config that does not declare a propagated field (e.g. an older one without
    # `seed`) simply doesn't receive it — never a constructor error.
    _llm_base = {
        k: v for k, v in vars(llm_config).items() if k in _llm_fields
    }

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

    # Optional capability sub-configs, each a top-level EvaluationConfig field. The three
    # LLM-derived features inherit the shared backend defaults; the rest are plain.
    sub_configs["audio_synthesis"] = AudioSynthesisConfig(
        **(config_dict.get("audio_synthesis") or {})
    )
    sub_configs["augmentation"] = AudioAugmentationConfig(
        **(config_dict.get("augmentation") or {})
    )
    sub_configs["answer_generation"] = AnswerGenerationConfig(
        **_merge_llm(config_dict.get("answer_generation") or {})
    )
    sub_configs["judge"] = JudgeConfig(**_merge_llm(config_dict.get("judge") or {}))
    sub_configs["query_optimization"] = QueryOptimizationConfig(**_merge_llm(_qo_raw))
    sub_configs["query_correction"] = QueryCorrectionConfig(
        **_merge_llm(config_dict.get("query_correction") or {})
    )
    sub_configs["embedding_fusion"] = EmbeddingFusionConfig(
        **(config_dict.get("embedding_fusion") or {})
    )
    sub_configs["rag"] = RagFlowConfig(**(config_dict.get("rag") or {}))

    device_pool_dict = config_dict.get("device_pool")
    sub_configs["device_pool"] = (
        DevicePoolConfig(**device_pool_dict) if device_pool_dict else None
    )

    # Remaining keys are scalar EvaluationConfig fields; exclude feature keys consumed above.
    _consumed = set(sub_configs) | set(FEATURE_SUBCONFIGS)
    main_config = {k: v for k, v in config_dict.items() if k not in _consumed}

    config = _construct_subconfig(cls, {**main_config, **sub_configs}, "")

    if validate:
        for warning in config.validate():
            # Surface non-fatal config warnings (dim mismatch, dataset compat, …) at load.
            get_logger(__name__).warning("config: %s", warning)

    return config


def build_from_yaml(cls, yaml_path: str, validate: bool = True) -> "Any":
    """Load configuration from a node-centric YAML file.

    The on-disk schema is the node-centric shape (experiment/dataset/graph/nodes/
    runtime); ``build_evaluation_config_kwargs`` translates it before construction.
    Legacy-shape keys still pass through, so this is the single load chokepoint for
    the CLI, public API, and presets.

    Returns:
        EvaluationConfig instance.

    Raises:
        ConfigurationError: If validate=True and validation fails.
    """

    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)
    if config_dict is None:
        raise ConfigurationError("Empty configuration file.")
    try:
        kwargs = build_evaluation_config_kwargs(_expand_env(config_dict))
    except GraphConfigError as exc:
        # Name the file: a translate failure otherwise reads as a bare schema complaint with
        # no clue which of a campaign's configs produced it.
        raise GraphConfigError(f"{yaml_path}: {exc}") from exc
    return cls.from_dict(
        kwargs, validate=validate
    )


def _expand_env(value: Any) -> Any:
    """Expand ``${VAR}``/``$VAR`` in every string of a config tree; an unset variable is
    left verbatim (``os.path.expandvars`` semantics) so the downstream path check reports
    the literal, un-substituted path that names what was missing."""
    import os

    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


# Config sections addressable by the legacy underscore override notation. Matched in
# declaration order; "vector_db_k" resolves to the vector_db section simply because no
# "vector" section exists. A key matching no section is set at the top level.
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
    """Map a legacy ``section_subkey`` override to ``(section, sub_key)``.

    Returns ``(None, key)`` for a plain top-level key.
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

    ``**overrides`` use underscore or dotted notation for nested values (e.g.
    ``model_asr_device`` or ``model.asr_device``) to override a preset's nested fields.

    Returns:
        EvaluationConfig instance.

    Raises:
        ConfigurationError: If validate=True and validation fails.
    """

    config_dict = get_preset(preset_name)

    # Apply overrides — dotted paths ("model.asr_device") take priority;
    # underscore notation is kept for backward compatibility.
    for key, value in overrides.items():
        if "." in key:
            segments = key.split(".")
            d = config_dict
            for seg in segments[:-1]:
                if not isinstance(d.get(seg), dict):
                    d[seg] = {}
                d = d[seg]
            d[segments[-1]] = value
            continue

        # Legacy underscore notation → generic section/sub_key resolution.
        section, sub_key = _resolve_underscore_override(key)
        if section is None:
            config_dict[key] = value
        else:
            config_dict.setdefault(section, {})[sub_key] = value

    return cls.from_dict(config_dict, validate=validate)
