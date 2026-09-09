"""Validation logic for :class:`EvaluationConfig`.

Holds ``validate`` and ``preflight_check`` bodies plus their device/model/data
sub-checks. :mod:`evaluator.config.evaluation` keeps thin delegators with the
public signatures; the heavy logic lives here.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from ..constants import ROLLUP_REDUCERS
from ..errors import ConfigurationError
from .base import (
    estimate_model_memory_gb,
    DEVICE_PATTERN,
)
from .node_config import KIND_KEEP, resolve_node_config
from . import evaluation as _evaluation
from .dataset_descriptors import get_descriptor, resolve_dataset_descriptor
from ..logging_config import get_logger
import evaluator.config as config_module

# Validation thresholds.
GPU_MEMORY_SAFETY_FRACTION = 0.9
LARGE_BATCH_SIZE_WARNING = 256

#: ``config -> error strings`` checks contributed by layers above config. The name checks that
#: need a registry (metric names, model types, corrector/combine-strategy names) live in
#: :mod:`evaluator.evaluation.validation`, which cannot be imported from here; it appends itself
#: on import instead, so every surface that calls :func:`collect_problems` — ``from_yaml(
#: validate=True)``, the builder's non-blocking validate — runs them exactly as before.
EXTRA_VALIDATORS: List[Callable[[Any], List[str]]] = []


def _detect_cuda():
    """Return (cuda_available, device_count); (False, 0) when torch is absent."""
    try:
        import torch

        if torch.cuda.is_available():
            return True, torch.cuda.device_count()
    except ImportError:
        get_logger(__name__).info(
            "torch not importable during config validation — assuming CPU-only "
            "(device checks will reject cuda devices)"
        )
    return False, 0


def device_fields(config: Any):
    return [
        ("model.asr_device", config.model.asr_device),
        ("model.text_emb_device", config.model.text_emb_device),
        ("model.audio_emb_device", config.model.audio_emb_device),
    ]


def validate(config: Any) -> List[str]:
    """Validate configuration and return list of warnings.

    Fatal errors raise ConfigurationError; non-fatal issues are returned as warnings.
    """
    errors, warnings = collect_problems(config)
    if errors:
        raise ConfigurationError(
            "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        )
    return warnings


def collect_problems(config: Any) -> Tuple[List[str], List[str]]:
    """Run all config checks and RETURN ``(errors, warnings)`` without raising — for the UI's
    non-blocking validate: the preview/forms load even with errors so the user can fix the nodes
    and validate again. :func:`validate` is the raising wrapper used on the run path."""
    errors: List[str] = []
    warnings: List[str] = []

    cuda_available, cuda_count = _detect_cuda()
    _validate_devices(config, errors, warnings, cuda_available, cuda_count)
    if cuda_available:
        _validate_gpu_memory(config, warnings, cuda_count)
    _validate_data_params(config, errors, warnings)
    _validate_dataset_compat(config, warnings)
    _validate_variant_rollup(config, errors)
    _validate_llm_concurrency(config, warnings)
    for extra in EXTRA_VALIDATORS:
        errors.extend(extra(config))
    return errors, warnings


#: Seconds of head-room a single in-flight LLM request is assumed to need. With N in flight the
#: server interleaves them, so each one's WALL time grows roughly N-fold.
_LLM_SECONDS_PER_WORKER = 60


# node_kind → (EvaluationConfig flat field, force_enabled) for every feature node a graph node
# can override — mirrors graph_config.FEATURE_NODE_CONFIG / config.node_config._FEATURE_BASES,
# kept as its own copy since both of those key off different objects (a RunState / a raw dict)
# and this one resolves straight off `config` at load time, before any RunState exists.
_FEATURE_NODE_KINDS = {
    "answer_judge": ("judge", True),
    "answer_gen": ("answer_generation", True),
    "query_optimization": ("query_optimization", False),
    "query_refine": ("query_optimization", False),
    "multi_query_retrieval": ("query_optimization", False),
    "query_correction": ("query_correction", False),
}


def configured_feature_configs(config, flat_attr):
    """The flat default (once, if set) plus every graph node's resolved override for whichever
    feature ``flat_attr`` names (e.g. ``"judge"``) — so a node-level override that enables a
    feature (or sets its own ``api_key_env``/``concurrency``/``timeout_s``) is caught here, not
    only the flat default. Graph-less configs just yield the flat default (``config.graph``
    raises for those — caught, not propagated: a config with no graph can't run either way)."""
    from ..pipeline.graph.operators import node_kind

    flat = getattr(config, flat_attr, None)
    if flat is not None:
        yield flat
    try:
        nodes = config.graph.nodes
    except Exception:
        return
    for node in nodes:
        kind = node_kind(node.stage, node.params)
        entry = _FEATURE_NODE_KINDS.get(kind)
        if entry is None or entry[0] != flat_attr:
            continue
        cfg = resolve_node_config(
            flat, node.params, force_enabled=entry[1],
            keep_on_node=KIND_KEEP.get(kind, frozenset()),
        )
        if cfg is not None:
            yield cfg


def _validate_llm_concurrency(config, warnings):
    """Warn when a component's concurrency inflates its effective wait past its timeout_s
    (each request's wall time grows with concurrency, so a timeout sized for serial calls
    storms under load)."""
    seen = set()
    for name in ("answer_generation", "judge", "query_optimization", "query_correction"):
        for cfg in configured_feature_configs(config, name):
            if not getattr(cfg, "enabled", False):
                continue
            workers = int(getattr(cfg, "concurrency", 1) or 1)
            timeout = int(getattr(cfg, "timeout_s", 0) or 0)
            needed = workers * _LLM_SECONDS_PER_WORKER
            key = (name, workers, timeout)
            if workers > 1 and timeout < needed and key not in seen:
                seen.add(key)
                warnings.append(
                    f"{name}.concurrency={workers} with timeout_s={timeout}: each request's wall "
                    f"time grows with concurrency, so this will time out and retry under load. "
                    f"Use timeout_s >= {needed}, or lower the concurrency."
                )


def _validate_variant_rollup(config, errors):
    """`variant_rollup:` must name a known reducer."""
    how = getattr(config, "variant_rollup", "mean")
    if how not in ROLLUP_REDUCERS:
        errors.append(
            f"unknown variant_rollup '{how}'. Valid: {', '.join(sorted(ROLLUP_REDUCERS))}"
        )


def _validate_devices(config, errors, warnings, cuda_available, cuda_count):
    """Check device-string format and CUDA availability."""
    for field_name, device in device_fields(config):
        if not DEVICE_PATTERN.match(device):
            errors.append(
                f"Invalid device format for {field_name}: '{device}'. "
                f"Expected 'cpu', 'cuda', 'cuda:N', or 'mps'. "
                f"Tip: call with_auto_devices() to auto-select valid devices."
            )
        elif device.startswith("cuda"):
            if not cuda_available:
                warnings.append(
                    f"{field_name} is set to '{device}' but CUDA is not available. "
                    f"Consider using 'cpu' or calling with_auto_devices()."
                )
            elif ":" in device and int(device.split(":")[1]) >= cuda_count:
                import os

                # CUDA_VISIBLE_DEVICES is what usually explains a surprising count (a launcher
                # or scheduler narrowing the set), so name it rather than leaving a bare number.
                visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                warnings.append(
                    f"{field_name} is set to '{device}' but only {cuda_count} "
                    f"CUDA device(s) available."
                    + (f" CUDA_VISIBLE_DEVICES={visible!r}." if visible else "")
                )


def _validate_gpu_memory(config, warnings, cuda_count):
    """Warn when estimated per-GPU memory exceeds 90% of capacity."""
    device_memory_usage: Dict[int, float] = {}
    model_assignments = [
        (config.model.asr_device, "asr", config.model.asr_model_type),
        (
            config.model.text_emb_device,
            "text_embedding",
            config.model.text_emb_model_type,
        ),
        (
            config.model.audio_emb_device,
            "audio_embedding",
            config.model.audio_emb_model_type,
        ),
    ]
    for device_str, category, model_type in model_assignments:
        if device_str.startswith("cuda"):
            device_idx = int(device_str.split(":")[1]) if ":" in device_str else 0
            estimated_mem = estimate_model_memory_gb(category, model_type)
            device_memory_usage[device_idx] = (
                device_memory_usage.get(device_idx, 0.0) + estimated_mem
            )
    # Resolve via the evaluation module so test patches of
    # ``evaluator.config.evaluation.get_gpu_memory_gb`` take effect.

    for device_idx, estimated_total in device_memory_usage.items():
        if device_idx < cuda_count:
            gpu_memory = _evaluation.get_gpu_memory_gb(device_idx)
            if (
                gpu_memory is not None
                and estimated_total > gpu_memory * GPU_MEMORY_SAFETY_FRACTION
            ):
                warnings.append(
                    f"GPU {device_idx}: estimated memory usage ({estimated_total:.1f}GB) "
                    f"exceeds 90% of available memory ({gpu_memory:.1f}GB). "
                    f"Consider distributing models across devices."
                )


def _validate_data_params(config, errors, warnings):
    """Check data paths, batch size, and k bounds."""
    if (
        config.data.questions_path is not None
        and not Path(config.data.questions_path).exists()
    ):
        errors.append(
            f"Questions path does not exist: '{config.data.questions_path}'. "
            f"Tip: verify file path, or set data.prepared_dataset_dir for prepared datasets."
        )
    if (
        config.data.corpus_path is not None
        and not Path(config.data.corpus_path).exists()
    ):
        errors.append(
            f"Corpus path does not exist: '{config.data.corpus_path}'. "
            f"Tip: verify file path and ensure corpus JSON/JSONL file exists."
        )
    if config.data.batch_size <= 0:
        errors.append(
            f"Batch size must be positive, got: {config.data.batch_size}. "
            f"Tip: start with batch_size=16 or batch_size=32."
        )
    elif config.data.batch_size > LARGE_BATCH_SIZE_WARNING:
        warnings.append(
            f"Very large batch size ({config.data.batch_size}) may cause memory issues."
        )
    if config.vector_db.k <= 0:
        errors.append(
            f"vector_db.k must be positive, got: {config.vector_db.k}. "
            f"Tip: use k=5 or k=10 for most evaluations."
        )
    if config.vector_db.reranker_top_k <= 0:
        errors.append(
            f"vector_db.reranker_top_k must be positive, got: {config.vector_db.reranker_top_k}. "
            f"Tip: use reranker_top_k=20 or reranker_top_k=50."
        )
    # Multi-source configs keep paths under data.datasets[role] (builder/Config&Run shape);
    # check those too so an edited path is caught.
    for role, entry in (getattr(config.data, "datasets", None) or {}).items():
        get = entry.get if isinstance(entry, dict) else (lambda k: getattr(entry, k, None))
        for key, label in (("questions_path", "Questions"), ("corpus_path", "Corpus")):
            path = get(key)
            if path is not None and not Path(path).exists():
                errors.append(f"{label} path does not exist: '{path}' (dataset '{role}').")


def _validate_dataset_compat(config, warnings):
    """Warn when the dataset doesn't support audio generation. Resolves the real descriptor
    from the single ``data.datasets[role]`` entry (builder/Config&Run shape); a genuinely
    multi-source map (>1 entry) has no single descriptor and is skipped."""
    try:
        datasets = getattr(config.data, "datasets", None)
        if datasets:
            if len(datasets) > 1:
                return  # genuinely multi-source: no single descriptor to check against
            entry = next(iter(datasets.values()))
            name = entry.get("dataset_name") if isinstance(entry, dict) \
                else getattr(entry, "dataset_name", None)
            descriptor = get_descriptor(name) if name else None
        else:
            descriptor = resolve_dataset_descriptor(config.data)
        if descriptor is None:
            return
        if config.audio_synthesis.enabled and not descriptor.supports_generation:
            warnings.append(
                f"audio_synthesis.enabled=True but dataset '{descriptor.id}' does not "
                f"support audio generation (supports_generation=False)."
            )
    except (ConfigurationError, ImportError, AttributeError, ValueError) as exc:
        # Unresolvable / unregistered dataset: skip the compat check rather than block
        # validation, but log it so a silently-skipped check is visible.
        get_logger(__name__).debug("dataset compat check skipped: %s", exc)


def preflight_check(config: Any) -> List[str]:
    """Perform comprehensive pre-flight validation before evaluation starts (fail fast).

    Unlike ``config.validate()``, this always returns warnings for issues it discovers
    itself — but still raises ``ConfigurationError`` when the underlying ``validate()``
    call hits a fatal error.

    Returns:
        [] when all checks pass.
    """
    warnings = config.validate()

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_available else 0
    except ImportError:
        cuda_available = False
        cuda_count = 0
        warnings.append("PyTorch not available. GPU features will not work.")

    if cuda_available:
        device_assignments = []

        if config.model.asr_device.startswith("cuda") and config.model.asr_model_type:
            device_idx = (
                int(config.model.asr_device.split(":")[1])
                if ":" in config.model.asr_device
                else 0
            )
            device_assignments.append((device_idx, "asr", config.model.asr_model_type))

        if (
            config.model.text_emb_device.startswith("cuda")
            and config.model.text_emb_model_type
        ):
            device_idx = (
                int(config.model.text_emb_device.split(":")[1])
                if ":" in config.model.text_emb_device
                else 0
            )
            device_assignments.append(
                (device_idx, "text_embedding", config.model.text_emb_model_type)
            )

        if (
            config.model.audio_emb_device.startswith("cuda")
            and config.model.audio_emb_model_type
        ):
            device_idx = (
                int(config.model.audio_emb_device.split(":")[1])
                if ":" in config.model.audio_emb_device
                else 0
            )
            device_assignments.append(
                (device_idx, "audio_embedding", config.model.audio_emb_model_type)
            )

        device_usage: Dict[int, float] = {}
        for device_idx, category, model_type in device_assignments:
            estimated = estimate_model_memory_gb(category, model_type)
            device_usage[device_idx] = device_usage.get(device_idx, 0.0) + estimated

        for device_idx, estimated_total in device_usage.items():
            if device_idx < cuda_count:
                free_memory = config_module.get_gpu_free_memory_gb(device_idx)
                if free_memory is not None and estimated_total > free_memory:
                    warnings.append(
                        f"GPU {device_idx}: estimated model memory ({estimated_total:.1f}GB) "
                        f"exceeds current free memory ({free_memory:.1f}GB). "
                        f"Free up GPU memory or use a different device."
                    )

    if config.graph_label == "asr_text_retrieval":
        if config.model.asr_model_type is None:
            warnings.append(
                "pipeline_mode is 'asr_text_retrieval' but asr_model_type is None. "
                "Set an ASR model or change pipeline_mode."
            )
        if config.model.text_emb_model_type is None:
            warnings.append(
                "pipeline_mode is 'asr_text_retrieval' but text_emb_model_type is None. "
                "Set a text embedding model or change pipeline_mode."
            )
    elif config.graph_label == "audio_emb_retrieval":
        if config.model.audio_emb_model_type is None:
            warnings.append(
                "pipeline_mode is 'audio_emb_retrieval' but audio_emb_model_type is None. "
                "Set an audio embedding model or change pipeline_mode."
            )

    import os

    # Check the flat default AND every graph node's resolved override — a node override
    # that enables the feature or points at a different api_key_env is otherwise invisible.
    _checked_envs: set = set()

    for cfg in configured_feature_configs(config, "judge"):
        if cfg.enabled and cfg.api_key_env not in _checked_envs:
            _checked_envs.add(cfg.api_key_env)
            if not os.environ.get(cfg.api_key_env):
                warnings.append(
                    f"LLM judge is enabled but {cfg.api_key_env} environment variable "
                    f"is not set. Judge scoring will fail."
                )

    for cfg in configured_feature_configs(config, "answer_generation"):
        if cfg.enabled and cfg.api_key_env not in _checked_envs:
            _checked_envs.add(cfg.api_key_env)
            if not os.environ.get(cfg.api_key_env):
                warnings.append(
                    f"answer_generation is enabled but {cfg.api_key_env} "
                    f"environment variable is not set. Answer generation will fail."
                )

    for cfg in configured_feature_configs(config, "query_optimization"):
        if cfg.enabled and cfg.api_key_env not in _checked_envs:
            _checked_envs.add(cfg.api_key_env)
            if not os.environ.get(cfg.api_key_env):
                warnings.append(
                    f"query_optimization is enabled but {cfg.api_key_env} "
                    f"environment variable is not set. Query optimization will fail."
                )

    return warnings
