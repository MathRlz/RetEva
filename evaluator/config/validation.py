"""Validation logic for :class:`EvaluationConfig`.

Holds ``validate`` and ``preflight_check`` bodies plus their device/model/data
sub-checks. :mod:`evaluator.config.evaluation` keeps thin delegators with the
public signatures; the heavy logic lives here.
"""

from pathlib import Path
from typing import Any, Dict, List

from ..config.types import enum_to_str
from ..errors import ConfigurationError
from .base import (
    estimate_model_memory_gb,
    get_text_embedding_dim,
    DEVICE_PATTERN,
)

# Validation thresholds.
GPU_MEMORY_SAFETY_FRACTION = 0.9
LARGE_BATCH_SIZE_WARNING = 256


def _detect_cuda():
    """Return (cuda_available, device_count); (False, 0) when torch is absent."""
    try:
        import torch

        if torch.cuda.is_available():
            from ..devices.capability import usable_gpu_count

            return True, usable_gpu_count()
    except ImportError:
        pass
    return False, 0


def device_fields(config: Any):
    return [
        ("model.asr_device", config.model.asr_device),
        ("model.text_emb_device", config.model.text_emb_device),
        ("model.audio_emb_device", config.model.audio_emb_device),
    ]


def validate(config: Any) -> List[str]:
    """Validate configuration and return list of warnings.

    Checks for common configuration errors before evaluation starts.
    Fatal errors raise ConfigurationError, non-fatal issues are returned
    as warnings.

    Returns:
        List of warning messages for non-fatal issues.

    Raises:
        ConfigurationError: If validation fails with unrecoverable errors.
    """
    errors: List[str] = []
    warnings: List[str] = []

    cuda_available, cuda_count = _detect_cuda()
    _validate_devices(config, errors, warnings, cuda_available, cuda_count)
    _validate_model_types(config, errors)
    _validate_embedding_dims(config, warnings)
    if cuda_available:
        _validate_gpu_memory(config, warnings, cuda_count)
    _validate_data_params(config, errors, warnings)
    _validate_dataset_compat(config, warnings)

    if errors:
        raise ConfigurationError(
            "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        )
    return warnings


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
                warnings.append(
                    f"{field_name} is set to '{device}' but only {cuda_count} "
                    f"CUDA device(s) available."
                )


def _validate_model_types(config, errors):
    """Check each configured model type is registered."""
    from ..models.registry import (
        asr_registry,
        text_embedding_registry,
        audio_embedding_registry,
    )

    checks = [
        (config.model.asr_model_type, asr_registry, "ASR"),
        (config.model.text_emb_model_type, text_embedding_registry, "text embedding"),
        (
            config.model.audio_emb_model_type,
            audio_embedding_registry,
            "audio embedding",
        ),
    ]
    for model_type, registry, label in checks:
        if model_type is not None and not registry.is_registered(model_type):
            available = ", ".join(registry.list_types())
            errors.append(
                f"Unknown {label} model type: '{model_type}'. "
                f"Available types: {available}. "
                f"Tip: use one listed type or call EvaluationConfig.from_preset(...)."
            )


def _validate_embedding_dims(config, warnings):
    """Warn when audio/text embedding dimensions disagree."""
    if config.model.pipeline_mode != "audio_emb_retrieval":
        return
    text_emb_dim = get_text_embedding_dim(config.model.text_emb_model_type)
    audio_emb_dim = config.model.audio_emb_dim
    if text_emb_dim is not None and audio_emb_dim != text_emb_dim:
        warnings.append(
            f"Embedding dimension mismatch: audio_emb_dim ({audio_emb_dim}) != "
            f"text embedding dim for '{config.model.text_emb_model_type}' ({text_emb_dim}). "
            f"Ensure your audio embedding model projects to the correct dimension."
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
    from . import evaluation as _evaluation

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


def _validate_dataset_compat(config, warnings):
    """Warn when the dataset doesn't support the chosen pipeline mode."""
    try:
        from ..datasets.descriptor import resolve_dataset_descriptor

        descriptor = resolve_dataset_descriptor(config.data)
        mode = enum_to_str(config.model.pipeline_mode)
        if not descriptor.supports_pipeline_mode(mode):
            warnings.append(
                f"Pipeline mode '{mode}' is not in the compatible modes for dataset "
                f"'{descriptor.id}': {', '.join(descriptor.compatible_pipeline_modes)}. "
                f"Results may be incorrect."
            )
        if config.audio_synthesis.enabled and not descriptor.supports_generation:
            warnings.append(
                f"audio_synthesis.enabled=True but dataset '{descriptor.id}' does not "
                f"support audio generation (supports_generation=False)."
            )
    except Exception:
        pass  # Unresolvable dataset; skip check rather than block validation


def preflight_check(config: Any) -> List[str]:
    """Perform comprehensive pre-flight validation before evaluation starts.

    This function runs all validation checks and returns warnings. It's designed
    to be called before evaluation to catch configuration issues early (fail fast).

    Unlike config.validate(), this function:
    - Always returns warnings (never raises for warnings)
    - Performs additional runtime environment checks
    - Checks GPU memory availability in real-time

    Args:
        config: The evaluation configuration to check.

    Returns:
        List of warning messages. Empty list indicates all checks passed.

    Raises:
        ConfigurationError: If validation fails with unrecoverable errors.

    Example:
        >>> config = EvaluationConfig.from_preset("whisper_labse")
        >>> warnings = preflight_check(config)
        >>> if warnings:
        ...     for w in warnings:
        ...         print(f"Warning: {w}")
        >>> # Proceed with evaluation...
    """
    # Run standard validation first (may raise ConfigurationError)
    warnings = config.validate()

    # Additional runtime environment checks
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            from ..devices.capability import usable_gpu_count

            cuda_count = usable_gpu_count()
        else:
            cuda_count = 0
    except ImportError:
        cuda_available = False
        cuda_count = 0
        warnings.append("PyTorch not available. GPU features will not work.")

    # Check real-time GPU memory availability
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

        # Check each device's current free memory
        device_usage: Dict[int, float] = {}
        for device_idx, category, model_type in device_assignments:
            estimated = estimate_model_memory_gb(category, model_type)
            device_usage[device_idx] = device_usage.get(device_idx, 0.0) + estimated

        for device_idx, estimated_total in device_usage.items():
            if device_idx < cuda_count:
                import evaluator.config as config_module

                free_memory = config_module.get_gpu_free_memory_gb(device_idx)
                if free_memory is not None and estimated_total > free_memory:
                    warnings.append(
                        f"GPU {device_idx}: estimated model memory ({estimated_total:.1f}GB) "
                        f"exceeds current free memory ({free_memory:.1f}GB). "
                        f"Free up GPU memory or use a different device."
                    )

    # Validate pipeline mode consistency
    if config.model.pipeline_mode == "asr_text_retrieval":
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
    elif config.model.pipeline_mode == "audio_emb_retrieval":
        if config.model.audio_emb_model_type is None:
            warnings.append(
                "pipeline_mode is 'audio_emb_retrieval' but audio_emb_model_type is None. "
                "Set an audio embedding model or change pipeline_mode."
            )

    # Warn about AdvancedRetrievalConfig stub fields — declared but not implemented
    _UNIMPLEMENTED_RETRIEVAL_FEATURES = {
        "multi_vector_enabled": "multi-vector retrieval",
        "query_expansion_enabled": "query expansion",
        "pseudo_feedback_enabled": "pseudo-relevance feedback",
        "adaptive_fusion_enabled": "adaptive fusion",
    }
    for field, label in _UNIMPLEMENTED_RETRIEVAL_FEATURES.items():
        if getattr(config.vector_db, field, False):
            warnings.append(
                f"vector_db.{field}=True but {label} is not implemented in any "
                f"retrieval backend. The setting will have no effect."
            )

    import os

    # Check for judge API key if enabled
    if config.judge.enabled:
        api_key = os.environ.get(config.judge.api_key_env)
        if not api_key:
            warnings.append(
                f"LLM judge is enabled but {config.judge.api_key_env} environment variable "
                f"is not set. Judge scoring will fail."
            )

    # Check for answer_generation API key if enabled
    if config.answer_generation.enabled:
        api_key = os.environ.get(config.answer_generation.api_key_env)
        if not api_key:
            warnings.append(
                f"answer_generation is enabled but {config.answer_generation.api_key_env} "
                f"environment variable is not set. Answer generation will fail."
            )

    # Check for query_optimization API key if enabled
    if config.query_optimization.enabled:
        api_key = os.environ.get(config.query_optimization.api_key_env)
        if not api_key:
            warnings.append(
                f"query_optimization is enabled but {config.query_optimization.api_key_env} "
                f"environment variable is not set. Query optimization will fail."
            )

    return warnings
