"""Validation logic for :class:`EvaluationConfig`.

Holds ``validate`` and ``preflight_check`` bodies plus their device/model/data
sub-checks. :mod:`evaluator.config.evaluation` keeps thin delegators with the
public signatures; the heavy logic lives here.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

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
            return True, torch.cuda.device_count()
    except ImportError:
        from ..logging_config import get_logger

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

    Checks for common configuration errors before evaluation starts.
    Fatal errors raise ConfigurationError, non-fatal issues are returned
    as warnings.

    Returns:
        List of warning messages for non-fatal issues.

    Raises:
        ConfigurationError: If validation fails with unrecoverable errors.
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
    _validate_template(config, errors)
    _validate_devices(config, errors, warnings, cuda_available, cuda_count)
    _validate_model_types(config, errors)
    _validate_embedding_dims(config, warnings)
    if cuda_available:
        _validate_gpu_memory(config, warnings, cuda_count)
    _validate_data_params(config, errors, warnings)
    _validate_dataset_compat(config, warnings)
    _validate_metric_allowlist(config, errors)
    _validate_variant_rollup(config, errors)
    _validate_llm_concurrency(config, warnings)
    return errors, warnings


#: Seconds of head-room a single in-flight LLM request is assumed to need. With N in flight the
#: server interleaves them, so each one's WALL time grows roughly N-fold.
_LLM_SECONDS_PER_WORKER = 60


# node_kind → (EvaluationConfig flat field, force_enabled) for every feature node a graph node
# can override — mirrors graph_config._FEATURE_NODE_CONFIG / evaluation.node_config._FEATURE_BASES,
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


def _configured_feature_configs(config, flat_attr):
    """The flat default (once, if set) plus every graph node's resolved override for whichever
    feature ``flat_attr`` names (e.g. ``"judge"``) — so a node-level override that enables a
    feature (or sets its own ``api_key_env``/``concurrency``/``timeout_s``) is caught here, not
    only the flat default. Graph-less configs just yield the flat default (``config.graph``
    raises for those — caught, not propagated; see ``_configured_model_types``)."""
    from ..evaluation.node_config import _KIND_KEEP, resolve_node_config
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
            keep_on_node=_KIND_KEEP.get(kind, frozenset()),
        )
        if cfg is not None:
            yield cfg


def _validate_llm_concurrency(config, warnings):
    """Warn when a component's timeout was sized for serial calls but it runs concurrently.

    A timeout that is fine at concurrency 1 becomes a timeout STORM at 4: every request takes
    ~N x longer, each timeout retries twice, and the retries pile onto an already-saturated
    server. Cheap to check, expensive to discover at hour two of a sweep."""
    seen = set()
    for name in ("answer_generation", "judge", "query_optimization", "query_correction"):
        for cfg in _configured_feature_configs(config, name):
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
    """`variant_rollup:` (A1) must name a known reducer."""
    how = getattr(config, "variant_rollup", "mean")
    from ..evaluation.aggregate import _ROLLUP_REDUCERS

    if how not in _ROLLUP_REDUCERS:
        errors.append(
            f"unknown variant_rollup '{how}'. Valid: {', '.join(sorted(_ROLLUP_REDUCERS))}"
        )


def _validate_metric_allowlist(config, errors):
    """The `metrics:` allowlist (B1) must name registered metrics — caught at validate time
    with the registered list named, not as a silent no-op at report time."""
    names = getattr(config, "metrics", None)
    if not names:
        return
    from ..evaluation.metric_registry import list_metrics

    registered = {spec.name for spec in list_metrics()}
    unknown = [n for n in names if n not in registered]
    if unknown:
        errors.append(
            f"unknown metric(s) in `metrics:` allowlist: {', '.join(sorted(unknown))}. "
            f"Registered: {', '.join(sorted(registered))}"
        )


def _validate_template(config, errors):
    """A graph-template reference must name a known template — caught at validate time (with the
    available list) instead of only when the graph is built. An explicit ``graph.nodes`` wins, so
    skip the check then."""
    override = getattr(config, "graph_override", None) or {}
    template = override.get("template")
    if not template or override.get("nodes"):
        return
    from ..pipeline.graph.modes import GRAPH_TEMPLATE_SPECS

    if template not in GRAPH_TEMPLATE_SPECS:
        available = ", ".join(sorted(GRAPH_TEMPLATE_SPECS))
        errors.append(f"unknown graph template '{template}'. Available templates: {available}")


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


_KIND_FAMILY = {"asr": "asr", "text_embedding": "text_emb", "audio_embedding": "audio_emb"}


def _configured_model_types(config):
    """Yield (family, model_type) for the flat default (always, all three families) plus every
    graph node's resolved per-node override — so a branch-only model type (never promoted into
    the flat default, per the node-centric fold) fails here at validate time instead of only
    surfacing as a factory.py build crash. Graph-less configs (no explicit ``graph.nodes``, no
    template) simply contribute only the flat triple — ``config.graph`` raises for those, caught
    here rather than propagated (a config with no graph can't run either way; validate()
    shouldn't crash discovering that)."""
    yield "asr", config.model.asr_model_type
    yield "text_emb", config.model.text_emb_model_type
    yield "audio_emb", config.model.audio_emb_model_type

    from .graph_config import resolved_model_config
    from ..pipeline.graph.operators import node_kind

    try:
        nodes = config.graph.nodes
    except Exception:
        return
    for node in nodes:
        family = _KIND_FAMILY.get(node_kind(node.stage, node.params))
        if family is None:
            continue
        mcfg = resolved_model_config(config, node)
        yield family, getattr(mcfg, f"{family}_model_type")


def _validate_model_types(config, errors):
    """Check every configured model type (the flat default AND any per-node branch override)
    is registered, deduped by (family, model_type)."""
    from ..models.registry import (
        asr_registry,
        text_embedding_registry,
        audio_embedding_registry,
    )

    registries = {
        "asr": (asr_registry, "ASR"),
        "text_emb": (text_embedding_registry, "text embedding"),
        "audio_emb": (audio_embedding_registry, "audio embedding"),
    }
    seen = set()
    for family, model_type in _configured_model_types(config):
        if model_type is None or (family, model_type) in seen:
            continue
        seen.add((family, model_type))
        registry, label = registries[family]
        if not registry.is_registered(model_type):
            available = ", ".join(registry.list_types())
            errors.append(
                f"Unknown {label} model type: '{model_type}'. "
                f"Available types: {available}. "
                f"Tip: use one listed type or call EvaluationConfig.from_preset(...)."
            )


def _validate_embedding_dims(config, warnings):
    """Warn when audio/text embedding dimensions disagree."""
    if config.graph_template != "audio_emb_retrieval":
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
    # Multi-source configs keep paths under data.datasets[role] (the builder/Config&Run shape),
    # not the top-level data.*; validate those entries too so an edited path is checked.
    for role, entry in (getattr(config.data, "datasets", None) or {}).items():
        get = entry.get if isinstance(entry, dict) else (lambda k: getattr(entry, k, None))
        for key, label in (("questions_path", "Questions"), ("corpus_path", "Corpus")):
            path = get(key)
            if path is not None and not Path(path).exists():
                errors.append(f"{label} path does not exist: '{path}' (dataset '{role}').")


def _validate_dataset_compat(config, warnings):
    """Warn when the dataset doesn't support audio generation.

    The builder & Config&Run shape keeps the dataset under ``data.datasets[role]`` (a 1-entry map
    for a single-source graph) — resolve the *real* descriptor from the single map entry.
    Genuinely multi-source maps (>1 entry) have no single descriptor — skipped.

    No "pipeline mode compatibility" check here (dropped): the graph's own nodes/edges are
    the source of truth for what a run can do — a per-dataset-type hand-maintained
    "compatible modes" allowlist was a second, independently-drifting classification on top
    of that, and it went stale (e.g. pubmed_qa's override omitted audio_emb_retrieval even
    though pubmed_qa_rag_fulltext_apm.yaml runs it correctly via TTS gap-fill), producing
    false "Results may be incorrect" warnings on working graphs."""
    try:
        from ..datasets.descriptor import get_descriptor, resolve_dataset_descriptor

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
        # validation, but log it so a silently-skipped check is visible (C1/F20).
        from ..logging_config import get_logger

        get_logger(__name__).debug("dataset compat check skipped: %s", exc)


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
        cuda_count = torch.cuda.device_count() if cuda_available else 0
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
    if config.graph_template == "asr_text_retrieval":
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
    elif config.graph_template == "audio_emb_retrieval":
        if config.model.audio_emb_model_type is None:
            warnings.append(
                "pipeline_mode is 'audio_emb_retrieval' but audio_emb_model_type is None. "
                "Set an audio embedding model or change pipeline_mode."
            )

    import os

    # API-key-env checks: the flat default AND every graph node's resolved override (a
    # node-level override that enables the feature, or points at a different api_key_env, is
    # invisible to a check that only looks at the flat default — see _configured_feature_configs).
    _checked_envs: set = set()

    # Check for judge API key if enabled
    for cfg in _configured_feature_configs(config, "judge"):
        if cfg.enabled and cfg.api_key_env not in _checked_envs:
            _checked_envs.add(cfg.api_key_env)
            if not os.environ.get(cfg.api_key_env):
                warnings.append(
                    f"LLM judge is enabled but {cfg.api_key_env} environment variable "
                    f"is not set. Judge scoring will fail."
                )

    # Check for answer_generation API key if enabled
    for cfg in _configured_feature_configs(config, "answer_generation"):
        if cfg.enabled and cfg.api_key_env not in _checked_envs:
            _checked_envs.add(cfg.api_key_env)
            if not os.environ.get(cfg.api_key_env):
                warnings.append(
                    f"answer_generation is enabled but {cfg.api_key_env} "
                    f"environment variable is not set. Answer generation will fail."
                )

    # Check for query_optimization API key if enabled
    for cfg in _configured_feature_configs(config, "query_optimization"):
        if cfg.enabled and cfg.api_key_env not in _checked_envs:
            _checked_envs.add(cfg.api_key_env)
            if not os.environ.get(cfg.api_key_env):
                warnings.append(
                    f"query_optimization is enabled but {cfg.api_key_env} "
                    f"environment variable is not set. Query optimization will fail."
                )

    return warnings
