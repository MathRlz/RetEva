"""Evaluation configuration.

This module owns the :class:`EvaluationConfig` dataclass and its public method
signatures. The heavy method bodies live in focused sibling modules:

- :mod:`evaluator.config.loading` — ``from_dict`` / ``from_yaml`` / ``from_preset``
- :mod:`evaluator.config.validation` — ``validate`` / ``preflight_check``
- :mod:`evaluator.config.serialization` — ``to_dict`` / ``to_yaml`` / ``_serialize_*``

The methods here are thin delegators so the public API is unchanged.
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Dict, Any, List

from ..config.types import enum_to_str

from .cache import CacheConfig
from .logging import LoggingConfig
from .model import ModelConfig
from .data import DataConfig
from .audio_synthesis import AudioSynthesisConfig
from .audio_augmentation import AudioAugmentationConfig
from .answer_generation import AnswerGenerationConfig
from .llm_backend import LLMConfig
from .llm_server import LLMServerConfig
from .judge import JudgeConfig
from .query_optimization import QueryOptimizationConfig
from .query_correction import QueryCorrectionConfig
from .embedding_fusion import EmbeddingFusionConfig
from .rag_flow import RagFlowConfig
from .vector_db import VectorDBConfig
from .device_pool import DevicePoolConfig
from .tracking import TrackingConfig
from .service_runtime import ServiceRuntimeConfig
from .streaming import StreamingConfig
from .dataset_sink import DatasetSinkConfig
from .base import estimate_model_memory_gb
from .base import get_text_embedding_dim  # noqa: F401 — re-exported for importers
from .base import get_gpu_memory_gb  # noqa: F401 — re-exported; preflight patch target
from . import loading as _loading
from . import serialization as _serialization
from . import validation as _validation


# Feature sub-config field name -> class. Single source for from_dict/presets.
_FEATURE_SUBCONFIGS = {
    "audio_synthesis": AudioSynthesisConfig,
    "augmentation": AudioAugmentationConfig,
    "answer_generation": AnswerGenerationConfig,
    "judge": JudgeConfig,
    "query_optimization": QueryOptimizationConfig,
    "query_correction": QueryCorrectionConfig,
    "embedding_fusion": EmbeddingFusionConfig,
    "rag": RagFlowConfig,
}


@dataclass
class EvaluationConfig:
    """Complete evaluation configuration for audio-to-text retrieval.

    Holds all configuration parameters for running evaluations, including
    model settings, data paths, caching, and logging options. Configuration
    can be created from scratch, loaded from YAML files, or instantiated
    from presets.

    Attributes:
        experiment_name: Human-readable name for the evaluation run.
        output_dir: Directory for storing evaluation results.
        cache: Cache configuration for intermediate results.
        logging: Logging level and output configuration.
        model: Model types, names, and device assignments.
        data: Dataset paths and processing settings.
        audio_synthesis: Settings for synthesizing audio from text.
        judge: LLM-as-judge configuration for scoring.
        embedding_fusion: Configuration for fusing audio and text embeddings.
        vector_db: Vector database and retrieval settings.
        tracking: Experiment tracking configuration (MLflow).
        service_runtime: Service provider startup/offload runtime policy.
        checkpoint_enabled: Whether to save progress checkpoints.
        checkpoint_interval: Number of samples between checkpoints.
        resume_from_checkpoint: Whether to resume from saved state.
        parallel_enabled: Enable the DAG executor's intra-level per-branch concurrency
            (different branches/nodes run on their own devices in one process).

    Examples:
        Creating a config from scratch::

            >>> config = EvaluationConfig(
            ...     experiment_name="my_eval",
            ...     output_dir="results/",
            ... )
            >>> config.model.asr_model_type = "whisper"
            >>> config.data.batch_size = 16

        Loading from a YAML file::

            >>> config = EvaluationConfig.from_yaml("configs/whisper_eval.yaml")
            >>> config.experiment_name
            'whisper_evaluation'

        Using a preset with overrides::

            >>> config = EvaluationConfig.from_preset(
            ...     "whisper_labse",
            ...     experiment_name="custom_run",
            ...     data_batch_size=64
            ... )
            >>> config.model.asr_model_type
            'whisper'
            >>> config.data.batch_size
            64

        Auto-configuring devices based on hardware::

            >>> config = EvaluationConfig.from_preset("fast_dev")
            >>> config = config.with_auto_devices()
            >>> # Devices now set based on available GPUs
    """

    experiment_name: str = "evaluation"
    # Optional group id tying several runs into one logical experiment (e.g. a sweep); the
    # leaderboard can pivot/compare a group (architecture-improvements §3).
    experiment_group: Optional[str] = None
    output_dir: str = "evaluation_results"

    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    llm_server: LLMServerConfig = field(default_factory=LLMServerConfig)
    # Optional capability sub-configs — each its own default (disabled) so an un-drawn feature
    # still resolves for the handlers' global fall-back. A capability runs when its NODE is in the
    # graph (the spec); there is no `features:` grouping — every one is a top-level field.
    audio_synthesis: AudioSynthesisConfig = field(default_factory=AudioSynthesisConfig)
    augmentation: AudioAugmentationConfig = field(default_factory=AudioAugmentationConfig)
    answer_generation: AnswerGenerationConfig = field(default_factory=AnswerGenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    query_optimization: QueryOptimizationConfig = field(default_factory=QueryOptimizationConfig)
    query_correction: QueryCorrectionConfig = field(default_factory=QueryCorrectionConfig)
    embedding_fusion: EmbeddingFusionConfig = field(default_factory=EmbeddingFusionConfig)
    rag: RagFlowConfig = field(default_factory=RagFlowConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    device_pool: Optional[DevicePoolConfig] = None
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    service_runtime: ServiceRuntimeConfig = field(default_factory=ServiceRuntimeConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    dataset_sink: DatasetSinkConfig = field(default_factory=DatasetSinkConfig)

    checkpoint_enabled: bool = True
    checkpoint_interval: int = 50
    resume_from_checkpoint: bool = True

    # Explicit DAG override (config C2): {"nodes": [stage_id, ...], "edges": {to: [from]}}.
    # When set, the executor builds the graph from this spec (via build_graph_from_spec)
    # instead of deriving it from pipeline_mode. mode still drives handler behavior.
    graph_override: Optional[Dict[str, Any]] = None

    # Oracle baseline: re-run retrieval with GT transcriptions to separate ASR cost
    compute_oracle_baseline: bool = False

    # Statistical analysis
    compute_confidence_intervals: bool = False

    # Domain term weighting for TW-WER
    domain_term_weights_file: Optional[str] = None

    # `parallel_enabled` drives the DAG executor's intra-level branch concurrency.
    parallel_enabled: bool = False
    # CPU-bound per-item stage parallelism (Roadmap 4b): "sync" (default — today's in-line map),
    # "thread", or "process" (a ProcessPool for GIL-bound stages). The map is order-preserving and
    # determinism-neutral (per-item seeding makes each item independent of the worker).
    cpu_stage_executor: str = "sync"
    cpu_stage_workers: int = 0  # 0 = auto (os.cpu_count)

    @property
    def graph_template(self) -> Optional[str]:
        """The pipeline-mode LABEL for this config: derived from the explicit graph's node kinds
        (``label_from_graph``) — the graph is the spec, this is just its label. A back-compat
        template reference (``graph_override['template']``, legacy flat dicts only) wins when
        present. ``None`` only for a model-only / query-head-less custom graph."""
        override = self.graph_override or {}
        template = override.get("template")
        if template:
            return template
        from ..pipeline.graph.modes import label_from_graph

        return label_from_graph(override)

    def with_auto_devices(self) -> "EvaluationConfig":
        """Return a copy of this config with auto-configured device assignments.

        Creates a new EvaluationConfig with model devices automatically set
        based on available hardware (CUDA GPUs or CPU fallback). If device_pool
        is configured, uses GPUPool for allocation.

        Returns:
            New EvaluationConfig with auto-configured devices.
        """
        if self.device_pool is not None:
            # Use GPU pool for allocation
            return self._configure_devices_with_pool()

        # Legacy behavior: use ModelConfig.auto_configure_devices()
        new_model = replace(self.model)
        new_model.auto_configure_devices()

        # Return a new config with the updated model
        return replace(self, model=new_model)

    def _configure_devices_with_pool(self) -> "EvaluationConfig":
        """Configure devices using GPU pool allocation."""
        from .devices import GPUPool
        from .devices.strategy import create_strategy

        pool = GPUPool(
            devices=list(self.device_pool.available_devices),
            memory_buffer_percent=self.device_pool.memory_buffer_percent,
            allow_cpu_fallback=self.device_pool.allow_cpu_fallback,
        )

        # Set allocation strategy
        if self.device_pool.model_device_overrides:
            strategy = create_strategy(
                "manual", overrides=self.device_pool.model_device_overrides
            )
        else:
            strategy = create_strategy(
                enum_to_str(self.device_pool.allocation_strategy)
            )
        pool.set_strategy(strategy)

        new_model = replace(self.model)

        # Allocate a device for every model family that is actually configured (graph-first: the
        # graph carries the models; a family with no model_type set is simply skipped). Was keyed
        # off the removed pipeline_mode's required fields.
        _alloc = (
            ("asr", "asr_model_type", "asr_device"),
            ("text_embedding", "text_emb_model_type", "text_emb_device"),
            ("audio_embedding", "audio_emb_model_type", "audio_emb_device"),
        )
        for category, model_type_attr, device_attr in _alloc:
            model_type = getattr(self.model, model_type_attr)
            if model_type:
                mem = estimate_model_memory_gb(category, model_type)
                setattr(new_model, device_attr, pool.allocate(category, mem))

        return replace(self, model=new_model)

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings.

        Checks for common configuration errors before evaluation starts.
        Fatal errors raise ConfigurationError, non-fatal issues are returned
        as warnings.

        Returns:
            List of warning messages for non-fatal issues.

        Raises:
            ConfigurationError: If validation fails with unrecoverable errors.
        """
        return _validation.validate(self)

    # Sub-config fields that belong to runtime vs experiment dictionaries.
    _RUNTIME_FIELDS = frozenset(
        {
            "cache",
            "logging",
            "model",
            "data",
            "vector_db",
            "device_pool",
            "dataset_sink",
            "streaming",
        }
    )
    _RUNTIME_SCALARS = frozenset(
        {
            "checkpoint_enabled",
            "checkpoint_interval",
            "resume_from_checkpoint",
            "parallel_enabled",
            "cpu_stage_executor",
            "cpu_stage_workers",
            "compute_oracle_baseline",
            "compute_confidence_intervals",
            "domain_term_weights_file",
            "graph_override",
        }
    )
    _EXPERIMENT_SUBCONFIGS = frozenset(
        {
            "llm",
            "llm_server",
            "tracking",
            "service_runtime",
            *_FEATURE_SUBCONFIGS,
        }
    )
    # Plain sub-configs: key → class. No custom construction logic needed.
    # Adding a new sub-config here is the only change required in this file.
    # (The optional features live under `features` — see _FEATURE_SUBCONFIGS.)
    _PLAIN_SUBCONFIGS = {
        "cache": CacheConfig,
        "logging": LoggingConfig,
        "model": ModelConfig,
        "data": DataConfig,
        "llm": LLMConfig,
        "llm_server": LLMServerConfig,
        "vector_db": VectorDBConfig,
        "tracking": TrackingConfig,
        "service_runtime": ServiceRuntimeConfig,
        "streaming": StreamingConfig,
        "dataset_sink": DatasetSinkConfig,
    }

    def to_runtime_dict(self) -> Dict[str, Any]:
        """Return runtime execution configuration surface (auto-serialized)."""
        return _serialization.to_runtime_dict(self)

    def to_experiment_dict(self) -> Dict[str, Any]:
        """Return experiment/reporting configuration surface (auto-serialized)."""
        return _serialization.to_experiment_dict(self)

    @classmethod
    def from_dict(
        cls, config_dict: Dict[str, Any], validate: bool = True
    ) -> "EvaluationConfig":
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
        return _loading.build_from_dict(cls, config_dict, validate=validate)

    @classmethod
    def from_yaml(cls, yaml_path: str, validate: bool = True) -> "EvaluationConfig":
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
        return _loading.build_from_yaml(cls, yaml_path, validate=validate)

    @classmethod
    def from_preset(
        cls, preset_name: str, validate: bool = True, **overrides: Any
    ) -> "EvaluationConfig":
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
        return _loading.build_from_preset(
            cls, preset_name, validate=validate, **overrides
        )

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        _serialization.to_yaml(self, yaml_path)

    def to_dict(self, *, include_config: bool = False) -> Dict[str, Any]:
        """Convert to dictionary.

        Args:
            include_config: If True return full nested dict that can
                round-trip through ``from_dict``.  If False (default)
                return compact flat dict for telemetry/logging.
        """
        return _serialization.to_dict(self, include_config=include_config)


def preflight_check(config: EvaluationConfig) -> List[str]:
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
    return _validation.preflight_check(config)
