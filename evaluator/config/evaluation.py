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

    Aggregates the sub-configs (model, data, cache, capability nodes, vector_db,
    runtime, ...) plus run-level scalars; field comments below carry the per-field
    semantics. Build from scratch, ``from_dict``, ``from_yaml``, or ``from_preset``.
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
    # Opt-in (off by default — storage cost): additionally persist per-variant intermediate
    # artifacts (today: retrieved.jsonl) alongside the always-on answers.jsonl/metrics.json —
    # see evaluation/results_io.py:save_run_artifacts.
    persist_intermediate_artifacts: bool = False

    # Explicit DAG override (config C2): {"nodes": [stage_id, ...], "edges": {to: [from]}}.
    # When set, the executor builds the graph from this spec (via build_graph_from_spec)
    # instead of deriving it from pipeline_mode. mode still drives handler behavior.
    graph_override: Optional[Dict[str, Any]] = None

    # Statistical analysis
    compute_confidence_intervals: bool = False

    # Metric allowlist (B1): when set, the report computes EXACTLY these registry metrics
    # (explicit naming bypasses MetricSpec.opt_in; a metric whose inputs the graph doesn't
    # produce is skipped). None keeps collect-all — every satisfiable metric.
    metrics: Optional[List[str]] = None

    # Lineage-variant rollup (A1): how fan-out variants (q42·aug0/1) reduce to their
    # parent before pairing/CIs — "mean" (default) | "min" | "max" (worst-case is min or
    # max depending on the metric's direction).
    variant_rollup: str = "mean"

    # Domain term weighting for TW-WER
    domain_term_weights_file: Optional[str] = None

    # `parallel_enabled` drives the DAG executor's intra-level branch concurrency.
    parallel_enabled: bool = False
    # CPU-bound per-item stage parallelism (Roadmap 4b): "sync" (default — today's in-line map),
    # "thread", or "process" (a ProcessPool for GIL-bound stages). The map is order-preserving and
    # determinism-neutral (per-item seeding makes each item independent of the worker).
    cpu_stage_executor: str = "sync"
    cpu_stage_workers: int = 0  # 0 = auto (os.cpu_count)

    # Memoized resolved DAG (see the ``graph`` property below). ``init=False`` so
    # ``dataclasses.replace()`` (used throughout to derive a modified copy, e.g.
    # ``with_auto_devices``) can never carry a stale cached graph forward onto a copy whose
    # ``graph_override``/nodes might differ — every replace()'d copy starts uncached and
    # re-resolves lazily on first ``.graph`` access.
    _graph_cache: Any = field(default=None, repr=False, compare=False, init=False)

    @property
    def graph(self) -> Any:
        """The resolved execution DAG (a ``StageGraph``) for this config — memoized on first
        access so ``factory.py``/``validation.py`` don't each re-derive it. See
        ``build_graph_for_config``. A branched config's own ``build_branched_graph`` result is
        NOT cached here — callers that need the pre-CSE branched graph still call it directly."""
        if self._graph_cache is None:
            from ..pipeline.graph.modes import build_graph_for_config

            self._graph_cache = build_graph_for_config(self)
        return self._graph_cache

    @property
    def graph_template(self) -> Optional[str]:
        """The pipeline-mode LABEL for this config: derived from the explicit graph's node kinds
        (``label_from_graph``) — the graph is the spec, this is just its label. A back-compat
        template reference (``graph_override['template']``, legacy flat dicts only) wins when
        present. ``None`` only for a model-only / query-head-less custom graph."""
        from ..pipeline.graph.modes import resolve_template_label

        return resolve_template_label(self)

    def with_auto_devices(self) -> "EvaluationConfig":
        """Return a copy with device assignments auto-configured from available
        hardware — via GPU pool allocation when ``device_pool`` is set, else
        ``ModelConfig.auto_configure_devices()``."""
        if self.device_pool is not None:
            return self._configure_devices_with_pool()

        # Legacy behavior: use ModelConfig.auto_configure_devices()
        new_model = replace(self.model)
        new_model.auto_configure_devices()

        return replace(self, model=new_model)

    def _configure_devices_with_pool(self) -> "EvaluationConfig":
        """Configure devices using GPU pool allocation."""
        from .devices.pool import pool_from_config

        pool = pool_from_config(self.device_pool)

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
        """Validate the configuration: fatal errors raise ConfigurationError,
        non-fatal issues are returned as a list of warning messages."""
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
            "persist_intermediate_artifacts",
            "parallel_enabled",
            "cpu_stage_executor",
            "cpu_stage_workers",
            "compute_confidence_intervals",
            "metrics",
            "variant_rollup",
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
        """Create configuration from a dictionary (nested dicts build the
        sub-configs). Raises ConfigurationError if ``validate=True`` (default)
        and validation fails."""
        return _loading.build_from_dict(cls, config_dict, validate=validate)

    @classmethod
    def from_yaml(cls, yaml_path: str, validate: bool = True) -> "EvaluationConfig":
        """Load configuration from a node-centric YAML file.

        The on-disk schema is the node-centric shape (experiment/dataset/graph/nodes/
        runtime); ``build_evaluation_config_kwargs`` translates it before construction.
        Legacy-shape keys still pass through (the translator is backward-compatible), so
        this is the single load chokepoint for the CLI, public API, and presets. Raises
        ConfigurationError if ``validate=True`` (default) and validation fails.
        """
        return _loading.build_from_yaml(cls, yaml_path, validate=validate)

    @classmethod
    def from_preset(
        cls, preset_name: str, validate: bool = True, **overrides: Any
    ) -> "EvaluationConfig":
        """Create configuration from a named preset (e.g. 'whisper_labse',
        'fast_dev') with optional overrides. Nested overrides use underscore
        notation: ``model_asr_device='cpu'`` sets ``model.asr_device``. Raises
        ConfigurationError if ``validate=True`` (default) and validation fails.
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
