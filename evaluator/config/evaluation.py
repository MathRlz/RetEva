"""Evaluation configuration."""
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml

from ..config.types import enum_to_str
from ..errors import ConfigurationError


def _serialize_value(val: Any) -> Any:
    """Recursively serialize a value for dict output."""
    if isinstance(val, Enum):
        return str(val)
    if hasattr(val, '__dataclass_fields__'):
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
from .embedding_fusion import EmbeddingFusionConfig
from .vector_db import VectorDBConfig
from .device_pool import DevicePoolConfig
from .tracking import TrackingConfig
from .service_runtime import ServiceRuntimeConfig
from .base import (
    estimate_model_memory_gb,
    get_gpu_memory_gb,
    get_text_embedding_dim,
    DEVICE_PATTERN,
)

# Validation thresholds.
GPU_MEMORY_SAFETY_FRACTION = 0.9
LARGE_BATCH_SIZE_WARNING = 256


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


def _detect_cuda():
    """Return (cuda_available, device_count); (False, 0) when torch is absent."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.device_count()
    except ImportError:
        pass
    return False, 0


@dataclass
class FeaturesConfig:
    """Optional, toggleable pipeline features.

    Groups the capability sub-configs (each with its own ``enabled`` flag) under
    one namespace so the root ``EvaluationConfig`` is not flooded with always-off
    options. Access them as ``config.features.judge`` (canonical) or via the
    backward-compatible ``config.judge`` shortcut properties on EvaluationConfig.
    """
    audio_synthesis: AudioSynthesisConfig = field(default_factory=AudioSynthesisConfig)
    augmentation: AudioAugmentationConfig = field(default_factory=AudioAugmentationConfig)
    answer_generation: AnswerGenerationConfig = field(default_factory=AnswerGenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    query_optimization: QueryOptimizationConfig = field(default_factory=QueryOptimizationConfig)
    embedding_fusion: EmbeddingFusionConfig = field(default_factory=EmbeddingFusionConfig)


# Feature sub-config field name -> class. Single source for from_dict/presets.
_FEATURE_SUBCONFIGS = {
    "audio_synthesis": AudioSynthesisConfig,
    "augmentation": AudioAugmentationConfig,
    "answer_generation": AnswerGenerationConfig,
    "judge": JudgeConfig,
    "query_optimization": QueryOptimizationConfig,
    "embedding_fusion": EmbeddingFusionConfig,
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
        parallel_enabled: Whether to use multi-GPU parallel evaluation.
        num_parallel_workers: Number of parallel GPU workers (0 = auto-detect).
    
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
    output_dir: str = "evaluation_results"
    
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    llm_server: LLMServerConfig = field(default_factory=LLMServerConfig)
    # Optional toggleable features grouped under one namespace (#7).
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    device_pool: Optional[DevicePoolConfig] = None
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    service_runtime: ServiceRuntimeConfig = field(default_factory=ServiceRuntimeConfig)
    
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 50
    resume_from_checkpoint: bool = True

    # Oracle baseline: re-run retrieval with GT transcriptions to separate ASR cost
    compute_oracle_baseline: bool = False

    # Statistical analysis
    compute_confidence_intervals: bool = False

    # Domain term weighting for TW-WER
    domain_term_weights_file: Optional[str] = None

    # Parallel evaluation settings
    parallel_enabled: bool = False
    num_parallel_workers: int = 0  # 0 = auto-detect GPU count

    # Backward-compatible shortcuts: config.judge <-> config.features.judge, etc.
    # Keeps existing call sites working after the features grouping (#7).
    @property
    def audio_synthesis(self) -> AudioSynthesisConfig:
        return self.features.audio_synthesis

    @audio_synthesis.setter
    def audio_synthesis(self, value: AudioSynthesisConfig) -> None:
        self.features.audio_synthesis = value

    @property
    def augmentation(self) -> AudioAugmentationConfig:
        return self.features.augmentation

    @augmentation.setter
    def augmentation(self, value: AudioAugmentationConfig) -> None:
        self.features.augmentation = value

    @property
    def answer_generation(self) -> AnswerGenerationConfig:
        return self.features.answer_generation

    @answer_generation.setter
    def answer_generation(self, value: AnswerGenerationConfig) -> None:
        self.features.answer_generation = value

    @property
    def judge(self) -> JudgeConfig:
        return self.features.judge

    @judge.setter
    def judge(self, value: JudgeConfig) -> None:
        self.features.judge = value

    @property
    def query_optimization(self) -> QueryOptimizationConfig:
        return self.features.query_optimization

    @query_optimization.setter
    def query_optimization(self, value: QueryOptimizationConfig) -> None:
        self.features.query_optimization = value

    @property
    def embedding_fusion(self) -> EmbeddingFusionConfig:
        return self.features.embedding_fusion

    @embedding_fusion.setter
    def embedding_fusion(self, value: EmbeddingFusionConfig) -> None:
        self.features.embedding_fusion = value

    def with_auto_devices(self) -> 'EvaluationConfig':
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
    
    def _configure_devices_with_pool(self) -> 'EvaluationConfig':
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
            strategy = create_strategy("manual", overrides=self.device_pool.model_device_overrides)
        else:
            strategy = create_strategy(enum_to_str(self.device_pool.allocation_strategy))
        pool.set_strategy(strategy)
        
        new_model = replace(self.model)

        # Which models a mode needs is owned by PipelineModeSpec (the single
        # authority). Map each required model field to its (allocation category,
        # model-type attr, device attr) and allocate for the ones that are set.
        from ..pipeline.stage_graph import resolve_pipeline_mode_spec

        _field_to_alloc = {
            "model.asr_model_type": ("asr", "asr_model_type", "asr_device"),
            "model.text_emb_model_type": ("text_embedding", "text_emb_model_type", "text_emb_device"),
            "model.audio_emb_model_type": ("audio_embedding", "audio_emb_model_type", "audio_emb_device"),
        }
        mode = enum_to_str(self.model.pipeline_mode)
        spec = resolve_pipeline_mode_spec(mode)
        for field_name in spec.required_model_fields:
            category, model_type_attr, device_attr = _field_to_alloc[field_name]
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
        errors: List[str] = []
        warnings: List[str] = []

        cuda_available, cuda_count = _detect_cuda()
        self._validate_devices(errors, warnings, cuda_available, cuda_count)
        self._validate_model_types(errors)
        self._validate_embedding_dims(warnings)
        if cuda_available:
            self._validate_gpu_memory(warnings, cuda_count)
        self._validate_data_params(errors, warnings)
        self._validate_dataset_compat(warnings)

        if errors:
            raise ConfigurationError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )
        return warnings

    @property
    def _device_fields(self):
        return [
            ('model.asr_device', self.model.asr_device),
            ('model.text_emb_device', self.model.text_emb_device),
            ('model.audio_emb_device', self.model.audio_emb_device),
        ]

    def _validate_devices(self, errors, warnings, cuda_available, cuda_count):
        """Check device-string format and CUDA availability."""
        for field_name, device in self._device_fields:
            if not DEVICE_PATTERN.match(device):
                errors.append(
                    f"Invalid device format for {field_name}: '{device}'. "
                    f"Expected 'cpu', 'cuda', 'cuda:N', or 'mps'. "
                    f"Tip: call with_auto_devices() to auto-select valid devices."
                )
            elif device.startswith('cuda'):
                if not cuda_available:
                    warnings.append(
                        f"{field_name} is set to '{device}' but CUDA is not available. "
                        f"Consider using 'cpu' or calling with_auto_devices()."
                    )
                elif ':' in device and int(device.split(':')[1]) >= cuda_count:
                    warnings.append(
                        f"{field_name} is set to '{device}' but only {cuda_count} "
                        f"CUDA device(s) available."
                    )

    def _validate_model_types(self, errors):
        """Check each configured model type is registered."""
        from ..models.registry import (
            asr_registry, text_embedding_registry, audio_embedding_registry,
        )
        checks = [
            (self.model.asr_model_type, asr_registry, "ASR"),
            (self.model.text_emb_model_type, text_embedding_registry, "text embedding"),
            (self.model.audio_emb_model_type, audio_embedding_registry, "audio embedding"),
        ]
        for model_type, registry, label in checks:
            if model_type is not None and not registry.is_registered(model_type):
                available = ', '.join(registry.list_types())
                errors.append(
                    f"Unknown {label} model type: '{model_type}'. "
                    f"Available types: {available}. "
                    f"Tip: use one listed type or call EvaluationConfig.from_preset(...)."
                )

    def _validate_embedding_dims(self, warnings):
        """Warn when audio/text embedding dimensions disagree."""
        if self.model.pipeline_mode != "audio_emb_retrieval":
            return
        text_emb_dim = get_text_embedding_dim(self.model.text_emb_model_type)
        audio_emb_dim = self.model.audio_emb_dim
        if text_emb_dim is not None and audio_emb_dim != text_emb_dim:
            warnings.append(
                f"Embedding dimension mismatch: audio_emb_dim ({audio_emb_dim}) != "
                f"text embedding dim for '{self.model.text_emb_model_type}' ({text_emb_dim}). "
                f"Ensure your audio embedding model projects to the correct dimension."
            )

    def _validate_gpu_memory(self, warnings, cuda_count):
        """Warn when estimated per-GPU memory exceeds 90% of capacity."""
        device_memory_usage: Dict[int, float] = {}
        model_assignments = [
            (self.model.asr_device, "asr", self.model.asr_model_type),
            (self.model.text_emb_device, "text_embedding", self.model.text_emb_model_type),
            (self.model.audio_emb_device, "audio_embedding", self.model.audio_emb_model_type),
        ]
        for device_str, category, model_type in model_assignments:
            if device_str.startswith("cuda"):
                device_idx = int(device_str.split(":")[1]) if ":" in device_str else 0
                estimated_mem = estimate_model_memory_gb(category, model_type)
                device_memory_usage[device_idx] = (
                    device_memory_usage.get(device_idx, 0.0) + estimated_mem
                )
        for device_idx, estimated_total in device_memory_usage.items():
            if device_idx < cuda_count:
                gpu_memory = get_gpu_memory_gb(device_idx)
                if gpu_memory is not None and estimated_total > gpu_memory * GPU_MEMORY_SAFETY_FRACTION:
                    warnings.append(
                        f"GPU {device_idx}: estimated memory usage ({estimated_total:.1f}GB) "
                        f"exceeds 90% of available memory ({gpu_memory:.1f}GB). "
                        f"Consider distributing models across devices."
                    )

    def _validate_data_params(self, errors, warnings):
        """Check data paths, batch size, and k bounds."""
        if self.data.questions_path is not None and not Path(self.data.questions_path).exists():
            errors.append(
                f"Questions path does not exist: '{self.data.questions_path}'. "
                f"Tip: verify file path, or set data.prepared_dataset_dir for prepared datasets."
            )
        if self.data.corpus_path is not None and not Path(self.data.corpus_path).exists():
            errors.append(
                f"Corpus path does not exist: '{self.data.corpus_path}'. "
                f"Tip: verify file path and ensure corpus JSON/JSONL file exists."
            )
        if self.data.batch_size <= 0:
            errors.append(
                f"Batch size must be positive, got: {self.data.batch_size}. "
                f"Tip: start with batch_size=16 or batch_size=32."
            )
        elif self.data.batch_size > LARGE_BATCH_SIZE_WARNING:
            warnings.append(
                f"Very large batch size ({self.data.batch_size}) may cause memory issues."
            )
        if self.vector_db.k <= 0:
            errors.append(
                f"vector_db.k must be positive, got: {self.vector_db.k}. "
                f"Tip: use k=5 or k=10 for most evaluations."
            )
        if self.vector_db.reranker_top_k <= 0:
            errors.append(
                f"vector_db.reranker_top_k must be positive, got: {self.vector_db.reranker_top_k}. "
                f"Tip: use reranker_top_k=20 or reranker_top_k=50."
            )

    def _validate_dataset_compat(self, warnings):
        """Warn when the dataset doesn't support the chosen pipeline mode."""
        try:
            from ..datasets.descriptor import resolve_dataset_descriptor
            descriptor = resolve_dataset_descriptor(self.data)
            mode = enum_to_str(self.model.pipeline_mode)
            if not descriptor.supports_pipeline_mode(mode):
                warnings.append(
                    f"Pipeline mode '{mode}' is not in the compatible modes for dataset "
                    f"'{descriptor.id}': {', '.join(descriptor.compatible_pipeline_modes)}. "
                    f"Results may be incorrect."
                )
            if self.audio_synthesis.enabled and not descriptor.supports_generation:
                warnings.append(
                    f"audio_synthesis.enabled=True but dataset '{descriptor.id}' does not "
                    f"support audio generation (supports_generation=False)."
                )
        except Exception:
            pass  # Unresolvable dataset; skip check rather than block validation

    # Sub-config fields that belong to runtime vs experiment dictionaries.
    _RUNTIME_FIELDS = frozenset({
        "cache", "logging", "model", "data", "vector_db", "device_pool",
    })
    _RUNTIME_SCALARS = frozenset({
        "checkpoint_enabled", "checkpoint_interval", "resume_from_checkpoint",
        "parallel_enabled", "num_parallel_workers",
        "compute_oracle_baseline", "compute_confidence_intervals", "domain_term_weights_file",
    })
    _EXPERIMENT_SUBCONFIGS = frozenset({
        "features", "llm", "llm_server", "tracking", "service_runtime",
    })
    # Plain sub-configs: key → class. No custom construction logic needed.
    # Adding a new sub-config here is the only change required in this file.
    # (The optional features live under `features` — see _FEATURE_SUBCONFIGS.)
    _PLAIN_SUBCONFIGS = {
        'cache': CacheConfig,
        'logging': LoggingConfig,
        'model': ModelConfig,
        'data': DataConfig,
        'llm': LLMConfig,
        'llm_server': LLMServerConfig,
        'vector_db': VectorDBConfig,
        'tracking': TrackingConfig,
        'service_runtime': ServiceRuntimeConfig,
    }

    def to_runtime_dict(self) -> Dict[str, Any]:
        """Return runtime execution configuration surface (auto-serialized)."""
        result: Dict[str, Any] = {}
        for name in self._RUNTIME_FIELDS:
            val = getattr(self, name)
            if val is None:
                continue
            result[name] = _serialize_dataclass(val)
        for name in self._RUNTIME_SCALARS:
            result[name] = getattr(self, name)
        return result

    def to_experiment_dict(self) -> Dict[str, Any]:
        """Return experiment/reporting configuration surface (auto-serialized)."""
        result: Dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "output_dir": self.output_dir,
        }
        for name in self._EXPERIMENT_SUBCONFIGS:
            result[name] = _serialize_dataclass(getattr(self, name))
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], validate: bool = True) -> 'EvaluationConfig':
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
            key: cls_(**(config_dict.get(key) or {}))
            for key, cls_ in cls._PLAIN_SUBCONFIGS.items()
        }

        # Shared LLM backend — propagates to judge/answer_generation/query_optimization
        # as defaults; per-component values override.
        llm_config = LLMConfig(**(config_dict.get('llm') or {}))
        _llm_fields = {
            "model", "api_base", "api_key_env", "temperature",
            "timeout_s", "use_local_server", "local_server_url",
        }
        _llm_base = {k: v for k, v in vars(llm_config).items() if k in _llm_fields}

        def _merge_llm(comp_dict: dict) -> dict:
            merged = dict(_llm_base)
            merged.update(comp_dict)
            return merged

        # Backwards-compat: map old llm_* names in query_optimization dict
        _qo_raw = dict(config_dict.get('query_optimization') or {})
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

        sub_configs['llm'] = llm_config

        # Assemble the optional features into one FeaturesConfig (#7). The three
        # LLM-derived features inherit the shared backend defaults; the rest are
        # plain sub-configs.
        features = FeaturesConfig(
            audio_synthesis=AudioSynthesisConfig(**(config_dict.get('audio_synthesis') or {})),
            augmentation=AudioAugmentationConfig(**(config_dict.get('augmentation') or {})),
            answer_generation=AnswerGenerationConfig(**_merge_llm(config_dict.get('answer_generation') or {})),
            judge=JudgeConfig(**_merge_llm(config_dict.get('judge') or {})),
            query_optimization=QueryOptimizationConfig(**_merge_llm(_qo_raw)),
            embedding_fusion=EmbeddingFusionConfig(**(config_dict.get('embedding_fusion') or {})),
        )
        sub_configs['features'] = features

        # device_pool is optional
        device_pool_dict = config_dict.get('device_pool')
        sub_configs['device_pool'] = DevicePoolConfig(**device_pool_dict) if device_pool_dict else None

        # All remaining keys are scalar fields on EvaluationConfig itself.
        # Exclude the feature keys consumed above (they're not EvaluationConfig fields).
        _consumed = set(sub_configs) | set(_FEATURE_SUBCONFIGS)
        main_config = {k: v for k, v in config_dict.items() if k not in _consumed}

        config = cls(**main_config, **sub_configs)
        
        if validate:
            config.validate()
        
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: str, validate: bool = True) -> 'EvaluationConfig':
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file.
            validate: If True, validate configuration after loading. Defaults to True.
            
        Returns:
            EvaluationConfig instance.
            
        Raises:
            ConfigurationError: If validate=True and validation fails.
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        if config_dict is None:
            raise AttributeError("Empty configuration file.")
        return cls.from_dict(config_dict, validate=validate)
    
    @classmethod
    def from_preset(cls, preset_name: str, validate: bool = True, **overrides: Any) -> 'EvaluationConfig':
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
            if '.' in key:
                # Dotted path: navigate / create nested dicts
                segments = key.split('.')
                d = config_dict
                for seg in segments[:-1]:
                    if not isinstance(d.get(seg), dict):
                        d[seg] = {}
                    d = d[seg]
                d[segments[-1]] = value
                continue

            # Legacy underscore notation
            parts = key.split('_', 1)
            if len(parts) == 2 and parts[0] in ['cache', 'logging', 'model', 'data', 'judge', 'vector', 'device', 'tracking', 'service']:
                section = parts[0]
                # Handle 'vector_db' special case
                if section == 'vector':
                    remaining = parts[1]
                    if remaining.startswith('db_'):
                        section = 'vector_db'
                        sub_key = remaining[3:]  # Remove 'db_' prefix
                    else:
                        # Not a vector_db key, treat as top-level
                        config_dict[key] = value
                        continue
                # Handle 'device_pool' special case
                elif section == 'device':
                    remaining = parts[1]
                    if remaining.startswith('pool_'):
                        section = 'device_pool'
                        sub_key = remaining[5:]  # Remove 'pool_' prefix
                    else:
                        # Not a device_pool key, treat as top-level
                        config_dict[key] = value
                        continue
                elif section == 'service':
                    remaining = parts[1]
                    if remaining.startswith('runtime_'):
                        section = 'service_runtime'
                        sub_key = remaining[8:]  # Remove 'runtime_' prefix
                    else:
                        config_dict[key] = value
                        continue
                else:
                    sub_key = parts[1]

                if section not in config_dict:
                    config_dict[section] = {}
                config_dict[section][sub_key] = value
            elif len(parts) == 2 and parts[0] == 'audio' and parts[1].startswith('synthesis_'):
                # Handle audio_synthesis section
                sub_key = parts[1][10:]  # Remove 'synthesis_' prefix
                if 'audio_synthesis' not in config_dict:
                    config_dict['audio_synthesis'] = {}
                config_dict['audio_synthesis'][sub_key] = value
            else:
                config_dict[key] = value
        
        return cls.from_dict(config_dict, validate=validate)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            "experiment_name": self.experiment_name,
            "output_dir": self.output_dir,
            "runtime": self.to_runtime_dict(),
            "experiment": self.to_experiment_dict(),
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self, *, include_config: bool = False) -> Dict[str, Any]:
        """Convert to dictionary.

        Args:
            include_config: If True return full nested dict that can
                round-trip through ``from_dict``.  If False (default)
                return compact flat dict for telemetry/logging.
        """
        if include_config:
            return self._to_nested_dict()

        result = {
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'checkpoint_enabled': self.checkpoint_enabled,
            'checkpoint_interval': self.checkpoint_interval,
            'parallel_enabled': self.parallel_enabled,
            'num_parallel_workers': self.num_parallel_workers,
            'cache_enabled': self.cache.enabled,
            'pipeline_mode': enum_to_str(self.model.pipeline_mode),
            'asr_model': f"{self.model.asr_model_type}:{self.model.asr_model_name or self.model.asr_size or 'default'}",
            'text_emb_model': f"{self.model.text_emb_model_type}:{self.model.text_emb_model_name or self.model.text_emb_size or 'default'}",
            'audio_emb_model': f"{self.model.audio_emb_model_type}:{self.model.audio_emb_model_name or self.model.audio_emb_size or 'default'}",
            'dataset': self.data.dataset_name,
            'questions_path': self.data.questions_path,
            'corpus_path': self.data.corpus_path,
            'batch_size': self.data.batch_size,
            'trace_limit': self.data.trace_limit,
            'judge_enabled': self.judge.enabled,
            'tracking_enabled': self.tracking.enabled,
            'tracking_backend': self.tracking.backend,
            'service_startup_mode': self.service_runtime.startup_mode,
            'service_offload_policy': self.service_runtime.offload_policy,
        }
        if self.device_pool is not None:
            result['device_pool_strategy'] = enum_to_str(self.device_pool.allocation_strategy)
        return result

    def _to_nested_dict(self) -> Dict[str, Any]:
        """Full nested dict that round-trips through ``from_dict``."""
        result: Dict[str, Any] = {
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'checkpoint_enabled': self.checkpoint_enabled,
            'checkpoint_interval': self.checkpoint_interval,
            'resume_from_checkpoint': self.resume_from_checkpoint,
            'parallel_enabled': self.parallel_enabled,
            'num_parallel_workers': self.num_parallel_workers,
        }
        for key in self._PLAIN_SUBCONFIGS:
            result[key] = _serialize_dataclass(getattr(self, key))
        result['llm'] = _serialize_dataclass(self.llm)
        result['features'] = _serialize_dataclass(self.features)
        if self.device_pool is not None:
            result['device_pool'] = _serialize_dataclass(self.device_pool)
        return result


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
        warnings.append(
            "PyTorch not available. GPU features will not work."
        )
    
    # Check real-time GPU memory availability
    if cuda_available:
        device_assignments = []
        
        if config.model.asr_device.startswith("cuda") and config.model.asr_model_type:
            device_idx = int(config.model.asr_device.split(":")[1]) if ":" in config.model.asr_device else 0
            device_assignments.append((device_idx, "asr", config.model.asr_model_type))
        
        if config.model.text_emb_device.startswith("cuda") and config.model.text_emb_model_type:
            device_idx = int(config.model.text_emb_device.split(":")[1]) if ":" in config.model.text_emb_device else 0
            device_assignments.append((device_idx, "text_embedding", config.model.text_emb_model_type))
        
        if config.model.audio_emb_device.startswith("cuda") and config.model.audio_emb_model_type:
            device_idx = int(config.model.audio_emb_device.split(":")[1]) if ":" in config.model.audio_emb_device else 0
            device_assignments.append((device_idx, "audio_embedding", config.model.audio_emb_model_type))
        
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
