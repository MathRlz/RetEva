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
    audio_synthesis: AudioSynthesisConfig = field(default_factory=AudioSynthesisConfig)
    augmentation: AudioAugmentationConfig = field(default_factory=AudioAugmentationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    answer_generation: AnswerGenerationConfig = field(default_factory=AnswerGenerationConfig)
    llm_server: LLMServerConfig = field(default_factory=LLMServerConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    query_optimization: QueryOptimizationConfig = field(default_factory=QueryOptimizationConfig)
    embedding_fusion: EmbeddingFusionConfig = field(default_factory=EmbeddingFusionConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    device_pool: Optional[DevicePoolConfig] = None
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    service_runtime: ServiceRuntimeConfig = field(default_factory=ServiceRuntimeConfig)
    
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 50
    resume_from_checkpoint: bool = True
    
    # Parallel evaluation settings
    parallel_enabled: bool = False
    num_parallel_workers: int = 0  # 0 = auto-detect GPU count
    
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
        
        # Allocate devices for each model type based on pipeline mode
        mode = enum_to_str(self.model.pipeline_mode)
        
        if mode in ("asr_text_retrieval", "asr_only") and self.model.asr_model_type:
            asr_mem = estimate_model_memory_gb("asr", self.model.asr_model_type)
            new_model.asr_device = pool.allocate("asr", asr_mem)
        
        if mode in ("asr_text_retrieval", "audio_text_retrieval") and self.model.text_emb_model_type:
            text_mem = estimate_model_memory_gb("text_embedding", self.model.text_emb_model_type)
            new_model.text_emb_device = pool.allocate("text_embedding", text_mem)
        
        if mode in ("audio_emb_retrieval", "audio_text_retrieval") and self.model.audio_emb_model_type:
            audio_mem = estimate_model_memory_gb("audio_embedding", self.model.audio_emb_model_type)
            new_model.audio_emb_device = pool.allocate("audio_embedding", audio_mem)
        
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
        
        # Validate device strings
        device_fields = [
            ('model.asr_device', self.model.asr_device),
            ('model.text_emb_device', self.model.text_emb_device),
            ('model.audio_emb_device', self.model.audio_emb_device),
        ]
        for field_name, device in device_fields:
            if not DEVICE_PATTERN.match(device):
                errors.append(
                    f"Invalid device format for {field_name}: '{device}'. "
                    f"Expected 'cpu', 'cuda', 'cuda:N', or 'mps'. "
                    f"Tip: call with_auto_devices() to auto-select valid devices."
                )
        
        # Check CUDA availability warnings
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_count = torch.cuda.device_count() if cuda_available else 0
        except ImportError:
            cuda_available = False
            cuda_count = 0
        
        for field_name, device in device_fields:
            if device.startswith('cuda'):
                if not cuda_available:
                    warnings.append(
                        f"{field_name} is set to '{device}' but CUDA is not available. "
                        f"Consider using 'cpu' or calling with_auto_devices()."
                    )
                elif ':' in device:
                    device_idx = int(device.split(':')[1])
                    if device_idx >= cuda_count:
                        warnings.append(
                            f"{field_name} is set to '{device}' but only {cuda_count} "
                            f"CUDA device(s) available."
                        )
        
        # Validate model types against registries
        from ..models.registry import asr_registry, text_embedding_registry, audio_embedding_registry
        
        if self.model.asr_model_type is not None:
            if not asr_registry.is_registered(self.model.asr_model_type):
                available = ', '.join(asr_registry.list_types())
                errors.append(
                    f"Unknown ASR model type: '{self.model.asr_model_type}'. "
                    f"Available types: {available}. "
                    f"Tip: use one listed type or call EvaluationConfig.from_preset(...)."
                )
        
        if self.model.text_emb_model_type is not None:
            if not text_embedding_registry.is_registered(self.model.text_emb_model_type):
                available = ', '.join(text_embedding_registry.list_types())
                errors.append(
                    f"Unknown text embedding model type: '{self.model.text_emb_model_type}'. "
                    f"Available types: {available}. "
                    f"Tip: use one listed type or call EvaluationConfig.from_preset(...)."
                )
        
        if self.model.audio_emb_model_type is not None:
            if not audio_embedding_registry.is_registered(self.model.audio_emb_model_type):
                available = ', '.join(audio_embedding_registry.list_types())
                errors.append(
                    f"Unknown audio embedding model type: '{self.model.audio_emb_model_type}'. "
                    f"Available types: {available}. "
                    f"Tip: use one listed type or call EvaluationConfig.from_preset(...)."
                )
        
        # Validate embedding dimension compatibility
        if self.model.pipeline_mode == "audio_emb_retrieval":
            text_emb_dim = get_text_embedding_dim(self.model.text_emb_model_type)
            audio_emb_dim = self.model.audio_emb_dim
            
            if text_emb_dim is not None and audio_emb_dim != text_emb_dim:
                warnings.append(
                    f"Embedding dimension mismatch: audio_emb_dim ({audio_emb_dim}) != "
                    f"text embedding dim for '{self.model.text_emb_model_type}' ({text_emb_dim}). "
                    f"Ensure your audio embedding model projects to the correct dimension."
                )
        
        # GPU memory validation
        if cuda_available:
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
                    if gpu_memory is not None and estimated_total > gpu_memory * 0.9:
                        warnings.append(
                            f"GPU {device_idx}: estimated memory usage ({estimated_total:.1f}GB) "
                            f"exceeds 90% of available memory ({gpu_memory:.1f}GB). "
                            f"Consider distributing models across devices."
                        )
        
        # Validate file paths exist when provided
        if self.data.questions_path is not None:
            path = Path(self.data.questions_path)
            if not path.exists():
                errors.append(
                    f"Questions path does not exist: '{self.data.questions_path}'. "
                    f"Tip: verify file path, or set data.prepared_dataset_dir for prepared datasets."
                )
        
        if self.data.corpus_path is not None:
            path = Path(self.data.corpus_path)
            if not path.exists():
                errors.append(
                    f"Corpus path does not exist: '{self.data.corpus_path}'. "
                    f"Tip: verify file path and ensure corpus JSON/JSONL file exists."
                )
        
        # Validate batch size is positive
        if self.data.batch_size <= 0:
            errors.append(
                f"Batch size must be positive, got: {self.data.batch_size}. "
                f"Tip: start with batch_size=16 or batch_size=32."
            )
        elif self.data.batch_size > 256:
            warnings.append(
                f"Very large batch size ({self.data.batch_size}) may cause memory issues."
            )
        
        # Validate k values for metrics are positive
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
        
        # Raise errors if any
        if errors:
            raise ConfigurationError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )
        
        return warnings

    # Sub-config fields that belong to runtime vs experiment dictionaries.
    _RUNTIME_FIELDS = frozenset({
        "cache", "logging", "model", "data", "vector_db", "device_pool",
    })
    _RUNTIME_SCALARS = frozenset({
        "checkpoint_enabled", "checkpoint_interval", "resume_from_checkpoint",
        "parallel_enabled", "num_parallel_workers",
    })
    _EXPERIMENT_SUBCONFIGS = frozenset({
        "audio_synthesis", "augmentation", "answer_generation", "llm_server", "judge",
        "query_optimization", "embedding_fusion", "tracking", "service_runtime",
    })

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

        runtime_dict = config_dict.pop("runtime", None)
        if isinstance(runtime_dict, dict):
            for key, value in runtime_dict.items():
                if isinstance(value, dict) and isinstance(config_dict.get(key), dict):
                    merged = dict(config_dict[key])
                    merged.update(value)
                    config_dict[key] = merged
                else:
                    config_dict[key] = value

        experiment_dict = config_dict.pop("experiment", None)
        if isinstance(experiment_dict, dict):
            for key, value in experiment_dict.items():
                if isinstance(value, dict) and isinstance(config_dict.get(key), dict):
                    merged = dict(config_dict[key])
                    merged.update(value)
                    config_dict[key] = merged
                else:
                    config_dict[key] = value
        
        cache_config = CacheConfig(**config_dict.get('cache', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        # Handle model config
        model_dict = dict(config_dict.get('model', {}))
        
        model_config = ModelConfig(**model_dict)
        data_config = DataConfig(**config_dict.get('data', {}))
        audio_synthesis_config = AudioSynthesisConfig(**config_dict.get('audio_synthesis', {}))
        augmentation_config = AudioAugmentationConfig(**config_dict.get('augmentation', {}))
        # Shared LLM backend — propagates to judge/answer_generation/query_optimization
        # as defaults; per-component values override.
        llm_config = LLMConfig(**config_dict.get('llm', {}))
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
        _qo_raw = dict(config_dict.get('query_optimization', {}))
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

        answer_generation_config = AnswerGenerationConfig(**_merge_llm(config_dict.get('answer_generation', {})))
        llm_server_config = LLMServerConfig(**config_dict.get('llm_server', {}))
        judge_config = JudgeConfig(**_merge_llm(config_dict.get('judge', {})))
        query_optimization_config = QueryOptimizationConfig(**_merge_llm(_qo_raw))
        embedding_fusion_config = EmbeddingFusionConfig(**config_dict.get('embedding_fusion', {}))
        vector_db_config = VectorDBConfig(**config_dict.get('vector_db', {}))
        tracking_config = TrackingConfig(**config_dict.get('tracking', {}))
        service_runtime_config = ServiceRuntimeConfig(**config_dict.get('service_runtime', {}))

        # Handle device_pool (optional)
        device_pool_dict = config_dict.get('device_pool')
        device_pool_config = DevicePoolConfig(**device_pool_dict) if device_pool_dict else None

        main_config = {
            k: v for k, v in config_dict.items()
            if k not in [
                'cache',
                'logging',
                'model',
                'data',
                'audio_synthesis',
                'augmentation',
                'llm',
                'answer_generation',
                'llm_server',
                'judge',
                'query_optimization',
                'embedding_fusion',
                'vector_db',
                'device_pool',
                'tracking',
                'service_runtime',
            ]
        }

        config = cls(
            **main_config,
            cache=cache_config,
            logging=logging_config,
            model=model_config,
            data=data_config,
            audio_synthesis=audio_synthesis_config,
            augmentation=augmentation_config,
            llm=llm_config,
            answer_generation=answer_generation_config,
            llm_server=llm_server_config,
            judge=judge_config,
            query_optimization=query_optimization_config,
            embedding_fusion=embedding_fusion_config,
            vector_db=vector_db_config,
            device_pool=device_pool_config,
            tracking=tracking_config,
            service_runtime=service_runtime_config,
        )
        
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
        
        # Apply overrides
        for key, value in overrides.items():
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
        from dataclasses import asdict

        def _section(obj: Any) -> Dict[str, Any]:
            d = asdict(obj)
            # Convert enums to strings
            for k, v in d.items():
                if hasattr(v, 'value'):
                    d[k] = v.value
            return d

        result: Dict[str, Any] = {
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'checkpoint_enabled': self.checkpoint_enabled,
            'checkpoint_interval': self.checkpoint_interval,
            'parallel_enabled': self.parallel_enabled,
            'num_parallel_workers': self.num_parallel_workers,
            'model': _section(self.model),
            'data': _section(self.data),
            'cache': _section(self.cache),
            'logging': _section(self.logging),
            'vector_db': _section(self.vector_db),
            'llm': _section(self.llm),
            'judge': _section(self.judge),
            'tracking': _section(self.tracking),
            'service_runtime': _section(self.service_runtime),
            'audio_synthesis': _section(self.audio_synthesis),
            'augmentation': _section(self.augmentation),
            'answer_generation': _section(self.answer_generation),
            'llm_server': _section(self.llm_server),
            'query_optimization': _section(self.query_optimization),
            'embedding_fusion': _section(self.embedding_fusion),
        }
        if self.device_pool is not None:
            result['device_pool'] = _section(self.device_pool)
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
    
    # Check for judge API key if enabled
    if config.judge.enabled:
        import os
        api_key = os.environ.get(config.judge.api_key_env)
        if not api_key:
            warnings.append(
                f"LLM judge is enabled but {config.judge.api_key_env} environment variable "
                f"is not set. Judge scoring will fail."
            )
    
    return warnings
