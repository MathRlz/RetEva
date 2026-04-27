"""Configuration management for evaluation framework.

This package contains domain-specific configuration modules split from the
original monolithic config.py for better organization and maintainability.

Backward Compatibility:
    All configuration classes are re-exported from this package to maintain
    backward compatibility with existing code that imports from evaluator.config.
"""

# Import config types and errors first
from ..config.types import (
    PipelineMode,
    VectorDBType,
    AllocationStrategy,
    DatasetType,
    to_enum,
    enum_to_str,
)
from ..errors import ConfigurationError

# Import all configuration classes
from .cache import CacheConfig
from .logging import LoggingConfig
from .model import ModelConfig
from .data import DataConfig
from .audio_synthesis import AudioSynthesisConfig
from .audio_augmentation import AudioAugmentationConfig
from .llm_server import LLMServerConfig
from .llm_backend import LLMConfig
from .query_optimization import QueryOptimizationConfig
from .judge import JudgeConfig
from .embedding_fusion import EmbeddingFusionConfig
from .vector_db import VectorDBConfig
from .device_pool import DevicePoolConfig
from .tracking import TrackingConfig
from .service_runtime import ServiceRuntimeConfig
from .evaluation import EvaluationConfig, preflight_check
from .templates import ConfigTemplates

# Import utilities and helpers
from .base import (
    validate_device_string,
    detect_device,
    get_available_gpu_count,
    get_gpu_memory_gb,
    get_gpu_free_memory_gb,
    estimate_model_memory_gb,
    get_text_embedding_dim,
    DEVICE_PATTERN,
    MODEL_MEMORY_ESTIMATES_GB,
    MODEL_EMBEDDING_DIMS,
)

__all__ = [
    # Enums and types
    "PipelineMode",
    "VectorDBType",
    "AllocationStrategy",
    "DatasetType",
    "to_enum",
    "enum_to_str",
    # Errors
    "ConfigurationError",
    # Config classes
    "LLMConfig",
    "CacheConfig",
    "LoggingConfig",
    "ModelConfig",
    "DataConfig",
    "AudioSynthesisConfig",
    "AudioAugmentationConfig",
    "LLMServerConfig",
    "QueryOptimizationConfig",
    "JudgeConfig",
    "EmbeddingFusionConfig",
    "VectorDBConfig",
    "DevicePoolConfig",
    "TrackingConfig",
    "ServiceRuntimeConfig",
    "EvaluationConfig",
    "ConfigTemplates",
    # Utilities
    "validate_device_string",
    "detect_device",
    "get_available_gpu_count",
    "get_gpu_memory_gb",
    "get_gpu_free_memory_gb",
    "estimate_model_memory_gb",
    "get_text_embedding_dim",
    "preflight_check",
    "DEVICE_PATTERN",
    "MODEL_MEMORY_ESTIMATES_GB",
    "MODEL_EMBEDDING_DIMS",
]
