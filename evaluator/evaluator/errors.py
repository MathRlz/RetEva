"""Custom exceptions for the evaluator framework."""


class EvaluatorError(Exception):
    """
    Base exception for all evaluator-specific errors.
    
    Use this as the base class for all custom exceptions in the evaluator
    framework to allow for easy catching of all evaluator-related errors.
    """
    pass


class ConfigurationError(EvaluatorError):
    """
    Raised when configuration validation fails with unrecoverable errors.
    
    This includes:
    - Invalid device strings
    - Unknown model types
    - Missing required configuration fields
    - Invalid parameter combinations
    - File paths that don't exist
    
    Examples:
        >>> raise ConfigurationError("Invalid device format: 'gpu:0'. Expected 'cuda:0'")
        >>> raise ConfigurationError("Unknown ASR model type: 'whisper3'. Available: whisper, wav2vec2")
    """
    pass


class ModelLoadError(EvaluatorError):
    """
    Raised when a model fails to load or initialize.
    
    This includes:
    - HuggingFace model download failures
    - Invalid model names or paths
    - PEFT/LoRA adapter loading errors
    - Model architecture mismatches
    - Missing dependencies
    
    Examples:
        >>> raise ModelLoadError("Failed to load model 'invalid/model-name' from HuggingFace")
        >>> raise ModelLoadError("PEFT adapter incompatible with base model architecture")
    """
    pass


class DeviceError(EvaluatorError):
    """
    Raised when device allocation or operations fail.
    
    This includes:
    - CUDA device not available
    - Out of memory errors
    - Invalid device indices
    - Device transfer failures
    
    Examples:
        >>> raise DeviceError("CUDA device 2 not available, only 2 GPUs detected")
        >>> raise DeviceError("Out of memory on cuda:0. Try reducing batch size")
    """
    pass


class CacheError(EvaluatorError):
    """
    Raised when cache operations fail.
    
    This includes:
    - Cache directory creation failures
    - Corrupted cache files
    - Cache read/write errors
    - Insufficient disk space
    
    Examples:
        >>> raise CacheError("Failed to create cache directory: Permission denied")
        >>> raise CacheError("Corrupted cache file: .cache/embeddings/abc123.npy")
    """
    pass


class VectorStoreError(EvaluatorError):
    """
    Raised when vector store operations fail.
    
    This includes:
    - Vector database initialization failures
    - Index build errors
    - Query/retrieval failures
    - Dimension mismatches
    - Connection errors (for remote stores like Qdrant)
    
    Examples:
        >>> raise VectorStoreError("Failed to connect to Qdrant server at localhost:6333")
        >>> raise VectorStoreError("Dimension mismatch: query vector (768) vs index (1024)")
    """
    pass
