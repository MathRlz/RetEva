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
        >>> raise ConfigurationError(
        ...     "Unknown ASR model type: 'whisper3'. Available: whisper, wav2vec2")
    """
    pass
