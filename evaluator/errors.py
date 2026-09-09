"""Custom exceptions for the evaluator framework."""


class EvaluatorError(Exception):
    """Base exception for all evaluator-specific errors."""
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
    """
    pass
