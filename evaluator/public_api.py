"""Stable public API exports for evaluator."""

from .api import (
    evaluate_from_config,
    evaluate_from_preset,
    quick_evaluate,
    run_evaluation,
    run_evaluation_matrix,
    EvaluationError,
)
from .config import EvaluationConfig
from .errors import ConfigurationError
from .evaluation.results import EvaluationResults
from .config.model_presets import get_preset, list_presets

__all__ = [
    # High-level execution API
    "evaluate_from_config",
    "evaluate_from_preset",
    "quick_evaluate",
    "run_evaluation",
    "run_evaluation_matrix",
    # Core types
    "EvaluationConfig",
    "EvaluationResults",
    # Preset discovery
    "get_preset",
    "list_presets",
    # Exceptions
    "EvaluationError",
    "ConfigurationError",
]

__version__ = "2.0.0"
