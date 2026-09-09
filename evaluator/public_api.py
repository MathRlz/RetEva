"""Stable public API exports for evaluator."""

from .api import (
    evaluate_from_config,
    evaluate_from_preset,
    quick_evaluate,
    run_evaluation,
    EvaluationError,
)
from .config import EvaluationConfig
from .errors import ConfigurationError
from .evaluation.results import EvaluationResults
from .config.model_presets import get_preset, list_presets

__all__ = [
    "evaluate_from_config",
    "evaluate_from_preset",
    "quick_evaluate",
    "run_evaluation",
    "EvaluationConfig",
    "EvaluationResults",
    "get_preset",
    "list_presets",
    "EvaluationError",
    "ConfigurationError",
]

__version__ = "2.0.0"
