"""Evaluator package - Audio-to-Text Retrieval Evaluation Framework for medical-domain
audio-to-text retrieval systems. See :mod:`evaluator.public_api` for the stable API.
"""

from .public_api import (
    evaluate_from_config,
    evaluate_from_preset,
    quick_evaluate,
    run_evaluation,
    EvaluationError,
    EvaluationConfig,
    ConfigurationError,
    EvaluationResults,
    get_preset,
    list_presets,
)

__all__ = [
    "evaluate_from_config",
    "evaluate_from_preset",
    "quick_evaluate",
    "run_evaluation",
    "EvaluationError",
    "EvaluationConfig",
    "ConfigurationError",
    "EvaluationResults",
    "get_preset",
    "list_presets",
]
