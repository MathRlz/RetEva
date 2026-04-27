"""Evaluator package - Audio-to-Text Retrieval Evaluation Framework.

This package provides a comprehensive framework for evaluating audio-to-text retrieval
systems in the medical domain. It supports multiple pipeline modes including ASR-based
and direct audio embedding approaches.

Public API
----------
The top-level ``evaluator`` namespace exposes the stable, high-level API:

- evaluate_from_config() - Run evaluation from YAML config
- evaluate_from_preset() - Quick evaluation with predefined presets
- quick_evaluate() - One-line evaluation for simple cases
- run_evaluation() - Run evaluation from a prepared EvaluationConfig
- EvaluationConfig / EvaluationResults - Core input/output types
    
    Example::
    
        from evaluator import evaluate_from_preset
        
        results = evaluate_from_preset(
            "whisper_labse",
            data_path="questions.json",
            corpus_path="corpus.json"
        )
        print(f"MRR: {results.get_metric('MRR', 0.0):.4f}")

For advanced/internal extension points, import explicit subpackages such as
``evaluator.pipeline``, ``evaluator.models``, ``evaluator.storage``, or
``evaluator.advanced_api``.
"""

from .public_api import (
    evaluate_from_config,
    evaluate_from_preset,
    quick_evaluate,
    run_evaluation,
    run_evaluation_matrix,
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
    "run_evaluation_matrix",
    "EvaluationError",
    "EvaluationConfig",
    "ConfigurationError",
    "EvaluationResults",
    "get_preset",
    "list_presets",
]
