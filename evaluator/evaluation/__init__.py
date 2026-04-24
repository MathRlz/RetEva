"""Evaluation package - modular evaluation functions."""

from .metrics import (
    reciprocal_rank,
    recall_at_k,
    precision_at_k,
    dcg_at_k,
    ndcg_at_k,
    average_precision,
    word_error_rate,
    character_error_rate,
)

from .helpers import (
    _payload_to_key,
    _search_results_to_keys,
    _build_relevant_from_item,
    collate_fn,
    asr_collate_fn,
)
from ..metrics import compute_ir_metrics, log_ir_metrics

from .sample_wise import evaluate_with_pipeline
from .phased import evaluate_phased, evaluate_from_bundle
from .results import EvaluationResults


__all__ = [
    "EvaluationResults",
    # Main evaluation functions
    "evaluate_with_pipeline",
    "evaluate_phased",
    "evaluate_from_bundle",
    # Metric functions (re-exported from ir_metrics/stt_metrics)
    "reciprocal_rank",
    "recall_at_k",
    "precision_at_k",
    "dcg_at_k",
    "ndcg_at_k",
    "average_precision",
    "word_error_rate",
    "character_error_rate",
    # Helper functions
    "_payload_to_key",
    "_search_results_to_keys",
    "_build_relevant_from_item",
    "collate_fn",
    "asr_collate_fn",
    "compute_ir_metrics",
    "log_ir_metrics",
]
