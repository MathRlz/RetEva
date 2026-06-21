"""Evaluation package - modular evaluation functions."""

from ..metrics import (
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

from .executor import run_graph, run_from_bundle
from .results import EvaluationResults

__all__ = [
    "EvaluationResults",
    "run_graph",
    "run_from_bundle",
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
]
