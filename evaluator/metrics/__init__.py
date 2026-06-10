"""Evaluation metrics module.

This package provides comprehensive metrics for evaluating retrieval quality
and speech recognition accuracy.

Main Components:
    - ir: Information retrieval metrics (MRR, nDCG, Recall@k, etc.)
    - ir_vectorized: Vectorized implementations for batch processing
    - stt: Speech-to-text metrics (WER, CER, etc.)

Usage:
    from evaluator.metrics import compute_metrics, compute_wer

    # IR metrics
    metrics = compute_metrics(relevance_dict, k_values=[1, 5, 10])
    print(f"MRR: {metrics['MRR']:.4f}")

    # STT metrics
    wer = compute_wer(reference, hypothesis)
    print(f"WER: {wer:.2%}")
"""

# Import IR metrics
from .ir import (
    reciprocal_rank,
    precision_at_k,
    recall_at_k,
    dcg_at_k,
    ndcg_at_k,
    average_precision,
)

# Import vectorized IR metrics
from .ir_vectorized import (
    compute_metrics_batch,
    VectorizedIRMetrics,
)

# Import STT metrics
from .stt import (
    word_error_rate,
    character_error_rate,
    term_weighted_wer,
)

from .ir_aggregate import (
    compute_ir_metrics,
    log_ir_metrics,
)

from .selection import select_metrics_for_dataset_type

from .diagnostics import (
    first_relevant_rank_distribution,
    wer_recall_correlation,
    categorize_failures,
    embedding_alignment,
    per_speaker_breakdown,
    judge_calibration,
)

__all__ = [
    # IR metrics
    "reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "dcg_at_k",
    "ndcg_at_k",
    "average_precision",
    # Vectorized IR metrics
    "compute_metrics_batch",
    "VectorizedIRMetrics",
    # STT metrics
    "word_error_rate",
    "character_error_rate",
    "term_weighted_wer",
    # Aggregate helpers
    "compute_ir_metrics",
    "log_ir_metrics",
    # Diagnostic metrics
    "first_relevant_rank_distribution",
    "wer_recall_correlation",
    "categorize_failures",
    "embedding_alignment",
    "per_speaker_breakdown",
    "judge_calibration",
    # Metric selection
    "select_metrics_for_dataset_type",
]
