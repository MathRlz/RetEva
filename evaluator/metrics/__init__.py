"""Evaluation metrics module.

This package provides comprehensive metrics for evaluating retrieval quality
and speech recognition accuracy.

Main Components:
    - ir: Information retrieval metrics (MRR, nDCG, Recall@k, etc.)
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

# Import STT metrics
from .stt import (
    word_error_rate,
    character_error_rate,
)

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
    # STT metrics
    "word_error_rate",
    "character_error_rate",
    # Diagnostic metrics
    "first_relevant_rank_distribution",
    "wer_recall_correlation",
    "categorize_failures",
    "embedding_alignment",
    "per_speaker_breakdown",
    "judge_calibration",
]
