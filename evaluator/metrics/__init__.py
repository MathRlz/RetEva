"""Evaluation metrics module: information retrieval metrics (MRR, nDCG, Recall@k, …)
and speech-to-text metrics (WER, CER, …).
"""

from .ir import (
    reciprocal_rank,
    precision_at_k,
    recall_at_k,
    dcg_at_k,
    ndcg_at_k,
    average_precision,
)

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
    "reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "dcg_at_k",
    "ndcg_at_k",
    "average_precision",
    "word_error_rate",
    "character_error_rate",
    "first_relevant_rank_distribution",
    "wer_recall_correlation",
    "categorize_failures",
    "embedding_alignment",
    "per_speaker_breakdown",
    "judge_calibration",
]
