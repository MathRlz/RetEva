"""Metric computation functions for evaluation.

This module re-exports all metric functions from the metrics package
for convenient access within the evaluation package.
"""
# Re-export IR metrics
from ..metrics import (
    reciprocal_rank,
    recall_at_k,
    precision_at_k,
    dcg_at_k,
    ndcg_at_k,
    average_precision,
)

# Re-export STT metrics
from ..metrics import (
    word_error_rate,
    character_error_rate,
)

# Import DatasetType for metric selection
from ..config.types import DatasetType
from typing import List, Dict, Any


def select_metrics_for_dataset_type(dataset_type: DatasetType) -> Dict[str, List[str]]:
    """
    Select appropriate metrics based on dataset type.
    
    Args:
        dataset_type: Type of dataset being evaluated.
        
    Returns:
        Dictionary with 'primary' and 'secondary' metric lists.
        
    Examples:
        >>> from evaluator.config.types import DatasetType
        >>> metrics = select_metrics_for_dataset_type(DatasetType.AUDIO_QUERY_RETRIEVAL)
        >>> print(metrics['primary'])
        ['mrr', 'ndcg@10', 'recall@10', 'wer', 'cer']
    """
    if dataset_type == DatasetType.AUDIO_QUERY_RETRIEVAL:
        return {
            "primary": ["mrr", "ndcg@10", "recall@10", "wer", "cer"],
            "secondary": ["recall@5", "recall@20", "precision@10", "map"]
        }
    
    elif dataset_type == DatasetType.TEXT_QUERY_RETRIEVAL:
        return {
            "primary": ["mrr", "ndcg@10", "recall@10"],
            "secondary": ["map", "recall@5", "precision@10", "ndcg@5"]
        }
    
    elif dataset_type == DatasetType.AUDIO_TRANSCRIPTION:
        return {
            "primary": ["wer", "cer"],
            "secondary": []
        }
    
    elif dataset_type == DatasetType.QUESTION_ANSWERING:
        return {
            "primary": ["mrr", "ndcg@10", "recall@10"],
            "secondary": ["map", "precision@5"]
        }
    
    elif dataset_type == DatasetType.MULTIMODAL_QA:
        return {
            "primary": ["mrr", "ndcg@10", "recall@10", "wer"],
            "secondary": ["map", "recall@5", "cer"]
        }
    
    elif dataset_type == DatasetType.PASSAGE_RANKING:
        return {
            "primary": ["ndcg@10", "mrr", "map"],
            "secondary": ["ndcg@5", "ndcg@20", "recall@10"]
        }
    
    else:
        # Default: compute all available metrics
        return {
            "primary": ["mrr", "ndcg@10", "wer", "cer"],
            "secondary": ["recall@5", "recall@10", "map"]
        }


__all__ = [
    # IR metrics
    "reciprocal_rank",
    "recall_at_k",
    "precision_at_k",
    "dcg_at_k",
    "ndcg_at_k",
    "average_precision",
    # STT metrics
    "word_error_rate",
    "character_error_rate",
    # Metric selection
    "select_metrics_for_dataset_type",
]
