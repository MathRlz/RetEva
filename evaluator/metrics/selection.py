"""Default metric selection per dataset type.

Maps a :class:`~evaluator.config.types.DatasetType` to its recommended primary
+ secondary metric names. Used as a fallback when a config does not pin an
explicit metric set.
"""

from typing import Dict, List

from ..config.types import DatasetType


def select_metrics_for_dataset_type(dataset_type: DatasetType) -> Dict[str, List[str]]:
    """Select appropriate metrics based on dataset type.

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
            "secondary": ["recall@5", "recall@20", "precision@10", "map"],
        }

    elif dataset_type == DatasetType.TEXT_QUERY_RETRIEVAL:
        return {
            "primary": ["mrr", "ndcg@10", "recall@10"],
            "secondary": ["map", "recall@5", "precision@10", "ndcg@5"],
        }

    elif dataset_type == DatasetType.AUDIO_TRANSCRIPTION:
        return {"primary": ["wer", "cer"], "secondary": []}

    elif dataset_type == DatasetType.QUESTION_ANSWERING:
        return {
            "primary": ["mrr", "ndcg@10", "recall@10"],
            "secondary": ["map", "precision@5"],
        }

    elif dataset_type == DatasetType.MULTIMODAL_QA:
        return {
            "primary": ["mrr", "ndcg@10", "recall@10", "wer"],
            "secondary": ["map", "recall@5", "cer"],
        }

    elif dataset_type == DatasetType.PASSAGE_RANKING:
        return {
            "primary": ["ndcg@10", "mrr", "map"],
            "secondary": ["ndcg@5", "ndcg@20", "recall@10"],
        }

    else:
        # Default: compute all available metrics
        return {
            "primary": ["mrr", "ndcg@10", "wer", "cer"],
            "secondary": ["recall@5", "recall@10", "map"],
        }


__all__ = ["select_metrics_for_dataset_type"]
