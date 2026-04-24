"""Shared helper functions for evaluation.

This module contains utility functions used across evaluation functions:
- Payload key extraction
- Relevance mapping construction
- DataLoader collate functions
- IR metric aggregation
"""
from typing import Any, Dict, Iterable, List, Tuple, Union

from ..pipeline import RetrievalPayload
from ..models.retrieval.contracts import ScoredRetrievalResult, normalize_search_results


def _payload_to_key(payload: Union[RetrievalPayload, str]) -> str:
    """Map retrieval payload to relevance key (doc_id preferred, text fallback)."""
    if isinstance(payload, dict):
        if payload.get("doc_id") is not None:
            return str(payload["doc_id"])
        if payload.get("text") is not None:
            return str(payload["text"])
    return str(payload)


def _search_results_to_keys(
    results: Iterable[Union[ScoredRetrievalResult, Tuple[Any, float]]]
) -> List[str]:
    """Convert mixed retrieval result entries to canonical relevance keys."""
    normalized = normalize_search_results(list(results))
    return [_payload_to_key(result.payload) for result in normalized]


def _build_relevant_from_item(item: Dict[str, Any]) -> Dict[str, int]:
    """Build relevance mapping from dataset item with backwards compatibility.
    
    Supports multiple formats:
    - relevance_grades dict: {doc_id: grade} for graded relevance
    - groundtruth_doc_ids list: [doc_id, ...] for binary relevance
    - Falls back to transcription as single relevant document
    """
    relevance = item.get("relevance_grades")
    if isinstance(relevance, dict) and len(relevance) > 0:
        return {str(k): int(v) for k, v in relevance.items()}

    gt_doc_ids = item.get("groundtruth_doc_ids")
    if isinstance(gt_doc_ids, list) and len(gt_doc_ids) > 0:
        return {str(doc_id): 1 for doc_id in gt_doc_ids}

    return {str(item["transcription"]): 1}


def collate_fn(batch):
    """Simple collate function for DataLoader - returns batch as-is."""
    return batch


def asr_collate_fn(batch):
    """Collate function that groups audio data for audio embedding mode.
    
    Returns a dictionary with:
    - audio_arrays: List of audio tensors
    - sampling_rates: List of sampling rates
    - transcriptions: List of ground truth texts
    - language: Language code from first batch item
    """
    import torch
    
    audio_arrays = []
    sampling_rates = []
    transcriptions = []
    
    for item in batch:
        audio = item["audio_array"]
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio)
        audio_arrays.append(audio)
        sampling_rates.append(item["sampling_rate"])
        transcriptions.append(item["transcription"])
    
    return {
        "audio_arrays": audio_arrays,
        "sampling_rates": sampling_rates,
        "transcriptions": transcriptions,
        "language": batch[0].get("language", None)
    }


__all__ = [
    "_payload_to_key",
    "_search_results_to_keys",
    "_build_relevant_from_item",
    "collate_fn",
    "asr_collate_fn",
]
