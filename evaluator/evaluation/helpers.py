"""Shared helper functions for evaluation.

This module contains utility functions used across evaluation functions:
- Payload key extraction
- Relevance mapping construction
- DataLoader collate functions
- IR metric aggregation
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..pipeline import RetrievalPayload
from ..models.retrieval.contracts import ScoredRetrievalResult, normalize_search_results

# Human-readable label per pipeline mode (used for log messages).
PIPELINE_MODE_LABELS: Dict[str, str] = {
    "audio_text_retrieval": "Audio-to-Text Retrieval",
    "audio_emb_retrieval": "Audio Embedding Retrieval",
    "asr_text_retrieval": "ASR + Text Retrieval",
    "asr_only": "ASR Only",
}


def detect_pipeline_mode(
    retrieval_pipeline: Optional[Any],
    asr_pipeline: Optional[Any],
    text_embedding_pipeline: Optional[Any],
    audio_embedding_pipeline: Optional[Any],
    configured_mode: Optional[str] = None,
) -> str:
    """Resolve the pipeline mode string from the set of provided pipelines.

    Single source of truth for mode detection shared by every evaluation engine.
    Pure — performs no logging — so callers control their own log output.

    ``configured_mode`` (the config's explicit ``pipeline_mode``) takes precedence: the
    factory builds pipelines *for* that mode, and pipeline-presence detection alone cannot
    tell ``audio_emb_retrieval`` cross-modal (audio query + text corpus → both an audio AND a
    text pipeline) from ``audio_text_retrieval`` fusion. Detection is the fallback when no
    config mode is available (direct pipeline callers).

    Raises:
        ValueError: If neither an audio embedding nor an ASR pipeline is provided.
    """
    if configured_mode:
        return str(configured_mode)
    if audio_embedding_pipeline is not None and text_embedding_pipeline is not None:
        return "audio_text_retrieval"
    if audio_embedding_pipeline is not None:
        return "audio_emb_retrieval"
    if asr_pipeline is not None and text_embedding_pipeline is not None:
        if retrieval_pipeline is not None:
            return "asr_text_retrieval"
        return "asr_only"
    if asr_pipeline is not None:
        return "asr_only"
    raise ValueError(
        "Must provide either audio_embedding_pipeline OR asr_pipeline. "
        "Tip: provide (asr_pipeline + text_embedding_pipeline + retrieval_pipeline), "
        "(audio_embedding_pipeline + retrieval_pipeline), or asr_pipeline only."
    )


def _payload_to_key(payload: Union[RetrievalPayload, str]) -> str:
    """Map retrieval payload to relevance key (doc_id preferred, text fallback)."""
    if isinstance(payload, dict):
        if payload.get("doc_id") is not None:
            return str(payload["doc_id"])
        if payload.get("text") is not None:
            return str(payload["text"])
    return str(payload)


def _search_results_to_keys(
    results: Iterable[Union[ScoredRetrievalResult, Tuple[Any, float]]],
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
        "language": batch[0].get("language", None),
    }


__all__ = [
    "PIPELINE_MODE_LABELS",
    "detect_pipeline_mode",
    "_payload_to_key",
    "_search_results_to_keys",
    "_build_relevant_from_item",
    "collate_fn",
    "asr_collate_fn",
]
