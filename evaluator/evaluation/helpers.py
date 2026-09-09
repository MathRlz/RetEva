"""Shared helper functions for evaluation: payload-key extraction, relevance-mapping
construction, and DataLoader collate functions."""

from typing import Any, Dict, Iterable, List, Tuple, Union

from ..models.retrieval.contracts import (
    RetrievalPayload,
    ScoredRetrievalResult,
    normalize_search_results,
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
    """Relevance mapping from a dataset item: ``relevance_grades`` (graded) if present, else
    ``groundtruth_doc_ids`` (binary), else the item's transcription as the sole relevant doc."""
    relevance = item.get("relevance_grades")
    if isinstance(relevance, dict) and len(relevance) > 0:
        return {str(k): int(v) for k, v in relevance.items()}

    gt_doc_ids = item.get("groundtruth_doc_ids")
    if isinstance(gt_doc_ids, list) and len(gt_doc_ids) > 0:
        return {str(doc_id): 1 for doc_id in gt_doc_ids}

    return {str(item["transcription"]): 1}


def collate_fn(batch):
    return batch


__all__ = [
    "_payload_to_key",
    "_search_results_to_keys",
    "_build_relevant_from_item",
    "_branch_of",
    "collate_fn",
]


def _branch_of(producer_id: str) -> str:
    """Branch label encoded in a node id (``retrieval@corr`` → ``corr``; else ``main``)."""
    return producer_id.split("@", 1)[1] if "@" in producer_id else "main"
