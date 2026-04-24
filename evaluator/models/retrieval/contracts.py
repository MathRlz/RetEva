"""Typed contracts for retrieval payloads and scored results."""

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


RetrievalPayload = Union[Mapping[str, Any], str]
RawSearchResult = Tuple[Any, float]


@dataclass(frozen=True)
class ScoredRetrievalResult:
    """Canonical scored retrieval output entry."""

    payload: Any
    score: float


def normalize_search_results(
    results: Sequence[Union[ScoredRetrievalResult, RawSearchResult]]
) -> List[ScoredRetrievalResult]:
    """Normalize mixed retrieval result entries into dataclass contracts."""
    normalized: List[ScoredRetrievalResult] = []
    for entry in results:
        if isinstance(entry, ScoredRetrievalResult):
            normalized.append(entry)
            continue
        payload, score = entry
        normalized.append(ScoredRetrievalResult(payload=payload, score=float(score)))
    return normalized


def normalize_batch_search_results(
    batch_results: Iterable[Sequence[Union[ScoredRetrievalResult, RawSearchResult]]]
) -> List[List[ScoredRetrievalResult]]:
    """Normalize batched retrieval outputs into contract form."""
    return [normalize_search_results(results) for results in batch_results]

