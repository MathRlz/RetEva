"""Tests for typed retrieval result contracts and adapters."""

import numpy as np

from evaluator.models.retrieval.contracts import (
    ScoredRetrievalResult,
    normalize_search_results,
    normalize_batch_search_results,
)
from evaluator.pipeline import RetrievalPipeline
from evaluator.storage import InMemoryVectorStore


def test_normalize_search_results_mixed_inputs():
    """Normalizer accepts both tuples and dataclass results."""
    entries = [
        ({"doc_id": "d1", "text": "alpha"}, 0.9),
        ScoredRetrievalResult(payload="beta", score=0.8),
    ]
    normalized = normalize_search_results(entries)
    assert all(isinstance(entry, ScoredRetrievalResult) for entry in normalized)
    assert normalized[0].payload["doc_id"] == "d1"
    assert normalized[1].payload == "beta"


def test_normalize_batch_search_results():
    """Batch normalizer keeps query grouping."""
    batch = [
        [({"doc_id": "d1"}, 0.9)],
        [ScoredRetrievalResult(payload={"doc_id": "d2"}, score=0.8)],
    ]
    normalized = normalize_batch_search_results(batch)
    assert len(normalized) == 2
    assert normalized[0][0].payload["doc_id"] == "d1"
    assert normalized[1][0].payload["doc_id"] == "d2"


def test_retrieval_pipeline_search_batch_records_returns_contracts():
    """Typed retrieval API returns ScoredRetrievalResult entries."""
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    payloads = [{"doc_id": "doc1", "text": "alpha"}, {"doc_id": "doc2", "text": "beta"}]
    store = InMemoryVectorStore()
    pipeline = RetrievalPipeline(vector_store=store)
    pipeline.build_index(vectors, metadata=payloads)

    query_embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    batch_results = pipeline.search_batch_records(query_embeddings, k=1)

    assert len(batch_results) == 1
    assert isinstance(batch_results[0][0], ScoredRetrievalResult)
    assert batch_results[0][0].payload["doc_id"] == "doc1"
