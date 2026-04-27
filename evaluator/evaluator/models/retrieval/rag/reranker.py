"""Cross-encoder reranking for retrieval results."""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import numpy as np

from ....logging_config import get_logger

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Tuple[Any, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """Rerank documents based on query relevance.

        Args:
            query: The query text.
            documents: List of (payload, initial_score) tuples from initial retrieval.
            top_k: Number of top results to return. If None, returns all.

        Returns:
            List of (payload, reranker_score) tuples sorted by relevance.
        """
        pass

    @abstractmethod
    def rerank_batch(
        self,
        queries: List[str],
        documents_batch: List[List[Tuple[Any, float]]],
        top_k: Optional[int] = None,
    ) -> List[List[Tuple[Any, float]]]:
        """Rerank documents for multiple queries.

        Args:
            queries: List of query texts.
            documents_batch: List of document lists for each query.
            top_k: Number of top results to return per query.

        Returns:
            List of reranked document lists.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the reranker model name."""
        pass


def _extract_text(payload: Any) -> str:
    """Extract text content from a payload object."""
    if isinstance(payload, dict):
        return str(payload.get("text", ""))
    return str(payload)


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using sentence-transformers.

    Supports models like:
    - cross-encoder/ms-marco-MiniLM-L-6-v2
    - BAAI/bge-reranker-base
    - cross-encoder/ms-marco-TinyBERT-L-2-v2
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        """Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model name or path.
            device: Device to run the model on (e.g., "cuda:0", "cpu").
                   If None, auto-detects.
            batch_size: Batch size for scoring.
            max_length: Maximum sequence length for tokenization.
        """
        self._model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers"
            ) from e

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )
        logger.info(f"Loaded CrossEncoder reranker: {model_name}")

    def name(self) -> str:
        """Return the reranker model name."""
        return self._model_name

    def rerank(
        self,
        query: str,
        documents: List[Tuple[Any, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """Rerank documents based on query relevance.

        Args:
            query: The query text.
            documents: List of (payload, initial_score) tuples.
            top_k: Number of top results to return.

        Returns:
            List of (payload, reranker_score) tuples sorted by relevance.
        """
        if not documents:
            return []

        # Extract texts for scoring
        texts = [_extract_text(payload) for payload, _ in documents]
        
        # Create query-document pairs
        pairs = [[query, text] for text in texts]

        # Get cross-encoder scores
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        # Handle both single and batch predictions
        if isinstance(scores, (int, float)):
            scores = [scores]
        scores = np.array(scores).flatten()

        # Combine payloads with new scores and sort
        scored = [
            (payload, float(score))
            for (payload, _), score in zip(documents, scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored

    def rerank_batch(
        self,
        queries: List[str],
        documents_batch: List[List[Tuple[Any, float]]],
        top_k: Optional[int] = None,
    ) -> List[List[Tuple[Any, float]]]:
        """Rerank documents for multiple queries.

        Args:
            queries: List of query texts.
            documents_batch: List of document lists for each query.
            top_k: Number of top results to return per query.

        Returns:
            List of reranked document lists.
        """
        if len(queries) != len(documents_batch):
            raise ValueError(
                f"Number of queries ({len(queries)}) must match "
                f"number of document batches ({len(documents_batch)})"
            )

        results = []
        for query, documents in zip(queries, documents_batch):
            results.append(self.rerank(query, documents, top_k))

        return results
