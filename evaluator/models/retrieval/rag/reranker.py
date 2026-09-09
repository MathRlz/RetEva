"""Cross-encoder reranking for retrieval results.

A reranker is duck-typed: implement ``rerank(query, documents, top_k=None) ->
List[(payload, score)]`` + ``name()`` and register with ``@register_reranker_model``
(see ``evaluator-architecture.md`` §15.1) — no base class required.
"""
from typing import Any, List, Optional, Tuple

import numpy as np

from ....logging_config import get_logger
from ...registry import register_reranker_model

logger = get_logger(__name__)


def _extract_text(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("text", ""))
    return str(payload)


@register_reranker_model(
    'cross_encoder',
    default_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
    description='Cross-encoder reranker using sentence-transformers',
)
class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
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
        return self._model_name

    def rerank(
        self,
        query: str,
        documents: List[Tuple[Any, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """Rerank documents based on query relevance."""
        if not documents:
            return []

        texts = [_extract_text(payload) for payload, _ in documents]
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # cross-encoder .predict returns a bare scalar for a single pair, an array for a batch
        if isinstance(scores, (int, float)):
            scores = [scores]
        scores = np.array(scores).flatten()

        scored = [
            (payload, float(score))
            for (payload, _), score in zip(documents, scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored
