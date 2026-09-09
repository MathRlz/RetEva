"""Iterative / multi-hop RAG flow configuration.

A RAG flow is retrieve→refine hops: embed, retrieve, reformulate the query from the
retrieved docs, retrieve again. ``rounds=1`` (default) is exactly single retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

_REFINE_METHODS = {"rewrite_with_context", "relevance_feedback", "self_rag_critique"}


@dataclass
class RagFlowConfig:
    """Multi-hop retrieve→refine flow."""

    rounds: int = 1
    refine_method: str = "rewrite_with_context"
    refine_context_top_k: int = 3

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError(f"rag.rounds must be >= 1, got {self.rounds}")
        if self.refine_method not in _REFINE_METHODS:
            raise ValueError(
                f"rag.refine_method must be one of {sorted(_REFINE_METHODS)}, "
                f"got {self.refine_method!r}"
            )
        if self.refine_context_top_k < 1:
            raise ValueError(
                f"rag.refine_context_top_k must be >= 1, got {self.refine_context_top_k}"
            )

    @property
    def enabled(self) -> bool:
        """A RAG flow is 'on' (i.e. unrolls extra hops) only when rounds > 1."""
        return self.rounds > 1
