"""Iterative / multi-hop RAG flow configuration.

A RAG flow is a sequence of retrieve→refine hops: embed the query, retrieve, reformulate
the query using the retrieved documents, retrieve again. ``rounds`` controls how many
retrieval passes run; the graph builder UNROLLS them into distinct node instances (the
OneOf + prior-producer wiring picks each hop's inputs automatically). ``rounds=1`` (the
default) is exactly today's single retrieval — no behavior change.
"""

from __future__ import annotations

from dataclasses import dataclass

_REFINE_METHODS = {"rewrite_with_context", "relevance_feedback", "self_rag_critique"}


@dataclass
class RagFlowConfig:
    """Multi-hop retrieve→refine flow.

    Attributes:
        rounds: number of retrieval passes (>= 1). 1 == single retrieval (default).
        refine_method: the query_refine strategy used between hops.
        refine_context_top_k: how many top retrieved docs feed the refinement.
    """

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
