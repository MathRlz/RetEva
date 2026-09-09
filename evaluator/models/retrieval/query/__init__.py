"""Query optimization: rewriting, HyDE, decomposition, and multi-query generation via LLMs
to improve retrieval."""

from .optimization import (
    rewrite_query,
    generate_hypothetical_document,
    decompose_query,
    generate_multi_queries,
    combine_retrieval_results,
)

try:
    from .prompts import (  # noqa: F401
        QUERY_REWRITE_PROMPT,
        HYDE_PROMPT,
        QUERY_DECOMPOSE_PROMPT,
        MULTI_QUERY_PROMPT,
    )
    _has_prompts = True
except (ImportError, AttributeError):
    _has_prompts = False

__all__ = [
    "rewrite_query",
    "generate_hypothetical_document",
    "decompose_query",
    "generate_multi_queries",
    "combine_retrieval_results",
]

if _has_prompts:
    __all__.extend([
        "QUERY_REWRITE_PROMPT",
        "HYDE_PROMPT",
        "QUERY_DECOMPOSE_PROMPT",
        "MULTI_QUERY_PROMPT",
    ])
