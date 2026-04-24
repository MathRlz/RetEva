"""Query optimization module.

This package provides advanced query optimization techniques including query
rewriting, HyDE (Hypothetical Document Embeddings), query decomposition, and
multi-query generation using LLMs to improve retrieval performance.

Main Components:
    - optimization: Query optimization implementations
    - prompts: Prompt templates for query optimization

Usage:
    from evaluator.models.retrieval.query import rewrite_query, generate_hypothetical_document
    
    optimized = rewrite_query(query, config)
"""

# Import optimization functions
from .optimization import (
    rewrite_query,
    generate_hypothetical_document,
    decompose_query,
    generate_multi_queries,
    combine_retrieval_results,
    clear_llm_cache,
)

# Import prompts if they exist
try:
    from .prompts import (
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
    "clear_llm_cache",
]

if _has_prompts:
    __all__.extend([
        "QUERY_REWRITE_PROMPT",
        "HYDE_PROMPT",
        "QUERY_DECOMPOSE_PROMPT",
        "MULTI_QUERY_PROMPT",
    ])
