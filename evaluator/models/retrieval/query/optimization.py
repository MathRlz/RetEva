"""Advanced query optimization for improved retrieval.

This module implements various query optimization techniques including:
- Query rewriting: LLM-based iterative refinement
- HyDE: Hypothetical Document Embeddings
- Query decomposition: Breaking complex queries into sub-queries
- Multi-query generation: Creating query variations
"""

import os
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
import requests

from ....logging_config import get_logger
from ....config import QueryOptimizationConfig
from .prompts import (
    get_rewrite_prompt,
    get_hyde_prompt,
    get_decompose_prompt,
    get_multi_query_prompt,
)

logger = get_logger(__name__)

# Simple in-memory cache for LLM calls
_llm_cache: Dict[str, str] = {}


def _hash_request(system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
    """Create a hash key for caching LLM requests.
    
    Args:
        system_prompt: System prompt text.
        user_prompt: User prompt text.
        model: Model identifier.
        temperature: Sampling temperature.
        
    Returns:
        Hash string for cache key.
    """
    content = f"{system_prompt}|||{user_prompt}|||{model}|||{temperature}"
    return hashlib.sha256(content.encode()).hexdigest()


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    config: QueryOptimizationConfig,
    use_cache: bool = True
) -> str:
    """Call LLM API with caching.
    
    Args:
        system_prompt: System prompt text.
        user_prompt: User prompt text.
        config: Query optimization configuration.
        use_cache: Whether to use cache. Default: True.
        
    Returns:
        LLM response text.
        
    Raises:
        RuntimeError: If API call fails.
    """
    # Check cache
    if use_cache:
        cache_key = _hash_request(
            system_prompt, user_prompt, config.llm_model, config.llm_temperature
        )
        if cache_key in _llm_cache:
            logger.debug(f"LLM cache hit for query optimization")
            return _llm_cache[cache_key]
    
    # Get API key from environment
    api_key = os.getenv(config.llm_api_key_env)
    if not api_key:
        raise RuntimeError(
            f"API key not found in environment variable: {config.llm_api_key_env}"
        )
    
    # Prepare request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    payload = {
        "model": config.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.llm_temperature,
    }
    
    try:
        response = requests.post(
            config.get_api_base(),  # Use get_api_base() to support local servers
            headers=headers,
            json=payload,
            timeout=config.timeout_s,
        )
        response.raise_for_status()
        
        result = response.json()
        response_text = result["choices"][0]["message"]["content"].strip()
        
        # Cache the result
        if use_cache:
            _llm_cache[cache_key] = response_text
        
        return response_text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API call failed: {e}")
        raise RuntimeError(f"Failed to call LLM API: {e}")


def rewrite_query(
    query: str,
    config: QueryOptimizationConfig,
    context: Optional[List[str]] = None
) -> str:
    """Rewrite query using LLM for improved retrieval.
    
    Implements iterative query refinement, optionally using retrieval
    context from initial results to guide refinement.
    
    Args:
        query: Original query text.
        config: Query optimization configuration.
        context: Optional list of retrieved texts for context-aware refinement.
        
    Returns:
        Refined query text.
        
    Examples:
        >>> config = QueryOptimizationConfig(enabled=True, method="rewrite")
        >>> refined = rewrite_query("htn treatment", config)
        >>> # Returns something like: "hypertension management guidelines..."
    """
    if not config.enabled or config.method != "rewrite":
        return query
    
    logger.info(f"Rewriting query: {query}")
    
    current_query = query
    
    for iteration in range(config.max_iterations):
        # Prepare context if available and configured
        context_str = None
        if config.use_initial_context and context and iteration > 0:
            # Use top-k results as context for refinement
            top_contexts = context[:config.context_top_k]
            context_str = "\n".join([f"- {ctx}" for ctx in top_contexts])
        
        # Get prompts
        system_prompt, user_prompt = get_rewrite_prompt(current_query, context_str)
        
        # Call LLM
        try:
            refined_query = _call_llm(system_prompt, user_prompt, config)
            
            # Check if query changed significantly
            if refined_query.lower() == current_query.lower():
                logger.debug(f"Query converged after {iteration + 1} iterations")
                break
            
            current_query = refined_query
            logger.debug(f"Iteration {iteration + 1}: {current_query}")
            
        except (RuntimeError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Query rewriting failed: {e}. Using original query.")
            return query
    
    logger.info(f"Final refined query: {current_query}")
    return current_query


def generate_hypothetical_document(
    query: str,
    config: QueryOptimizationConfig
) -> str:
    """Generate hypothetical document using HyDE technique.
    
    Creates a hypothetical answer to the query, which is then embedded
    for retrieval. This can improve retrieval by matching the style and
    content of actual documents.
    
    Args:
        query: Query text.
        config: Query optimization configuration.
        
    Returns:
        Hypothetical document text to embed.
        
    Examples:
        >>> config = QueryOptimizationConfig(enabled=True, method="hyde")
        >>> doc = generate_hypothetical_document("What causes diabetes?", config)
        >>> # Returns a brief medical explanation
    """
    if not config.enabled or config.method != "hyde":
        return query
    
    logger.info(f"Generating hypothetical document for: {query}")
    
    system_prompt, user_prompt = get_hyde_prompt(query)
    
    try:
        hypothetical_doc = _call_llm(system_prompt, user_prompt, config)
        logger.debug(f"Generated document: {hypothetical_doc}")
        return hypothetical_doc
        
    except (RuntimeError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"HyDE generation failed: {e}. Using original query.")
        return query


def decompose_query(
    query: str,
    config: QueryOptimizationConfig
) -> List[str]:
    """Decompose complex query into simpler sub-queries.
    
    Breaks down a complex question into multiple focused sub-queries
    that can be searched independently and combined.
    
    Args:
        query: Complex query text.
        config: Query optimization configuration.
        
    Returns:
        List of sub-query texts.
        
    Examples:
        >>> config = QueryOptimizationConfig(enabled=True, method="decompose")
        >>> subqueries = decompose_query(
        ...     "What are the symptoms and treatments for diabetes?",
        ...     config
        ... )
        >>> # Returns: ["diabetes symptoms", "diabetes treatment options"]
    """
    if not config.enabled or config.method != "decompose":
        return [query]
    
    logger.info(f"Decomposing query: {query}")
    
    system_prompt, user_prompt = get_decompose_prompt(query)
    
    try:
        response = _call_llm(system_prompt, user_prompt, config)
        
        # Parse sub-queries (one per line)
        subqueries = [
            line.strip()
            for line in response.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        
        # Remove numbering if present (e.g., "1. ", "- ")
        subqueries = [
            line.lstrip("0123456789.-) ").strip()
            for line in subqueries
        ]
        
        # Filter out empty strings
        subqueries = [q for q in subqueries if q]
        
        if not subqueries:
            logger.warning("No sub-queries generated. Using original query.")
            return [query]
        
        logger.info(f"Generated {len(subqueries)} sub-queries")
        for i, subq in enumerate(subqueries, 1):
            logger.debug(f"  {i}. {subq}")
        
        return subqueries
        
    except (RuntimeError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Query decomposition failed: {e}. Using original query.")
        return [query]


def generate_multi_queries(
    query: str,
    config: QueryOptimizationConfig
) -> List[str]:
    """Generate multiple query variations.
    
    Creates alternative phrasings of the same query to retrieve
    diverse relevant documents.
    
    Args:
        query: Original query text.
        config: Query optimization configuration.
        
    Returns:
        List of query variations (including original).
        
    Examples:
        >>> config = QueryOptimizationConfig(enabled=True, method="multi_query")
        >>> queries = generate_multi_queries("heart attack symptoms", config)
        >>> # Returns: ["heart attack symptoms", "myocardial infarction signs", ...]
    """
    if not config.enabled or config.method != "multi_query":
        return [query]
    
    logger.info(f"Generating query variations for: {query}")
    
    system_prompt, user_prompt = get_multi_query_prompt(query)
    
    try:
        response = _call_llm(system_prompt, user_prompt, config)
        
        # Parse variations (one per line)
        variations = [
            line.strip()
            for line in response.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        
        # Remove numbering if present
        variations = [
            line.lstrip("0123456789.-) ").strip()
            for line in variations
        ]
        
        # Filter out empty strings
        variations = [q for q in variations if q]
        
        # Include original query
        all_queries = [query] + [v for v in variations if v.lower() != query.lower()]
        
        logger.info(f"Generated {len(all_queries)} query variations (including original)")
        for i, q in enumerate(all_queries, 1):
            logger.debug(f"  {i}. {q}")
        
        return all_queries
        
    except (RuntimeError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Multi-query generation failed: {e}. Using original query.")
        return [query]


def combine_retrieval_results(
    results_list: List[List[Tuple[Any, float]]],
    strategy: str = "rrf",
    k: int = 5,
    rrf_k: int = 60,
    weights: Optional[List[float]] = None
) -> List[Tuple[Any, float]]:
    """Combine results from multiple queries.
    
    Merges retrieval results from multiple queries using various strategies.
    
    Args:
        results_list: List of result lists, each containing (payload, score) tuples.
        strategy: Combination strategy. Options: "rrf", "weighted", "union", "intersection".
        k: Number of final results to return.
        rrf_k: RRF k parameter (for "rrf" strategy).
        weights: Optional weights for each query (for "weighted" strategy).
        
    Returns:
        Combined list of (payload, score) tuples.
        
    Examples:
        >>> results1 = [(doc1, 0.9), (doc2, 0.7)]
        >>> results2 = [(doc2, 0.8), (doc3, 0.6)]
        >>> combined = combine_retrieval_results([results1, results2], strategy="rrf")
    """
    if not results_list:
        return []
    
    if len(results_list) == 1:
        return results_list[0][:k]
    
    if strategy == "rrf":
        # Reciprocal Rank Fusion
        from ..rag.hybrid import reciprocal_rank_fusion
        
        # RRF expects List[List[Tuple[Any, float]]]
        combined = reciprocal_rank_fusion(results_list, k=rrf_k, top_n=k)
        return combined
    
    elif strategy == "weighted":
        # Weighted combination
        if weights is None:
            weights = [1.0 / len(results_list)] * len(results_list)
        
        combined_scores: Dict[Any, float] = {}
        
        for results, weight in zip(results_list, weights):
            for payload, score in results:
                # Use hash of str representation as key
                key = str(payload)
                if key not in combined_scores:
                    combined_scores[key] = 0.0
                combined_scores[key] += score * weight
        
        # Convert back and sort
        combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return combined[:k]
    
    elif strategy == "union":
        # Union: include all unique documents, sorted by best score
        combined_scores: Dict[str, Tuple[Any, float]] = {}
        
        for results in results_list:
            for payload, score in results:
                key = str(payload)
                if key not in combined_scores or score > combined_scores[key][1]:
                    combined_scores[key] = (payload, score)
        
        # Sort by score
        combined = sorted(combined_scores.values(), key=lambda x: x[1], reverse=True)
        return combined[:k]
    
    elif strategy == "intersection":
        # Intersection: only documents appearing in all queries
        # Track payloads and their scores across queries
        payload_scores: Dict[str, List[float]] = {}
        
        for results in results_list:
            for payload, score in results:
                key = str(payload)
                if key not in payload_scores:
                    payload_scores[key] = []
                payload_scores[key].append(score)
        
        # Keep only payloads that appear in all queries
        n_queries = len(results_list)
        intersection = [
            (key, sum(scores) / len(scores))  # Average score
            for key, scores in payload_scores.items()
            if len(scores) == n_queries
        ]
        
        # Sort by average score
        combined = sorted(intersection, key=lambda x: x[1], reverse=True)
        return combined[:k]
    
    else:
        raise ValueError(f"Unknown combination strategy: {strategy}")


def clear_llm_cache():
    """Clear the LLM call cache."""
    _llm_cache.clear()
    logger.info("Cleared LLM cache")
