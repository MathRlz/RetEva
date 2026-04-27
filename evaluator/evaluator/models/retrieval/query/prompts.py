"""Medical-domain specific prompt templates for query optimization.

This module provides prompt templates for various query optimization techniques
including query rewriting, HyDE (Hypothetical Document Embeddings), query
decomposition, and multi-query generation. All prompts are tailored for
medical information retrieval.
"""

# Query Rewriting Prompts
QUERY_REWRITE_SYSTEM_PROMPT = """You are a medical information retrieval expert. Your task is to refine search queries to improve medical document retrieval.

Guidelines:
- Expand medical abbreviations and acronyms
- Add relevant medical terminology
- Clarify ambiguous terms
- Keep queries concise and focused
- Preserve the original intent
"""

QUERY_REWRITE_USER_PROMPT = """Refine this medical search query to improve retrieval results:

Original Query: {query}

Provide an improved version that is more specific and uses appropriate medical terminology. Return only the refined query without explanation."""

QUERY_REWRITE_WITH_CONTEXT_PROMPT = """Refine this medical search query based on initial retrieval results:

Original Query: {query}

Top Retrieved Results:
{context}

Provide an improved query that addresses gaps or ambiguities revealed by these results. Return only the refined query without explanation. Query is a question."""


# HyDE (Hypothetical Document Embeddings) Prompts
HYDE_SYSTEM_PROMPT = """You are a medical expert writing concise, factual responses to medical queries.

Guidelines:
- Write as if you're answering from a medical textbook or clinical guideline
- Use appropriate medical terminology
- Be specific and factual
- Keep responses focused (2-3 sentences)
- Do not add disclaimers or warnings
"""

HYDE_USER_PROMPT = """Write a brief, factual answer to this medical query as it would appear in a medical reference document:

Query: {query}

Provide a concise, authoritative answer (2-3 sentences) using appropriate medical terminology."""


# Query Decomposition Prompts
QUERY_DECOMPOSE_SYSTEM_PROMPT = """You are a medical information specialist. Your task is to break down complex medical queries into simpler sub-queries.

Guidelines:
- Identify distinct medical concepts or questions
- Each sub-query should be independently searchable
- Preserve medical terminology
- Return 2-4 sub-queries for most questions
- Each sub-query should contribute to answering the original question
"""

QUERY_DECOMPOSE_USER_PROMPT = """Break down this complex medical query into simpler sub-queries:

Query: {query}

Provide 2-4 focused sub-queries that together address the original question. Return only the sub-queries, one per line, without numbering or explanation."""


# Multi-Query Generation Prompts
MULTI_QUERY_SYSTEM_PROMPT = """You are a medical information retrieval expert. Your task is to generate query variations that capture different perspectives of the same medical question.

Guidelines:
- Use synonyms and alternative medical terms
- Vary the phrasing while keeping the meaning
- Include both technical and lay terminology when appropriate
- Generate 3-5 variations
- Each variation should retrieve potentially different but relevant documents
"""

MULTI_QUERY_USER_PROMPT = """Generate variations of this medical search query:

Original Query: {query}

Provide 3-5 alternative phrasings that capture the same information need. Use different medical terminology and perspectives. Return only the query variations, one per line, without numbering or explanation."""


# Medical Domain Context
MEDICAL_CONTEXT_EXAMPLES = """Example medical query refinements:

1. Original: "htn treatment"
   Refined: "hypertension management guidelines and pharmacological treatment options"

2. Original: "covid symptoms"
   Refined: "SARS-CoV-2 infection clinical presentation and symptomatology"

3. Original: "diabetes meds"
   Refined: "pharmacological management of diabetes mellitus including oral hypoglycemic agents and insulin therapy"

4. Original: "heart attack signs"
   Refined: "myocardial infarction clinical presentation and diagnostic criteria"
"""


def get_rewrite_prompt(query: str, context: str = None) -> tuple[str, str]:
    """Get system and user prompts for query rewriting.
    
    Args:
        query: Original query to rewrite.
        context: Optional context from initial retrieval results.
        
    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = QUERY_REWRITE_SYSTEM_PROMPT
    
    if context:
        user_prompt = QUERY_REWRITE_WITH_CONTEXT_PROMPT.format(
            query=query,
            context=context
        )
    else:
        user_prompt = QUERY_REWRITE_USER_PROMPT.format(query=query)
    
    return system_prompt, user_prompt


def get_hyde_prompt(query: str) -> tuple[str, str]:
    """Get system and user prompts for HyDE.
    
    Args:
        query: Query for which to generate hypothetical document.
        
    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = HYDE_SYSTEM_PROMPT
    user_prompt = HYDE_USER_PROMPT.format(query=query)
    return system_prompt, user_prompt


def get_decompose_prompt(query: str) -> tuple[str, str]:
    """Get system and user prompts for query decomposition.
    
    Args:
        query: Complex query to decompose.
        
    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = QUERY_DECOMPOSE_SYSTEM_PROMPT
    user_prompt = QUERY_DECOMPOSE_USER_PROMPT.format(query=query)
    return system_prompt, user_prompt


def get_multi_query_prompt(query: str) -> tuple[str, str]:
    """Get system and user prompts for multi-query generation.
    
    Args:
        query: Query for which to generate variations.
        
    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = MULTI_QUERY_SYSTEM_PROMPT
    user_prompt = MULTI_QUERY_USER_PROMPT.format(query=query)
    return system_prompt, user_prompt
