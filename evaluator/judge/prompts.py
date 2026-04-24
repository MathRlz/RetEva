"""Prompt templates for LLM-as-judge evaluation.

This module provides prompt templates for multi-aspect judging of
retrieval results in medical domain contexts.
"""

from typing import List, Dict

# Base system prompt for judging
JUDGE_SYSTEM_PROMPT = """You are an expert medical information specialist evaluating the quality of document retrieval for medical queries.

Your task is to assess retrieved documents against specific quality criteria on a scale of 1-5:
- 1: Very poor / Not relevant
- 2: Poor / Minimally relevant
- 3: Acceptable / Moderately relevant
- 4: Good / Highly relevant
- 5: Excellent / Perfectly relevant

Be objective and consistent in your evaluations."""


# Aspect-specific evaluation prompts
ASPECT_PROMPTS = {
    "relevance": {
        "name": "Relevance",
        "description": "How well does the retrieved document match the query intent?",
        "criteria": """
- 5: Document directly addresses the query with precise information
- 4: Document covers the query topic with relevant details
- 3: Document is related but only partially addresses the query
- 2: Document is tangentially related to the query
- 1: Document is unrelated to the query
""",
    },
    "accuracy": {
        "name": "Accuracy",
        "description": "How factually correct is the information in the retrieved document?",
        "criteria": """
- 5: All information is factually accurate and up-to-date
- 4: Information is mostly accurate with minor imprecisions
- 3: Information is generally accurate but has some errors
- 2: Information contains significant inaccuracies
- 1: Information is mostly incorrect or misleading
""",
    },
    "completeness": {
        "name": "Completeness",
        "description": "How comprehensively does the document answer the query?",
        "criteria": """
- 5: Document provides complete, thorough coverage of all query aspects
- 4: Document covers most important aspects of the query
- 3: Document covers some key aspects but misses others
- 2: Document only addresses a small part of the query
- 1: Document does not address the query meaningfully
""",
    },
    "clarity": {
        "name": "Clarity",
        "description": "How clearly is the information presented?",
        "criteria": """
- 5: Information is crystal clear, well-organized, and easy to understand
- 4: Information is clear with good organization
- 3: Information is understandable but could be clearer
- 2: Information is confusing or poorly organized
- 1: Information is very unclear or incomprehensible
""",
    },
    "factuality": {
        "name": "Factuality",
        "description": "Does the document cite evidence and maintain medical accuracy?",
        "criteria": """
- 5: All claims are well-supported with proper medical evidence
- 4: Most claims are supported, minor assertions without citation
- 3: Some claims are supported, others are assertions
- 2: Few claims have supporting evidence
- 1: No evidence provided, mostly unsupported claims
""",
    },
}


def get_judge_prompt(
    query: str,
    retrieved_text: str,
    aspect: str = "relevance",
    chain_of_thought: bool = False,
    few_shot_examples: List[Dict] = None
) -> tuple[str, str]:
    """Get system and user prompts for judging.
    
    Args:
        query: User query text.
        retrieved_text: Retrieved document text.
        aspect: Evaluation aspect.
        chain_of_thought: Include chain-of-thought prompting.
        few_shot_examples: Optional few-shot examples.
        
    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    if aspect not in ASPECT_PROMPTS:
        raise ValueError(f"Unknown aspect: {aspect}. Valid: {list(ASPECT_PROMPTS.keys())}")
    
    aspect_info = ASPECT_PROMPTS[aspect]
    
    # Build system prompt
    system_prompt = JUDGE_SYSTEM_PROMPT
    
    if few_shot_examples:
        system_prompt += "\n\nHere are some examples of evaluations:\n"
        for i, example in enumerate(few_shot_examples, 1):
            system_prompt += f"\nExample {i}:"
            system_prompt += f"\nQuery: {example['query']}"
            system_prompt += f"\nDocument: {example['document']}"
            system_prompt += f"\nScore: {example['score']}"
            if 'reasoning' in example:
                system_prompt += f"\nReasoning: {example['reasoning']}"
    
    # Build user prompt
    user_prompt = f"""Evaluate the following retrieved document for {aspect_info['name'].lower()}:

Query: {query}

Retrieved Document:
{retrieved_text}

Evaluation Criteria:
{aspect_info['criteria']}

"""
    
    if chain_of_thought:
        user_prompt += """Please think through your evaluation step by step:
1. What is the query asking for?
2. What does the document contain?
3. How well does it match the query?
4. What score is most appropriate?

Provide your reasoning and then assign a score from 1-5."""
    else:
        user_prompt += "Provide only the numeric score (1-5) without explanation."
    
    return system_prompt, user_prompt


def get_multi_aspect_prompt(
    query: str,
    retrieved_text: str,
    aspects: List[str],
    chain_of_thought: bool = False
) -> tuple[str, str]:
    """Get prompt for evaluating multiple aspects at once.
    
    Args:
        query: User query text.
        retrieved_text: Retrieved document text.
        aspects: List of aspects to evaluate.
        chain_of_thought: Include chain-of-thought prompting.
        
    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = JUDGE_SYSTEM_PROMPT
    
    # Build criteria section
    criteria_text = "Evaluation Criteria:\n\n"
    for aspect in aspects:
        if aspect not in ASPECT_PROMPTS:
            raise ValueError(f"Unknown aspect: {aspect}")
        
        aspect_info = ASPECT_PROMPTS[aspect]
        criteria_text += f"{aspect_info['name']}:\n{aspect_info['criteria']}\n"
    
    user_prompt = f"""Evaluate the following retrieved document across multiple aspects:

Query: {query}

Retrieved Document:
{retrieved_text}

{criteria_text}

"""
    
    if chain_of_thought:
        user_prompt += "For each aspect, provide your reasoning and score (1-5).\n\n"
        user_prompt += "Format your response as:\n"
        for aspect in aspects:
            aspect_name = ASPECT_PROMPTS[aspect]['name']
            user_prompt += f"{aspect_name}: [reasoning] Score: [1-5]\n"
    else:
        user_prompt += "Provide scores for each aspect in this format:\n"
        for aspect in aspects:
            aspect_name = ASPECT_PROMPTS[aspect]['name']
            user_prompt += f"{aspect_name}: [score]\n"
    
    return system_prompt, user_prompt


def get_structured_judge_prompt(
    query: str,
    retrieved_text: str,
    reference_answer: str = None
) -> tuple[str, str]:
    """Get prompt for structured evaluation with detailed output.
    
    Args:
        query: User query text.
        retrieved_text: Retrieved document text.
        reference_answer: Optional reference/ground truth answer.
        
    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = JUDGE_SYSTEM_PROMPT + """

You will provide a structured evaluation in JSON format with the following fields:
- relevance_score: 1-5
- accuracy_score: 1-5
- completeness_score: 1-5
- overall_score: 1-5
- strengths: list of key strengths
- weaknesses: list of key weaknesses
- reasoning: brief explanation of scores
"""
    
    user_prompt = f"""Evaluate this retrieval result:

Query: {query}

Retrieved Document:
{retrieved_text}
"""
    
    if reference_answer:
        user_prompt += f"\nReference Answer:\n{reference_answer}\n"
    
    user_prompt += """
Provide a structured evaluation in JSON format with:
- relevance_score (1-5)
- accuracy_score (1-5)
- completeness_score (1-5)
- overall_score (1-5)
- strengths (list)
- weaknesses (list)
- reasoning (string)
"""
    
    return system_prompt, user_prompt


# Few-shot examples for medical domain
MEDICAL_FEW_SHOT_EXAMPLES = [
    {
        "query": "What are the symptoms of hypertension?",
        "document": "Hypertension, or high blood pressure, often presents with headaches, dizziness, shortness of breath, and nosebleeds. However, many patients are asymptomatic, making regular blood pressure monitoring essential.",
        "score": 5,
        "reasoning": "Document directly addresses the query, providing specific symptoms and noting the important point about asymptomatic cases.",
    },
    {
        "query": "Treatment options for type 2 diabetes",
        "document": "Diabetes is a chronic condition affecting millions worldwide. It can lead to various complications if not properly managed.",
        "score": 2,
        "reasoning": "Document mentions diabetes but does not address treatment options, which is what the query asks for.",
    },
    {
        "query": "Side effects of metformin",
        "document": "Common side effects of metformin include gastrointestinal issues such as nausea, diarrhea, and abdominal discomfort. These effects often diminish with continued use.",
        "score": 4,
        "reasoning": "Document lists specific side effects but could be more comprehensive by mentioning rarer but serious effects.",
    },
]


def get_few_shot_examples(n: int = 3) -> List[Dict]:
    """Get n few-shot examples for medical domain.
    
    Args:
        n: Number of examples to return.
        
    Returns:
        List of example dictionaries.
    """
    return MEDICAL_FEW_SHOT_EXAMPLES[:n]
