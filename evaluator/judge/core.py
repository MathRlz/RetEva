"""LLM-as-judge utilities for retrieval evaluation traces."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List


def _extract_retrieved_doc_keys(trace: Dict[str, Any]) -> List[str]:
    items = trace.get("retrieved", [])
    return [str(it.get("doc_key", "")) for it in items if isinstance(it, dict)]


def judge_trace_with_openai_compatible(
    trace: Dict[str, Any],
    *,
    api_base: str,
    model: str,
    api_key: str,
    temperature: float,
    timeout_s: int,
) -> Dict[str, Any]:
    """Judge a single trace and return structured verdict.

    Expects an OpenAI-compatible chat completions endpoint.
    """
    relevant = trace.get("relevant", {})
    retrieved = _extract_retrieved_doc_keys(trace)

    prompt = (
        "You are evaluating retrieval quality for a medical QA benchmark. "
        "Given expected relevant doc ids and retrieved doc ids, score relevance.\n"
        "Return strict JSON with fields: score (0..1), verdict (PASS|FAIL), reason (short).\n"
        f"Expected relevant: {json.dumps(relevant, ensure_ascii=False)}\n"
        f"Retrieved: {json.dumps(retrieved, ensure_ascii=False)}"
    )

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "Respond with strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    }

    req = urllib.request.Request(
        api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(f"LLM judge request failed: {exc}") from exc

    try:
        body = json.loads(raw)
        content = body["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Invalid LLM judge response payload: {raw[:500]}") from exc

    try:
        verdict = json.loads(content)
    except Exception as exc:
        raise RuntimeError(
            "LLM judge returned non-JSON content. "
            f"Expected strict JSON, got: {content[:500]}"
        ) from exc

    if "score" not in verdict or "verdict" not in verdict:
        raise RuntimeError(f"LLM judge JSON missing required fields: {verdict}")

    return {
        "score": float(verdict["score"]),
        "verdict": str(verdict["verdict"]),
        "reason": str(verdict.get("reason", "")),
    }


def run_llm_judging(
    traces: List[Dict[str, Any]],
    *,
    api_base: str,
    model: str,
    api_key_env: str,
    temperature: float,
    max_cases: int,
    timeout_s: int,
) -> Dict[str, Any]:
    """Run LLM-as-judge on top traces. Raises on any critical configuration/runtime error."""
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"LLM judge is enabled but environment variable {api_key_env} is not set"
        )

    if max_cases <= 0:
        raise RuntimeError("LLM judge max_cases must be > 0 when judge is enabled")

    selected = traces[:max_cases]
    outputs: List[Dict[str, Any]] = []
    for trace in selected:
        verdict = judge_trace_with_openai_compatible(
            trace,
            api_base=api_base,
            model=model,
            api_key=api_key,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        outputs.append(
            {
                "query_id": trace.get("query_id"),
                "judge": verdict,
            }
        )

    scores: List[float] = []
    for row in outputs:
        judge_data = row.get("judge")
        if isinstance(judge_data, dict) and "score" in judge_data:
            scores.append(float(judge_data["score"]))
    return {
        "model": model,
        "api_base": api_base,
        "cases": len(outputs),
        "mean_score": (sum(scores) / len(scores)) if scores else 0.0,
        "details": outputs,
    }


def judge_multi_aspect(
    query: str,
    retrieved_text: str,
    aspects: List[str],
    *,
    api_base: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    timeout_s: int = 60,
    chain_of_thought: bool = False,
    score_aggregation: str = "average",
    aspect_weights: Dict[str, float] = None
) -> Dict[str, Any]:
    """Judge a retrieved document across multiple aspects.
    
    Args:
        query: User query text.
        retrieved_text: Retrieved document text.
        aspects: List of aspects to evaluate (relevance, accuracy, etc.).
        api_base: API base URL.
        model: Model identifier.
        api_key: API key.
        temperature: Sampling temperature.
        timeout_s: Request timeout.
        chain_of_thought: Enable chain-of-thought prompting.
        score_aggregation: How to combine aspect scores.
        aspect_weights: Optional weights for aspects.
        
    Returns:
        Dictionary with scores per aspect and overall score.
    """
    from .prompts import get_multi_aspect_prompt
    
    system_prompt, user_prompt = get_multi_aspect_prompt(
        query, retrieved_text, aspects, chain_of_thought
    )
    
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    
    req = urllib.request.Request(
        api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    response_text = data["choices"][0]["message"]["content"]
    
    # Parse scores from response
    aspect_scores = {}
    for aspect in aspects:
        # Simple extraction: look for "AspectName: X" pattern
        import re
        pattern = rf"{aspect}.*?(\d+)"
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            aspect_scores[aspect] = float(match.group(1))
        else:
            aspect_scores[aspect] = 3.0  # Default
    
    # Compute overall score
    if score_aggregation == "average":
        overall = sum(aspect_scores.values()) / len(aspect_scores)
    elif score_aggregation == "weighted":
        if not aspect_weights:
            overall = sum(aspect_scores.values()) / len(aspect_scores)
        else:
            overall = sum(
                aspect_scores.get(aspect, 0) * aspect_weights.get(aspect, 0)
                for aspect in aspects
            )
    elif score_aggregation == "min":
        overall = min(aspect_scores.values())
    elif score_aggregation == "max":
        overall = max(aspect_scores.values())
    else:
        overall = sum(aspect_scores.values()) / len(aspect_scores)
    
    return {
        "aspect_scores": aspect_scores,
        "overall_score": overall,
        "raw_response": response_text if chain_of_thought else None,
    }


def judge_single_aspect(
    query: str,
    retrieved_text: str,
    aspect: str = "relevance",
    *,
    api_base: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    timeout_s: int = 60,
    chain_of_thought: bool = False,
    few_shot_examples: List[Dict] = None
) -> Dict[str, Any]:
    """Judge a retrieved document on a single aspect.
    
    Args:
        query: User query text.
        retrieved_text: Retrieved document text.
        aspect: Aspect to evaluate.
        api_base: API base URL.
        model: Model identifier.
        api_key: API key.
        temperature: Sampling temperature.
        timeout_s: Request timeout.
        chain_of_thought: Enable chain-of-thought prompting.
        few_shot_examples: Optional few-shot examples.
        
    Returns:
        Dictionary with score and optional reasoning.
    """
    from .prompts import get_judge_prompt
    
    system_prompt, user_prompt = get_judge_prompt(
        query, retrieved_text, aspect, chain_of_thought, few_shot_examples
    )
    
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    
    req = urllib.request.Request(
        api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    response_text = data["choices"][0]["message"]["content"].strip()
    
    # Extract score
    import re
    score_match = re.search(r"(\d+)", response_text)
    score = float(score_match.group(1)) if score_match else 3.0
    
    return {
        "score": score,
        "aspect": aspect,
        "reasoning": response_text if chain_of_thought else None,
    }
