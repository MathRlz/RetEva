"""RAG-generation metrics for the ``answer_gen`` stage.

Retrieval metrics score *which docs* came back; these score *what the model said* with them.
Two heuristic metrics plus three LLM-judged ones (RAGAS-style, injected ``client``), each in
``[0, 1]`` (1 = best).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Sequence

from .clinical import DEFAULT_CRITICAL_TERMS

logger = logging.getLogger(__name__)

# A dose mention = a number followed by a clinically-critical unit (mg, mcg, ml, units, …).
_DOSE = re.compile(
    r"(\d+(?:\.\d+)?)\s*("
    + "|".join(sorted(map(re.escape, DEFAULT_CRITICAL_TERMS), key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z0-9]+")


def _dose_pairs(text: str) -> set:
    """Set of ``(number, unit)`` dose mentions in ``text`` (unit lowercased)."""
    return {(m.group(1), m.group(2).lower()) for m in _DOSE.finditer(text or "")}


def drug_dosage_safety(answer: str, reference: str) -> float:
    """Fraction of the reference's drug doses reproduced exactly in the answer (1 = all safe);
    a unit slip (``mg``->``mcg``) or changed number drops the pair, catching order-of-magnitude
    dosing errors. No doses in the reference -> 1.0 (nothing to get wrong)."""
    ref = _dose_pairs(reference)
    if not ref:
        return 1.0
    ans = _dose_pairs(answer)
    return len(ref & ans) / len(ref)


def context_recall(reference_answer: str, contexts: Sequence[str]) -> float:
    """Fraction of the reference answer's content words present in the retrieved context; low
    recall means the answer can't be grounded, separating a generation failure from a
    retrieval one."""
    ref_words = {w.lower() for w in _WORD.findall(reference_answer or "")}
    if not ref_words:
        return 1.0
    ctx_words = {w.lower() for w in _WORD.findall(" ".join(contexts or []))}
    return len(ref_words & ctx_words) / len(ref_words)


def _judged_score(client: Any, system: str, user: str) -> Optional[float]:
    """Call the LLM judge and parse a ``[0,1]`` score from its reply (``None`` on failure)."""
    try:
        content = client.call(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
    except Exception as exc:
        # transport failure ≠ unparseable reply — make the judge being down visible
        logger.warning("judge call failed: %s", exc)
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", content or "")
    if not m:
        return None
    val = float(m.group(1))
    if val > 1.0:  # a 0–100 or 0–10 scale → normalise
        val = val / 100.0 if val > 10.0 else val / 10.0
    return max(0.0, min(1.0, val))


def faithfulness(answer: str, contexts: Sequence[str], client: Any) -> Optional[float]:
    """LLM-judged: is every claim in ``answer`` supported by the retrieved ``contexts``?"""
    sys = (
        "You are a strict RAG grader. Score how faithfully the ANSWER is supported by the "
        "CONTEXT, 0.0 (hallucinated) to 1.0 (fully grounded). Reply with only the number."
    )
    user = f"CONTEXT:\n{chr(10).join(contexts or [])}\n\nANSWER:\n{answer}"
    return _judged_score(client, sys, user)


def answer_relevance(answer: str, question: str, client: Any) -> Optional[float]:
    """LLM-judged: does ``answer`` actually address ``question``?"""
    sys = (
        "Score how directly the ANSWER addresses the QUESTION, 0.0 (off-topic) to 1.0 "
        "(directly answers). Reply with only the number."
    )
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    return _judged_score(client, sys, user)


def factual_correctness(answer: str, reference: str, client: Any) -> Optional[float]:
    """LLM-judged: does ``answer`` agree with the REFERENCE answer?"""
    sys = (
        "Score how factually consistent the ANSWER is with the REFERENCE, 0.0 (contradicts) "
        "to 1.0 (matches). Reply with only the number."
    )
    user = f"REFERENCE:\n{reference}\n\nANSWER:\n{answer}"
    return _judged_score(client, sys, user)
