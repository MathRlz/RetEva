"""RAG-generation metrics for the ``answer_gen`` stage (task C4).

Retrieval metrics score *which docs* came back; these score *what the model said* with them — the
other half of a medical RAG system's quality. Two heuristic, self-contained metrics plus three
LLM-judged ones (RAGAS-style), each in ``[0, 1]`` (1 = best):

- **drug_dosage_safety** (heuristic, the high-stakes medical one) — are the reference's drug doses
  reproduced *exactly* in the answer? A ``mg``→``mcg`` slip or a changed number scores it down; this
  is the answer-side analogue of CEER.
- **context_recall** (heuristic) — how much of the reference answer's content is actually present in
  the retrieved context (can the answer even be grounded?).
- **faithfulness / answer_relevance / factual_correctness** (LLM-judged) — claim support in context,
  on-topicness, and agreement with the reference. These take an injected ``client`` (anything with
  ``.call(messages) -> str``) so they're testable without a live endpoint and reuse the judge stack.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from .clinical import DEFAULT_CRITICAL_TERMS

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
    """Fraction of the reference's drug doses reproduced exactly in the answer (1 = all safe).

    A unit slip (``mg``→``mcg``) or a changed number drops the matching pair, so the score falls —
    catching exactly the order-of-magnitude dosing errors that make a medical answer dangerous.
    No doses in the reference → 1.0 (nothing to get wrong)."""
    ref = _dose_pairs(reference)
    if not ref:
        return 1.0
    ans = _dose_pairs(answer)
    return len(ref & ans) / len(ref)


def context_recall(reference_answer: str, contexts: Sequence[str]) -> float:
    """Fraction of the reference answer's content words present in the retrieved context.

    Low recall means the answer *cannot* be faithfully grounded — the retrieval didn't bring back
    what's needed — separating a generation failure from a retrieval one."""
    ref_words = {w.lower() for w in _WORD.findall(reference_answer or "")}
    if not ref_words:
        return 1.0
    ctx_words = {w.lower() for w in _WORD.findall(" ".join(contexts or []))}
    return len(ref_words & ctx_words) / len(ref_words)


# ── LLM-judged (RAGAS-style); client injected so they're testable + endpoint-agnostic ──


def _judged_score(client: Any, system: str, user: str) -> Optional[float]:
    """Call the LLM judge and parse a ``[0,1]`` score from its reply (``None`` on failure)."""
    try:
        content = client.call(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
    except Exception:
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
