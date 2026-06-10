"""Post-ASR query correction (architecture C1).

Repairs domain errors in the ASR hypothesis before embedding/retrieval. Three pluggable
correctors:

- ``rule`` — deterministic whole-word replacement (case-insensitive, word-boundary safe), seeded
  with medical abbreviation/unit normalizations, extensible via config.
- ``kb`` — fuzzy-match each (long enough) word against a knowledge-base of canonical medical terms
  and snap near-misses back (the classic ASR error: ``metformin`` → ``met foreman``). Deterministic.
- ``llm`` — ask an LLM to repair transcription errors using only medical knowledge (injected client,
  so it's testable + endpoint-agnostic).

Every method emits a **correction diff** (raw → corrected, per item) so the experiment can report
*what* the correction changed — the evidence behind the ref/asr/asr+correction comparison.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Starter, deliberately-conservative medical normalizations (unambiguous expansions /
# abbreviation casing). The research value is the plug point; extend via config.replacements.
DEFAULT_MEDICAL_RULES: Dict[str, str] = {
    "micrograms": "microgram",
    "milligrams": "milligram",
    "iv": "IV",
    "i.v.": "IV",
    "im": "IM",
    "po": "PO",
    "bid": "BID",
    "tid": "TID",
    "qid": "QID",
}


def _build_rules(config: Any) -> Dict[str, str]:
    rules: Dict[str, str] = {}
    if getattr(config, "use_default_rules", True):
        rules.update(DEFAULT_MEDICAL_RULES)
    rules.update(getattr(config, "replacements", None) or {})
    return rules


def _compile(rules: Dict[str, str]) -> List[Tuple[re.Pattern, str]]:
    # longest-first so multi-word keys win; word-boundary, case-insensitive.
    compiled: List[Tuple[re.Pattern, str]] = []
    for wrong in sorted(rules, key=len, reverse=True):
        compiled.append(
            (re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE), rules[wrong])
        )
    return compiled


def correct_text(text: str, compiled: List[Tuple[re.Pattern, str]]) -> str:
    """Apply compiled replacement rules to one text."""
    out = text
    for pattern, repl in compiled:
        out = pattern.sub(repl, out)
    return out


# ── KB (fuzzy) corrector ──────────────────────────────────────────────
# A starter glossary of canonical medical terms an ASR commonly garbles. Extend via
# config.kb_terms. Only words ≥ 4 chars and within the edit-distance budget are snapped.
DEFAULT_KB_TERMS: Tuple[str, ...] = (
    "metformin",
    "ibuprofen",
    "amoxicillin",
    "paracetamol",
    "acetaminophen",
    "lisinopril",
    "atorvastatin",
    "omeprazole",
    "prednisone",
    "warfarin",
    "hypertension",
    "hypotension",
    "tachycardia",
    "bradycardia",
    "myocardial",
    "ischemia",
    "anticoagulant",
    "milligram",
    "microgram",
    "intravenous",
)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _kb_correct_text(text: str, terms: Tuple[str, ...], max_dist: int) -> str:
    """Snap each long-enough word to its nearest KB term within ``max_dist`` edits."""

    def repl(match: "re.Match") -> str:
        word = match.group(0)
        if len(word) < 4 or word.lower() in {t.lower() for t in terms}:
            return word
        best, best_d = None, max_dist + 1
        for term in terms:
            d = _levenshtein(word.lower(), term.lower())
            if d < best_d:
                best, best_d = term, d
        return best if best is not None and best_d <= max_dist else word

    return re.sub(r"[A-Za-z]+", repl, text)


def _llm_correct(texts: List[str], config: Any, client: Any) -> List[str]:
    """Repair transcription errors with an LLM (one call per text; empty/failed → unchanged)."""
    system = (
        "You correct speech-recognition errors in a medical query using ONLY medical "
        "knowledge. Fix mis-heard drug names, units, and terms. Preserve meaning and dosage "
        "numbers exactly. Reply with ONLY the corrected query, no preamble."
    )
    out: List[str] = []
    for t in texts:
        if not t:
            out.append(t)
            continue
        try:
            fixed = client.call(
                [{"role": "system", "content": system}, {"role": "user", "content": t}]
            )
            out.append((fixed or t).strip() or t)
        except Exception:
            out.append(t)  # never let a correction failure drop the query
    return out


def correct_query_texts(
    texts: List[str], config: Any, client: Optional[Any] = None
) -> List[str]:
    """Correct a batch of query texts via the configured method. Returns a new same-length list."""
    method = getattr(config, "method", "rule")
    if method == "rule":
        compiled = _compile(_build_rules(config))
        return (
            [correct_text(t or "", compiled) for t in texts]
            if compiled
            else list(texts)
        )
    if method == "kb":
        terms = tuple(getattr(config, "kb_terms", None) or DEFAULT_KB_TERMS)
        max_dist = int(getattr(config, "kb_max_distance", 1))
        return [_kb_correct_text(t or "", terms, max_dist) for t in texts]
    if method == "llm":
        if client is None:
            from ..llm.client import LLMClient

            client = LLMClient(config.to_llm_config(), component="query_correction")
        return _llm_correct(texts, config, client)
    raise ValueError(f"unsupported correction method {method!r}")


def correction_diff(raw: List[str], corrected: List[str]) -> List[Dict[str, Any]]:
    """Per-item record of what the corrector changed (the evidence for the correction branch)."""
    return [
        {"raw": r, "corrected": c, "changed": r != c} for r, c in zip(raw, corrected)
    ]
