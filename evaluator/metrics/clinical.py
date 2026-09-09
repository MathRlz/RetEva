"""Clinical / safety-oriented metrics.

CEER (Critical Entity Error Rate): per-item error rate over the reference's clinically
critical tokens (drug doses/units) — a low overall WER can still hide a dangerous error (e.g.
"100 micrograms" -> "100 grams"). Extend the default term set via ``domain_terms.py``.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Set

# Conservative default set of dose units / forms whose mistranscription is clinically critical.
DEFAULT_CRITICAL_TERMS: Set[str] = {
    "microgram",
    "micrograms",
    "mcg",
    "µg",
    "ug",
    "milligram",
    "milligrams",
    "mg",
    "gram",
    "grams",
    "g",
    "kilogram",
    "kilograms",
    "kg",
    "milliliter",
    "milliliters",
    "ml",
    "millilitre",
    "millilitres",
    "liter",
    "liters",
    "litre",
    "litres",
    "l",
    "unit",
    "units",
    "iu",
    "microliter",
    "microliters",
    "nanogram",
    "nanograms",
    "ng",
    "milligrams/kg",
    "mg/kg",
    "mg/ml",
}

_WORD = re.compile(r"[A-Za-zµ0-9/]+")


def _tokens(text: str) -> list:
    return [t.lower() for t in _WORD.findall(text or "")]


def critical_entity_error_rate(
    reference: str,
    hypothesis: str,
    terms: Optional[Iterable[str]] = None,
) -> float:
    """Per-item CEER: fraction of the reference's *critical* tokens not preserved (by count)
    in the hypothesis. 0.0 when the reference has no critical tokens (nothing at risk).
    """
    # A frozenset is trusted pre-lowercased (the cached rx set) — no per-item rebuild.
    if isinstance(terms, frozenset):
        critical = terms
    else:
        critical = set(
            t.lower() for t in (terms if terms is not None else DEFAULT_CRITICAL_TERMS)
        )
    ref_tokens = _tokens(reference)
    hyp_tokens = _tokens(hypothesis)
    ref_critical = [t for t in ref_tokens if t in critical]
    if not ref_critical:
        return 0.0
    # Count-based: a critical term is "lost" once per missing occurrence in the hypothesis.
    errors = 0
    from collections import Counter

    hyp_counts = Counter(hyp_tokens)
    needed = Counter(ref_critical)
    for term, n in needed.items():
        have = hyp_counts.get(term, 0)
        if have < n:
            errors += n - have
    return errors / len(ref_critical)
