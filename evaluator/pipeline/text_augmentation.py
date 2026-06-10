"""Text perturbation for robustness testing (task C2).

The thesis asks how voice→retrieval degrades under realistic query corruption. ``AudioAugmenter``
covers the acoustic side; this is the text side: inject the kinds of errors an ASR system makes on
medical speech, deterministically (seeded per item) so a corrupted variant is reproducible.

Two perturbation families, both medically motivated:

- **homophones** — words an ASR confuses acoustically (``to``/``two``, ``ileum``/``ilium``,
  ``hypo``/``hyper`` prefixes), the everyday source of retrieval drift.
- **unit/dose corruption** — the *dangerous* class: ``mg``↔``mcg``, ``ml``↔``l``, ``once``↔``twice``.
  A 1000× dose error from a unit swap is exactly what a clinical retrieval system must be robust to,
  and what CEER (`metrics/clinical.py`) measures the cost of.

Seeded via ``item_seed(seed, query_id, node_id, variant)`` so the same item always corrupts the
same way regardless of order/parallelism.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, Optional

# ASR-confusable word pairs (bidirectional). Lowercase; matched case-insensitively per word.
HOMOPHONES: Dict[str, str] = {
    "to": "two",
    "for": "four",
    "their": "there",
    "no": "know",
    "by": "buy",
    "hour": "our",
    "ileum": "ilium",
    "ilium": "ileum",
    "mucus": "mucous",
    "humerus": "humorous",
    "perineal": "peroneal",
    "dysphagia": "dysphasia",
    "hypotension": "hypertension",
    "hyperkalemia": "hypokalemia",
    "afferent": "efferent",
}

# Dangerous dose/unit confusions (bidirectional). These change clinical meaning by orders of
# magnitude or frequency — the high-stakes corruption class CEER targets.
UNIT_CONFUSIONS: Dict[str, str] = {
    "mg": "mcg",
    "mcg": "mg",
    "ml": "l",
    "g": "mg",
    "once": "twice",
    "twice": "once",
    "daily": "weekly",
    "qd": "bid",
    "bid": "tid",
}


@dataclass
class TextAugmentConfig:
    """Which text perturbations to apply + how aggressively."""

    homophones: bool = True
    unit_corruption: bool = True
    char_swap_prob: float = (
        0.0  # per-word chance of an adjacent-character transposition
    )
    max_edits: int = 2  # cap total word substitutions so the query stays recognisable


class TextAugmenter:
    """Deterministic, medically-motivated text corruption for robustness branches (C2)."""

    def __init__(self, config: Optional[TextAugmentConfig] = None) -> None:
        self.config = config or TextAugmentConfig()
        self._table: Dict[str, str] = {}
        if self.config.homophones:
            self._table.update(HOMOPHONES)
        if self.config.unit_corruption:
            self._table.update(UNIT_CONFUSIONS)

    def augment(self, text: str, seed: Optional[int] = None) -> str:
        """Return a perturbed copy of ``text`` (deterministic for a given ``seed``)."""
        if not text:
            return text
        rng = random.Random(seed)
        tokens = re.findall(r"\w+|\W+", text)  # keep whitespace/punctuation tokens
        word_idx = [i for i, t in enumerate(tokens) if t.strip() and t.isalnum()]
        rng.shuffle(word_idx)

        edits = 0
        for i in word_idx:
            if edits >= self.config.max_edits:
                break
            repl = self._table.get(tokens[i].lower())
            if repl is not None:
                tokens[i] = _match_case(tokens[i], repl)
                edits += 1
            elif (
                self.config.char_swap_prob and rng.random() < self.config.char_swap_prob
            ):
                tokens[i] = _swap_adjacent(tokens[i], rng)
                edits += 1
        return "".join(tokens)


def _match_case(original: str, repl: str) -> str:
    if original.isupper():
        return repl.upper()
    if original[:1].isupper():
        return repl.capitalize()
    return repl


def _swap_adjacent(word: str, rng: random.Random) -> str:
    if len(word) < 2:
        return word
    j = rng.randrange(len(word) - 1)
    chars = list(word)
    chars[j], chars[j + 1] = chars[j + 1], chars[j]
    return "".join(chars)
