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

import contextvars
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: When set (by correct_one_text_status), _llm_correct appends one entry per item whose
#: LLM call failed — the handler surfaces the count as provenance.correction_fallbacks.
_llm_fallback_sink: "contextvars.ContextVar[Optional[list]]" = contextvars.ContextVar(
    "_llm_fallback_sink", default=None
)

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
        best, best_d, tied = None, max_dist + 1, False
        for term in terms:
            d = _levenshtein(word.lower(), term.lower())
            if d < best_d:
                best, best_d, tied = term, d, False
            elif best is not None and d == best_d:
                tied = True
        # skip ambiguous ties instead of silently taking config order (matches the
        # phonetic corrector's tie policy)
        return best if best is not None and best_d <= max_dist and not tied else word

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
        except Exception as exc:
            # never let a correction failure drop the query — but a dead endpoint
            # must not silently read as "correction ran and changed nothing"
            logger.warning("llm correction failed, keeping original text: %s", exc)
            sink = _llm_fallback_sink.get()
            if sink is not None:
                sink.append(1)
            out.append(t)
    # C7.5 — constrained decode, post-hoc: an LLM must not invent non-existent drugs.
    # Snap any out-of-vocab drug-shaped token in the OUTPUT through the phonetic corrector
    # (server-agnostic; real grammar-constrained decoding stays a deferred extension).
    if getattr(config, "constrain_to_vocab", False):
        out = _phonetic_corrector(out, config)
    return out


# ── Corrector registry (§11, C7 groundwork) ───────────────────────────
# A corrector is `(texts, config, client=None) → corrected texts` (same length).
# Register a custom one with @register_corrector("my_method") and select it via the
# query_correction node's `method` param — no core edit.

_CORRECTOR_REGISTRY: Dict[str, Any] = {}


def register_corrector(method: str):
    """Decorator: register a corrector callable under ``method``."""

    def decorator(fn):
        _CORRECTOR_REGISTRY[method] = fn
        return fn

    return decorator


def list_correctors() -> List[str]:
    """Registered corrector method names (drives the node's `method` choices)."""
    return sorted(_CORRECTOR_REGISTRY)


@register_corrector("rule")
def _rule_corrector(
    texts: List[str], config: Any, client: Optional[Any] = None
) -> List[str]:
    compiled = _compile(_build_rules(config))
    return (
        [correct_text(t or "", compiled) for t in texts] if compiled else list(texts)
    )


@register_corrector("kb")
def _kb_corrector(
    texts: List[str], config: Any, client: Optional[Any] = None
) -> List[str]:
    terms = tuple(getattr(config, "kb_terms", None) or DEFAULT_KB_TERMS)
    max_dist = int(getattr(config, "kb_max_distance", 1))
    return [_kb_correct_text(t or "", terms, max_dist) for t in texts]


@register_corrector("llm")
def _llm_corrector(
    texts: List[str], config: Any, client: Optional[Any] = None
) -> List[str]:
    if client is None:
        from ..llm_client.client import LLMClient

        client = LLMClient(config.to_llm_config(), component="query_correction")
    return _llm_correct(texts, config, client)


# ── Phonetic (drug sound-alike) corrector (C7.1) ──────────────────────
# Snaps a word to a canonical drug/term when it SOUNDS the same — the confusion class the
# edit-distance kb corrector can't reach safely (ph/f, soft/hard c, z/s, x/ks, vowel slips
# at edit distance 2+). Code equality gates the looser edit budget.


def _phonetic_code(word: str) -> str:
    """Compact Metaphone-style code. Not a full Metaphone — just the sound-alike classes
    ASR actually produces for medical terms; both the misheard word and the canonical term
    go through the same encoder, so consistency matters more than phonetic fidelity."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    w = re.sub(r"^(kn|gn|pn|wr|ps)", lambda m: m.group(1)[1], w)
    w = w.replace("ph", "f").replace("x", "ks").replace("q", "k")
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = w.replace("c", "k").replace("z", "s")
    w = re.sub(r"g(?=[eiy])", "j", w)
    w = w.replace("th", "t")
    first, rest = w[0], w[1:]
    rest = re.sub(r"[aeiouhwy]", "", rest)  # vowels + weak letters carry the slips
    return re.sub(r"(.)\1+", r"\1", first + rest)


def _phonetic_vocab(config: Any) -> Tuple[str, ...]:
    """Snap vocabulary: the bundled/overridden drug terms ∪ the ``kb_terms`` config channel."""
    from ..metrics.domain_terms import load_drug_terms

    terms = list(load_drug_terms(getattr(config, "drug_terms_path", None) or None))
    terms += [str(t) for t in (getattr(config, "kb_terms", None) or ())]
    return tuple(dict.fromkeys(t.lower() for t in terms))


@lru_cache(maxsize=8)  # keyed by the full term tuple; a process sees only a few vocabs
def _phonetic_index(terms: Tuple[str, ...]) -> Tuple[frozenset, Dict[str, List[str]]]:
    """Vocab set + code→terms index, cached per vocabulary (correction runs per item)."""
    by_code: Dict[str, List[str]] = {}
    for term in terms:
        by_code.setdefault(_phonetic_code(term), []).append(term)
    return frozenset(terms), by_code


def _phonetic_correct_text(text: str, terms: Tuple[str, ...], max_edits: int) -> str:
    """Snap each long-enough word to the unique same-sounding vocab term within
    ``max_edits`` residual edits. Ambiguity (two equally-near candidates) skips the word."""
    vocab, by_code = _phonetic_index(terms)

    def repl(match: "re.Match") -> str:
        word = match.group(0)
        low = word.lower()
        if len(word) < 4 or low in vocab:
            return word
        candidates = by_code.get(_phonetic_code(low), ())
        best, best_d, tied = None, max_edits + 1, False
        for term in candidates:
            d = _levenshtein(low, term)
            if d < best_d:
                best, best_d, tied = term, d, False
            elif d == best_d and term != best:
                tied = True
        return best if best is not None and not tied else word

    return re.sub(r"[A-Za-z]+", repl, text)


@register_corrector("phonetic")
def _phonetic_corrector(
    texts: List[str], config: Any, client: Optional[Any] = None
) -> List[str]:
    terms = _phonetic_vocab(config)
    max_edits = int(getattr(config, "phonetic_max_edits", 2))
    return [_phonetic_correct_text(t or "", terms, max_edits) for t in texts]


# ── Dose/unit plausibility (C7.2) ─────────────────────────────────────
# Repairs the units-slip class CEER exists for ("levothyroxine 125 mg" for mcg): when a
# recognised drug's dose is implausible in the stated unit and EXACTLY ONE metric-mass unit
# makes the number plausible, the unit — never the number — is rewritten. Anything
# ambiguous, unknown, or cross-family (insulin units vs mg) is left alone.

_MASS_FACTORS: Dict[str, float] = {"mcg": 0.001, "mg": 1.0, "g": 1000.0}
_UNIT_ALIASES: Dict[str, str] = {
    "microgram": "mcg", "micrograms": "mcg", "mcg": "mcg", "ug": "mcg", "µg": "mcg",
    "milligram": "mg", "milligrams": "mg", "mg": "mg",
    "gram": "g", "grams": "g", "g": "g",
}
_UNIT_ALT = "|".join(sorted(_UNIT_ALIASES, key=len, reverse=True))


def _repair_unit(drug_spec: Dict[str, Any], num: str, unit_word: str) -> Optional[str]:
    """The repaired unit string, or None to leave the text alone."""
    canonical = str(drug_spec.get("unit", "mg"))
    if canonical not in _MASS_FACTORS:
        return None  # non-mass dose family (insulin units/IU) — no safe conversion
    stated = _UNIT_ALIASES[unit_word.lower()]
    lo, hi = float(drug_spec["min"]), float(drug_spec["max"])
    val = float(num)
    scale = _MASS_FACTORS[stated] / _MASS_FACTORS[canonical]
    if lo <= val * scale <= hi:
        return None  # plausible as stated
    fits = [
        u for u in _MASS_FACTORS
        if lo <= val * _MASS_FACTORS[u] / _MASS_FACTORS[canonical] <= hi
    ]
    return fits[0] if len(fits) == 1 else None


_DOSE_NUM = r"(\d+(?:\.\d+)?)"


@lru_cache(maxsize=8)  # keyed by the full doses table; a process sees only a few
def _dose_patterns(drugs: Tuple[str, ...]) -> Tuple["re.Pattern", "re.Pattern"]:
    """Two compiled scans per doses table: "metoprolol 50 g" / "50 g (of) metoprolol"."""
    alt = "|".join(re.escape(d) for d in sorted(drugs, key=len, reverse=True))
    drug_first = re.compile(
        rf"\b({alt})\s+{_DOSE_NUM}\s*({_UNIT_ALT})\b", re.IGNORECASE
    )
    dose_first = re.compile(
        rf"\b{_DOSE_NUM}\s*({_UNIT_ALT})(?=\s+(?:of\s+)?({alt})\b)", re.IGNORECASE
    )
    return drug_first, dose_first


def _dose_correct_text(text: str, doses: Dict[str, Dict[str, Any]]) -> str:
    if not doses:
        return text
    drug_first, dose_first = _dose_patterns(tuple(sorted(doses)))

    def fix(m: "re.Match", drug_g: int, num_g: int, unit_g: int) -> str:
        spec = doses.get(m.group(drug_g).lower())
        repaired = spec and _repair_unit(spec, m.group(num_g), m.group(unit_g))
        if not repaired:
            return m.group(0)
        return m.group(0)[: m.start(unit_g) - m.start(0)] + repaired

    out = drug_first.sub(lambda m: fix(m, drug_g=1, num_g=2, unit_g=3), text)
    return dose_first.sub(lambda m: fix(m, drug_g=3, num_g=1, unit_g=2), out)


@register_corrector("clinical")
def _clinical_corrector(
    texts: List[str], config: Any, client: Optional[Any] = None
) -> List[str]:
    """Phonetic snap + dose/unit plausibility repair — the CEER-targeted combination
    (phonetic first, so a snapped drug name anchors the dose check)."""
    from ..metrics.domain_terms import load_drug_doses

    corrected = _phonetic_corrector(texts, config, client)
    doses = load_drug_doses(getattr(config, "drug_doses_path", None) or None)
    return [_dose_correct_text(t, doses) for t in corrected]


def correct_query_texts(
    texts: List[str], config: Any, client: Optional[Any] = None
) -> List[str]:
    """Correct a batch of query texts via the configured method. Returns a new same-length list."""
    method = getattr(config, "method", "rule")
    corrector = _CORRECTOR_REGISTRY.get(method)
    if corrector is None:
        known = ", ".join(list_correctors())
        raise ValueError(
            f"unsupported correction method {method!r}. Registered: {known}"
        )
    return corrector(texts, config, client)


def resolve_correction_client(config: Any) -> Optional[Any]:
    """Build the batch-level client a corrector needs **once**, so a per-item map (4b) doesn't
    rebuild it per call. Only the ``llm`` method needs one; ``rule``/``kb`` return None (and stay
    picklable for the ``process`` backend)."""
    if getattr(config, "method", "rule") != "llm":
        return None
    from ..llm_client.client import LLMClient

    return LLMClient(config.to_llm_config(), component="query_correction")


def correct_one_text(text: str, config: Any, client: Optional[Any] = None) -> str:
    """Correct a single query text — the per-item unit for the 4b ``parallel_map`` (sync default →
    byte-identical to :func:`correct_query_texts`). Pass a ``client`` from
    :func:`resolve_correction_client` to reuse one across items (the ``llm`` method; rule/kb ignore
    it). Top-level (not a closure) so the ``process`` backend can pickle it."""
    return correct_query_texts([text or ""], config, client)[0]


def correct_one_text_status(
    text: str, config: Any, client: Optional[Any] = None
) -> Tuple[str, int]:
    """Like :func:`correct_one_text`, plus the number of LLM-call fallbacks for this item
    (0 for non-llm methods) — the handler's provenance counter."""
    sink: list = []
    token = _llm_fallback_sink.set(sink)
    try:
        return correct_query_texts([text or ""], config, client)[0], len(sink)
    finally:
        _llm_fallback_sink.reset(token)


def correction_diff(raw: List[str], corrected: List[str]) -> List[Dict[str, Any]]:
    """Per-item record of what the corrector changed (the evidence for the correction branch)."""
    return [
        {"raw": r, "corrected": c, "changed": r != c} for r, c in zip(raw, corrected)
    ]
