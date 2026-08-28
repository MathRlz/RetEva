"""A5 pins (CRITIQUE.md): CEER against a hand-computed table over its error classes.

CEER has no reference implementation, so its semantic is pinned exhaustively instead:
count-based loss over the reference's critical-token occurrences, per error class —
including the documented ceiling that a dose-VALUE error is invisible to default CEER
(numbers are not critical terms; the unit is).
"""

import pytest

from evaluator.metrics.clinical import critical_entity_error_rate


@pytest.mark.parametrize(
    "reference,hypothesis,expected",
    [
        # No critical tokens in the reference → nothing at risk → 0.
        ("what is the capital of france", "what is the capitol of france", 0.0),
        # All critical tokens preserved → 0 (word errors elsewhere don't count).
        ("take 50 mg daily", "take 50 mg dailyy", 0.0),
        # Unit swap: the reference's one critical token is lost → 1.
        ("take 50 mg daily", "take 50 mcg daily", 1.0),
        # Documented ceiling: a dose-VALUE error alone is invisible (unit preserved).
        ("take 100 mg daily", "take 200 mg daily", 0.0),
        # Partial loss: one of two critical tokens lost → 1/2.
        ("dilute 5 ml with 2 mg", "dilute 5 milliliter with 2 mg", 0.5),
        # Count-based: mg needed twice, present once → 1/2.
        ("5 mg in the morning and 10 mg at night", "5 mg in the morning and 10 at night", 0.5),
        # Extra critical tokens in the hypothesis never reduce the error.
        ("take 50 mg", "take 50 mg or 60 mcg", 0.0),
        # µg is a critical token (and the tokenizer keeps the µ).
        ("levothyroxine 125 µg", "levothyroxine 125 mg", 1.0),
        # Compound unit is one token.
        ("dose is 5 mg/kg", "dose is 5 mg per kg", 1.0),
        # Case-insensitive on both sides.
        ("take 50 MG", "take 50 Mg", 0.0),
        # Insulin-style units.
        ("inject 10 units of insulin", "inject 10 unit of insulin", 1.0),
        # Everything critical lost → 1 (both of two).
        ("10 mcg and 2 ml", "ten and two", 1.0),
        # Empty hypothesis loses every critical token.
        ("take 50 mg", "", 1.0),
        # Empty reference has nothing at risk.
        ("", "take 50 mg", 0.0),
    ],
)
def test_ceer_hand_computed_table(reference, hypothesis, expected):
    assert critical_entity_error_rate(reference, hypothesis) == pytest.approx(expected)


def test_custom_terms_extend_the_critical_set():
    # Drug names are not critical by default…
    assert critical_entity_error_rate("take warfarin", "take warfaring") == 0.0
    # …but count once passed via terms (the ceer_rx mechanism).
    assert critical_entity_error_rate(
        "take warfarin", "take warfaring", terms={"warfarin"}
    ) == 1.0


def test_frozenset_terms_are_trusted_prelowercased():
    # The cached rx path passes a frozenset that is used as-is (no per-item rebuild) —
    # so it must already be lowercase to match the lowercased tokens.
    assert critical_entity_error_rate(
        "take Warfarin 5 mg", "take 5 mg", terms=frozenset({"warfarin"})
    ) == 1.0
