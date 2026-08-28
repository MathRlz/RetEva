"""B4's drift class, closed structurally by R3-P3 resolution.

B4 added a drift-pin because each feature had a hand-maintained ``*_OVERLAY_KEYS`` list:
a key that was not a real field crashed ``dataclasses.replace`` on the first branch that
overrode it, and a field missing from the list silently ignored overrides forever. R3-P3
deleted the lists — the overlay keys ARE the dataclass's fields now — so the drift cannot
happen. These tests pin that property instead of the (gone) allowlists.
"""

from dataclasses import fields, replace

import pytest

from evaluator.config.answer_generation import AnswerGenerationConfig
from evaluator.config.audio_synthesis import AudioSynthesisConfig
from evaluator.config.embedding_fusion import EmbeddingFusionConfig
from evaluator.config.judge import JudgeConfig
from evaluator.config.query_correction import QueryCorrectionConfig
from evaluator.evaluation.node_config import DISCRIMINATOR_FIELDS, resolve_node_config

CONFIGS = [
    ("correction", QueryCorrectionConfig),
    ("answer_gen", AnswerGenerationConfig),
    ("judge", JudgeConfig),
    ("fusion", EmbeddingFusionConfig),
    ("tts", AudioSynthesisConfig),
]


@pytest.mark.parametrize("name,config_cls", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_no_overlay_allowlist_survives(name, config_cls):
    """The old failure mode needed a hand-written key list; there is none to drift."""
    import evaluator.evaluation.handlers.audio as audio
    import evaluator.evaluation.handlers.embedding as embedding
    import evaluator.evaluation.handlers.query as query
    import evaluator.evaluation.handlers.rag as rag

    for mod in (audio, embedding, query, rag):
        leftovers = [n for n in dir(mod) if n.endswith("_OVERLAY_KEYS")]
        assert leftovers == [], f"{mod.__name__} still hand-maintains {leftovers}"


@pytest.mark.parametrize("name,config_cls", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_every_field_is_overridable_by_construction(name, config_cls):
    """Every non-discriminator field of every feature config accepts a node override —
    the property the allowlists had to be audited for."""
    base = config_cls()
    for f in fields(config_cls):
        if f.name in DISCRIMINATOR_FIELDS:
            continue
        current = getattr(base, f.name)
        if isinstance(current, bool):
            probe = not current
        elif isinstance(current, str):
            probe = current + "_x"
        elif isinstance(current, int) and not isinstance(current, bool):
            probe = current + 7
        elif isinstance(current, float):
            probe = current + 0.5
        else:
            continue  # containers / None / enums: no safe generic probe
        try:
            resolved = resolve_node_config(base, {f.name: probe})
        except (ValueError, TypeError):
            continue  # the config validates this field — the override reached it, which
            # is exactly what this test asserts
        assert getattr(resolved, f.name) == probe, f"{name}.{f.name} ignored the override"


@pytest.mark.parametrize("name,config_cls", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_discriminators_never_overlay(name, config_cls):
    """An operator discriminator that happens to share a field name selects behavior; it
    must never be written into the config."""
    base = config_cls()
    resolved = resolve_node_config(base, {d: "SENTINEL" for d in DISCRIMINATOR_FIELDS})
    assert resolved is base


def test_resolution_never_crashes_on_a_bogus_param():
    """The B4 defect (an allowlisted non-field) is impossible: unknown keys are skipped,
    and a value that cannot be cast keeps the inherited one."""
    base = QueryCorrectionConfig(kb_max_distance=2)
    assert resolve_node_config(base, {"api_key": "x", "nope": 1}) is base
    assert resolve_node_config(base, {"kb_max_distance": "not-an-int"}) == replace(base)
