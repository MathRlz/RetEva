"""C7.1 — phonetic (drug sound-alike) corrector + the opt-in corrected/rx metrics.

The phonetic corrector catches what the edit-distance kb corrector can't reach safely:
same-sound substitutions (ph/f, soft c, z/s, x/ks, vowel slips) at edit distance 2+,
gated on phonetic-code equality. The corrected_*/ceer_rx metrics are opt-in via
``query_correction.corrected_metrics`` — default-off reports are byte-identical.
"""
from types import SimpleNamespace

from evaluator.config.query_correction import QueryCorrectionConfig
from evaluator.evaluation.query_correction import (
    _phonetic_code,
    correct_query_texts,
    list_correctors,
)


def _cfg(**kw):
    return QueryCorrectionConfig(enabled=True, method="phonetic", **kw)


# ── the corrector ─────────────────────────────────────────────────────

def test_phonetic_registered():
    assert "phonetic" in list_correctors()


def test_sound_alike_drugs_snap():
    out = correct_query_texts(
        ["patient takes fenytoin and prednizone",
         "prescribed amoksicillin 500 mg",
         "on warferin for afib"],
        _cfg(),
    )
    assert out[0] == "patient takes phenytoin and prednisone"
    assert out[1] == "prescribed amoxicillin 500 mg"
    assert out[2] == "on warfarin for afib"


def test_different_sounding_drug_not_snapped():
    # klonopin ≠ clonidine phonetically — merging different drugs would be the dangerous
    # failure mode; kb-at-distance-1 also must not fire here (distance is 4).
    out = correct_query_texts(["took klonopin last night"], _cfg())
    assert out[0] == "took klonopin last night"


def test_valid_term_and_short_words_untouched():
    out = correct_query_texts(["metoprolol 50 mg bid for hypertension"], _cfg())
    assert out[0] == "metoprolol 50 mg bid for hypertension"


def test_edit_budget_respected():
    # Same phonetic code but 3 residual edits > default budget 2 → no snap.
    word = "prednizzzone"
    assert _phonetic_code(word) == _phonetic_code("prednisone")
    out = correct_query_texts([f"takes {word}"], _cfg())
    assert out[0] == f"takes {word}"
    out = correct_query_texts([f"takes {word}"], _cfg(phonetic_max_edits=3))
    assert out[0] == "takes prednisone"


def test_kb_terms_channel_extends_the_vocab():
    out = correct_query_texts(["injected ozempik weekly"], _cfg(kb_terms=["ozempic"]))
    assert out[0] == "injected ozempic weekly"


def test_branch_override_reaches_the_corrector():
    # Per-branch params must reach the corrector, or a branch silently runs the global
    # method (R3-P3: resolved once at setup; was the overlay allowlist).
    from types import MethodType

    from evaluator.evaluation.executor.state import RunState
    from evaluator.evaluation.handlers.query import _node_correction_config
    from evaluator.evaluation.node_config import resolve_node_config

    base = QueryCorrectionConfig(enabled=True, method="rule")
    resolved = resolve_node_config(
        base, {"method": "phonetic", "phonetic_max_edits": 3}
    )
    s = SimpleNamespace(
        current_node=SimpleNamespace(id="query_correction"),
        node_configs={"query_correction": resolved},
        query_correction_config=base,
    )
    s.resolved_config = MethodType(RunState.resolved_config, s)
    cfg = _node_correction_config(s)
    assert cfg.method == "phonetic"
    assert cfg.phonetic_max_edits == 3


# ── the opt-in metrics ────────────────────────────────────────────────

def _score(flag: bool, with_corrected: bool):
    from evaluator.evaluation.handlers.metrics import _branch_scores
    from evaluator.evaluation.item_set import ItemSet

    ids = ["q1"]
    artifacts = {
        "reference_transcription": ItemSet(ids, ["took metoprolol 50 mg"]),
        "query_text": ItemSet(ids, ["took metroprolol 50 mg"]),
    }
    if with_corrected:
        artifacts["corrected_query_text"] = ItemSet(ids, ["took metoprolol 50 mg"])
    s = SimpleNamespace(
        disable_ir_metrics=False,
        query_correction_config=QueryCorrectionConfig(corrected_metrics=flag),
        metric_allowlist=None,
        variant_rollup="mean",
    )
    return _branch_scores(s, artifacts)


def test_c7_metrics_absent_by_default():
    scores = _score(flag=False, with_corrected=True)
    assert "wer" in scores and "ceer" in scores
    assert not {"ceer_rx", "corrected_wer", "corrected_ceer",
                "corrected_ceer_rx"} & set(scores)


def test_c7_metrics_fire_when_opted_in():
    scores = _score(flag=True, with_corrected=True)
    # Default CEER counts dose units only: 'mg' survives in the raw hypothesis → 0.0.
    # The drug-aware set sees the mangled drug name → raw 0.5, corrected 0.0. This is the
    # gap that made the vocab extension load-bearing for C7.
    assert scores["ceer"].values[0] == 0.0
    assert scores["ceer_rx"].values[0] == 0.5
    assert scores["corrected_ceer_rx"].values[0] == 0.0
    assert scores["corrected_wer"].values[0] == 0.0
    assert scores["wer"].values[0] > 0.0


def test_corrected_metrics_need_the_artifact():
    scores = _score(flag=True, with_corrected=False)
    assert "ceer_rx" in scores
    assert "corrected_wer" not in scores


# ── C7.2: dose/unit plausibility (`clinical`) ─────────────────────────

def _clinical(texts, **kw):
    return correct_query_texts(
        texts, QueryCorrectionConfig(enabled=True, method="clinical", **kw))


def test_unit_slip_repaired():
    assert _clinical(["metoprolol 50 g twice daily"])[0] == \
        "metoprolol 50 mg twice daily"
    # mcg-canonical drug misheard as mg
    assert _clinical(["levothyroxine 125 mg every morning"])[0] == \
        "levothyroxine 125 mcg every morning"


def test_dose_then_drug_order_repaired():
    assert _clinical(["took 50 g of metoprolol"])[0] == "took 50 mg of metoprolol"


def test_plausible_dose_untouched():
    # 1 g metformin = 1000 mg, inside 250–2550 — stated unit is fine after conversion.
    assert _clinical(["metformin 1 g with meals"])[0] == "metformin 1 g with meals"
    assert _clinical(["metoprolol 50 mg"])[0] == "metoprolol 50 mg"


def test_units_family_and_unknown_drug_left_alone():
    # insulin doses are in 'units' — no mass conversion is safe; unknown drugs untouched.
    assert _clinical(["insulin 500 units at night"])[0] == "insulin 500 units at night"
    assert _clinical(["fakedrugol 900 g daily"])[0] == "fakedrugol 900 g daily"


def test_clinical_composes_phonetic_then_dose():
    # The phonetic snap anchors the dose check: fenytoin → phenytoin, then g → mg.
    # (metroprolol would NOT compose: its extra consonant changes the phonetic code —
    # that's the kb corrector's edit-distance-1 catch, by design.)
    assert _clinical(["fenytoin 500 g twice daily"])[0] == \
        "phenytoin 500 mg twice daily"


def test_number_is_never_rewritten():
    # 5000 mg metoprolol is implausible in every unit scale that fits... except none —
    # ambiguous/none → text left alone (we repair units, never invent numbers).
    out = _clinical(["metoprolol 5000 mg stat"])[0]
    assert "5000" in out


# ── C7.5: constrained decode (post-hoc vocab snap on LLM output) ──────

class _FakeClient:
    def __init__(self, reply):
        self.reply = reply

    def call(self, messages):
        return self.reply


def test_llm_output_snapped_to_vocab_when_constrained():
    cfg = QueryCorrectionConfig(
        enabled=True, method="llm", constrain_to_vocab=True)
    out = correct_query_texts(
        ["whatever"], cfg, client=_FakeClient("take prednizone 20 mg"))
    assert out[0] == "take prednisone 20 mg"


def test_llm_output_untouched_without_flag_and_when_valid():
    cfg = QueryCorrectionConfig(enabled=True, method="llm")
    out = correct_query_texts(
        ["whatever"], cfg, client=_FakeClient("take prednizone 20 mg"))
    assert out[0] == "take prednizone 20 mg"  # flag off → LLM output verbatim
    cfg2 = QueryCorrectionConfig(
        enabled=True, method="llm", constrain_to_vocab=True)
    out2 = correct_query_texts(
        ["whatever"], cfg2, client=_FakeClient("take prednisone 20 mg"))
    assert out2[0] == "take prednisone 20 mg"  # valid output stays


# ── C7.6: drift pins (registry ↔ builder ↔ config ↔ overlay) ──────────

def test_builder_method_choices_track_the_registry():
    from evaluator.pipeline.graph.operators_catalog import _transform_param_spec

    spec = _transform_param_spec({"op": "correct"})
    assert spec["method"]["choices"] == list_correctors()


def test_every_correction_field_is_branch_overridable():
    # R3-P3 replaced the hand-maintained allowlist (and this test's bucket audit) with
    # field-derived resolution: EVERY QueryCorrectionConfig field now reaches the corrector
    # from a node param, including the LLM-backend ones (a branch may point one corrector
    # at a different model). `corrected_metrics` stays run-level — the metric opt-in reads
    # the GLOBAL config — so overlaying it on a node is inert by design.
    import dataclasses

    from evaluator.evaluation.node_config import resolve_node_config

    base = QueryCorrectionConfig(enabled=True, method="rule")
    for f in dataclasses.fields(QueryCorrectionConfig):
        if f.name in ("method", "replacements", "kb_terms"):
            continue  # validated / container fields: covered by the dedicated tests above
        current = getattr(base, f.name)
        probe = (not current) if isinstance(current, bool) else (
            (current or 0) + 1 if isinstance(current, (int, float)) else "probe"
        )
        resolved = resolve_node_config(base, {f.name: probe})
        assert getattr(resolved, f.name) == probe, f"{f.name} ignored a branch override"


def test_new_keys_survive_the_config_fold():
    # graph node params → build_evaluation_config_kwargs fold → EvaluationConfig.query_correction
    from evaluator.config.evaluation import EvaluationConfig
    from evaluator.config.graph_config import build_evaluation_config_kwargs
    from tests.graph_test_helpers import explicit_graph

    cfg = EvaluationConfig.from_dict(build_evaluation_config_kwargs({
        "graph": explicit_graph([
            "dataset_source", "asr",
            {"id": "query_correction", "type": "query_correction",
             "params": {"enabled": True, "method": "clinical",
                        "phonetic_max_edits": 3, "constrain_to_vocab": True,
                        "corrected_metrics": True}},
        ]),
    }), validate=False)
    qc = cfg.query_correction
    assert qc.method == "clinical"
    assert qc.phonetic_max_edits == 3
    assert qc.constrain_to_vocab is True
    assert qc.corrected_metrics is True
