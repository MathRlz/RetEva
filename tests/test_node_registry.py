"""Stage-node registry: the 11 operators, their ports, taxonomy, and handlers."""

import pytest

from evaluator.pipeline.graph.operators import OPERATORS
from evaluator.pipeline.graph.registry import (
    _NODE_REGISTRY,
    get_stage_node_def,
    is_structural,
    node_category,
    stages_in_category,
)


def test_all_operators_registered():
    assert len(OPERATORS) == 12   # asr + tts are first-class (the convert collapse retired)
    for op in OPERATORS:
        assert op in _NODE_REGISTRY, op


@pytest.mark.parametrize("op", sorted(OPERATORS))
def test_every_operator_has_a_node_def_with_ports(op):
    nd = get_stage_node_def(op)
    assert hasattr(nd, "inputs") and hasattr(nd, "outputs")


@pytest.mark.parametrize("op", sorted(OPERATORS))
def test_every_operator_has_a_handler(op):
    # A registered operator must dispatch to a handler (else a run would crash mid-graph).
    import evaluator.evaluation.handlers  # noqa: F401  (populate the handler registry)
    from evaluator.evaluation.stage_registry import get_stage_spec

    spec = get_stage_spec(op)
    assert spec is not None


def test_node_taxonomy_categories():
    # category == "model" is exactly the device/offload-managed encoder set.
    assert node_category("embed", {"axis": "corpus"}) == "model"
    assert node_category("source", {}) == "source"
    assert node_category("sink", {"target": "finalize"}) == "sink"
    assert node_category("measure", {"family": "retrieval"}) == "metric"


def test_embed_is_in_the_model_category():
    assert "embed" in stages_in_category("model")


@pytest.mark.parametrize("stage,params", [
    ("metrics", None), ("finalize", None), ("aggregate", None),
    ("transcription_metrics", None), ("retrieval_metrics", None),
    ("embedding_alignment_metrics", None), ("answer_metrics", None),
    ("build_query_traces", None), ("dataset_sink", None),
    ("leaderboard_sink", None), ("tracking_sink", None),
    ("measure", {"family": "report"}), ("sink", {"target": "aggregate"}),
])
def test_plumbing_nodes_are_structural(stage, params):
    # P2.1: the measure/report/trace + sink plumbing is auto-derived, hidden from the palette.
    assert is_structural(stage, params) is True


@pytest.mark.parametrize("stage,params", [
    ("answer_judge", None), ("measure", {"family": "judge"}),  # the judge is a Model, not plumbing
    ("asr", None), ("tts", None), ("text_embedding", None), ("audio_embedding", None),
    ("retrieval", None), ("vector_db", None), ("rerank", None), ("answer_gen", None),
    ("dataset_source", None), ("query_correction", None),
])
def test_meaningful_nodes_are_not_structural(stage, params):
    assert is_structural(stage, params) is False


def test_ground_truth_is_a_real_optional_input():
    """Ground truth is a real (optional) input PORT on the measurable node, wired from the source —
    NOT a declared annotation: retrieval←relevant_docs, asr←reference_transcription,
    answer_gen←short_answers; tts produces audio and has none."""
    from evaluator.pipeline.graph.artifacts import (
        ARTIFACT_REFERENCE_TRANSCRIPTION,
        ARTIFACT_RELEVANT_DOCS,
        ARTIFACT_RETRIEVED,
        ARTIFACT_SHORT_ANSWERS,
    )
    from evaluator.pipeline.graph.operators_catalog import _search_optional

    assert ARTIFACT_RELEVANT_DOCS in _search_optional({})                       # retrieval GT
    assert get_stage_node_def("asr").optional_inputs == \
        (ARTIFACT_REFERENCE_TRANSCRIPTION,)                                     # asr GT
    assert get_stage_node_def("tts").optional_inputs == ()                      # tts none
    # context + answer GT + the relevance the handler reads for its trace payload (R3-P2)
    assert get_stage_node_def("generate").optional_inputs == \
        (ARTIFACT_RETRIEVED, ARTIFACT_SHORT_ANSWERS, ARTIFACT_RELEVANT_DOCS)


def test_no_ground_truth_inputs_concept():
    """The GT-declaration machinery (GroundTruth record / ground_truth_inputs field /
    node_ground_truth) is gone — GT is a plain optional input, nothing special."""
    import evaluator.pipeline.introspection as intro
    from evaluator.pipeline.graph.registry import StageNodeDef

    assert not hasattr(intro, "node_ground_truth") and not hasattr(intro, "GroundTruth")
    assert "ground_truth_inputs" not in StageNodeDef.__dataclass_fields__


def test_tts_provider_choices_match_registry():
    """The tts node's provider picker mirrors the @register_tts_model registry — guard against drift
    (a newly-registered provider not offered, or a removed one still listed)."""
    from evaluator.models.registry import tts_registry

    choices = get_stage_node_def("tts").param_spec["provider"]["choices"]
    assert set(choices) == set(tts_registry.list_types())


def test_tts_is_gated_on_audio_synthesis():
    """P3.3: tts is explicit — the text→audio bridge is in the graph ONLY when audio_synthesis is
    enabled (no silent gap-fill). An audio dataset carries its own clips and needs no tts node."""
    from evaluator.pipeline.graph.assembly import FeatureSet, assemble_specs

    assert "tts" in assemble_specs(
        "asr_text_retrieval", FeatureSet(audio_synthesis_enabled=True))
    assert "tts" not in assemble_specs(
        "asr_text_retrieval", FeatureSet(audio_synthesis_enabled=False))
