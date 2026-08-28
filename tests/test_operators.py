"""Operator vocabulary: ALIASES (forward) ⇄ node_kind (reverse) bijection."""

from evaluator.pipeline.graph.operators import (
    ALIASES,
    expand_alias,
    is_operator,
    node_kind,
)


def test_node_kind_resolves_discriminator_fields():
    assert node_kind("embed", {"axis": "corpus"}) == "corpus_embedding"
    assert node_kind("embed", {"axis": "query", "modality": "audio"}) == "audio_embedding"
    assert node_kind("embed", {"axis": "query", "modality": "text"}) == "text_embedding"
    assert node_kind("refine", {"op": "rerank"}) == "rerank"
    assert node_kind("refine", {"op": "mmr"}) == "mmr"
    assert node_kind("search", {"method": "multi_query"}) == "multi_query_retrieval"
    assert node_kind("convert", {"op": "asr"}) == "asr"


def test_alias_roundtrip_for_every_legacy_name():
    # Each alias (operator, fixed-fields) must resolve back to its own legacy name.
    for legacy_name, (operator, fixed) in ALIASES.items():
        assert node_kind(operator, dict(fixed)) == legacy_name, legacy_name


def test_expand_alias_merges_fixed_and_explicit_params():
    op, params = expand_alias("corpus_embedding", None)
    assert op == "embed"
    assert params == {"axis": "corpus"}
    # explicit params win over the fixed discriminators
    op2, params2 = expand_alias("rerank", {"model": "ce"})
    assert op2 == "refine"
    assert params2 == {"op": "rerank", "model": "ce"}


def test_operators_are_recognized():
    assert is_operator("embed")
    assert is_operator("search")
    assert not is_operator("corpus_embedding")  # that's a legacy alias, not an operator
