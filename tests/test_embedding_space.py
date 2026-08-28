"""Embedding-space typing + validation + the per-instance override."""

import pytest

from evaluator.config.evaluation import EvaluationConfig
from evaluator.errors import ConfigurationError
from evaluator.evaluation.validation import validate_graph_embedding_spaces
from evaluator.pipeline.graph import build_graph_for_config


def _config(mode, *, text_emb=None, audio_emb=None, retrieval_mode="dense"):
    cfg = EvaluationConfig()
    cfg.graph_override = {"template": mode}
    if text_emb:
        cfg.model.text_emb_model_type = text_emb
    if audio_emb:
        cfg.model.audio_emb_model_type = audio_emb
    cfg.vector_db.retrieval_mode = retrieval_mode
    return cfg


def validate_embedding_spaces(cfg):
    """The per-node graph-level check (the config-level duplicate was deleted)."""
    validate_graph_embedding_spaces(build_graph_for_config(cfg), cfg)


def test_audio_emb_space_mismatch_raises():
    cfg = _config("audio_emb_retrieval", text_emb="labse", audio_emb="wavlm")
    with pytest.raises(ConfigurationError, match="Embedding-space mismatch"):
        validate_embedding_spaces(cfg)


def test_sparse_mode_skips_check():
    cfg = _config(
        "audio_emb_retrieval", text_emb="labse", audio_emb="wavlm", retrieval_mode="sparse"
    )
    validate_embedding_spaces(cfg)  # no vector comparison → no raise


def test_asr_text_same_embedder_ok():
    validate_embedding_spaces(_config("asr_text_retrieval", text_emb="labse"))


def test_embedding_space_override_declares_shared_space():
    # An APM trained to project audio into jina_v4's text space: the encoder-derived spaces
    # differ, but the SAME explicit override on both nodes declares the shared space.
    cfg = _config("audio_emb_retrieval", text_emb="jina_v4", audio_emb="attention_pool")
    with pytest.raises(ConfigurationError, match="Embedding-space mismatch"):
        validate_embedding_spaces(cfg)  # without the override → mismatch
    cfg.model.audio_emb_embedding_space = "jina_v4_space"
    cfg.model.text_emb_embedding_space = "jina_v4_space"
    validate_embedding_spaces(cfg)  # shared override → no raise


def test_embedding_space_override_must_match_on_both_sides():
    cfg = _config("audio_emb_retrieval", text_emb="jina_v4", audio_emb="attention_pool")
    cfg.model.audio_emb_embedding_space = "jina_v4_space"
    cfg.model.text_emb_embedding_space = "labse_space"
    with pytest.raises(ConfigurationError, match="Embedding-space mismatch"):
        validate_embedding_spaces(cfg)


# ── Compatible-space registry + runtime guard (Roadmap 2b) ───────────


import evaluator.models.embedding_space as es  # noqa: E402


@pytest.fixture
def _isolated_compat_registry():
    """Snapshot/restore the global compatible-space registry around a test."""
    saved = set(es._COMPATIBLE_SPACES)
    try:
        yield
    finally:
        es._COMPATIBLE_SPACES.clear()
        es._COMPATIBLE_SPACES.update(saved)


def test_spaces_compatible_basics():
    assert es.spaces_compatible("x", "x")
    assert es.spaces_compatible(None, "x") and es.spaces_compatible("x", None)
    assert not es.spaces_compatible("a", "b")


def test_registered_pair_makes_distinct_spaces_compatible(_isolated_compat_registry):
    assert not es.spaces_compatible("space_a", "space_b")
    es.register_compatible_spaces("space_a", "space_b")
    assert es.spaces_compatible("space_a", "space_b")
    assert es.spaces_compatible("space_b", "space_a")  # symmetric


def test_validator_accepts_registered_compatible_pair(_isolated_compat_registry):
    # Distinct overridden spaces would mismatch …
    cfg = _config("audio_emb_retrieval", text_emb="jina_v4", audio_emb="attention_pool")
    cfg.model.audio_emb_embedding_space = "audio_only_space"
    cfg.model.text_emb_embedding_space = "text_only_space"
    with pytest.raises(ConfigurationError, match="Embedding-space mismatch"):
        validate_embedding_spaces(cfg)
    # … until declared cross-comparable.
    es.register_compatible_spaces("audio_only_space", "text_only_space")
    validate_embedding_spaces(cfg)  # no raise


def _dense_pipeline(index_space):
    from evaluator.pipeline.retrieval_pipeline import RetrievalPipeline
    from evaluator.storage.vector_store import InMemoryVectorStore

    rp = RetrievalPipeline(vector_store=InMemoryVectorStore())
    rp.index_space_id = index_space
    return rp


def test_runtime_guard_raises_on_incompatible_query_space():
    rp = _dense_pipeline("sonar")
    with pytest.raises(es.EmbeddingSpaceMismatch, match="mismatch"):
        rp.assert_query_space("wavlm:base")


def test_runtime_guard_passes_same_or_unknown_space():
    rp = _dense_pipeline("sonar")
    rp.assert_query_space("sonar")   # identical → ok
    rp.assert_query_space(None)      # unknown query → unchecked
    _dense_pipeline(None).assert_query_space("sonar")  # unknown index → unchecked


def test_runtime_guard_passes_registered_compatible(_isolated_compat_registry):
    rp = _dense_pipeline("clap_audio")
    with pytest.raises(es.EmbeddingSpaceMismatch):
        rp.assert_query_space("clap_text")
    es.register_compatible_spaces("clap_audio", "clap_text")
    rp.assert_query_space("clap_text")  # now compatible → ok
