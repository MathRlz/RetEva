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


# ── resolve_query_space per-node resolution (regression: false-positive mismatch on a
# multi-variant graph's own branches — reported live: pubmed_qa_tts_asr_embed_cmp.yaml's
# jina_v4 branches asserted the flat default "labse" instead of their own embedder) ──

import os  # noqa: E402

from evaluator.evaluation.validation import resolve_query_space  # noqa: E402
from evaluator.pipeline.graph import build_graph_for_config as _build_graph  # noqa: E402

_REAL_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "configs", "pubmed_qa_tts_asr_embed_cmp.yaml"
)


def _real_cfg_and_graph():
    cfg = EvaluationConfig.from_yaml(_REAL_CONFIG, validate=False)
    return cfg, _build_graph(cfg)


def test_resolve_query_space_without_producer_id_uses_flat_default():
    # Old behavior, preserved for callers with no live producer_id.
    cfg, _ = _real_cfg_and_graph()
    assert resolve_query_space(cfg, "text_query_vectors") == "labse:sentence-transformers/LaBSE"


def test_resolve_query_space_with_producer_id_uses_the_actual_branch_embedder():
    # THE bug: every branch used to assert the flat default regardless of its own
    # per-node model override — a jina_v4 branch's query space came back "labse", which
    # then mismatched its own jina_v4-built index and raised EmbeddingSpaceMismatch.
    cfg, graph = _real_cfg_and_graph()
    jina_node = next(n for n in graph.nodes if n.id == "text_embedding_xtts_whisper_jina")
    assert jina_node is not None
    space = resolve_query_space(cfg, "text_query_vectors", producer_id=jina_node.id)
    assert space == "jina_v4:jinaai/jina-embeddings-v4"

    labse_node_id = "text_embedding_xtts_whisper_labse"
    space2 = resolve_query_space(cfg, "text_query_vectors", producer_id=labse_node_id)
    assert space2 == "labse:sentence-transformers/LaBSE"


def test_resolve_query_space_fused_stream_stays_unknown_regardless_of_producer_id():
    cfg, graph = _real_cfg_and_graph()
    some_id = graph.nodes[0].id
    assert resolve_query_space(cfg, "fused_query_vectors", producer_id=some_id) is None


def test_full_graph_has_zero_embedding_space_mismatches_across_all_variants():
    # The pre-flight, per-node-aware checker (already correct) confirms the graph itself
    # is sound — the bug was purely in the SEPARATE runtime guard, resolve_query_space.
    from evaluator.evaluation.validation import check_embedding_spaces

    cfg, graph = _real_cfg_and_graph()
    assert check_embedding_spaces(graph, cfg) == []


def test_runtime_guard_no_longer_false_positives_on_a_jina_branch(monkeypatch):
    # End-to-end-ish: simulate what _stage_retrieval actually does for a jina branch's
    # retrieval node — must NOT raise now (used to raise EmbeddingSpaceMismatch).
    from evaluator.pipeline.retrieval_pipeline import RetrievalPipeline
    from evaluator.storage.vector_store import InMemoryVectorStore

    cfg, graph = _real_cfg_and_graph()
    rp = RetrievalPipeline(vector_store=InMemoryVectorStore())
    rp.index_space_id = "jina_v4:jinaai/jina-embeddings-v4"  # what vector_db_jina's index holds

    query_space = resolve_query_space(
        cfg, "text_query_vectors", producer_id="text_embedding_xtts_whisper_jina",
    )
    rp.assert_query_space(query_space)  # must not raise
