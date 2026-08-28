"""create_pipeline_from_config wires the right pipelines — the build decision is derived from the
execution graph's nodes, not the ``pipeline_mode`` string (graph-first Phase 2)."""

from unittest.mock import MagicMock, patch

from evaluator.config.evaluation import EvaluationConfig


def _build(mode, text_type):
    with patch("evaluator.pipeline.factory.create_gpu_pool_from_config", return_value=None), \
         patch("evaluator.pipeline.factory.create_reranker_from_config", return_value=None), \
         patch("evaluator.pipeline.factory._create_retrieval_pipeline"), \
         patch("evaluator.pipeline.factory.create_asr_model"), \
         patch("evaluator.pipeline.factory.create_text_embedding_model"), \
         patch("evaluator.pipeline.factory.create_audio_embedding_model"), \
         patch("evaluator.pipeline.factory.ASRPipeline"), \
         patch("evaluator.pipeline.factory.TextEmbeddingPipeline") as m_text, \
         patch("evaluator.pipeline.factory.AudioEmbeddingPipeline"):
        from evaluator.pipeline.factory import create_pipeline_from_config

        cfg = EvaluationConfig()
        cfg.graph_override = {"template": mode}
        cfg.model.asr_model_type = "wav2vec2"
        cfg.model.audio_emb_model_type = "attention_pool"
        cfg.model.text_emb_model_type = text_type
        bundle = create_pipeline_from_config(cfg, cache_manager=MagicMock())
        return bundle, m_text


def test_audio_emb_builds_text_pipeline_for_crossmodal_corpus():
    # With a text embedder configured, corpus_embedding text-embeds the corpus (the cross-modal
    # APM self-retrieval setup) instead of falling to the audio-corpus TTS path.
    bundle, m_text = _build("audio_emb_retrieval", "jina_v4")
    assert bundle.audio_embedding_pipeline is not None
    assert bundle.text_embedding_pipeline is not None
    m_text.assert_called_once()


def test_audio_emb_without_text_embedder_skips_text_pipeline():
    bundle, m_text = _build("audio_emb_retrieval", None)
    assert bundle.audio_embedding_pipeline is not None
    assert bundle.text_embedding_pipeline is None
    m_text.assert_not_called()


def test_audio_emb_builds_no_asr_even_when_asr_type_set():
    # The graph for audio_emb_retrieval has no ASR (convert) node, so no ASR pipeline is built —
    # even though config.model.asr_model_type defaults to a real value. (Kills the
    # "wav2vec2 loaded for audio_emb" concern at the source.)
    bundle, _ = _build("audio_emb_retrieval", "jina_v4")
    assert bundle.asr_pipeline is None


def test_asr_text_retrieval_builds_asr_text_retrieval_not_audio():
    bundle, _ = _build("asr_text_retrieval", "jina_v4")
    assert bundle.asr_pipeline is not None
    assert bundle.text_embedding_pipeline is not None
    assert bundle.retrieval_pipeline is not None
    assert bundle.audio_embedding_pipeline is None


def test_asr_only_builds_asr_alone():
    bundle, _ = _build("asr_only", "jina_v4")
    assert bundle.asr_pipeline is not None
    assert bundle.audio_embedding_pipeline is None
    assert bundle.text_embedding_pipeline is None   # no corpus/query embed node in asr_only
    assert bundle.retrieval_pipeline is None         # no search node


def test_audio_text_retrieval_builds_audio_text_retrieval():
    bundle, _ = _build("audio_text_retrieval", "clap_text")
    assert bundle.audio_embedding_pipeline is not None
    assert bundle.text_embedding_pipeline is not None
    assert bundle.retrieval_pipeline is not None
    assert bundle.asr_pipeline is None
