"""End-to-end audio_emb_retrieval with mock embedders (no models loaded).

Guards the cross-modal audio-query → text-corpus path: the query embeddings must reach
retrieval, and the scoring/finalize tail must run (the node-list footgun where an explicit
graph drops retrieval_metrics/metrics/finalize → no metrics).
"""
import numpy as np
import pytest

from evaluator.config.evaluation import EvaluationConfig
from evaluator.datasets.loaders.base import AudioSample
from evaluator.datasets.runtime import AudioSamplesQueryDataset
from evaluator.evaluation.executor.run import run_graph
from evaluator.evaluation.executor.state import EvaluationContext
from evaluator.models.retrieval.strategy import RetrievalStrategyConfig
from evaluator.pipeline.retrieval_pipeline import RetrievalPipeline
from evaluator.storage.cache import CacheManager
from evaluator.storage.vector_store import InMemoryVectorStore


class _Named:
    def __init__(self, n):
        self._n = n

    def name(self):
        return self._n


class _MockAudioEmb:  # mimics AudioEmbeddingPipeline: process_batch + a wrapped .model
    model = _Named("mock-audio")

    def process_batch(self, audio_list, sampling_rates):
        return [np.eye(8, dtype="float32")[i % 8] for i, _ in enumerate(audio_list)]


class _MockTextEmb:
    model = _Named("mock-text")

    def process_batch(self, texts, show_progress=False, desc=None):
        return np.array([np.eye(8, dtype="float32")[i % 8] for i in range(len(texts))])


def _run(graph_override=None):
    samples = [
        AudioSample(
            audio_array=np.zeros(8000, dtype="float32"), sampling_rate=16000,
            transcription=f"sentence number {i}", sample_id=f"s{i}", language="en", metadata={},
        )
        for i in range(8)
    ]
    cache = CacheManager(enabled=False)
    cfg = EvaluationConfig()
    cfg.graph_override = {"template": "audio_emb_retrieval"}
    cfg.model.audio_emb_model_type = "attention_pool"
    cfg.model.text_emb_model_type = "jina_v4"
    # Cross-modal APM: the audio embedder is trained to project into the text corpus space, so
    # declare the shared space on both sides (the 2b contract; the mock vectors already align).
    cfg.model.audio_emb_embedding_space = "shared_apm_space"
    cfg.model.text_emb_embedding_space = "shared_apm_space"
    context = EvaluationContext(
        retrieval_pipeline=RetrievalPipeline(
            InMemoryVectorStore(), cache, strategy_config=RetrievalStrategyConfig()
        ),
        audio_embedding_pipeline=_MockAudioEmb(),
        text_embedding_pipeline=_MockTextEmb(),
        cache_manager=cache, k=5, batch_size=4,
    )
    return run_graph(
        AudioSamplesQueryDataset(samples), context,
        eval_config=cfg, graph_override=graph_override,
    )


def test_audio_emb_default_graph_produces_metrics():
    res = _run()
    assert res["pipeline_mode"] == "audio_emb_retrieval"
    assert res.get("MRR") is not None  # query vectors reached retrieval + metrics ran
    assert res.get("Recall@5") is not None
    # model info recorded for audio_emb mode (was a gap), incl. structured provenance (C6/F30)
    assert res.get("audio_embedder") == "mock-audio"
    prov = res["report"]["provenance"]["models"]
    assert prov["audio_emb"]["type"] == "attention_pool"
    assert "text_emb" in prov and prov["text_emb"]["type"] == "jina_v4"


def test_audio_emb_complete_nodelist_graph_produces_metrics():
    from evaluator.config.graph_config import build_evaluation_config_kwargs
    from tests.graph_test_helpers import explicit_graph

    nodes = [
        "dataset_source",
        {"id": "audio_embedding", "type": "audio_embedding",
         "params": {"model": "attention_pool", "dim": 8}},
        "corpus_embedding",
        {"id": "vector_db", "type": "vector_db", "params": {"store": "inmemory"}},
        {"id": "retrieval", "type": "retrieval", "params": {"k": 5}},
        "retrieval_metrics", "metrics", "finalize",
    ]
    legacy = build_evaluation_config_kwargs(
        {"graph": explicit_graph(nodes),
         "nodes": {"text_embedding": {"model": "jina_v4"}}}
    )
    res = _run(graph_override=legacy["graph_override"])
    assert res.get("MRR") is not None


def test_retrieval_without_embedder_fails_graph_validation():
    # An explicit graph that retrieves without any query embedder is caught at build time
    # with a clear "needs artifact 'query_vectors'" error (the runtime guard in _stage_retrieval
    # is the backstop for a producer that publishes None — e.g. a stale cache).
    from tests.graph_test_helpers import explicit_graph
    from evaluator.config.graph_config import build_evaluation_config_kwargs

    nodes = [
        "dataset_source", "corpus_embedding",
        {"id": "vector_db", "type": "vector_db", "params": {"store": "inmemory"}},
        {"id": "retrieval", "type": "retrieval", "params": {"k": 5}},
    ]
    legacy = build_evaluation_config_kwargs(
        {"graph": explicit_graph(nodes),
         "nodes": {"text_embedding": {"model": "jina_v4"}}}
    )
    with pytest.raises(ValueError, match="query_vectors"):
        _run(graph_override=legacy["graph_override"])
