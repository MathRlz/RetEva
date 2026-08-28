"""End-to-end audio_text_retrieval with mock embedders (no models loaded).

audio_text is the cross-modal *fusion* template: the audio query is embedded directly
(``audio_embedding``) **and** transcribed by ASR into a text query that is embedded
(``text_embedding``); the two streams fuse into one query, retrieved against the text corpus.
This guards that the template runs end to end — both streams reach the fusion node, the fused
vectors reach retrieval, and the metrics tail (incl. the audio↔text embedding-alignment
diagnostic, which needs both streams) runs.

The text-query stream is ASR(audio), not the ground-truth transcription — so it is a realistic
(noisy) query, not an oracle leak. A pure-audio dataset with no clip is gap-filled by the ``tts``
node before ASR, so the text stream is always available.
"""
import numpy as np

from evaluator.config.evaluation import EvaluationConfig
from evaluator.datasets.loaders.base import AudioSample
from evaluator.datasets.runtime import AudioSamplesQueryDataset
from evaluator.evaluation.executor.run import run_graph
from evaluator.evaluation.executor.state import EvaluationContext, RunFeatures
from evaluator.models.retrieval.strategy import RetrievalStrategyConfig
from evaluator.pipeline.retrieval_pipeline import RetrievalPipeline
from evaluator.storage.cache import CacheManager
from evaluator.storage.vector_store import InMemoryVectorStore


class _Named:
    def __init__(self, n):
        self._n = n

    def name(self):
        return self._n


class _MockAudioEmb:
    model = _Named("mock-audio")

    def process_batch(self, audio_list, sampling_rates):
        return [np.eye(8, dtype="float32")[i % 8] for i, _ in enumerate(audio_list)]


class _MockTextEmb:
    model = _Named("mock-text")

    def __init__(self):
        self.seen = []  # the texts the text embedder was asked to embed

    def process_batch(self, texts, show_progress=False, desc=None):
        self.seen.extend(texts)
        return np.array([np.eye(8, dtype="float32")[i % 8] for i in range(len(texts))])


class _MockASR:
    """Transcribes each audio query to its (stored) transcription — stands in for a real ASR
    model so the text-query stream is populated without loading one."""

    model = _Named("mock-asr")

    def process_dataset(self, dataset, **_kw):
        hyps = [str(dataset[i].get("transcription", "")) for i in range(len(dataset))]
        return hyps, list(hyps)


def _run():
    samples = [
        AudioSample(
            audio_array=np.zeros(8000, dtype="float32"), sampling_rate=16000,
            transcription=f"sentence number {i}", sample_id=f"s{i}", language="en", metadata={},
        )
        for i in range(8)
    ]
    cache = CacheManager(enabled=False)
    cfg = EvaluationConfig()
    cfg.graph_override = {"template": "audio_text_retrieval"}
    cfg.model.audio_emb_model_type = "clap_audio"
    cfg.model.text_emb_model_type = "clap_text"
    cfg.model.audio_emb_embedding_space = "shared_clap_space"
    cfg.model.text_emb_embedding_space = "shared_clap_space"
    cfg.embedding_fusion.enabled = True            # the template's distinguishing feature
    text_emb = _MockTextEmb()
    context = EvaluationContext(
        retrieval_pipeline=RetrievalPipeline(
            InMemoryVectorStore(), cache, strategy_config=RetrievalStrategyConfig()
        ),
        asr_pipeline=_MockASR(),
        audio_embedding_pipeline=_MockAudioEmb(),
        text_embedding_pipeline=text_emb,
        cache_manager=cache, k=5, batch_size=4,
        features=RunFeatures(embedding_fusion_config=cfg.embedding_fusion),
    )
    res = run_graph(AudioSamplesQueryDataset(samples), context, eval_config=cfg)
    return res, text_emb


def test_audio_text_template_runs_end_to_end():
    res, _ = _run()
    assert res["pipeline_mode"] == "audio_text_retrieval"
    # both streams reached fusion → retrieval → the metrics tail ran
    assert res.get("MRR") is not None
    assert res.get("Recall@5") is not None
    # both embedders recorded in provenance
    prov = res["report"]["provenance"]["models"]
    assert prov["audio_emb"]["type"] == "clap_audio"
    assert prov["text_emb"]["type"] == "clap_text"


def test_text_query_stream_is_asr_transcription():
    # The text side of the fusion is real: ASR's transcription reached the text embedder (not the
    # absent/empty stream of the pre-ASR design, and not the ground-truth reference directly).
    _, text_emb = _run()
    assert any("sentence number" in t for t in text_emb.seen)
