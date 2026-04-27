from types import SimpleNamespace

import pytest

from evaluator import EvaluationConfig
from evaluator.errors import ConfigurationError
from evaluator.datasets.runtime import _load_corpus_entries
from evaluator.services.evaluation_service import load_dataset


class _FakeLoader:
    def __init__(self, samples):
        self._samples = samples

    def load(self):
        return self._samples


def test_load_corpus_entries_from_json(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        '[{"doc_id":"d1","text":"alpha"},{"id":"d2","content":"beta"}]',
        encoding="utf-8",
    )

    corpus = _load_corpus_entries(str(corpus_path))
    assert len(corpus) == 2
    assert corpus[0]["doc_id"] == "d1"
    assert corpus[1]["doc_id"] == "d2"


def test_load_dataset_local_source_asr_only(monkeypatch, tmp_path):
    sample = SimpleNamespace(
        audio_array=[0.1, 0.2],
        sampling_rate=16000,
        transcription="hello world",
        sample_id="s1",
        language="en",
        metadata={},
    )

    monkeypatch.setattr(
        "evaluator.datasets.runtime.create_dataset_loader",
        lambda **kwargs: _FakeLoader([sample]),
    )

    config = EvaluationConfig()
    config.data.dataset_name = "generic_audio_transcription"
    config.data.dataset_source = "local"
    config.data.audio_dir = str(tmp_path)
    config.model.pipeline_mode = "asr_only"

    dataset = load_dataset(config, retrieval_pipeline=None, text_emb_pipeline=None)
    assert len(dataset) == 1
    item = dataset[0]
    assert item["question_id"] == "s1"
    assert item["transcription"] == "hello world"


def test_load_dataset_retrieval_requires_corpus(monkeypatch, tmp_path):
    sample = SimpleNamespace(
        audio_array=[0.1],
        sampling_rate=16000,
        transcription="query",
        sample_id="s2",
        language="en",
        metadata={},
    )
    monkeypatch.setattr(
        "evaluator.datasets.runtime.create_dataset_loader",
        lambda **kwargs: _FakeLoader([sample]),
    )

    config = EvaluationConfig()
    config.data.dataset_name = "generic_audio_query_retrieval"
    config.data.dataset_source = "local"
    config.data.audio_dir = str(tmp_path)
    config.model.pipeline_mode = "asr_text_retrieval"

    with pytest.raises(ConfigurationError, match="Retrieval mode requires non-empty corpus"):
        load_dataset(config, retrieval_pipeline=object(), text_emb_pipeline=None)
