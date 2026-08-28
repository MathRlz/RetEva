"""Self-retrieval corpus = unique sentences (AudioSamplesQueryDataset.get_corpus)."""

import numpy as np

from evaluator.datasets.runtime import AudioSamplesQueryDataset
from evaluator.datasets.loaders.base import AudioSample


def _sample(text, sid):
    return AudioSample(
        audio_array=np.zeros(16, dtype="float32"),
        sampling_rate=16000,
        transcription=text,
        sample_id=sid,
        language="en",
        metadata={},
    )


def test_corpus_dedups_to_unique_sentences():
    # Two audios share a sentence → ONE corpus doc; doc_id is the sentence text.
    ds = AudioSamplesQueryDataset(
        [_sample("alpha", "a"), _sample("beta", "b"), _sample("alpha", "c")]
    )
    corpus = ds.get_corpus()
    texts = sorted(d["doc_id"] for d in corpus)
    assert texts == ["alpha", "beta"]  # deduped
    assert all(d["doc_id"] == d["text"] for d in corpus)


def test_query_relevant_doc_is_its_sentence():
    ds = AudioSamplesQueryDataset([_sample("the patient is stable", "a")])
    item = ds[0]
    # the corpus doc_id equals the sentence, matching the {sentence: 1} relevance fallback
    assert item["transcription"] == "the patient is stable"
    assert ds.get_corpus()[0]["doc_id"] == "the patient is stable"


def test_corpus_doc_ids_logs_on_broken_get_corpus():
    # H1: a raising get_corpus() must be logged, not silently swallowed (which would disable
    # IR metrics with no trace). Capture off the logger directly (propagate=False once
    # setup_logging runs, so pytest caplog can't see it).
    import io
    import logging
    from evaluator.datasets.runtime import _corpus_doc_ids

    class _Broken:
        def get_corpus(self):
            raise RuntimeError("corpus file truncated")

    lg = logging.getLogger("evaluator")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.WARNING)
    lg.addHandler(handler)
    old = lg.level
    lg.setLevel(logging.WARNING)
    try:
        assert _corpus_doc_ids(_Broken()) == set()
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old)
    assert "could not read corpus" in buf.getvalue()
