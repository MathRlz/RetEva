"""RefAudioDatasetView must serve items from bus refs even when the base questions carry
no audio_path — the journal-resume case: tts levels skipped, refs restored, dataset
reloaded fresh (regression for the audioemb noisy_cmp resume crash)."""

import numpy as np
import pytest
import soundfile as sf

from evaluator.evaluation.audio_refs import RefAudioDatasetView
from evaluator.evaluation.item_set import ItemSet


class _Question:
    def __init__(self, qid, text):
        self.question_id = qid
        self.question_text = text
        self.audio_path = None  # journal-resumed run: tts never ran in this process
        self.groundtruth_doc_ids = [f"doc_{qid}"]
        self.relevance_grades = {f"doc_{qid}": 1}
        self.language = "en"
        self.metadata = {}


class _Base:
    """Mimics LazyAudioQueryDataset: __getitem__ REFUSES questions without audio_path."""

    def __init__(self, questions):
        self.questions = questions

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        if not q.audio_path:
            raise ValueError(f"Question {q.question_id} has no audio_path.")
        raise AssertionError("view must not fetch items through base.__getitem__")


@pytest.fixture
def ref_wav(tmp_path):
    path = tmp_path / "q1.wav"
    sf.write(path, np.zeros(1600, dtype=np.float32), 16000)
    return str(path)


def test_view_serves_items_without_base_audio_path(ref_wav):
    base = _Base([_Question("q1", "what is x?")])
    view = RefAudioDatasetView(base, ItemSet(["q1"], [ref_wav]))

    item = view[0]
    assert item["question_id"] == "q1"
    assert item["question_text"] == "what is x?"
    assert item["transcription"] == "what is x?"
    assert item["groundtruth_doc_ids"] == ["doc_q1"]
    assert item["relevance_grades"] == {"doc_q1": 1}
    assert item["sampling_rate"] == 16000
    assert item["audio_array"].shape == (1600,)


def test_view_resolves_variant_lineage(ref_wav):
    # fan-out variant id maps to its lineage parent's metadata
    base = _Base([_Question("q1", "what is x?")])
    view = RefAudioDatasetView(base, ItemSet(["q1·aug0"], [ref_wav]))
    item = view[0]
    assert item["question_id"] == "q1·aug0"
    assert item["question_text"] == "what is x?"
