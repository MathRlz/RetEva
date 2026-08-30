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


class _Sample:
    def __init__(self, sid, text):
        self.sample_id = sid
        self.transcription = text
        self.language = "pl"
        self.metadata = {"groundtruth_doc_ids": [sid]}


class _SampleBase:
    """Mimics AudioSamplesQueryDataset: samples (sample_id), no questions attr."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)


def test_view_resolves_sample_based_datasets(ref_wav):
    # admed_voice shape: samples with sample_id/transcription, no questions/audio_path.
    base = _SampleBase([_Sample("s1", "podaj dawkę leku")])
    view = RefAudioDatasetView(base, ItemSet(["s1"], [ref_wav]))
    item = view[0]
    assert item["question_id"] == "s1"
    assert item["transcription"] == "podaj dawkę leku"
    assert item["question_text"] == "podaj dawkę leku"
    assert item["groundtruth_doc_ids"] == ["s1"]
    assert item["audio_array"].shape == (1600,)


def test_augment_handles_inmemory_audio(tmp_path):
    # A sample-based dataset publishes no path refs; the augment per-item fn must accept
    # an (array, sr) payload and still write the perturbed wav.
    import numpy as np

    from evaluator.config.audio_augmentation import AudioAugmentationConfig
    from evaluator.evaluation.handlers.audio import _augment_audio_one, _inmemory_audio_refs
    from evaluator.pipeline.audio.augmentation import AudioAugmenter

    aug = AudioAugmenter(AudioAugmentationConfig(
        enabled=True, add_noise=True, noise_type="white", snr_db=15.0))
    audio = np.zeros(1600, dtype=np.float32) + 0.01
    pairs = _augment_audio_one(
        ("s1", (audio, 16000)), augmenter=aug, n_variants=1,
        base_seed=42, node_id="augment", out_dir=str(tmp_path),
    )
    assert len(pairs) == 1
    vid, out_path = pairs[0]
    assert vid == "s1"
    import soundfile as sf
    data, sr = sf.read(out_path)
    assert sr == 16000 and len(data) > 0

    # _inmemory_audio_refs over a dict-item dataset
    class _DS:
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return {"audio_array": audio, "sampling_rate": 16000, "question_id": f"s{i}"}

    from types import SimpleNamespace
    refs = _inmemory_audio_refs(SimpleNamespace(dataset=_DS()))
    assert refs is not None
    assert list(refs.ids) == ["s0", "s1"]
    assert isinstance(refs.values[0], tuple)
