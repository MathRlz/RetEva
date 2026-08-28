"""Dataset split selection + unique sample_ids (admed / hani self-retrieval)."""

import numpy as np
import pytest

from evaluator.config.data import DataConfig


class _Wav:
    def squeeze(self):
        return self

    def numpy(self):
        return np.zeros(16, dtype="float32")


def test_admed_declares_train_test_default_both():
    from evaluator.datasets.descriptor import get_descriptor

    d = get_descriptor("admed_voice")
    assert list(d.splits) == ["train", "test"]
    assert d.default_split == "both"  # full corpus unless a holdout is asked for


def test_admed_validation_split_rejected():
    from evaluator.datasets.builtins.admed_voice import AdmedVoiceDataset

    with pytest.raises(ValueError, match="no 'validation' split"):
        AdmedVoiceDataset.from_config(
            DataConfig(dataset_name="admed_voice", split="validation")
        )


def test_admed_sample_ids_unique_despite_duplicate_rows(monkeypatch):
    # admed filenames repeat across category/speaker dirs, and the corpus has genuinely
    # duplicated file rows — ids must still be unique (else dataset_source drops all GT).
    import pandas as pd

    from evaluator.datasets import core
    from evaluator.datasets.builtins.admed_voice import AdmedVoiceDataset

    def fake_corpus(test_size=0.3, base_path=None):
        df = pd.DataFrame(
            {
                # row 0/1: same filename, different dirs; row 2: a FULL duplicate of row 0
                "source": ["natural", "natural", "natural"],
                "cat_code": ["5", "6", "5"],
                "rec_place": ["rp", "rp", "rp"],
                "speaker_id": ["s1", "s2", "s1"],
                "filename": ["1.wav", "1.wav", "1.wav"],
                "phrase": ["alpha", "beta", "alpha"],
                "file_path": ["/f/a.wav", "/f/b.wav", "/f/a.wav"],
            }
        )
        return df, pd.DataFrame([])

    monkeypatch.setattr(core, "load_admed_voice_corpus", fake_corpus)
    monkeypatch.setattr(core, "load_audio_file", lambda p: (_Wav(), 16000))

    ds = AdmedVoiceDataset.from_config(DataConfig(dataset_name="admed_voice", split="both"))
    ids = [ds[i]["question_id"] for i in range(len(ds))]
    assert len(set(ids)) == len(ids) == 3  # unique even for the full-duplicate row


def test_admed_passes_config_path_to_loader(monkeypatch):
    import pandas as pd

    from evaluator.datasets import core
    from evaluator.datasets.builtins.admed_voice import AdmedVoiceDataset

    captured = {}

    def fake_corpus(test_size=0.3, base_path=None):
        captured["base_path"] = base_path
        df = pd.DataFrame(
            {
                "source": ["natural"], "cat_code": ["5"], "rec_place": ["rp"],
                "speaker_id": ["s1"], "filename": ["1.wav"], "phrase": ["alpha"],
                "file_path": ["/f/a.wav"],
            }
        )
        return df, pd.DataFrame([])

    monkeypatch.setattr(core, "load_admed_voice_corpus", fake_corpus)
    monkeypatch.setattr(core, "load_audio_file", lambda p: (_Wav(), 16000))

    AdmedVoiceDataset.from_config(
        DataConfig(dataset_name="admed_voice", data_path="/data/admed", split="both")
    )
    assert captured["base_path"] == "/data/admed"


def test_hani_both_split_namespaces_ids(monkeypatch):
    # The HF loader restarts its positional sample_id at 0 per split — the "both" union
    # must namespace them so the per-split id collision doesn't drop ground truth.
    from evaluator.datasets.loaders import factory
    from evaluator.datasets.loaders.base import AudioSample
    from evaluator.datasets.builtins.hani_medical import HaniMedicalDataset

    seen = []

    class _Loader:
        def __init__(self, split):
            self.split = split

        def load(self):
            return [
                AudioSample(
                    audio_array=np.zeros(16, dtype="float32"),
                    sampling_rate=16000,
                    transcription=f"t-{self.split}",
                    sample_id="0",  # collides across splits without namespacing
                    language="en",
                    metadata={},
                )
            ]

    def fake_factory(**kw):
        seen.append(kw["huggingface_split"])
        return _Loader(kw["huggingface_split"])

    monkeypatch.setattr(factory, "create_dataset_loader", fake_factory)

    ds = HaniMedicalDataset.from_config(
        DataConfig(dataset_name="hani_medical", split="both")
    )
    assert seen == ["train", "test"]
    ids = [ds[i]["question_id"] for i in range(len(ds))]
    assert len(set(ids)) == 2
