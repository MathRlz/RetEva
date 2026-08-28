"""HF loader stereo→mono downmix (the multi-channel ASR crash fix)."""

import numpy as np

from evaluator.datasets.loaders.huggingface import HuggingFaceDatasetLoader, _to_mono


def test_to_mono_downmixes_both_channel_layouts():
    assert _to_mono(np.zeros((2, 8000), dtype="float32")).shape == (8000,)  # (channels, samples)
    assert _to_mono(np.zeros((8000, 2), dtype="float32")).shape == (8000,)  # (samples, channels)
    assert _to_mono(np.zeros((8000,), dtype="float32")).shape == (8000,)  # already mono


def test_to_mono_averages_channels():
    stereo = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 2.0, 1.0]])  # (channels, samples)
    np.testing.assert_allclose(_to_mono(stereo), [0.5, 1.0, 2.0, 2.0])


def test_extract_audio_downmixes_stereo():
    loader = HuggingFaceDatasetLoader(dataset_name="test/dataset")
    stereo = np.array([[0.0, 2.0], [2.0, 0.0]])
    audio, _ = loader._extract_audio({"audio": {"array": stereo}}, "audio")
    assert audio.ndim == 1
    np.testing.assert_allclose(audio, [1.0, 1.0])
