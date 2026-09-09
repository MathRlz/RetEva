"""Torchaudio-backed helpers keep the librosa semantics —
pitch-preserving stretch, duration-preserving pitch shift, plain resample."""
import numpy as np
import pytest

pytest.importorskip("torchaudio")

from evaluator.utils.audio import pitch_shift, resample_audio, time_stretch  # noqa: E402


def _sine(sr=16000, secs=1.0, hz=440.0):
    t = np.arange(int(sr * secs)) / sr
    return np.sin(2 * np.pi * hz * t).astype(np.float32)


def test_time_stretch_shortens_at_higher_rate():
    audio = _sine()
    fast = time_stretch(audio, 1.25)
    assert 0.7 * len(audio) < len(fast) < 0.9 * len(audio)  # ~len/1.25


def test_pitch_shift_preserves_length_and_moves_signal():
    audio = _sine()
    shifted = pitch_shift(audio, 16000, 2.5)  # fractional semitones supported
    assert len(shifted) == len(audio)
    assert not np.allclose(shifted[:1000], audio[:1000], atol=1e-3)


def test_resample_halves_length_and_same_rate_is_identity():
    audio = _sine()
    assert len(resample_audio(audio, 16000, 8000)) == len(audio) // 2
    assert resample_audio(audio, 16000, 16000) is audio


def test_to_mono_16k_matches_the_inline_sites():
    import torch, torchaudio
    from evaluator.utils.audio import to_mono_16k
    stereo = torch.randn(2, 8000)
    out = to_mono_16k(stereo, 8000, downmix="mean", as_numpy=True)
    ref = torchaudio.functional.resample(stereo, 8000, 16000).squeeze().numpy().mean(axis=0)
    assert out.shape == ref.shape and abs(out - ref).max() < 1e-6
    mono = torch.randn(4000)
    assert to_mono_16k(mono, 16000, downmix="mean", as_numpy=False) is mono
