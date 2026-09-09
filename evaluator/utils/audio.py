"""Torchaudio-backed audio DSP helpers.

Graceful-degradation contract: when torchaudio is missing, warn and return the input
unchanged rather than fail the run.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _torch():
    try:
        import torch
        import torchaudio
        return torch, torchaudio
    except ImportError:
        return None, None


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono float audio; no-op when rates match or torchaudio is absent."""
    if source_sr == target_sr:
        return audio
    torch, torchaudio = _torch()
    if torch is None:
        logger.warning("torchaudio not installed, cannot resample %dHz -> %dHz",
                       source_sr, target_sr)
        return audio
    try:
        tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        return torchaudio.functional.resample(tensor, source_sr, target_sr).numpy()
    except (ValueError, RuntimeError) as e:
        logger.warning("Audio resampling failed (%d -> %d): %s", source_sr, target_sr, e)
        return audio


def to_mono_16k(audio, sr: int, *, downmix: str = "mean", as_numpy: bool = False):
    """Resample to 16 kHz and collapse channels. Returns the same tensor object unchanged
    when the rate already matches and ``as_numpy`` is False (no resample, squeeze, or copy)."""
    torch, torchaudio = _torch()
    if torch is None:
        logger.warning("torchaudio not installed, to_mono_16k returning input unchanged")
        return audio
    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio)
    changed = False
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
        changed = True
    if not as_numpy and not changed and audio.ndim == 1:
        return audio
    x = audio.squeeze()
    if x.ndim > 1:
        x = x.mean(dim=0) if downmix == "mean" else x[0]
    return x.numpy() if as_numpy else x


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Pitch-preserving time stretch (phase vocoder); rate > 1 speeds up."""
    torch, torchaudio = _torch()
    if torch is None:
        logger.warning("torchaudio not installed, time stretch skipped")
        return audio
    n_fft, hop = 2048, 512
    window = torch.hann_window(n_fft)
    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    spec = torch.stft(tensor, n_fft, hop, window=window, return_complex=True)
    phase_advance = torch.linspace(0, np.pi * hop, n_fft // 2 + 1)[..., None]
    stretched = torchaudio.functional.phase_vocoder(spec, rate, phase_advance)
    return torch.istft(stretched, n_fft, hop, window=window).numpy().astype(np.float32)


def pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by ``n_steps`` semitones (fractional OK), preserving duration."""
    torch, torchaudio = _torch()
    if torch is None:
        logger.warning("torchaudio not installed, pitch shift skipped")
        return audio
    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    # torchaudio wants integer steps; work in cents for fractional semitones.
    out = torchaudio.functional.pitch_shift(
        tensor, sr, n_steps=int(round(n_steps * 100)), bins_per_octave=1200)
    return out.numpy().astype(np.float32)
