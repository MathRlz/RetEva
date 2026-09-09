"""Audio augmentation for robustness testing and data expansion.

Includes noise addition, speed perturbation, pitch shifting, and codec simulation.
"""

import numpy as np
import logging
from typing import Optional
from ...utils.audio import time_stretch, pitch_shift

logger = logging.getLogger(__name__)


class AudioAugmenter:
    """Audio augmentation pipeline for robustness testing."""

    def __init__(self, config):
        self.config = config
        logger.info("Initialized AudioAugmenter")

    def augment(
        self, audio: np.ndarray, sr: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """Apply augmentation chain to audio.

        ``seed`` (e.g. ``item_seed(...)``) makes the perturbation deterministic/reproducible
        regardless of order. Returns audio of the same shape as the input.
        """
        # A per-call local RNG (not the global np.random): thread/process-safe (no
        # global-state race) and draws stay order-deterministic.
        rng = (
            np.random.RandomState(int(seed) % (2**32))
            if seed is not None
            else np.random.RandomState()
        )
        augmented = audio.copy()

        # Apply each augmentation if enabled (draws consumed in this fixed order from ``rng``)
        if self.config.add_noise:
            augmented = self._add_noise(augmented, sr, rng)

        if self.config.speed_perturbation:
            augmented = self._speed_perturbation(augmented, sr, rng)

        if self.config.pitch_shift:
            augmented = self._pitch_shift(augmented, sr, rng)

        if self.config.volume_change:
            augmented = self._volume_change(augmented, rng)

        augmented = augmented.astype(np.float32)
        augmented = np.clip(augmented, -1.0, 1.0)

        return augmented

    def _add_noise(
        self, audio: np.ndarray, sr: int, rng: np.random.RandomState
    ) -> np.ndarray:
        noise_type = self.config.noise_type.lower()

        if noise_type == "white":
            noise = self._generate_white_noise(len(audio), rng)
        elif noise_type == "pink":
            noise = self._generate_pink_noise(len(audio), rng)
        elif noise_type == "brown":
            noise = self._generate_brown_noise(len(audio), rng)
        else:
            logger.warning(f"Unknown noise type: {noise_type}, using white noise")
            noise = self._generate_white_noise(len(audio), rng)

        signal_power = np.mean(audio**2)
        noise_power = np.mean(noise**2)

        snr_linear = 10 ** (self.config.snr_db / 10.0)
        noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))

        noisy_audio = audio + noise_scale * noise

        logger.debug(f"Added {noise_type} noise at {self.config.snr_db} dB SNR")
        return noisy_audio

    def _generate_white_noise(
        self, length: int, rng: np.random.RandomState
    ) -> np.ndarray:
        return rng.randn(length).astype(np.float32)

    def _generate_pink_noise(
        self, length: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Generate pink (1/f) noise using Voss-McCartney algorithm."""
        num_sources = 16
        sources = np.zeros((num_sources, length))

        for i in range(num_sources):
            # Each source updates at rate 2^i
            update_rate = 2**i
            num_updates = length // update_rate

            values = rng.randn(num_updates + 1)

            # Upsample to full length
            for j in range(num_updates):
                start = j * update_rate
                end = min((j + 1) * update_rate, length)
                sources[i, start:end] = values[j]

        pink_noise = np.sum(sources, axis=0)

        pink_noise = pink_noise / np.std(pink_noise)

        return pink_noise.astype(np.float32)

    def _generate_brown_noise(
        self, length: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Generate brown (1/f^2) noise via cumulative sum of white noise."""
        white_noise = rng.randn(length)
        brown_noise = np.cumsum(white_noise)

        brown_noise = brown_noise / np.std(brown_noise)

        return brown_noise.astype(np.float32)

    def _speed_perturbation(
        self, audio: np.ndarray, sr: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Apply speed perturbation (time stretching)."""
        speed_min, speed_max = self.config.speed_range
        speed_factor = rng.uniform(speed_min, speed_max)

        stretched = time_stretch(audio, speed_factor)
        logger.debug(f"Applied speed perturbation: {speed_factor:.2f}x")
        return stretched

    def _pitch_shift(
        self, audio: np.ndarray, sr: int, rng: np.random.RandomState
    ) -> np.ndarray:
        pitch_min, pitch_max = self.config.pitch_semitones_range
        n_steps = rng.uniform(pitch_min, pitch_max)

        shifted = pitch_shift(audio, sr, n_steps)
        logger.debug(f"Applied pitch shift: {n_steps:.1f} semitones")
        return shifted

    def _volume_change(
        self, audio: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        vol_min, vol_max = self.config.volume_range
        volume_factor = rng.uniform(vol_min, vol_max)

        logger.debug(f"Applied volume change: {volume_factor:.2f}x")
        return audio * volume_factor
