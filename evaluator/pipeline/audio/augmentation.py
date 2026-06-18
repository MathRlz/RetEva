"""Audio augmentation for robustness testing and data expansion.

This module provides audio augmentation techniques to test ASR robustness
and expand training datasets. Augmentations include noise addition,
speed perturbation, pitch shifting, and codec simulation.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class AudioAugmenter:
    """Audio augmentation pipeline for robustness testing."""

    def __init__(self, config):
        """
        Initialize audio augmenter.

        Args:
            config: AudioAugmentationConfig instance.
        """
        self.config = config
        logger.info("Initialized AudioAugmenter")

    def augment(
        self, audio: np.ndarray, sr: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply augmentation chain to audio.

        Args:
            audio: Audio waveform (1D numpy array, float32).
            sr: Sample rate in Hz.
            seed: Optional per-item seed (e.g. ``item_seed(...)``) making the perturbation
                deterministic/reproducible regardless of order (architecture A10).

        Returns:
            Augmented audio waveform (same shape as input).
        """
        # A per-call local RNG (not the global np.random): identical Mersenne-Twister sequence to
        # the former ``np.random.seed(seed)`` path, but local — so the same shared augmenter is
        # thread/process-safe (no global-state race) and draws stay order-deterministic.
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

        # Ensure output is still float32 and in valid range
        augmented = augmented.astype(np.float32)
        augmented = np.clip(augmented, -1.0, 1.0)

        return augmented

    def create_variants(
        self,
        audio: np.ndarray,
        sr: int,
        n_variants: Optional[int] = None,
        *,
        base_seed: Optional[int] = None,
        query_id: Optional[str] = None,
        node_id: str = "augment_audio",
    ) -> List[np.ndarray]:
        """
        Create multiple augmented variants of the same audio.

        Args:
            audio: Original audio waveform.
            sr: Sample rate.
            n_variants: Number of variants to create (uses config if None).
            base_seed: Run seed. When given (with ``query_id``), each variant ``i`` is
                perturbed with ``item_seed(base_seed, query_id, node_id, i)`` so the variant
                is reproducible regardless of order/parallelism (architecture A10/S5). Without
                it, variants stay random (back-compat).

        Returns:
            List of augmented audio arrays.
        """
        n = n_variants or self.config.n_variants
        variants = []

        for i in range(n):
            logger.debug(f"Creating variant {i+1}/{n}")
            seed: Optional[int] = None
            if base_seed is not None and query_id is not None:
                from ...evaluation.provenance import item_seed

                seed = item_seed(int(base_seed), query_id, node_id, i)
            variant = self.augment(audio, sr, seed=seed)
            variants.append(variant)

        return variants

    def _add_noise(
        self, audio: np.ndarray, sr: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Add noise to audio signal."""
        noise_type = self.config.noise_type.lower()

        # Generate noise
        if noise_type == "white":
            noise = self._generate_white_noise(len(audio), rng)
        elif noise_type == "pink":
            noise = self._generate_pink_noise(len(audio), rng)
        elif noise_type == "brown":
            noise = self._generate_brown_noise(len(audio), rng)
        else:
            logger.warning(f"Unknown noise type: {noise_type}, using white noise")
            noise = self._generate_white_noise(len(audio), rng)

        # Calculate signal and noise power
        signal_power = np.mean(audio**2)
        noise_power = np.mean(noise**2)

        # Calculate scaling factor for desired SNR
        snr_linear = 10 ** (self.config.snr_db / 10.0)
        noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))

        # Add scaled noise
        noisy_audio = audio + noise_scale * noise

        logger.debug(f"Added {noise_type} noise at {self.config.snr_db} dB SNR")
        return noisy_audio

    def _generate_white_noise(
        self, length: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Generate white (Gaussian) noise."""
        return rng.randn(length).astype(np.float32)

    def _generate_pink_noise(
        self, length: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Generate pink (1/f) noise using Voss-McCartney algorithm."""
        # Number of random sources
        num_sources = 16

        # Initialize sources
        sources = np.zeros((num_sources, length))

        # Generate noise
        for i in range(num_sources):
            # Each source updates at rate 2^i
            update_rate = 2**i
            num_updates = length // update_rate

            # Generate random values
            values = rng.randn(num_updates + 1)

            # Upsample to full length
            for j in range(num_updates):
                start = j * update_rate
                end = min((j + 1) * update_rate, length)
                sources[i, start:end] = values[j]

        # Sum all sources
        pink_noise = np.sum(sources, axis=0)

        # Normalize
        pink_noise = pink_noise / np.std(pink_noise)

        return pink_noise.astype(np.float32)

    def _generate_brown_noise(
        self, length: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Generate brown (1/f^2) noise via cumulative sum of white noise."""
        white_noise = rng.randn(length)
        brown_noise = np.cumsum(white_noise)

        # Normalize
        brown_noise = brown_noise / np.std(brown_noise)

        return brown_noise.astype(np.float32)

    def _speed_perturbation(
        self, audio: np.ndarray, sr: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Apply speed perturbation (time stretching)."""
        # Random speed factor within range
        speed_min, speed_max = self.config.speed_range
        speed_factor = rng.uniform(speed_min, speed_max)

        try:
            import librosa

            # Time stretch (preserves pitch)
            stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
            logger.debug(f"Applied speed perturbation: {speed_factor:.2f}x")
            return stretched
        except ImportError:
            logger.warning("librosa not installed, speed perturbation skipped")
            return audio

    def _pitch_shift(
        self, audio: np.ndarray, sr: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Apply pitch shifting."""
        # Random pitch shift within range
        pitch_min, pitch_max = self.config.pitch_semitones_range
        n_steps = rng.uniform(pitch_min, pitch_max)

        try:
            import librosa

            # Pitch shift
            shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            logger.debug(f"Applied pitch shift: {n_steps:.1f} semitones")
            return shifted
        except ImportError:
            logger.warning("librosa not installed, pitch shift skipped")
            return audio

    def _volume_change(
        self, audio: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """Apply random volume change."""
        vol_min, vol_max = self.config.volume_range
        volume_factor = rng.uniform(vol_min, vol_max)

        logger.debug(f"Applied volume change: {volume_factor:.2f}x")
        return audio * volume_factor


def augment_batch(
    audio_list: List[np.ndarray],
    sampling_rates: List[int],
    config,
    n_variants_per_sample: int = 1,
    *,
    base_seed: Optional[int] = None,
    query_ids: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Augment a batch of audio samples.

    Args:
        audio_list: List of audio waveforms.
        sampling_rates: List of sample rates (one per audio).
        config: AudioAugmentationConfig instance.
        n_variants_per_sample: Number of variants to create per sample.
        base_seed: Run seed; with ``query_ids`` (aligned to ``audio_list``) makes every
            variant deterministic per item (A10/S5).

    Returns:
        Tuple of (augmented_audio_list, augmented_sampling_rates).
    """
    if not config.enabled:
        return audio_list, sampling_rates

    augmenter = AudioAugmenter(config)
    augmented_audio = []
    augmented_srs = []

    for idx, (audio, sr) in enumerate(zip(audio_list, sampling_rates)):
        qid = query_ids[idx] if query_ids is not None and idx < len(query_ids) else None
        # Create variants
        variants = augmenter.create_variants(
            audio, sr, n_variants_per_sample, base_seed=base_seed, query_id=qid
        )

        # Add original + variants
        augmented_audio.append(audio)  # Original
        augmented_srs.append(sr)

        for variant in variants:
            augmented_audio.append(variant)
            augmented_srs.append(sr)

    logger.info(
        f"Augmented {len(audio_list)} samples to "
        f"{len(augmented_audio)} samples "
        f"({n_variants_per_sample} variants per sample)"
    )

    return augmented_audio, augmented_srs


def create_augmentation_config(
    add_noise: bool = True,
    snr_db: float = 20.0,
    speed_perturbation: bool = True,
    speed_range: Tuple[float, float] = (0.9, 1.1),
    **kwargs,
):
    """
    Convenience function to create augmentation config.

    Args:
        add_noise: Enable noise addition.
        snr_db: Signal-to-noise ratio in dB.
        speed_perturbation: Enable speed perturbation.
        speed_range: Speed multiplier range.
        **kwargs: Additional config parameters.

    Returns:
        AudioAugmentationConfig instance.

    Examples:
        >>> config = create_augmentation_config(
        ...     add_noise=True,
        ...     snr_db=15.0,
        ...     speed_perturbation=True
        ... )
    """
    from evaluator.config import AudioAugmentationConfig

    return AudioAugmentationConfig(
        enabled=True,
        add_noise=add_noise,
        snr_db=snr_db,
        speed_perturbation=speed_perturbation,
        speed_range=speed_range,
        **kwargs,
    )
