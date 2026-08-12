"""Audio augmentation configuration."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AudioAugmentationConfig:
    """
    Configuration for audio augmentation during evaluation.

    Adds noise, speed perturbation, and codec simulation to test ASR robustness.

    Attributes:
        enabled: Whether audio augmentation is enabled. Default: False.
        add_noise: Add background noise. Default: False.
        noise_type: Type of noise to add. Default: "white".
            Options: "white", "pink", "brown", "background".
        snr_db: Signal-to-noise ratio in dB. Default: 20.0.
        speed_perturbation: Apply speed perturbation. Default: False.
        speed_range: Speed multiplier range. Default: (0.9, 1.1).
        pitch_shift: Apply pitch shift. Default: False.
        pitch_semitones_range: Pitch shift range in semitones. Default: (-2, 2).
        volume_change: Apply volume changes. Default: False.
        volume_range: Volume multiplier range. Default: (0.8, 1.2).
        n_variants: Number of augmented variants per sample. Default: 1.

    Examples:
        >>> config = AudioAugmentationConfig(
        ...     enabled=True,
        ...     add_noise=True,
        ...     snr_db=15.0,
        ...     speed_perturbation=True
        ... )
    """
    enabled: bool = False
    add_noise: bool = False
    noise_type: str = "white"
    snr_db: float = 20.0
    speed_perturbation: bool = False
    speed_range: Tuple[float, float] = (0.9, 1.1)
    pitch_shift: bool = False
    pitch_semitones_range: Tuple[int, int] = (-2, 2)
    volume_change: bool = False
    volume_range: Tuple[float, float] = (0.8, 1.2)
    n_variants: int = 1
