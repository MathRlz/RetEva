"""Audio augmentation configuration."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AudioAugmentationConfig:
    """Configuration for audio augmentation during evaluation: noise, speed perturbation,
    and codec simulation to test ASR robustness."""
    enabled: bool = False
    add_noise: bool = False
    noise_type: str = "white"  # white | pink | brown | background
    snr_db: float = 20.0
    speed_perturbation: bool = False
    speed_range: Tuple[float, float] = (0.9, 1.1)
    pitch_shift: bool = False
    pitch_semitones_range: Tuple[int, int] = (-2, 2)
    volume_change: bool = False
    volume_range: Tuple[float, float] = (0.8, 1.2)
    n_variants: int = 1
