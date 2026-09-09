"""Audio synthesis configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioSynthesisConfig:
    """Configuration for offline audio synthesis from text (for text-only benchmark
    questions)."""
    enabled: bool = False
    provider: str = "piper"  # piper | xtts_v2 | mms
    voice: str = "en_US-lessac-medium"
    sample_rate: int = 16000
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    language: str = "en"
    # multi-speaker vocoders only (m4t: 0-199); None = provider default
    speaker_id: Optional[int] = None
    seed: int = 42
    output_dir: str = "prepared_benchmarks/audio"
    cache_dir: Optional[str] = None  # None = auto-derived from output_dir
    skip_cache: bool = False
    device: str = "cpu"  # GPU-capable providers only (m4t/mms/xtts_v2); ignored by piper (CPU-only)
