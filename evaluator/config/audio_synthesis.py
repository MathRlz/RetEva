"""Audio synthesis configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioSynthesisConfig:
    """
    Configuration for offline audio synthesis from text.
    
    Used for text-only benchmark questions where audio needs to be synthesized.
    Supports various TTS providers for generating synthetic speech.
    
    Attributes:
        enabled: Whether audio synthesis is enabled. Default: False.
        provider: TTS provider to use. Default: "piper".
            Options: "piper", "xtts_v2", "mms".
        voice: Voice identifier or language code. Default: "en_US-lessac-medium".
        sample_rate: Audio sample rate in Hz. Default: 16000.
        speed: Speech speed multiplier. Default: 1.0.
        pitch: Pitch shift multiplier. Default: 1.0.
        volume: Volume multiplier. Default: 1.0.
        language: Language code for TTS. Default: "en".
        seed: Random seed for reproducibility. Default: 42.
        output_dir: Directory for synthesized audio files. Default: "prepared_benchmarks/audio".
        cache_dir: Directory for caching synthesized audio. Default: None (no caching).
        api_key: API key for cloud TTS providers. Default: None.
    
    Examples:
        >>> config = AudioSynthesisConfig(
        ...     enabled=True,
        ...     provider="piper",
        ...     voice="en_US-lessac-medium",
        ...     cache_dir=".tts_cache"
        ... )
        >>> xtts_cfg = AudioSynthesisConfig(
        ...     enabled=True,
        ...     provider="xtts_v2",
        ...     language="en",
        ... )
        >>> mms_cfg = AudioSynthesisConfig(
        ...     enabled=True,
        ...     provider="mms",
        ...     language="pl",
        ... )
    """
    enabled: bool = False
    provider: str = "piper"
    voice: str = "en_US-lessac-medium"
    sample_rate: int = 16000
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    language: str = "en"
    seed: int = 42
    output_dir: str = "prepared_benchmarks/audio"
    cache_dir: Optional[str] = None
    api_key: Optional[str] = None
