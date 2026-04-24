"""Audio synthesis from text using TTS models."""
from typing import Optional, List
import numpy as np
import hashlib
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AudioSynthesizer:
    """Base class for text-to-speech synthesis with caching support."""
    
    def __init__(self, config):
        """
        Initialize audio synthesizer.
        
        Args:
            config: AudioSynthesisConfig instance with TTS settings.
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir) if config.cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"TTS cache enabled at: {self.cache_dir}")
        
        self.provider = self._create_provider()
        logger.info(f"Initialized TTS model: {config.provider}")
    
    def _create_provider(self):
        """Factory method to create TTS model backend based on config."""
        provider_name = self.config.provider.lower()
        
        if provider_name == "piper":
            from evaluator.models.tts.piper_tts import PiperTTS
            return PiperTTS(self.config)
        if provider_name in {"xtts", "xtts_v2", "xtts-v2"}:
            from evaluator.models.tts.xtts_v2_tts import XTTSv2TTS
            return XTTSv2TTS(self.config)
        if provider_name in {"mms", "mms_tts", "mms-tts"}:
            from evaluator.models.tts.mms_tts import MMSTTS
            return MMSTTS(self.config)
        else:
            raise ValueError(
                f"Unknown TTS provider: {provider_name}. "
                f"Supported providers: piper, xtts_v2, mms"
            )
    
    def synthesize(self, text: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Synthesize audio from text.
        
        Args:
            text: Text to synthesize.
            output_path: Optional path to save audio file.
            
        Returns:
            audio_array: Float32 waveform in [-1, 1] range.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis, returning silence")
            return np.zeros(self.config.sample_rate, dtype=np.float32)
        
        # Check cache
        if self.cache_dir:
            cache_key = self._get_cache_key(text)
            cached_audio = self._load_from_cache(cache_key)
            if cached_audio is not None:
                logger.debug(f"Loaded audio from cache (key: {cache_key[:16]}...)")
                if output_path:
                    self._save_audio(output_path, cached_audio)
                return cached_audio
        
        # Synthesize
        logger.debug(f"Synthesizing: '{text[:50]}...'")
        audio = self.provider.synthesize(text)
        provider_sr = int(getattr(self.provider, "output_sample_rate", self.config.sample_rate))
        audio = self._resample_if_needed(audio, provider_sr)
        
        # Normalize to [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        # Cache
        if self.cache_dir:
            self._save_to_cache(cache_key, audio)
            logger.debug(f"Cached audio (key: {cache_key[:16]}...)")
        
        # Save to file if requested
        if output_path:
            self._save_audio(output_path, audio)
            logger.debug(f"Saved audio to: {output_path}")
        
        return audio

    def _resample_if_needed(self, audio: np.ndarray, source_sr: int) -> np.ndarray:
        """Resample provider output to configured sample rate when required."""
        target_sr = int(self.config.sample_rate)
        if source_sr == target_sr:
            return audio
        try:
            import librosa
        except ImportError:
            logger.warning(
                f"Provider output sample rate is {source_sr}Hz but target is {target_sr}Hz; "
                "librosa not installed, audio will not be resampled."
            )
            return audio
        try:
            resampled = librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)
            logger.debug(f"Resampled audio from {source_sr}Hz to {target_sr}Hz")
            return np.asarray(resampled, dtype=np.float32)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Audio resampling failed ({source_sr}->{target_sr}): {e}")
            return audio
    
    def synthesize_batch(
        self, 
        texts: List[str], 
        output_dir: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Batch synthesis with progress tracking.
        
        Args:
            texts: List of texts to synthesize.
            output_dir: Optional directory to save audio files.
            
        Returns:
            List of audio arrays.
        """
        results = []
        total = len(texts)
        
        logger.info(f"Synthesizing {total} texts...")
        
        for i, text in enumerate(texts):
            output_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"synth_{i:05d}.wav")
            
            audio = self.synthesize(text, output_path)
            results.append(audio)
            
            if (i + 1) % 10 == 0 or (i + 1) == total:
                logger.info(f"Synthesized {i + 1}/{total} texts")
        
        logger.info(f"Batch synthesis complete: {total} texts")
        return results
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text and config settings."""
        config_str = (
            f"{self.config.provider}_"
            f"{self.config.voice}_"
            f"{self.config.speed}_"
            f"{self.config.pitch}_"
            f"{self.config.sample_rate}"
        )
        combined = f"{text}_{config_str}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load audio from cache."""
        cache_path = self.cache_dir / f"{cache_key}.npy"
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to load from cache: {e}")
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, audio: np.ndarray):
        """Save audio to cache."""
        try:
            cache_path = self.cache_dir / f"{cache_key}.npy"
            np.save(cache_path, audio)
        except OSError as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def _save_audio(self, path: str, audio: np.ndarray):
        """Save audio to WAV file."""
        try:
            import soundfile as sf
            sf.write(path, audio, self.config.sample_rate)
        except ImportError:
            logger.warning("soundfile not installed, cannot save audio to WAV")
        except OSError as e:
            logger.error(f"Failed to save audio: {e}")
