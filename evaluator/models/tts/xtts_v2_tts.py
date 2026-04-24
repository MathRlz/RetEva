"""XTTS-v2 TTS provider (multilingual, voice cloning capable)."""

from pathlib import Path
import logging

import numpy as np

logger = logging.getLogger(__name__)


class XTTSv2TTS:
    """Coqui XTTS-v2 wrapper.

    Notes:
    - Uses Coqui `TTS` package.
    - `config.voice` can be a speaker WAV path for voice cloning.
    - `config.language` controls synthesis language (e.g. "en", "pl", "de").
    """

    DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(self, config):
        self.config = config
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise RuntimeError(
                "XTTS-v2 provider requires Coqui TTS. Install with: pip install TTS"
            ) from exc

        self._tts = TTS(model_name=self.DEFAULT_MODEL, progress_bar=False)
        self.output_sample_rate = int(
            getattr(getattr(self._tts, "synthesizer", None), "output_sample_rate", 24000)
        )
        logger.info("XTTS-v2 initialized")

    def synthesize(self, text: str) -> np.ndarray:
        speaker_wav = None
        if self.config.voice:
            voice_path = Path(self.config.voice)
            if voice_path.exists() and voice_path.is_file():
                speaker_wav = str(voice_path)

        kwargs = {
            "text": text,
            "language": self.config.language or "en",
        }
        if speaker_wav is not None:
            kwargs["speaker_wav"] = speaker_wav

        audio = self._tts.tts(**kwargs)
        return np.asarray(audio, dtype=np.float32)
