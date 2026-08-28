"""XTTS-v2 TTS provider (multilingual, voice cloning capable)."""

from pathlib import Path
import logging

import numpy as np

from ..registry import register_tts_model
from .base_tts import BaseTTSModel

logger = logging.getLogger(__name__)


@register_tts_model(
    'xtts_v2',
    aliases=['xtts', 'xtts-v2'],
    default_name='tts_models/multilingual/multi-dataset/xtts_v2',
    capabilities=['speech_synthesis'],
    requires_path=False,
    description='Coqui XTTS-v2 — multilingual, voice-cloning capable',
)
class XTTSv2TTS(BaseTTSModel):
    """Coqui XTTS-v2 wrapper.

    Notes:
    - Uses Coqui `TTS` package.
    - `config.voice` can be a speaker WAV path for voice cloning.
    - `config.language` controls synthesis language (e.g. "en", "pl", "de").
    """

    DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(self, config):
        super().__init__(config)
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise RuntimeError(
                "XTTS-v2 provider requires Coqui TTS. Install with: pip install TTS"
            ) from exc

        self._tts = TTS(model_name=self.DEFAULT_MODEL, progress_bar=False)
        self.device = str(getattr(config, "device", "cpu") or "cpu")
        self._tts.to(self.device)
        self.output_sample_rate = int(
            getattr(getattr(self._tts, "synthesizer", None), "output_sample_rate", 24000)
        )
        logger.info("XTTS-v2 initialized (device=%s)", self.device)

    def to(self, device: str) -> "XTTSv2TTS":
        self.device = str(device)
        self._tts.to(self.device)
        return self

    def synthesize(self, text: str) -> np.ndarray:
        speaker_wav = None
        speaker = None
        available_speakers = getattr(self._tts, "speakers", None) or []

        if self.config.voice:
            voice_path = Path(self.config.voice)
            if voice_path.exists() and voice_path.is_file():
                speaker_wav = str(voice_path)
            elif self.config.voice in available_speakers:
                speaker = self.config.voice

        if speaker_wav is None and speaker is None:
            # XTTS-v2 is multi-speaker and refuses to synthesize with neither
            # `speaker_wav` nor `speaker_id` set. `config.voice` is shared across TTS
            # providers in a comparison config and often isn't a valid XTTS speaker name
            # (e.g. a Piper voice id) — fall back to the model's own first built-in
            # speaker rather than crashing, and log so the choice is visible.
            if not available_speakers:
                raise RuntimeError(
                    "XTTS-v2 requires a speaker reference: set `voice` to a WAV file "
                    "path for voice cloning, or to one of the model's built-in speaker "
                    "names, but this model exposes no built-in speakers."
                )
            speaker = available_speakers[0]
            logger.info("XTTS-v2: no usable voice/speaker in config, defaulting to built-in speaker %r", speaker)

        kwargs = {
            "text": text,
            "language": self.config.language or "en",
        }
        if speaker_wav is not None:
            kwargs["speaker_wav"] = speaker_wav
        else:
            kwargs["speaker"] = speaker

        audio = self._tts.tts(**kwargs)
        return np.asarray(audio, dtype=np.float32)
