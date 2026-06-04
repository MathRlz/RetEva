"""SeamlessM4T-v2 TTS provider (text-to-speech) using HuggingFace transformers.

Counterpart to the SeamlessM4T-v2 ASR model (``models/asr/seamless_m4t.py``):
the same family generates speech from text in ~100 languages.
"""
import logging
from typing import ClassVar, Dict

import numpy as np

from ..registry import register_tts_model
from .._seamless_lang import SEAMLESS_LANG_ALIASES
from .base_tts import BaseTTSModel, require_torch_transformers, model_sampling_rate

logger = logging.getLogger(__name__)


@register_tts_model(
    'm4t',
    aliases=['seamless_m4t', 'seamless-m4t', 'm4t_v2'],
    default_name='facebook/seamless-m4t-v2-large',
    capabilities=['speech_synthesis'],
    description='SeamlessM4T-v2 multilingual text-to-speech (HF transformers)',
)
class M4TTTS(BaseTTSModel):
    """Meta SeamlessM4T-v2 text-to-speech wrapper.

    Notes:
    - Uses `transformers.SeamlessM4Tv2ForTextToSpeech`.
    - `config.language` sets both source and target language (e.g. "en", "pl").
    - `config.voice` may set a numeric `speaker_id`.
    """

    _LANG_ALIASES: ClassVar[Dict[str, str]] = SEAMLESS_LANG_ALIASES

    def __init__(self, config):
        super().__init__(config)
        self._torch = require_torch_transformers("M4T TTS")
        from transformers import AutoProcessor, SeamlessM4Tv2ForTextToSpeech

        model_id = (config.voice or "").strip() or "facebook/seamless-m4t-v2-large"
        if not model_id.startswith("facebook/"):
            model_id = "facebook/seamless-m4t-v2-large"

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = SeamlessM4Tv2ForTextToSpeech.from_pretrained(model_id)
        self._model.eval()
        self.lang = self._LANG_ALIASES.get((config.language or "en").lower(), (config.language or "eng"))
        self.output_sample_rate = model_sampling_rate(self._model)
        logger.info(f"M4T TTS initialized with model: {model_id} (lang={self.lang})")

    def synthesize(self, text: str) -> np.ndarray:
        inputs = self._processor(text=text, src_lang=self.lang, return_tensors="pt")
        with self._torch.no_grad():
            output = self._model.generate(**inputs, tgt_lang=self.lang)
        # generate() returns a tuple/list of waveform tensors.
        waveform = output[0] if isinstance(output, (list, tuple)) else output
        audio = waveform.squeeze().cpu().numpy()
        return np.asarray(audio, dtype=np.float32)
