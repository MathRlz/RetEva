"""MMS TTS provider using Hugging Face transformers."""

import logging
from typing import Dict

import numpy as np

from ..registry import register_tts_model
from .base_tts import BaseTTSModel, require_torch_transformers, model_sampling_rate

logger = logging.getLogger(__name__)


@register_tts_model(
    'mms',
    aliases=['mms_tts', 'mms-tts'],
    default_name='facebook/mms-tts-eng',
    capabilities=['speech_synthesis'],
    description='Meta MMS TTS — multilingual (HF transformers)',
)
class MMSTTS(BaseTTSModel):
    """Meta MMS TTS wrapper.

    Notes:
    - By default, picks model from language code.
    - `config.voice` may be set directly to HF model id (e.g. facebook/mms-tts-eng).
    """

    LANGUAGE_TO_MODEL: Dict[str, str] = {
        "en": "facebook/mms-tts-eng",
        "de": "facebook/mms-tts-deu",
        "fr": "facebook/mms-tts-fra",
        "es": "facebook/mms-tts-spa",
        "it": "facebook/mms-tts-ita",
        "pl": "facebook/mms-tts-pol",
        "pt": "facebook/mms-tts-por",
        "nl": "facebook/mms-tts-nld",
        "cs": "facebook/mms-tts-ces",
        "uk": "facebook/mms-tts-ukr",
        "ru": "facebook/mms-tts-rus",
    }

    def __init__(self, config):
        super().__init__(config)
        self._torch = require_torch_transformers("MMS provider")
        from transformers import AutoTokenizer, VitsModel

        model_id = self._resolve_model_id()
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = VitsModel.from_pretrained(model_id)
        self._model.eval()
        self.output_sample_rate = model_sampling_rate(self._model)
        logger.info(f"MMS TTS initialized with model: {model_id}")

    def _resolve_model_id(self) -> str:
        voice = (self.config.voice or "").strip()
        if voice.startswith("facebook/mms-tts-"):
            return voice
        language = (self.config.language or "en").lower()
        return self.LANGUAGE_TO_MODEL.get(language, "facebook/mms-tts-eng")

    def synthesize(self, text: str) -> np.ndarray:
        inputs = self._tokenizer(text=text, return_tensors="pt")
        with self._torch.no_grad():
            output = self._model(**inputs)
        audio = output.waveform.squeeze().cpu().numpy()
        return np.asarray(audio, dtype=np.float32)
