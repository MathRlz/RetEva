"""SeamlessM4T-v2 ASR / speech-to-text translation model implementation."""
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional
import torch
from .base_asr import HuggingFaceASRModel
from ..registry import register_asr_model
from .._seamless_lang import SEAMLESS_LANG_ALIASES


@register_asr_model(
    'seamless_m4t',
    default_name='facebook/seamless-m4t-v2-large',
    description='SeamlessM4T-v2 multilingual ASR / speech-to-text translation',
)
class SeamlessM4TASRModel(HuggingFaceASRModel):
    """SeamlessM4T-v2 speech-to-text model.

    ASR and speech-to-text translation are the same model with one knob:
    ``tgt_lang``. Set it to the source language for same-language ASR, or keep
    the default ``"eng"`` to translate any of the ~100 supported source languages
    into English text.

    ``tgt_lang`` is the *output* language and is fixed by config — the per-call
    ``language`` the pipeline passes (the audio's source language) does not change
    it. SeamlessM4T uses 3-letter codes; common 2-letter codes are normalized.
    """

    _LANG_ALIASES: ClassVar[Dict[str, str]] = SEAMLESS_LANG_ALIASES

    @dataclass
    class Params:
        size: str = "v2-large"
        tgt_lang: str = "eng"
        SIZES: ClassVar[Dict[str, str]] = {
            "v2-large": "facebook/seamless-m4t-v2-large",
        }

    def __init__(
        self,
        model_name: str = "facebook/seamless-m4t-v2-large",
        adapter_path: Optional[str] = None,
        tgt_lang: str = "eng",
    ):
        """
        Initialize SeamlessM4T-v2 ASR model.

        Args:
            model_name: HuggingFace model identifier.
            adapter_path: Optional path to PEFT/LoRA adapter weights.
            tgt_lang: Output language code. "eng" turns non-English audio into
                English text (speech-to-text translation).
        """
        self.tgt_lang = self._LANG_ALIASES.get(tgt_lang, tgt_lang)
        super().__init__(model_name, adapter_path)

    def _create_processor(self):
        """Create SeamlessM4T processor."""
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(self.model_name)

    def _create_model(self):
        """Create SeamlessM4T-v2 speech-to-text model."""
        from transformers import SeamlessM4Tv2ForSpeechToText
        return SeamlessM4Tv2ForSpeechToText.from_pretrained(self.model_name)

    def _extract_features(self, processed_audio: List):
        """Extract features using the SeamlessM4T processor."""
        inputs = self.processor(
            audio=processed_audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return inputs.input_features, inputs.attention_mask

    def _generate_transcriptions(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        language: Optional[str],
    ) -> List[str]:
        """Generate transcriptions/translations via the model's generate method.

        ``language`` is the audio's *source* language and is intentionally
        ignored: SeamlessM4T's ``tgt_lang`` is the *output* language, fixed by
        config (one output language per run, as in a normal evaluation).
        """
        generated_ids = self.model.generate(
            input_features=features,
            attention_mask=attention_mask,
            tgt_lang=self.tgt_lang,
        )
        # SeamlessM4T generate returns a tensor (or sequence); decode token ids.
        if isinstance(generated_ids, (list, tuple)):
            generated_ids = generated_ids[0]
        transcripts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return transcripts
