"""Wav2Vec2 ASR model implementation."""
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional
import torch
from .base_asr import HuggingFaceASRModel
from ..registry import register_asr_model


@register_asr_model('wav2vec2', default_name='jonatasgrosman/wav2vec2-large-xlsr-53-polish', description='Facebook Wav2Vec2 ASR model')
class Wav2Vec2Model(HuggingFaceASRModel):
    """Wav2Vec2 automatic speech recognition model."""

    @dataclass
    class Params:
        size: str = "large-polish"
        SIZES: ClassVar[Dict[str, str]] = {
            "base": "facebook/wav2vec2-base-960h",
            "large": "facebook/wav2vec2-large-960h",
            "large-polish": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
        }

    def __init__(self, model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-polish", adapter_path: Optional[str] = None):
        """
        Initialize Wav2Vec2 ASR model.
        
        Args:
            model_name: HuggingFace model identifier. Default: "jonatasgrosman/wav2vec2-large-xlsr-53-polish".
            adapter_path: Optional path to PEFT/LoRA adapter weights
        """
        super().__init__(model_name, adapter_path)
    
    def _create_processor(self):
        """Create Wav2Vec2 processor."""
        from transformers import Wav2Vec2Processor
        return Wav2Vec2Processor.from_pretrained(self.model_name)
    
    def _create_model(self):
        """Create Wav2Vec2 CTC model."""
        from transformers import Wav2Vec2ForCTC
        return Wav2Vec2ForCTC.from_pretrained(self.model_name)
    
    def _extract_features(self, processed_audio: List):
        """Extract features using Wav2Vec2 processor."""
        inputs = self.processor(
            processed_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        return inputs.input_values, inputs.attention_mask
    
    def _generate_transcriptions(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        language: Optional[str]
    ) -> List[str]:
        """Generate transcriptions using Wav2Vec2's CTC approach."""
        logits = self.model(features, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcripts = self.processor.batch_decode(predicted_ids)
        return transcripts
