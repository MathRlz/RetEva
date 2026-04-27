"""Whisper ASR model implementation."""
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional
import torch
from .base_asr import HuggingFaceASRModel
from ..registry import register_asr_model


@register_asr_model('whisper', default_name='openai/whisper-medium', description='OpenAI Whisper ASR model')
class WhisperModel(HuggingFaceASRModel):
    """Whisper automatic speech recognition model."""

    @dataclass
    class Params:
        size: str = "small"
        SIZES: ClassVar[Dict[str, str]] = {
            "tiny": "openai/whisper-tiny",
            "base": "openai/whisper-base",
            "small": "openai/whisper-small",
            "medium": "openai/whisper-medium",
            "large-v2": "openai/whisper-large-v2",
            "large-v3": "openai/whisper-large-v3",
        }

    def __init__(self, model_name: str = "openai/whisper-small", adapter_path: Optional[str] = None):
        """
        Initialize Whisper ASR model.
        
        Args:
            model_name: HuggingFace model identifier (e.g., 'openai/whisper-small')
            adapter_path: Optional path to PEFT/LoRA adapter weights
        """
        super().__init__(model_name, adapter_path)
        
        # Whisper-specific: also load feature extractor for direct access
        from transformers import WhisperFeatureExtractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    
    def _create_processor(self):
        """Create Whisper processor."""
        from transformers import WhisperProcessor
        return WhisperProcessor.from_pretrained(self.model_name)
    
    def _create_model(self):
        """Create Whisper model for conditional generation."""
        from transformers import WhisperForConditionalGeneration
        return WhisperForConditionalGeneration.from_pretrained(self.model_name)
    
    def _extract_features(self, processed_audio: List):
        """Extract features using Whisper feature extractor."""
        inputs = self.feature_extractor(
            processed_audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True
        )
        return inputs.input_features, inputs.attention_mask
    
    def _generate_transcriptions(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        language: Optional[str]
    ) -> List[str]:
        """Generate transcriptions using Whisper's generate method."""
        predicted_ids = self.model.generate(
            features,
            attention_mask=attention_mask,
            language=language
        )
        transcripts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcripts
