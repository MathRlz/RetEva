"""Base class for HuggingFace-based ASR models."""
from typing import List, Optional
import torch
import torchaudio
from ..base import ASRModel


class HuggingFaceASRModel(ASRModel):
    """Base class for HuggingFace Transformer-based ASR models.

    Handles device management, PEFT/LoRA adapter loading, and audio preprocessing;
    subclasses implement _create_processor, _create_model, _extract_features,
    and _generate_transcriptions.
    """

    def __init__(self, model_name: str, adapter_path: Optional[str] = None):
        """model_name: HuggingFace identifier (e.g. 'openai/whisper-small');
        adapter_path: optional PEFT/LoRA adapter weights."""
        self.model_name = model_name
        self.adapter_path = adapter_path

        self.processor = self._create_processor()
        self.model = self._create_model()

        if adapter_path:
            self._load_adapter(adapter_path)

        # Inference-only: pin eval mode (no dropout). `from_pretrained` already defaults to eval
        # and the parity gate proves ASR is deterministic — but PEFT `merge_and_unload` can hand
        # back a train-mode model, so be explicit. Hardening, not a behavior change.
        self.model.eval()

        self.device = torch.device("cpu")

    def _create_processor(self):
        """Create and return the model-specific processor. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_processor()")

    def _create_model(self):
        """Create and return the model architecture. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _create_model()")

    def _load_adapter(self, adapter_path: str):
        """Load PEFT/LoRA adapter weights and merge into the base model.
        Raises ImportError if peft is not installed."""
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
        except ImportError:
            raise ImportError(
                "peft library required for adapter loading. Install with: pip install peft"
            )

    def to(self, device: torch.device):
        """Move model to device; returns self for chaining."""
        self.model.to(device)
        self.device = device
        return self

    def preprocess(self, audio_list: List, sampling_rates: List[int]):
        """Convert audio to 16kHz mono tensors and extract features;
        returns (features, attention_mask) ready for model input."""
        processed_audio = []

        for idx in range(len(audio_list)):
            if not isinstance(audio_list[idx], torch.Tensor):
                audio_list[idx] = torch.tensor(audio_list[idx])

            sr = sampling_rates[idx]
            audio_sample = audio_list[idx]

            if sr != 16000:
                audio_list[idx] = torchaudio.functional.resample(audio_sample, sr, 16000)
                sampling_rates[idx] = 16000

            # Ensure 1-D mono: the feature extractors require one channel per clip; a
            # multi-channel clip survives squeeze() and breaks batched padding. Downmix
            # over the channel axis (the smaller dimension: channels << samples).
            audio = audio_list[idx].squeeze()
            if audio.dim() > 1:
                ch_axis = int(min(range(audio.dim()), key=lambda d: audio.shape[d]))
                audio = audio.mean(dim=ch_axis)
            processed_audio.append(audio.numpy())

        return self._extract_features(processed_audio)

    def _extract_features(self, processed_audio: List):
        """Extract (features, attention_mask) from 16kHz numpy audio.
        Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _extract_features()")

    def transcribe_from_features(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        language: Optional[str] = None
    ) -> List[str]:
        """Generate transcriptions from preprocessed features;
        returns one transcript string per clip."""
        with torch.no_grad():
            features = features.to(self.device)
            attention_mask = attention_mask.to(self.device) if attention_mask is not None else None

            transcripts = self._generate_transcriptions(features, attention_mask, language)

        torch.cuda.empty_cache()
        return transcripts

    def _generate_transcriptions(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        language: Optional[str]
    ) -> List[str]:
        """Generate transcriptions from on-device features.
        Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _generate_transcriptions()")

    def transcribe(
        self,
        audio: List[torch.Tensor],
        sampling_rates: List[int],
        language: Optional[str] = None
    ) -> List[str]:
        """Transcribe raw audio into text (preprocess + transcribe_from_features)."""
        features, attention_mask = self.preprocess(audio, sampling_rates)
        return self.transcribe_from_features(features, attention_mask, language)

    def name(self) -> str:
        """Return human-readable model name, including adapter info if present."""
        class_name = self.__class__.__name__
        name = f"{class_name} - {self.model_name}"
        if self.adapter_path:
            name += f" + adapter({self.adapter_path})"
        return name
