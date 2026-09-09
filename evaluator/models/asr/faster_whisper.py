"""Faster Whisper ASR model implementation using CTranslate2."""
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple
import numpy as np
import torch
import torchaudio
from ..base import ASRModel
from ..registry import register_asr_model


# Mapping from model size names to faster-whisper model identifiers
WHISPER_MODEL_SIZES = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3",
    "distil-large-v2", "distil-large-v3",
    "distil-medium.en", "distil-small.en",
]


@register_asr_model('faster_whisper', default_name='large-v3', description='Faster Whisper ASR model using CTranslate2')
class FasterWhisperModel(ASRModel):

    @dataclass
    class Params:
        size: str = "large-v3"
        compute_type: str = "float16"
        SIZES: ClassVar[Dict[str, str]] = {
            "tiny": "tiny", "base": "base", "small": "small",
            "medium": "medium", "large-v1": "large-v1",
            "large-v2": "large-v2", "large-v3": "large-v3",
            "distil-large-v2": "distil-large-v2",
            "distil-large-v3": "distil-large-v3",
        }
    """Faster Whisper automatic speech recognition model using CTranslate2.

    This model uses the faster-whisper library which provides significantly
    faster inference compared to the original OpenAI Whisper implementation
    through CTranslate2 optimization.
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        adapter_path: Optional[str] = None,
        compute_type: str = "float16",
        device: str = "auto",
        cpu_threads: int = 4,
        num_workers: int = 1,
    ):
        """``model_name`` accepts a size name (see ``WHISPER_MODEL_SIZES``) or a path to a
        CTranslate2-converted model. ``adapter_path`` is not supported for faster-whisper
        (kept for interface compatibility). ``compute_type`` options: int8, int8_float16,
        int8_float32, int8_bfloat16, int16, float16, bfloat16, float32.
        """
        try:
            from faster_whisper import WhisperModel as FWWhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper library is required for FasterWhisperModel. "
                "Install with: pip install faster-whisper"
            )

        self.model_name = model_name
        self.adapter_path = adapter_path
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers

        if adapter_path:
            import warnings
            warnings.warn(
                "adapter_path is not supported for FasterWhisperModel and will be ignored. "
                "Use the standard WhisperModel for adapter support."
            )

        if device == "auto":
            self._device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device_str = device

        self.device = torch.device(self._device_str if self._device_str != "cuda" else "cuda:0")

        self.model = FWWhisperModel(
            model_name,
            device=self._device_str,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )

    def to(self, device: torch.device) -> "FasterWhisperModel":
        """faster-whisper models are initialized on a specific device and cannot be easily
        moved; this method recreates the model on the new device instead."""
        from faster_whisper import WhisperModel as FWWhisperModel

        new_device_str = "cuda" if device.type == "cuda" else "cpu"

        if new_device_str != self._device_str:
            self._device_str = new_device_str
            self.device = device

            self.model = FWWhisperModel(
                self.model_name,
                device=new_device_str,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers,
            )
        else:
            self.device = device

        return self

    def preprocess(
        self,
        audio_list: List[torch.Tensor],
        sampling_rates: List[int]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Resample audio to 16kHz if necessary and convert to numpy arrays."""
        processed_audio = []

        for idx in range(len(audio_list)):
            audio = audio_list[idx]
            sr = sampling_rates[idx]

            if not isinstance(audio, torch.Tensor):
                audio = torch.tensor(audio)

            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
                sampling_rates[idx] = 16000

            # Not routed through to_mono_16k: faster-whisper's ctranslate2 backend requires
            # float32 arrays, so this keeps the explicit dtype cast the helper doesn't do.
            audio_np = audio.squeeze().numpy().astype(np.float32)
            processed_audio.append(audio_np)

        return processed_audio, sampling_rates

    def transcribe_from_features(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        language: Optional[str] = None
    ) -> List[str]:
        """faster-whisper does not support feature-based transcription directly: this method
        exists for interface compatibility and raises unless ``features`` is actually a list
        of numpy arrays (raw audio) — use ``transcribe()`` directly with raw audio instead.
        ``attention_mask`` is ignored.
        """
        # Allows transcribe() to delegate to this method.
        if isinstance(features, (list, tuple)) and len(features) > 0:
            if isinstance(features[0], np.ndarray):
                return self._transcribe_audio_arrays(features, language)

        raise NotImplementedError(
            "FasterWhisperModel does not support transcribe_from_features with mel spectrograms. "
            "Use transcribe() with raw audio instead."
        )

    def _transcribe_audio_arrays(
        self,
        audio_arrays: List[np.ndarray],
        language: Optional[str] = None
    ) -> List[str]:
        """Transcribe audio arrays already resampled to 16kHz."""
        transcripts = []

        for audio in audio_arrays:
            segments, _ = self.model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,
            )

            text = " ".join(segment.text.strip() for segment in segments)
            transcripts.append(text)

        return transcripts

    def transcribe(
        self,
        audio: List[torch.Tensor],
        sampling_rates: List[int],
        language: Optional[str] = None
    ) -> List[str]:
        processed_audio, _ = self.preprocess(audio, sampling_rates)
        return self._transcribe_audio_arrays(processed_audio, language)

    def name(self) -> str:
        name = f"FasterWhisperModel - {self.model_name}"
        name += f" ({self.compute_type})"
        return name
