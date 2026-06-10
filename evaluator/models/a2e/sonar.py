"""SONAR speech embedding model implementation.

The SONAR speech encoder lands in the same 1024-d space as the SONAR text
encoder (``t2e/sonar.py``), so audio queries can be retrieved against a text
corpus zero-shot — no trained projection head required.

SONAR speech encoders are language-specific (``sonar_speech_encoder_<lang>``).
"""
import os
import tempfile
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional
import torch
import numpy as np
from ..base import AudioEmbeddingModel
from ..registry import register_audio_embedding_model
from .._sonar_base import SonarPipelineMixin, SONAR_INSTALL_HINT
from .attention_pool import _resample_to_16k


@register_audio_embedding_model(
    'sonar_speech',
    default_name='sonar_speech_encoder_eng',
    description='SONAR speech encoder — same 1024-d space as SONAR text (zero-shot cross-modal)',
    embedding_space='sonar',  # shared with sonar text encoder → cross-modal retrieval valid
)
class SonarSpeechModel(SonarPipelineMixin, AudioEmbeddingModel):
    """SONAR speech encoder for zero-shot cross-modal retrieval."""

    @dataclass
    class Params:
        size: str = "eng"
        SIZES: ClassVar[Dict[str, str]] = {
            "eng": "sonar_speech_encoder_eng",
        }

    def __init__(self, model_name: str = "sonar_speech_encoder_eng"):
        """
        Initialize SONAR speech encoder.

        Args:
            model_name: SONAR speech encoder identifier (language-specific).
        """
        self.model_name = model_name
        self.device = torch.device("cpu")
        self._pipeline = self._build_pipeline(self.device)

    def _build_pipeline(self, device: torch.device):
        try:
            from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
        except ImportError as exc:
            raise ImportError(SONAR_INSTALL_HINT) from exc
        return SpeechToEmbeddingModelPipeline(encoder=self.model_name, device=device)

    def preprocess_audio(self, audio_list: List[torch.Tensor], sampling_rates: List[int]):
        """Resample clips to 16 kHz mono numpy arrays (SONAR fbank input rate)."""
        return _resample_to_16k(audio_list, sampling_rates)

    def _write_temp_wavs(self, processed: List[np.ndarray]) -> List[str]:
        """Persist 16 kHz clips to temp WAVs; SONAR speech pipeline reads files."""
        import soundfile as sf
        paths = []
        for clip in processed:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(path, np.asarray(clip, dtype=np.float32), 16000)
            paths.append(path)
        return paths

    def encode_from_features(self, features, attention_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """Encode preprocessed 16 kHz clips into SONAR embeddings.

        ``features`` is the list of 16 kHz numpy clips from ``preprocess_audio``.
        """
        wav_paths = self._write_temp_wavs(features)
        try:
            embeddings = self._pipeline.predict(wav_paths)
        finally:
            for path in wav_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        return np.asarray(embeddings, dtype=np.float32)

    def encode_audio(self, audio_list: List[torch.Tensor], sampling_rates: List[int],
                     show_progress: bool = False) -> np.ndarray:
        processed = self.preprocess_audio(audio_list, sampling_rates)
        return self.encode_from_features(processed)

    def name(self) -> str:
        return f"SonarSpeechModel - {self.model_name}"
