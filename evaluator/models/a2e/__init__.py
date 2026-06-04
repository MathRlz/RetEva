"""A2E (Audio-to-Embedding) models."""

from .attention_pool import (
    AttentionPoolAudioModel,
    M4TAttentionPoolAudioModel,
    AttentionPooling,
    ProjectionHead,
)
from .clap_style import MultimodalClapStyleModel
from .hubert import HuBERTAudioModel
from .wavlm import WavLMAudioModel
from .sonar import SonarSpeechModel

__all__ = [
    "AttentionPoolAudioModel",
    "M4TAttentionPoolAudioModel",
    "AttentionPooling",
    "ProjectionHead",
    "MultimodalClapStyleModel",
    "HuBERTAudioModel",
    "WavLMAudioModel",
    "SonarSpeechModel",
]
