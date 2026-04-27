"""A2E (Audio-to-Embedding) models."""

from .attention_pool import AttentionPoolAudioModel, AttentionPooling, ProjectionHead
from .clap_style import MultimodalClapStyleModel
from .hubert import HuBERTAudioModel
from .wavlm import WavLMAudioModel

__all__ = [
    "AttentionPoolAudioModel",
    "AttentionPooling",
    "ProjectionHead",
    "MultimodalClapStyleModel",
    "HuBERTAudioModel",
    "WavLMAudioModel",
]
