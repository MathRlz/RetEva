"""WavLM audio embedding (microsoft/wavlm-*)."""
from dataclasses import dataclass
from typing import ClassVar, Dict

from ..registry import register_audio_embedding_model
from ._hf_ssl import HFSpeechEncoderEmbedding


@register_audio_embedding_model(
    "wavlm", default_name="microsoft/wavlm-base",
    description="WavLM self-supervised audio representation model from Microsoft",
)
class WavLMAudioModel(HFSpeechEncoderEmbedding):
    HF_MODEL_CLASS = "WavLMModel"
    DEFAULT_MODEL_NAME = "microsoft/wavlm-base"

    @dataclass
    class Params:
        size: str = "base"
        pooling: str = "mean"
        SIZES: ClassVar[Dict[str, str]] = {
            "base": "microsoft/wavlm-base",
            "base-plus": "microsoft/wavlm-base-plus",
            "large": "microsoft/wavlm-large",
        }
        CHOICES: ClassVar[Dict[str, list]] = {"pooling": ["mean", "cls"]}
