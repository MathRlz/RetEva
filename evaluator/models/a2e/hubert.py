"""HuBERT audio embedding (facebook/hubert-*)."""
from dataclasses import dataclass
from typing import ClassVar, Dict

from ..registry import register_audio_embedding_model
from ._hf_ssl import HFSpeechEncoderEmbedding


@register_audio_embedding_model(
    "hubert", default_name="facebook/hubert-base-ls960",
    description="HuBERT self-supervised audio representation model",
)
class HuBERTAudioModel(HFSpeechEncoderEmbedding):
    HF_MODEL_CLASS = "HubertModel"
    DEFAULT_MODEL_NAME = "facebook/hubert-base-ls960"

    @dataclass
    class Params:
        size: str = "base"
        pooling: str = "mean"
        SIZES: ClassVar[Dict[str, str]] = {
            "base": "facebook/hubert-base-ls960",
            "large": "facebook/hubert-large-ll60k",
        }
        CHOICES: ClassVar[Dict[str, list]] = {"pooling": ["mean", "cls"]}
