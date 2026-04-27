"""CLIP text embedding model implementation."""
from dataclasses import dataclass
from typing import ClassVar, Dict, List
import torch
import numpy as np
from ..base import TextEmbeddingModel
from ..registry import register_text_embedding_model


@register_text_embedding_model('clip', default_name='openai/clip-vit-base-patch32', description='OpenAI CLIP text encoder')
class ClipModel(TextEmbeddingModel):
    """CLIP text embedding model."""

    @dataclass
    class Params:
        size: str = "base"
        SIZES: ClassVar[Dict[str, str]] = {
            "base": "openai/clip-vit-base-patch32",
            "large": "openai/clip-vit-large-patch14",
        }

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPTokenizer

        self.model_name = model_name
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        self.model.to(device)
        self.device = device
        return self

    def encode(self, texts: List[str], show_progress: bool = False, desc: str = "Embedding") -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        torch.cuda.empty_cache()

        return embeddings.cpu().numpy()

    def name(self) -> str:
        return f"ClipModel - {self.model_name}"
