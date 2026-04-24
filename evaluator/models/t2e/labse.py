"""LaBSE text embedding model implementation."""
from dataclasses import dataclass
from typing import ClassVar, Dict, List
import torch
import numpy as np
from ..base import TextEmbeddingModel
from ..registry import register_text_embedding_model


@register_text_embedding_model('labse', default_name='sentence-transformers/LaBSE', description='Language-agnostic BERT Sentence Embedding')
class LabseModel(TextEmbeddingModel):
    """LaBSE (Language-agnostic BERT Sentence Embedding) text embedding model."""

    @dataclass
    class Params:
        size: str = "default"
        SIZES: ClassVar[Dict[str, str]] = {
            "default": "sentence-transformers/LaBSE",
        }

    def __init__(self, model_name: str = "sentence-transformers/LaBSE"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)
        torch.cuda.empty_cache()
        return embeddings

    def name(self) -> str:
        return f"LaBseModel - {self.model_name}"
