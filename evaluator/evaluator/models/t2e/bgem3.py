"""BGE-M3 text embedding model implementation."""
from dataclasses import dataclass
from typing import ClassVar, Dict, List
import torch
import numpy as np
from ..base import TextEmbeddingModel
from ..registry import register_text_embedding_model


@register_text_embedding_model('bge_m3', default_name='BAAI/bge-m3', description='BGE-M3 multilingual embedding model')
class BgeM3Model(TextEmbeddingModel):
    """BGE-M3 text embedding model."""

    @dataclass
    class Params:
        size: str = "default"
        SIZES: ClassVar[Dict[str, str]] = {
            "default": "BAAI/bge-m3",
        }

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from FlagEmbedding import BGEM3FlagModel
        
        self.model_name = model_name
        self.model = BGEM3FlagModel(model_name, use_fp16=True)

    def to(self, device: torch.device):
        # BgeM3FlagModel handles device internally
        return self

    def encode(self, texts: List[str], show_progress: bool = False, desc: str = "Embedding") -> np.ndarray:
        with torch.no_grad():
            embeddings = self.model.encode(texts)["dense_vecs"]
        torch.cuda.empty_cache()
        return embeddings

    def name(self) -> str:
        return f"BgeM3Model - {self.model_name}"
