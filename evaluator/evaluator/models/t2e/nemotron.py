"""Nemotron text embedding model implementation."""
from dataclasses import dataclass
from typing import ClassVar, Dict, List
import torch
import numpy as np
from ..base import TextEmbeddingModel
from ..registry import register_text_embedding_model


@register_text_embedding_model('nemotron', default_name='nvidia/llama-embed-nemotron-8b', description='NVIDIA Nemotron embedding model')
class NemotronModel(TextEmbeddingModel):
    """NVIDIA Nemotron text embedding model."""

    @dataclass
    class Params:
        size: str = "8b"
        SIZES: ClassVar[Dict[str, str]] = {
            "8b": "nvidia/llama-embed-nemotron-8b",
        }

    def __init__(self, model_name: str = "nvidia/llama-embed-nemotron-8b"):
        from sentence_transformers import SentenceTransformer
        attn_implementation = "flash_attention_2"
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs={
                "attn_implementation": attn_implementation,
                "torch_dtype": torch.bfloat16,
            },
            tokenizer_kwargs={"padding_side": "left"},
        )

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def encode(self, texts: List[str], show_progress: bool = False, desc: str = "Embedding") -> np.ndarray:
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress, tqdm_kwargs={"desc": desc} if show_progress else {})
        torch.cuda.empty_cache()
        return embeddings

    def name(self) -> str:
        return f"NemotronModel - {self.model_name}"
