"""JINA v4 text embedding model implementation."""
from dataclasses import dataclass
from typing import ClassVar, Dict, List
import torch
import numpy as np
from transformers import AutoModel, AutoConfig
from ..base import TextEmbeddingModel
from ..registry import register_text_embedding_model


@register_text_embedding_model('jina_v4', default_name='jinaai/jina-embeddings-v4', description='JINA Embeddings v4')
class JinaV4Model(TextEmbeddingModel):
    """JINA Embeddings v4 text embedding model."""

    @dataclass
    class Params:
        size: str = "default"
        SIZES: ClassVar[Dict[str, str]] = {
            "default": "jinaai/jina-embeddings-v4",
        }

    def __init__(self, model_name: str = "jinaai/jina-embeddings-v4"):
        jina_config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
        jina_config.verbosity = 0  # disable "encoding texts" message
        self.model = AutoModel.from_pretrained(
            model_name,
            config=jina_config,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        self.model_name = model_name

    def to(self, device: torch.device):
        self.model.to(device)
        return self

    def encode(self, texts: List[str], show_progress: bool = False, desc: str = "Embedding") -> np.ndarray:
        from tqdm import tqdm
        task = "retrieval"
        prompt_name = "query"
        return_numpy = False
        if show_progress:
            with tqdm(total=len(texts), desc=desc, unit="text") as pbar:
                torch_emb = self.model.encode_text(
                    texts=texts, task=task, prompt_name=prompt_name, return_numpy=return_numpy
                )
                pbar.update(len(texts))
        else:
            torch_emb = self.model.encode_text(
                texts=texts, task=task, prompt_name=prompt_name, return_numpy=return_numpy
            )
        torch.cuda.empty_cache()
        return torch.stack(torch_emb).cpu().numpy()

    def name(self) -> str:
        return "JinaEmbeddingsV4"
