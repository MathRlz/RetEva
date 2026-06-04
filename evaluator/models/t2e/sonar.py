"""SONAR text embedding model implementation.

SONAR (Sentence-level multimOdal and laNguage-Agnostic Representations) provides
a shared text/speech embedding space. This text encoder lands in the same 1024-d
space as the SONAR speech encoder (``a2e/sonar.py``), enabling zero-shot
cross-modal retrieval.
"""
from dataclasses import dataclass
from typing import ClassVar, Dict, List
import torch
import numpy as np
from ..base import TextEmbeddingModel
from ..registry import register_text_embedding_model
from .._sonar_base import SonarPipelineMixin, SONAR_INSTALL_HINT


@register_text_embedding_model(
    'sonar',
    default_name='text_sonar_basic_encoder',
    description='SONAR multilingual text encoder (shared text/speech space, 1024-d)',
)
class SonarTextModel(SonarPipelineMixin, TextEmbeddingModel):
    """SONAR multilingual text encoder."""

    @dataclass
    class Params:
        size: str = "basic"
        source_lang: str = "eng_Latn"
        SIZES: ClassVar[Dict[str, str]] = {
            "basic": "text_sonar_basic_encoder",
        }

    def __init__(
        self,
        model_name: str = "text_sonar_basic_encoder",
        source_lang: str = "eng_Latn",
        tokenizer: str = "text_sonar_basic_encoder",
    ):
        """
        Initialize SONAR text encoder.

        Args:
            model_name: SONAR text encoder identifier.
            source_lang: SONAR language code for the input text (e.g. "eng_Latn").
            tokenizer: SONAR tokenizer identifier (defaults to the encoder).
        """
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.source_lang = source_lang
        self.device = torch.device("cpu")
        self._pipeline = self._build_pipeline(self.device)

    def _build_pipeline(self, device: torch.device):
        try:
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        except ImportError as exc:
            raise ImportError(SONAR_INSTALL_HINT) from exc
        return TextToEmbeddingModelPipeline(
            encoder=self.model_name,
            tokenizer=self.tokenizer,
            device=device,
        )

    def encode(self, texts: List[str], show_progress: bool = False, desc: str = "Embedding") -> np.ndarray:
        embeddings = self._pipeline.predict(texts, source_lang=self.source_lang)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        return np.asarray(embeddings, dtype=np.float32)

    def name(self) -> str:
        return f"SonarTextModel - {self.model_name} - lang:{self.source_lang}"
