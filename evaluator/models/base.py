"""Base abstract classes for models."""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import torch


class ASRModel(ABC):
    """Base class for Automatic Speech Recognition models."""
    
    @abstractmethod
    def transcribe(self, audio: List[torch.Tensor],
                   sampling_rates: List[int], language: Optional[str] = None) -> List[str]:
        """Transcribe raw audio into text."""
        pass

    @abstractmethod
    def preprocess(self, audio_list: List[torch.Tensor],
                   sampling_rates: List[int]):
        """Preprocess audio for the model."""
        pass

    @abstractmethod
    def transcribe_from_features(self, features: torch.Tensor,
                                 attention_mask: Optional[torch.Tensor],
                                 language: Optional[str] = None) -> List[str]:
        """Transcribe from preprocessed features."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return model name."""
        pass

    @abstractmethod
    def to(self, device: torch.device):
        """Move model to device."""
        pass


class TextEmbeddingModel(ABC):
    """Base class for text embedding models.

    Normalization contract: ``encode`` is NOT required to return L2-normalized vectors —
    some models do (clip, the audio embedders), most text models don't (labse/jina/bge/
    nemotron/sonar). Retrieval does not assume it: the vector stores normalize on build/search
    for cosine similarity (``storage/vector_store.py``). For cross-modal retrieval, pair models
    trained into the same space (joint embedders); raw cross-model cosine is not meaningful.
    """

    @abstractmethod
    def encode(self, texts: List[str], show_progress: bool = False, desc: str = "Embedding") -> np.ndarray:
        """Encode texts into vectors of shape (N, D); may or may not be L2-normalized
        (see the class docstring — the vector store normalizes for cosine)."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return model name."""
        pass

    def to(self, device: torch.device):
        """Move model to device.

        Default implementation moves ``self.model``; subclasses with a different
        layout (or that manage device internally) override this.
        """
        self.model.to(device)
        return self


class AudioEmbeddingModel(ABC):
    """
    Base class for models that directly embed audio into vectors without ASR.
    This is for end-to-end audio->embedding models that don't produce text transcriptions.
    """
    
    @abstractmethod
    def encode_audio(self, audio_list: List[torch.Tensor], 
                     sampling_rates: List[int], 
                     show_progress: bool = False) -> np.ndarray:
        """
        Encode audio directly into embeddings.
        
        Args:
            audio_list: List of audio tensors
            sampling_rates: Corresponding sampling rates
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings with shape (N, D)
        """
        pass
    
    @abstractmethod
    def preprocess_audio(self, audio_list: List[torch.Tensor],
                        sampling_rates: List[int]):
        """
        Preprocess audio for the model.
        
        Args:
            audio_list: List of audio tensors
            sampling_rates: Corresponding sampling rates
            
        Returns:
            Preprocessed features ready for encoding
        """
        pass
    
    @abstractmethod
    def encode_from_features(self, features: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Encode preprocessed features into embeddings.
        
        Args:
            features: Preprocessed audio features
            attention_mask: Optional attention mask
            
        Returns:
            Array of embeddings
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return model name."""
        pass
    
    @abstractmethod
    def to(self, device: torch.device):
        """Move model to device."""
        pass
