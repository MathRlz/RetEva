"""
Abstract base classes for pipeline components.

This module provides abstract base classes that combine the CacheMixin
functionality with a defined interface for pipeline components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from ..storage.cache import CacheManager
from ..utils.cache_helpers import CacheMixin


class BasePipelineABC(CacheMixin, ABC):
    """
    Abstract base class for all pipeline components.
    
    Provides:
    - Cache management via CacheMixin
    - Consistent interface for cache statistics
    - Abstract methods for pipeline-specific functionality
    
    Subclasses must:
    1. Set self.cache in __init__
    2. Call _init_cache_stats() with category names
    3. Implement process() and process_batch() methods
    """
    
    cache: Optional[CacheManager]
    
    @property
    def model_name(self) -> str:
        """
        Get the name of the underlying model.
        
        Subclasses should override this to return the appropriate model name.
        """
        if hasattr(self, 'model') and hasattr(self.model, 'name'):
            return self.model.name()
        return "unknown"


class EmbeddingPipelineABC(BasePipelineABC):
    """
    Abstract base class for embedding pipelines.
    
    Used as a base for TextEmbeddingPipeline and AudioEmbeddingPipeline.
    Embedding pipelines convert input data into dense vector representations.
    """
    
    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Process a single input and return its embedding.
        
        Returns:
            np.ndarray: Embedding vector
        """
        pass
    
    @abstractmethod
    def process_batch(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Process a batch of inputs and return their embeddings.
        
        Returns:
            np.ndarray: Array of embedding vectors
        """
        pass


class TranscriptionPipelineABC(BasePipelineABC):
    """
    Abstract base class for transcription/ASR pipelines.
    
    Transcription pipelines convert audio input into text.
    """
    
    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> str:
        """
        Process a single audio input and return its transcription.
        
        Returns:
            str: Transcription text
        """
        pass
    
    @abstractmethod
    def process_batch(self, *args: Any, **kwargs: Any) -> List[str]:
        """
        Process a batch of audio inputs and return their transcriptions.
        
        Returns:
            List[str]: List of transcription texts
        """
        pass
