"""
Protocol definitions for pipeline interfaces.

This module defines Protocol classes that specify the expected interfaces
for pipeline components, enabling type checking and ensuring consistency
across different pipeline implementations.
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, TypedDict, Union, runtime_checkable

import numpy as np
import torch
from ..models.retrieval.contracts import ScoredRetrievalResult


@runtime_checkable
class CacheStatsProvider(Protocol):
    """Protocol for components that provide cache statistics."""
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics. For single-category caches:
            - hits: Number of cache hits
            - misses: Number of cache misses  
            - hit_rate: Ratio of hits to total requests
            
            For multi-category caches, returns nested dict with per-category stats.
        """
        ...
    
    def reset_cache_stats(self) -> None:
        """Reset all cache statistics to zero."""
        ...


@runtime_checkable
class BasePipeline(Protocol):
    """
    Base protocol for all pipeline components.
    
    All pipelines should implement:
    - get_cache_stats() for cache statistics reporting
    - reset_cache_stats() for resetting cache statistics
    """
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...
    
    def reset_cache_stats(self) -> None:
        """Reset cache statistics."""
        ...


@runtime_checkable
class EmbeddingPipeline(Protocol):
    """
    Protocol for embedding pipelines (text and audio).
    
    Embedding pipelines convert input data (text or audio) into dense
    vector representations (embeddings).
    """
    
    def process(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Process a single input and return its embedding.
        
        Returns:
            np.ndarray: Embedding vector
        """
        ...
    
    def process_batch(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Process a batch of inputs and return their embeddings.
        
        Returns:
            np.ndarray: Array of embedding vectors with shape (batch_size, embedding_dim)
        """
        ...
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...


@runtime_checkable
class TextEmbeddingPipelineProtocol(Protocol):
    """Protocol for text embedding pipelines."""
    
    def process(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            np.ndarray: Text embedding vector
        """
        ...
    
    def process_batch(
        self, 
        texts: List[str], 
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            np.ndarray: Array of text embeddings with shape (len(texts), embedding_dim)
        """
        ...
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...


@runtime_checkable
class AudioEmbeddingPipelineProtocol(Protocol):
    """Protocol for audio embedding pipelines."""
    
    def process(
        self, 
        audio: torch.Tensor, 
        sampling_rate: int
    ) -> np.ndarray:
        """
        Embed a single audio sample.
        
        Args:
            audio: Audio tensor
            sampling_rate: Audio sampling rate in Hz
            
        Returns:
            np.ndarray: Audio embedding vector
        """
        ...
    
    def process_batch(
        self,
        audio_list: List[torch.Tensor],
        sampling_rates: List[int],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed a batch of audio samples.
        
        Args:
            audio_list: List of audio tensors
            sampling_rates: List of sampling rates
            show_progress: Whether to show progress bar
            
        Returns:
            np.ndarray: Array of audio embeddings with shape (len(audio_list), embedding_dim)
        """
        ...
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...


@runtime_checkable
class ASRPipelineProtocol(Protocol):
    """Protocol for ASR (Automatic Speech Recognition) pipelines."""
    
    def process(
        self,
        audio: torch.Tensor,
        sampling_rate: int,
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe a single audio sample.
        
        Args:
            audio: Audio tensor
            sampling_rate: Audio sampling rate in Hz
            language: Optional language code
            
        Returns:
            str: Transcription text
        """
        ...
    
    def process_batch(
        self,
        audio_list: List[torch.Tensor],
        sampling_rates: List[int],
        language: Optional[str] = None
    ) -> List[str]:
        """
        Transcribe a batch of audio samples.
        
        Args:
            audio_list: List of audio tensors
            sampling_rates: List of sampling rates
            language: Optional language code
            
        Returns:
            List[str]: List of transcriptions
        """
        ...
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...


@runtime_checkable
class RetrievalPipelineProtocol(Protocol):
    """Protocol for retrieval pipelines."""
    
    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Any]] = None
    ) -> None:
        """
        Build the retrieval index from embeddings.
        
        Args:
            embeddings: Array of embedding vectors
            metadata: Optional list of metadata/payloads for each embedding
        """
        ...
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Union[ScoredRetrievalResult, Tuple[Any, float]]]:
        """
        Search for similar items using a single query embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (payload, score) tuples
        """
        ...
    
    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        query_texts: Optional[List[str]] = None
    ) -> List[List[Union[ScoredRetrievalResult, Tuple[Any, float]]]]:
        """
        Search for similar items using multiple query embeddings.
        
        Args:
            query_embeddings: Array of query embedding vectors
            k: Number of results to return per query
            query_texts: Optional query texts for sparse/hybrid retrieval
            
        Returns:
            List of result lists, each containing (payload, score) tuples
        """
        ...
    
    def save(self, path: str) -> None:
        """Save the index to disk."""
        ...
    
    def load(self, path: str) -> None:
        """Load the index from disk."""
        ...
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache/index statistics."""
        ...


# Type aliases for common return types
EmbeddingArray = np.ndarray
SearchResult = Union[ScoredRetrievalResult, Tuple[Any, float]]
SearchResults = List[SearchResult]
BatchSearchResults = List[SearchResults]


class RetrievalPayload(TypedDict, total=False):
    """Canonical payload shape used by retrieval/evaluation."""
    doc_id: str
    text: str

# Cache stats type
CacheStats = Dict[str, Union[int, float, Dict[str, Union[int, float]]]]
