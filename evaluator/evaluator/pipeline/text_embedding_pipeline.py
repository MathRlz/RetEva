from typing import Dict, Any, List, Optional

import numpy as np

from ..storage.cache import CacheManager
from ..devices.memory import get_memory_manager
from ..logging_config import get_logger, TimingContext
from ..models import TextEmbeddingModel
from ..utils.cache_helpers import CacheMixin

logger = get_logger(__name__)


class TextEmbeddingPipeline(CacheMixin):
    """
    Pipeline for text embedding with caching.
    
    Converts text input into dense vector representations using the
    configured text embedding model. Supports caching of embeddings
    for efficiency.
    """
    
    def __init__(
        self, 
        model: TextEmbeddingModel, 
        cache_manager: Optional[CacheManager] = None
    ) -> None:
        self.model = model
        self.cache = cache_manager
        self._init_cache_stats(['embeddings'])
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model.name()
    
    def process(self, text: str) -> np.ndarray:
        """
        Process a single text and return its embedding.
        
        Args:
            text: Input text to embed
            
        Returns:
            np.ndarray: Text embedding vector
        """
        model_name = self.model.name()
        
        cached_emb, hit = self._check_cache(
            'embeddings',
            lambda: self.cache.get_embedding(text, model_name) if self.cache is not None else None,
            log_key=f"text: {text[:50]}..."
        )
        if hit and cached_emb is not None:
            return cached_emb
        
        with TimingContext("Text embedding", logger):
            embedding = self.model.encode([text])[0]
        
        if self.cache is not None:
            self._store_cache(lambda: self.cache.set_embedding(text, model_name, embedding))  # type: ignore[union-attr]
        return embedding
    
    def process_batch(self, texts: List[str], show_progress: bool = False, desc: str = "Embedding") -> np.ndarray:
        """
        Process a batch of texts and return their embeddings.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            desc: Label shown in progress bar and timing log

        Returns:
            np.ndarray: Array of text embeddings with shape (len(texts), embedding_dim)
        """
        model_name = self.model.name()
        memory_manager = get_memory_manager()
        
        # Try to get full batch from cache
        if self.cache is not None:
            cached_batch = self.cache.get_embeddings_batch(texts, model_name)
            if cached_batch is not None:
                self._record_hit('embeddings', len(texts))
                logger.debug(f"Full batch cache hit for {len(texts)} texts")
                return cached_batch
        
        # Pre-allocate output array instead of using list
        embeddings: List[np.ndarray | None] = [None] * len(texts)
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []
        
        # Hoist cache check outside loop for performance
        cache_enabled = self.cache is not None
        
        for idx, text in enumerate(texts):
            if cache_enabled:
                cached_emb = self.cache.get_embedding(text, model_name)
                if cached_emb is not None:
                    embeddings[idx] = cached_emb
                    self._record_hit('embeddings')
                    continue
            uncached_texts.append(text)
            uncached_indices.append(idx)
            if cache_enabled:
                self._record_miss('embeddings')
        
        if uncached_texts:
            with TimingContext(f"{desc} ({len(uncached_texts)} texts)", logger):
                batch_embeddings = self.model.encode(uncached_texts, show_progress=show_progress, desc=desc)
                memory_manager.record_operation()
                
            for idx, emb in zip(uncached_indices, batch_embeddings):
                embeddings[idx] = emb
                if cache_enabled:
                    self.cache.set_embedding(texts[idx], model_name, emb)
        
        result = np.array(embeddings)
        
        # Clear GPU cache after batch processing
        memory_manager.clear_gpu_cache()
        
        return result
