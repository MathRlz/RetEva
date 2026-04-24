from typing import Dict, Any, List, Optional

import numpy as np
import torch

from ..storage.cache import CacheManager
from ..devices.memory import get_memory_manager
from ..logging_config import get_logger, TimingContext
from ..models import AudioEmbeddingModel
from ..utils.cache_helpers import CacheMixin, compute_audio_hash

logger = get_logger(__name__)


class AudioEmbeddingPipeline(CacheMixin):
    """
    Pipeline for direct audio to embedding using AudioEmbeddingModel.
    
    This is for models that directly encode audio to embeddings without ASR.
    Implements caching of computed embeddings for efficiency.
    """
    
    def __init__(
        self, 
        audio_embedding_model: AudioEmbeddingModel, 
        cache_manager: Optional[CacheManager] = None
    ) -> None:
        self._model = audio_embedding_model
        self.cache = cache_manager
        self._init_cache_stats(['embeddings'])
        logger.info(f"AudioEmbedding pipeline initialized with model: {audio_embedding_model.name()}")
    
    @property
    def model(self) -> AudioEmbeddingModel:
        """Get the underlying audio embedding model."""
        return self._model
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model.name()
    
    def process(self, audio: torch.Tensor, sampling_rate: int) -> np.ndarray:
        """
        Process a single audio sample and return its embedding.
        
        Args:
            audio: Audio tensor
            sampling_rate: Audio sampling rate in Hz
            
        Returns:
            np.ndarray: Audio embedding vector
        """
        audio_hash = compute_audio_hash(audio, sampling_rate)
        
        cached_emb, hit = self._check_cache(
            'embeddings',
            lambda: self.cache.get_audio_embedding(audio_hash, self._model.name()) if self.cache_enabled else None,
            log_key=f"audio {audio_hash[:8]}"
        )
        if hit:
            return cached_emb
        
        with TimingContext("Direct audio embedding", logger):
            embedding = self._model.encode_audio([audio], [sampling_rate])[0]
        
        self._store_cache(lambda: self.cache.set_audio_embedding(audio_hash, self._model.name(), embedding))
        return embedding
    
    def process_batch(
        self, 
        audio_list: List[torch.Tensor], 
        sampling_rates: List[int], 
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Process a batch of audio samples and return their embeddings.
        
        Args:
            audio_list: List of audio tensors
            sampling_rates: List of sampling rates
            show_progress: Whether to show progress bar (not currently used)
            
        Returns:
            np.ndarray: Array of audio embeddings with shape (len(audio_list), embedding_dim)
        """
        memory_manager = get_memory_manager()
        embeddings: List[Optional[np.ndarray]] = [None] * len(audio_list)
        compute_needed: List[tuple[torch.Tensor, int, Optional[str]]] = []
        compute_indices: List[int] = []
        
        # Check cache for all items
        if self.cache_enabled:
            for idx, audio in enumerate(audio_list):
                audio_hash = compute_audio_hash(audio, sampling_rates[idx])
                cached_emb = self.cache.get_audio_embedding(audio_hash, self._model.name())
                
                if cached_emb is not None:
                    embeddings[idx] = cached_emb
                    self._record_hit('embeddings')
                else:
                    compute_needed.append((audio, sampling_rates[idx], audio_hash))
                    compute_indices.append(idx)
                    self._record_miss('embeddings')
            
            cache_hits = len(audio_list) - len(compute_needed)
            if cache_hits > 0:
                logger.info(f"Audio embedding cache: {cache_hits}/{len(audio_list)} hits")
        else:
            # No cache, compute all
            compute_needed = [(audio, sr, None) for audio, sr in zip(audio_list, sampling_rates)]
            compute_indices = list(range(len(audio_list)))
            self._record_miss('embeddings', len(audio_list))
        
        # Compute missing embeddings
        if compute_needed:
            with TimingContext(f"Computing {len(compute_needed)} audio embeddings", logger):
                audios_to_compute = [item[0] for item in compute_needed]
                srs_to_compute = [item[1] for item in compute_needed]
                computed_embs = self._model.encode_audio(audios_to_compute, srs_to_compute)
                memory_manager.record_operation()
                
                # Cache and insert computed embeddings
                for idx, (_, _, audio_hash), emb in zip(compute_indices, compute_needed, computed_embs):
                    embeddings[idx] = emb
                    if self.cache_enabled and audio_hash:
                        self.cache.set_audio_embedding(audio_hash, self._model.name(), emb)
        
        result = np.array(embeddings)
        
        # Clear GPU cache after batch processing
        memory_manager.clear_gpu_cache()
        
        return result
