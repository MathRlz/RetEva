from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch

from ..storage.cache import CacheManager
from ..devices.memory import get_memory_manager
from ..logging_config import get_logger, TimingContext
from ..models import ASRModel
from ..utils.cache_helpers import CacheMixin, compute_audio_hash

logger = get_logger(__name__)


class ASRPipeline(CacheMixin):
    """
    Pipeline for ASR with caching of features and transcriptions.
    
    Converts audio input into text transcriptions using the configured
    ASR model. Supports caching of both intermediate features and 
    final transcriptions for efficiency.
    """
    
    def __init__(
        self, 
        model: ASRModel, 
        cache_manager: Optional[CacheManager] = None
    ) -> None:
        self.model = model
        self.cache = cache_manager
        self._init_cache_stats(['features', 'transcriptions'])
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model.name()

    def process(
        self, 
        audio: torch.Tensor, 
        sampling_rate: int, 
        language: Optional[str] = None
    ) -> str:
        """
        Process a single audio sample and return its transcription.
        
        Args:
            audio: Audio tensor
            sampling_rate: Audio sampling rate in Hz
            language: Optional language code for transcription
            
        Returns:
            str: Transcription text
        """
        audio_hash = compute_audio_hash(audio, sampling_rate)
        model_name = self.model.name()
        
        # Check transcription cache
        cached_trans, hit = self._check_cache(
            'transcriptions',
            lambda: self.cache.get_transcription(audio_hash, model_name, language) if self.cache_enabled else None,
            log_key=f"audio {audio_hash[:8]}"
        )
        if hit:
            return cached_trans
        
        # Check features cache
        features, attention_mask = None, None
        cached_features, hit = self._check_cache(
            'features',
            lambda: self.cache.get_asr_features(audio_hash, model_name) if self.cache_enabled else None,
            log_key=f"audio {audio_hash[:8]}"
        )
        if hit:
            features, attention_mask = cached_features
        
        if features is None:
            with TimingContext("ASR preprocessing", logger):
                features, attention_mask = self.model.preprocess([audio], [sampling_rate])
            self._store_cache(lambda: self.cache.set_asr_features(
                audio_hash, model_name,
                features.cpu().numpy() if isinstance(features, torch.Tensor) else features,
                attention_mask.cpu().numpy() if isinstance(attention_mask, torch.Tensor) else attention_mask
            ))
        
        with TimingContext("ASR transcription", logger):
            transcriptions = self.model.transcribe_from_features(features, attention_mask, language)
        transcription = transcriptions[0]
        
        self._store_cache(lambda: self.cache.set_transcription(audio_hash, model_name, transcription, language))
        return transcription

    def process_batch(
        self, 
        audio_list: List[torch.Tensor], 
        sampling_rates: List[int], 
        language: Optional[str] = None
    ) -> List[str]:
        """
        Process a batch of audio samples and return their transcriptions.
        
        Args:
            audio_list: List of audio tensors
            sampling_rates: List of sampling rates
            language: Optional language code for transcription
            
        Returns:
            List[str]: List of transcription texts
        """
        model_name = self.model.name()
        memory_manager = get_memory_manager()
        transcriptions: List[Optional[str]] = [None] * len(audio_list)
        uncached_indices: List[int] = []
        audio_hashes: List[str] = []
        
        # Hoist cache check outside loop for performance
        cache_enabled = self.cache_enabled
        
        for idx, (audio, sr) in enumerate(zip(audio_list, sampling_rates)):
            audio_hash = compute_audio_hash(audio, sr)
            audio_hashes.append(audio_hash)
            
            if cache_enabled:
                cached_trans = self.cache.get_transcription(audio_hash, model_name, language)
                if cached_trans is not None:
                    transcriptions[idx] = cached_trans
                    self._record_hit('transcriptions')
                    continue
            
            uncached_indices.append(idx)
            if cache_enabled:
                self._record_miss('transcriptions')
        
        if uncached_indices:
            uncached_audio = [audio_list[i] for i in uncached_indices]
            uncached_sr = [sampling_rates[i] for i in uncached_indices]
            with TimingContext(f"ASR batch processing ({len(uncached_indices)} samples)", logger):
                features, attention_mask = self.model.preprocess(uncached_audio, uncached_sr)
                batch_transcriptions = self.model.transcribe_from_features(features, attention_mask, language)
                memory_manager.record_operation()
            
            for idx, trans in zip(uncached_indices, batch_transcriptions):
                transcriptions[idx] = trans
                if cache_enabled:
                    self.cache.set_transcription(audio_hashes[idx], model_name, trans, language)
        
        # Clear GPU cache after batch processing
        memory_manager.clear_gpu_cache()
        
        return transcriptions  # type: ignore[return-value]

    def process_dataset(
        self,
        dataset: Any,
        batch_size: int = 32,
        num_workers: int = 0,
        language: Optional[str] = None,
        checkpoint_interval: int = 500,
        experiment_id: Optional[str] = None,
        resume_from_checkpoint: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        Process entire dataset through ASR with caching and checkpointing.
        
        Uses DataLoader with preprocessing collate function for efficient batching.
        Checks cache first for each sample, only processes uncached samples.
        
        Args:
            dataset: Dataset with audio_array, sampling_rate, transcription fields
            batch_size: Batch size for processing
            num_workers: Number of DataLoader workers for parallel preprocessing
            language: Optional language code for ASR
            checkpoint_interval: Save checkpoint every N samples
            experiment_id: Unique ID for checkpointing
            resume_from_checkpoint: Whether to resume from existing checkpoint
            
        Returns:
            Tuple of (hypotheses, ground_truth) lists
        """
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        
        model_name = self.model.name()
        memory_manager = get_memory_manager()
        all_hypotheses = []
        all_ground_truth = []
        start_idx = 0
        
        # Check for checkpoint
        if resume_from_checkpoint and self.cache and experiment_id:
            checkpoint = self.cache.get_checkpoint(f"{experiment_id}_asr")
            if checkpoint:
                start_idx = checkpoint.get('last_idx', 0)
                all_hypotheses = checkpoint.get('hypotheses', [])
                all_ground_truth = checkpoint.get('ground_truth', [])
                logger.info(f"Resuming ASR from checkpoint at sample {start_idx}")
        
        # Create collate function that checks cache and preprocesses
        def preprocessing_collate_fn(batch):
            """Collate that checks cache first, then preprocesses uncached samples."""
            cached_results = []
            uncached_items = []
            uncached_indices = []
            transcriptions = []
            audio_hashes = []
            
            for i, item in enumerate(batch):
                audio = item["audio_array"]
                if not isinstance(audio, torch.Tensor):
                    audio = torch.tensor(audio)
                
                sr = item["sampling_rate"]
                transcriptions.append(item["transcription"])
                
                # Compute hash and check cache
                audio_hash = compute_audio_hash(audio, sr)
                audio_hashes.append(audio_hash)
                
                if self.cache_enabled:
                    cached_trans = self.cache.get_transcription(audio_hash, model_name, language)
                    if cached_trans is not None:
                        cached_results.append((i, cached_trans))
                        self._record_hit('transcriptions')
                        continue
                
                self._record_miss('transcriptions')
                uncached_items.append((audio, sr))
                uncached_indices.append(i)
            
            # Preprocess only uncached audio
            features = None
            attention_mask = None
            if uncached_items:
                uncached_audio = [item[0] for item in uncached_items]
                uncached_sr = [item[1] for item in uncached_items]
                features, attention_mask = self.model.preprocess(uncached_audio, uncached_sr)
            
            return {
                "features": features,
                "attention_mask": attention_mask,
                "transcriptions": transcriptions,
                "audio_hashes": audio_hashes,
                "cached_results": cached_results,
                "uncached_indices": uncached_indices,
                "language": batch[0].get("language", language)
            }
        
        # Create DataLoader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=preprocessing_collate_fn,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"Processing {len(dataset)} samples through ASR...")
        
        sample_idx = 0
        
        # Use batch processing context manager for optimized memory management
        with memory_manager.batch_processing_scope(cleanup_interval=max(1, checkpoint_interval // batch_size if batch_size > 0 else 10)) as batch_processor:
            with TimingContext("ASR Dataset Processing", logger):
                for batch in tqdm(data_loader, desc="ASR Processing"):
                    batch_size_actual = len(batch["transcriptions"])
                    
                    # Skip already processed samples (from checkpoint)
                    if sample_idx + batch_size_actual <= start_idx:
                        sample_idx += batch_size_actual
                        continue
                    
                    # Record batch for memory management
                    batch_processor.record_batch(batch_size=batch_size_actual)
                    
                    transcriptions = batch["transcriptions"]
                    audio_hashes = batch["audio_hashes"]
                    cached_results = batch["cached_results"]
                    uncached_indices = batch["uncached_indices"]
                    lang = batch["language"]
                    
                    # Initialize results with None
                    hypotheses = [None] * batch_size_actual
                    
                    # Fill in cached results
                    for idx, trans in cached_results:
                        hypotheses[idx] = trans
                    
                    # Process uncached samples through model
                    if uncached_indices and batch["features"] is not None:
                        features = batch["features"]
                        attention_mask = batch["attention_mask"]
                        
                        # Move to device if needed
                        if hasattr(self.model, 'device'):
                            features = features.to(self.model.device)
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(self.model.device)
                        
                        # Transcribe
                        batch_transcriptions = self.model.transcribe_from_features(
                            features, attention_mask, lang
                        )
                        
                        # Fill in results and cache
                        for i, orig_idx in enumerate(uncached_indices):
                            trans = batch_transcriptions[i]
                            hypotheses[orig_idx] = trans
                            
                            # Cache the transcription
                            if self.cache_enabled:
                                self.cache.set_transcription(
                                    audio_hashes[orig_idx], model_name, trans, lang
                                )
                    
                    all_hypotheses.extend(hypotheses)
                    all_ground_truth.extend(transcriptions)
                    
                    sample_idx += batch_size_actual
                    
                    # Checkpoint
                    if self.cache and experiment_id and sample_idx % checkpoint_interval == 0:
                        checkpoint_data = {
                            'last_idx': sample_idx,
                            'hypotheses': all_hypotheses,
                            'ground_truth': all_ground_truth
                        }
                        self.cache.set_checkpoint(f"{experiment_id}_asr", checkpoint_data)
                        logger.info(f"ASR checkpoint saved at sample {sample_idx}")
            
            # Get batch processing statistics
            batch_stats = batch_processor.get_stats()
            logger.info(f"ASR processing complete: {batch_stats['batches_processed']} batches processed")
        
        logger.info(f"ASR processing complete: {len(all_hypotheses)} transcriptions")
        
        # Clear checkpoint on success
        if self.cache and experiment_id:
            try:
                import os
                checkpoint_path = self.cache._get_cache_path("checkpoints", f"{experiment_id}_asr", ".json")
                if checkpoint_path.exists():
                    os.remove(checkpoint_path)
            except OSError as exc:
                logger.warning(f"Failed to remove ASR checkpoint '{checkpoint_path}': {exc}")
        
        return all_hypotheses, all_ground_truth
