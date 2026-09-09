from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from ..storage.cache import CacheManager
from ..storage.cache.helpers import CacheMixin, compute_audio_hash
from ..devices.memory import get_memory_manager
from ..logging_config import get_logger, TimingContext
from ..models import ASRModel
from ..utils.progress import progress_iter

logger = get_logger(__name__)


@dataclass
class _AsrLoopParams:
    """Per-run scalar inputs to ``ASRPipeline._run_checkpoint_loop``: bundles the
    six plumbing params the batch loop reads so the loop signature stays focused on the
    moving parts (the DataLoader, the accumulators, the memory manager)."""

    start_idx: int
    batch_size: int
    checkpoint_interval: int
    experiment_id: Optional[str]
    model_name: str
    model_version: Optional[str]


class ASRPipeline(CacheMixin):
    """Pipeline for ASR with caching of features and transcriptions."""

    def __init__(
        self, model: ASRModel, cache_manager: Optional[CacheManager] = None
    ) -> None:
        self.model = model
        self.cache = cache_manager
        self._init_cache_stats(["features", "transcriptions"])

    @property
    def model_name(self) -> str:
        return self.model.name()

    def process(
        self, audio: torch.Tensor, sampling_rate: int, language: Optional[str] = None
    ) -> str:
        audio_hash = compute_audio_hash(audio, sampling_rate)
        model_name = self.model.name()
        model_version = self.model_version

        cached_trans, hit = self._check_cache(
            "transcriptions",
            lambda: (
                self.cache.get_transcription(
                    audio_hash, model_name, language, model_version
                )
                if self.cache_enabled
                else None
            ),
            log_key=f"audio {audio_hash[:8]}",
        )
        if hit:
            return cached_trans

        features, attention_mask = None, None
        cached_features, hit = self._check_cache(
            "features",
            lambda: (
                self.cache.get_asr_features(audio_hash, model_name, model_version)
                if self.cache_enabled
                else None
            ),
            log_key=f"audio {audio_hash[:8]}",
        )
        if hit:
            features, attention_mask = cached_features

        if features is None:
            with TimingContext("ASR preprocessing", logger):
                features, attention_mask = self.model.preprocess(
                    [audio], [sampling_rate]
                )
            self._store_cache(
                lambda: self.cache.set_asr_features(
                    audio_hash,
                    model_name,
                    (
                        features.cpu().numpy()
                        if isinstance(features, torch.Tensor)
                        else features
                    ),
                    (
                        attention_mask.cpu().numpy()
                        if isinstance(attention_mask, torch.Tensor)
                        else attention_mask
                    ),
                    model_version=model_version,
                )
            )

        with TimingContext("ASR transcription", logger):
            transcriptions = self.model.transcribe_from_features(
                features, attention_mask, language
            )
        transcription = transcriptions[0]

        self._store_cache(
            lambda: self.cache.set_transcription(
                audio_hash, model_name, transcription, language, model_version
            )
        )
        return transcription

    def process_batch(
        self,
        audio_list: List[torch.Tensor],
        sampling_rates: List[int],
        language: Optional[str] = None,
    ) -> List[str]:
        model_name = self.model.name()
        model_version = self.model_version
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
                cached_trans = self.cache.get_transcription(
                    audio_hash, model_name, language, model_version
                )
                if cached_trans is not None:
                    transcriptions[idx] = cached_trans
                    self._record_hit("transcriptions")
                    continue

            uncached_indices.append(idx)
            if cache_enabled:
                self._record_miss("transcriptions")

        if uncached_indices:
            uncached_audio = [audio_list[i] for i in uncached_indices]
            uncached_sr = [sampling_rates[i] for i in uncached_indices]
            with TimingContext(
                f"ASR batch processing ({len(uncached_indices)} samples)", logger
            ):
                features, attention_mask = self.model.preprocess(
                    uncached_audio, uncached_sr
                )
                batch_transcriptions = self.model.transcribe_from_features(
                    features, attention_mask, language
                )
                memory_manager.record_operation()

            for idx, trans in zip(uncached_indices, batch_transcriptions):
                transcriptions[idx] = trans
                if cache_enabled:
                    self.cache.set_transcription(
                        audio_hashes[idx], model_name, trans, language, model_version
                    )

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
        resume_from_checkpoint: bool = True,
        warmup_batch_sizing: bool = False,
    ) -> Tuple[List[str], List[str]]:
        from torch.utils.data import DataLoader

        model_name = self.model.name()
        model_version = self.model_version
        memory_manager = get_memory_manager()
        all_hypotheses = []
        all_ground_truth = []
        start_idx = 0

        if resume_from_checkpoint and self.cache and experiment_id:
            checkpoint = self.cache.get_checkpoint(f"{experiment_id}_asr")
            if checkpoint:
                start_idx = checkpoint.get("last_idx", 0)
                all_hypotheses = checkpoint.get("hypotheses", [])
                all_ground_truth = checkpoint.get("ground_truth", [])
                logger.info(f"Resuming ASR from checkpoint at sample {start_idx}")

        # Warm-up batch sizing: estimate a GPU-fitting batch from a one-sample forward,
        # instead of the static `batch_size`. Opt-in, GPU-only, and fully guarded — any failure
        # (or CPU) falls back to the configured `batch_size`. Batch size is output-invariant.
        if warmup_batch_sizing and torch.cuda.is_available() and len(dataset) > 0:
            batch_size = self._warmup_batch_size(dataset, language, memory_manager, batch_size)

        preprocessing_collate_fn = self._make_preprocessing_collate_fn(
            model_name, model_version, language
        )

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=preprocessing_collate_fn,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        logger.info(f"Processing {len(dataset)} samples through ASR...")

        all_hypotheses, all_ground_truth = self._run_checkpoint_loop(
            data_loader,
            all_hypotheses,
            all_ground_truth,
            memory_manager,
            _AsrLoopParams(
                start_idx=start_idx,
                batch_size=batch_size,
                checkpoint_interval=checkpoint_interval,
                experiment_id=experiment_id,
                model_name=model_name,
                model_version=model_version,
            ),
        )

        logger.info(f"ASR processing complete: {len(all_hypotheses)} transcriptions")

        # Clear checkpoint on success
        if self.cache and experiment_id:
            try:
                import os

                checkpoint_path = self.cache._get_cache_path(
                    "checkpoints", f"{experiment_id}_asr", ".json"
                )
                if checkpoint_path.exists():
                    os.remove(checkpoint_path)
            except OSError as exc:
                logger.warning(
                    f"Failed to remove ASR checkpoint '{checkpoint_path}': {exc}"
                )

        return all_hypotheses, all_ground_truth

    def _make_preprocessing_collate_fn(
        self,
        model_name: str,
        model_version: Optional[str],
        language: Optional[str],
    ):
        """Build collate that checks cache first, then preprocesses uncached samples."""

        def preprocessing_collate_fn(batch):
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

                audio_hash = compute_audio_hash(audio, sr)
                audio_hashes.append(audio_hash)

                if self.cache_enabled:
                    cached_trans = self.cache.get_transcription(
                        audio_hash, model_name, language, model_version
                    )
                    if cached_trans is not None:
                        cached_results.append((i, cached_trans))
                        self._record_hit("transcriptions")
                        continue

                self._record_miss("transcriptions")
                uncached_items.append((audio, sr))
                uncached_indices.append(i)

            features = None
            attention_mask = None
            if uncached_items:
                uncached_audio = [item[0] for item in uncached_items]
                uncached_sr = [item[1] for item in uncached_items]
                features, attention_mask = self.model.preprocess(
                    uncached_audio, uncached_sr
                )

            return {
                "features": features,
                "attention_mask": attention_mask,
                "transcriptions": transcriptions,
                "audio_hashes": audio_hashes,
                "cached_results": cached_results,
                "uncached_indices": uncached_indices,
                "language": batch[0].get("language", language),
            }

        return preprocessing_collate_fn

    def _warmup_batch_size(self, dataset, language, memory_manager, fallback: int) -> int:
        """One-sample warm-up → estimated batch. Fully guarded: any error → ``fallback``."""
        try:
            sample = dataset[0]
            audio = torch.as_tensor(sample["audio_array"])
            sr = int(sample.get("sampling_rate", 16000))
            lang = sample.get("language", language)
            est = memory_manager.warm_up_batch_size(
                lambda: self.model.transcribe([audio], [sr], lang), max_batch=fallback,
            )
            if est != fallback:
                logger.info("ASR warm-up batch sizing: %d → %d", fallback, est)
            return est
        except Exception as exc:  # noqa: BLE001 - never let warm-up break the run
            logger.warning(
                "ASR warm-up batch sizing failed (%s); using batch_size=%d", exc, fallback
            )
            return fallback

    def _run_checkpoint_loop(
        self,
        data_loader: Any,
        all_hypotheses: List[str],
        all_ground_truth: List[str],
        memory_manager: Any,
        params: "_AsrLoopParams",
    ) -> Tuple[List[str], List[str]]:
        start_idx = params.start_idx
        batch_size = params.batch_size
        checkpoint_interval = params.checkpoint_interval
        experiment_id = params.experiment_id
        model_name = params.model_name
        model_version = params.model_version
        sample_idx = 0

        with memory_manager.batch_processing_scope(
            cleanup_interval=max(
                1, checkpoint_interval // batch_size if batch_size > 0 else 10
            )
        ) as batch_processor:
            with TimingContext("ASR Dataset Processing", logger):
                for batch in progress_iter(
                    data_loader, "ASR Processing", unit="batch", min_items=1
                ):
                    batch_size_actual = len(batch["transcriptions"])

                    # Skip already processed samples (from checkpoint)
                    if sample_idx + batch_size_actual <= start_idx:
                        sample_idx += batch_size_actual
                        continue

                    batch_processor.record_batch(batch_size=batch_size_actual)

                    transcriptions = batch["transcriptions"]
                    audio_hashes = batch["audio_hashes"]
                    cached_results = batch["cached_results"]
                    uncached_indices = batch["uncached_indices"]
                    lang = batch["language"]

                    hypotheses = [None] * batch_size_actual

                    for idx, trans in cached_results:
                        hypotheses[idx] = trans

                    if uncached_indices and batch["features"] is not None:
                        features = batch["features"]
                        attention_mask = batch["attention_mask"]

                        if hasattr(self.model, "device"):
                            features = features.to(self.model.device)
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(self.model.device)

                        batch_transcriptions = self.model.transcribe_from_features(
                            features, attention_mask, lang
                        )

                        for i, orig_idx in enumerate(uncached_indices):
                            trans = batch_transcriptions[i]
                            hypotheses[orig_idx] = trans

                            if self.cache_enabled:
                                self.cache.set_transcription(
                                    audio_hashes[orig_idx],
                                    model_name,
                                    trans,
                                    lang,
                                    model_version,
                                )

                    all_hypotheses.extend(hypotheses)
                    all_ground_truth.extend(transcriptions)

                    sample_idx += batch_size_actual

                    # Checkpoint (interval <= 0 disables it — avoids modulo by zero)
                    if (
                        self.cache
                        and experiment_id
                        and checkpoint_interval > 0
                        and sample_idx % checkpoint_interval == 0
                    ):
                        checkpoint_data = {
                            "last_idx": sample_idx,
                            "hypotheses": all_hypotheses,
                            "ground_truth": all_ground_truth,
                        }
                        self.cache.set_checkpoint(
                            f"{experiment_id}_asr", checkpoint_data
                        )
                        logger.info(f"ASR checkpoint saved at sample {sample_idx}")

            batch_stats = batch_processor.get_stats()
            logger.info(
                f"ASR processing complete: {batch_stats['batches_processed']} batches processed"
            )

        return all_hypotheses, all_ground_truth
