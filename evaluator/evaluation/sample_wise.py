"""Sample-wise evaluation using pipelines."""
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from ..datasets import QueryDataset
from ..pipeline import (
    ASRPipelineProtocol,
    TextEmbeddingPipelineProtocol,
    AudioEmbeddingPipelineProtocol,
    RetrievalPipelineProtocol,
    SearchResult,
)
from ..storage.cache import CacheManager
from ..logging_config import get_logger, TimingContext, log_cache_stats
from ..metrics import compute_ir_metrics, log_ir_metrics

from .helpers import _search_results_to_keys, _build_relevant_from_item, collate_fn
from .metrics import word_error_rate, character_error_rate

logger = get_logger(__name__)


def _detect_mode(
    retrieval_pipeline: Optional[RetrievalPipelineProtocol],
    asr_pipeline: Optional[ASRPipelineProtocol],
    text_embedding_pipeline: Optional[TextEmbeddingPipelineProtocol],
    audio_embedding_pipeline: Optional[AudioEmbeddingPipelineProtocol],
) -> str:
    if audio_embedding_pipeline is not None and text_embedding_pipeline is not None:
        logger.info("Evaluation mode: Audio-to-Text Retrieval")
        return "audio_text_retrieval"
    if audio_embedding_pipeline is not None:
        logger.info("Evaluation mode: Audio Embedding Retrieval")
        return "audio_emb_retrieval"
    if asr_pipeline is not None and text_embedding_pipeline is not None:
        if retrieval_pipeline is not None:
            logger.info("Evaluation mode: ASR + Text Retrieval")
            return "asr_text_retrieval"
        logger.info("Evaluation mode: ASR Only")
        return "asr_only"
    if asr_pipeline is not None:
        logger.info("Evaluation mode: ASR Only")
        return "asr_only"
    raise ValueError(
        "Must provide either audio_embedding_pipeline OR asr_pipeline. "
        "Tip: provide (asr_pipeline + text_embedding_pipeline + retrieval_pipeline), "
        "(audio_embedding_pipeline + retrieval_pipeline), or asr_pipeline only."
    )


def _load_checkpoint_state(
    cache_manager: Optional[CacheManager],
    experiment_id: Optional[str],
    resume_from_checkpoint: bool,
) -> Tuple[int, List[float], List[float], List[List[str]], List[Dict[str, float]]]:
    start_idx = 0
    wer_scores: List[float] = []
    cer_scores: List[float] = []
    all_retrieved: List[List[str]] = []
    all_relevant: List[Dict[str, float]] = []

    if resume_from_checkpoint and cache_manager and experiment_id:
        checkpoint = cache_manager.get_checkpoint(experiment_id)
        if checkpoint:
            logger.info(f"Resuming from checkpoint at sample {checkpoint['last_idx']}")
            start_idx = checkpoint["last_idx"]
            wer_scores = checkpoint.get("wer_scores", [])
            cer_scores = checkpoint.get("cer_scores", [])
            all_retrieved = checkpoint.get("all_retrieved", [])
            all_relevant = checkpoint.get("all_relevant", [])
    return start_idx, wer_scores, cer_scores, all_retrieved, all_relevant


def _collect_retrieval_batch(
    results_list: List[List[SearchResult]],
    all_retrieved: List[List[str]],
) -> None:
    for results in results_list:
        all_retrieved.append(_search_results_to_keys(results))


def _run_audio_text_retrieval_batch(
    batch: List[Dict[str, Any]],
    audio_embedding_pipeline: AudioEmbeddingPipelineProtocol,
    retrieval_pipeline: RetrievalPipelineProtocol,
    all_retrieved: List[List[str]],
    all_relevant: List[Dict[str, float]],
    k: int,
) -> None:
    audio_list = [item["audio_array"] for item in batch]
    sampling_rates = [item["sampling_rate"] for item in batch]
    transcriptions = [item["transcription"] for item in batch]
    relevance_batch = [_build_relevant_from_item(item) for item in batch]

    with TimingContext(f"Audio embedding batch ({len(batch)} samples)", logger):
        audio_embeddings = audio_embedding_pipeline.process_batch(audio_list, sampling_rates)

    with TimingContext(f"Retrieval ({len(audio_embeddings)} queries)", logger):
        results_list = retrieval_pipeline.search_batch(
            audio_embeddings,
            k,
            query_texts=transcriptions,
        )

    _collect_retrieval_batch(results_list, all_retrieved)
    all_relevant.extend(relevance_batch)


def _run_audio_embedding_retrieval_batch(
    batch: List[Dict[str, Any]],
    audio_embedding_pipeline: AudioEmbeddingPipelineProtocol,
    retrieval_pipeline: RetrievalPipelineProtocol,
    all_retrieved: List[List[str]],
    all_relevant: List[Dict[str, float]],
    k: int,
) -> None:
    retrieval_mode = retrieval_pipeline.strategy_config.core.mode
    if retrieval_mode != "dense":
        raise ValueError(
            "audio_emb_retrieval supports only retrieval_mode='dense'. "
            "Sparse/hybrid requires query text unavailable in this path. "
            "Tip: switch vector_db.retrieval_mode='dense' or use asr_text_retrieval/audio_text_retrieval."
        )

    audio_list = [item["audio_array"] for item in batch]
    sampling_rates = [item["sampling_rate"] for item in batch]
    relevance_batch = [_build_relevant_from_item(item) for item in batch]

    with TimingContext(f"Audio embedding batch ({len(batch)} samples)", logger):
        embeddings = audio_embedding_pipeline.process_batch(audio_list, sampling_rates)

    with TimingContext(f"Retrieval ({len(embeddings)} queries)", logger):
        results_list = retrieval_pipeline.search_batch(embeddings, k)

    _collect_retrieval_batch(results_list, all_retrieved)
    all_relevant.extend(relevance_batch)


def _run_asr_text_retrieval_batch(
    batch: List[Dict[str, Any]],
    asr_pipeline: ASRPipelineProtocol,
    text_embedding_pipeline: TextEmbeddingPipelineProtocol,
    retrieval_pipeline: RetrievalPipelineProtocol,
    wer_scores: List[float],
    cer_scores: List[float],
    all_retrieved: List[List[str]],
    all_relevant: List[Dict[str, float]],
    k: int,
) -> None:
    audio_list = [item["audio_array"] for item in batch]
    sampling_rates = [item["sampling_rate"] for item in batch]
    transcriptions = [item["transcription"] for item in batch]
    relevance_batch = [_build_relevant_from_item(item) for item in batch]
    language = batch[0].get("language", None)

    with TimingContext(f"ASR batch({len(batch)} samples)", logger):
        hypotheses = asr_pipeline.process_batch(audio_list, sampling_rates, language)

    for gt_text, hyp_text in zip(transcriptions, hypotheses):
        wer_scores.append(word_error_rate(gt_text, hyp_text))
        cer_scores.append(character_error_rate(gt_text, hyp_text))

    with TimingContext(f"Text embedding batch ({len(hypotheses)} texts)", logger):
        embeddings = text_embedding_pipeline.process_batch(hypotheses)

    with TimingContext(f"Retrieval ({len(embeddings)} queries)", logger):
        results_list = retrieval_pipeline.search_batch(
            embeddings,
            k,
            query_texts=hypotheses,
        )

    _collect_retrieval_batch(results_list, all_retrieved)
    all_relevant.extend(relevance_batch)


def _run_asr_only_batch(
    batch: List[Dict[str, Any]],
    asr_pipeline: ASRPipelineProtocol,
    wer_scores: List[float],
    cer_scores: List[float],
) -> None:
    audio_list = [item["audio_array"] for item in batch]
    sampling_rates = [item["sampling_rate"] for item in batch]
    transcriptions = [item["transcription"] for item in batch]
    language = batch[0].get("language", None)

    with TimingContext(f"ASR batch ({len(batch)} samples)", logger):
        hypotheses = asr_pipeline.process_batch(audio_list, sampling_rates, language)

    for gt_text, hyp_text in zip(transcriptions, hypotheses):
        wer_scores.append(word_error_rate(gt_text, hyp_text))
        cer_scores.append(character_error_rate(gt_text, hyp_text))


def _save_checkpoint_if_due(
    cache_manager: Optional[CacheManager],
    experiment_id: Optional[str],
    sample_idx: int,
    checkpoint_interval: int,
    wer_scores: List[float],
    cer_scores: List[float],
    all_retrieved: List[List[str]],
    all_relevant: List[Dict[str, float]],
) -> None:
    if not (cache_manager and experiment_id):
        return
    if sample_idx % checkpoint_interval != 0:
        return
    checkpoint_data = {
        "last_idx": sample_idx,
        "wer_scores": wer_scores,
        "cer_scores": cer_scores,
        "all_retrieved": all_retrieved,
        "all_relevant": all_relevant,
    }
    cache_manager.set_checkpoint(experiment_id, checkpoint_data)
    logger.info(f"Checkpoint saved at sample {sample_idx}")


def _build_results(
    mode: str,
    retrieval_pipeline: Optional[RetrievalPipelineProtocol],
    asr_pipeline: Optional[ASRPipelineProtocol],
    text_embedding_pipeline: Optional[TextEmbeddingPipelineProtocol],
    audio_embedding_pipeline: Optional[AudioEmbeddingPipelineProtocol],
    wer_scores: List[float],
    cer_scores: List[float],
    all_retrieved: List[List[str]],
    all_relevant: List[Dict[str, float]],
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"pipeline_mode": mode}

    if mode == "audio_text_retrieval":
        assert audio_embedding_pipeline is not None
        assert text_embedding_pipeline is not None
        results["audio_embedder"] = audio_embedding_pipeline.model.name()
        results["text_embedder"] = text_embedding_pipeline.model.name()
        logger.info("Audio-to-text retrieval mode - no ASR metrics")
    elif mode == "audio_emb_retrieval":
        assert audio_embedding_pipeline is not None
        results["audio_embedder"] = audio_embedding_pipeline.model.name()
        logger.info("Audio embedding mode - no ASR metrics")
    elif mode in ["asr_text_retrieval", "asr_only"]:
        assert asr_pipeline is not None
        results["asr"] = asr_pipeline.model.name()
        if wer_scores:
            results["WER"] = sum(wer_scores) / len(wer_scores)
            results["CER"] = sum(cer_scores) / len(cer_scores)
            logger.info(f"ASR Metrics - WER: {results['WER']:.4f}, CER: {results['CER']:.4f}")

    if mode == "asr_text_retrieval":
        assert text_embedding_pipeline is not None
        results["embedder"] = text_embedding_pipeline.model.name()

    if mode != "asr_only":
        assert retrieval_pipeline is not None
        ir_metrics = compute_ir_metrics(all_retrieved, all_relevant, k_values=[1, 5, 10])
        results.update(ir_metrics)
        log_ir_metrics(results, logger, k_values=[1, 5, 10])

    return results


def evaluate_with_pipeline(
    dataset: QueryDataset,
    retrieval_pipeline: Optional[RetrievalPipelineProtocol] = None,
    asr_pipeline: Optional[ASRPipelineProtocol] = None,
    text_embedding_pipeline: Optional[TextEmbeddingPipelineProtocol] = None,
    audio_embedding_pipeline: Optional[AudioEmbeddingPipelineProtocol] = None,
    cache_manager: Optional[CacheManager] = None,
    k: int = 10,
    batch_size: int = 32,
    checkpoint_interval: int = 50,
    experiment_id: Optional[str] = None,
    resume_from_checkpoint: bool = True
) -> Dict[str, Any]:
    """
    Evaluate using independent pipelines.
    
    Supports three modes:
    1. ASR + Text Retrieval: Provide asr_pipeline, text_embedding_pipeline, retrieval_pipeline
       - Converts audio to text, then text to embeddings, then performs retrieval
       - Computes ASR metrics (WER, CER) and IR metrics
    
    2. Audio Embedding Retrieval: Provide audio_embedding_pipeline, retrieval_pipeline
       - Directly converts audio to embeddings, then performs retrieval
       - Computes IR metrics only (no ASR metrics)
    
    3. ASR Only: Provide asr_pipeline only
       - Converts audio to text only
       - Computes ASR metrics only (no IR metrics)
    
    Args:
        dataset: QueryDataset instance
        retrieval_pipeline: RetrievalPipeline instance for vector search
        asr_pipeline: Optional ASRPipeline instance
        text_embedding_pipeline: Optional TextEmbeddingPipeline instance
        audio_embedding_pipeline: Optional AudioEmbeddingPipeline instance
        cache_manager: Optional CacheManager for checkpointing
        k: Number of retrieval results
        batch_size: Batch size for processing
        checkpoint_interval: Save checkpoint every N samples
        experiment_id: Unique ID for this experiment (for checkpointing)
        resume_from_checkpoint: Whether to resume from existing checkpoint
        
    Returns:
        Dictionary of evaluation metrics
    """
    mode = _detect_mode(
        retrieval_pipeline=retrieval_pipeline,
        asr_pipeline=asr_pipeline,
        text_embedding_pipeline=text_embedding_pipeline,
        audio_embedding_pipeline=audio_embedding_pipeline,
    )
    
    logger.info(f"Dataset size: {len(dataset)}, Batch size: {batch_size}, k: {k}")
    
    start_idx, wer_scores, cer_scores, all_retrieved, all_relevant = _load_checkpoint_state(
        cache_manager=cache_manager,
        experiment_id=experiment_id,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    sample_idx = 0
    with TimingContext("Evaluation loop", logger):
        for batch in tqdm(data_loader, desc="Evaluating", initial=start_idx):
            if sample_idx < start_idx:
                sample_idx += len(batch)
                continue

            if mode == "audio_text_retrieval":
                assert audio_embedding_pipeline is not None
                assert retrieval_pipeline is not None
                _run_audio_text_retrieval_batch(
                    batch=batch,
                    audio_embedding_pipeline=audio_embedding_pipeline,
                    retrieval_pipeline=retrieval_pipeline,
                    all_retrieved=all_retrieved,
                    all_relevant=all_relevant,
                    k=k,
                )
            elif mode == "audio_emb_retrieval":
                assert audio_embedding_pipeline is not None
                assert retrieval_pipeline is not None
                _run_audio_embedding_retrieval_batch(
                    batch=batch,
                    audio_embedding_pipeline=audio_embedding_pipeline,
                    retrieval_pipeline=retrieval_pipeline,
                    all_retrieved=all_retrieved,
                    all_relevant=all_relevant,
                    k=k,
                )
            elif mode == "asr_text_retrieval":
                assert asr_pipeline is not None
                assert text_embedding_pipeline is not None
                assert retrieval_pipeline is not None
                _run_asr_text_retrieval_batch(
                    batch=batch,
                    asr_pipeline=asr_pipeline,
                    text_embedding_pipeline=text_embedding_pipeline,
                    retrieval_pipeline=retrieval_pipeline,
                    wer_scores=wer_scores,
                    cer_scores=cer_scores,
                    all_retrieved=all_retrieved,
                    all_relevant=all_relevant,
                    k=k,
                )
            elif mode == "asr_only":
                assert asr_pipeline is not None
                _run_asr_only_batch(
                    batch=batch,
                    asr_pipeline=asr_pipeline,
                    wer_scores=wer_scores,
                    cer_scores=cer_scores,
                )
            
            sample_idx += len(batch)
            _save_checkpoint_if_due(
                cache_manager=cache_manager,
                experiment_id=experiment_id,
                sample_idx=sample_idx,
                checkpoint_interval=checkpoint_interval,
                wer_scores=wer_scores,
                cer_scores=cer_scores,
                all_retrieved=all_retrieved,
                all_relevant=all_relevant,
            )
    
    logger.info("Computing final metrics...")
    results = _build_results(
        mode=mode,
        retrieval_pipeline=retrieval_pipeline,
        asr_pipeline=asr_pipeline,
        text_embedding_pipeline=text_embedding_pipeline,
        audio_embedding_pipeline=audio_embedding_pipeline,
        wer_scores=wer_scores,
        cer_scores=cer_scores,
        all_retrieved=all_retrieved,
        all_relevant=all_relevant,
    )
    
    if cache_manager:
        log_cache_stats(cache_manager, logger)
    
    return results


__all__ = ["evaluate_with_pipeline"]
