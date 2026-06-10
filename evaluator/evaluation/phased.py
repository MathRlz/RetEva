"""Phased evaluation for efficient batch processing.

This module contains the evaluate_phased() function which processes the entire
dataset in sequential phases (ASR/Embedding, Text Embedding, Retrieval, Metrics)
for better GPU utilization and throughput.
"""
from typing import Optional, Dict, Any, List
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..datasets import QueryDataset
from ..pipeline import (
    ASRPipelineProtocol,
    TextEmbeddingPipelineProtocol,
    AudioEmbeddingPipelineProtocol,
    RetrievalPipelineProtocol,
    build_stage_graph,
)
from ..storage.cache import CacheManager
from ..logging_config import get_logger, TimingContext, log_cache_stats
from ..judge import run_llm_judging
from ..metrics import compute_ir_metrics, log_ir_metrics

from .helpers import _payload_to_key, _search_results_to_keys, _build_relevant_from_item, collate_fn
from .metrics import word_error_rate, character_error_rate

logger = get_logger(__name__)


def evaluate_phased(
    dataset: QueryDataset,
    retrieval_pipeline: Optional[RetrievalPipelineProtocol] = None,
    asr_pipeline: Optional[ASRPipelineProtocol] = None,
    text_embedding_pipeline: Optional[TextEmbeddingPipelineProtocol] = None,
    audio_embedding_pipeline: Optional[AudioEmbeddingPipelineProtocol] = None,
    cache_manager: Optional[CacheManager] = None,
    k: int = 10,
    batch_size: int = 32,
    trace_limit: int = 0,
    judge_config: Optional[Any] = None,
    answer_gen_config: Optional[Any] = None,
    query_opt_config: Optional[Any] = None,
    embedding_fusion_config: Optional[Any] = None,
    num_workers: int = 0,
    checkpoint_interval: int = 500,
    experiment_id: Optional[str] = None,
    resume_from_checkpoint: bool = True
) -> Dict[str, Any]:
    """Optimized phased evaluation - processes entire dataset in phases.
    
    Runs evaluation in four sequential phases for efficiency:
    
    - Phase 1: ASR - Transcribe all audio samples (or audio embedding)
    - Phase 2: Embedding - Embed all transcriptions  
    - Phase 3: Retrieval - Perform all vector searches
    - Phase 4: Metrics - Compute all evaluation metrics
    
    Args:
        dataset: QueryDataset instance containing audio samples and ground truth.
        retrieval_pipeline: RetrievalPipeline instance for vector search.
        asr_pipeline: Optional ASRPipeline instance for transcription.
        text_embedding_pipeline: Optional TextEmbeddingPipeline instance.
        audio_embedding_pipeline: Optional AudioEmbeddingPipeline instance.
        cache_manager: Optional CacheManager for checkpointing.
        k: Number of retrieval results to return. Default: 10.
        batch_size: Batch size for processing. Default: 32.
        trace_limit: Limit number of samples (0 = no limit). Default: 0.
        judge_config: Optional LLM judge configuration for scoring.
        embedding_fusion_config: Optional config for fusing audio and text embeddings.
        num_workers: Number of DataLoader workers. Default: 0.
        checkpoint_interval: Save checkpoint every N samples. Default: 500.
        experiment_id: Unique ID for this experiment (for checkpointing).
        resume_from_checkpoint: Whether to resume from existing checkpoint.
        
    Returns:
        Dictionary of evaluation metrics including:
        - pipeline_mode: Mode used (asr_text_retrieval, audio_emb_retrieval, etc.)
        - WER, CER: Word/Character error rates (ASR modes)
        - MRR: Mean Reciprocal Rank
        - MAP: Mean Average Precision
        - Recall@k: Recall at k for various k values
        - NDCG@k: Normalized Discounted Cumulative Gain at k
        - total_samples: Number of samples processed
        - duration_seconds: Total evaluation time
    
    Raises:
        ValueError: If neither audio_embedding_pipeline nor asr_pipeline provided.
    
    Examples:
        Basic usage with ASR and text retrieval::
        
            >>> from evaluator.evaluation import evaluate_phased
            >>> from evaluator.pipeline import create_pipeline_from_config
            >>> 
            >>> bundle = create_pipeline_from_config(config, cache_manager)
            >>> results = evaluate_phased(
            ...     dataset=dataset,
            ...     retrieval_pipeline=bundle.retrieval_pipeline,
            ...     asr_pipeline=bundle.asr_pipeline,
            ...     text_embedding_pipeline=bundle.text_embedding_pipeline,
            ...     k=5,
            ...     batch_size=32
            ... )
        
        Interpreting results::
        
            >>> print(f"Pipeline: {results['pipeline_mode']}")
            Pipeline: asr_text_retrieval
            >>> print(f"ASR Quality - WER: {results['WER']:.2%}")
            ASR Quality - WER: 12.34%
            >>> print(f"Retrieval - MRR: {results['MRR']:.4f}")
            Retrieval - MRR: 0.7523
            >>> print(f"Recall@5: {results['Recall@5']:.4f}")
            Recall@5: 0.8912
        
        With checkpointing for long evaluations::
        
            >>> results = evaluate_phased(
            ...     dataset=large_dataset,
            ...     retrieval_pipeline=retrieval,
            ...     asr_pipeline=asr,
            ...     text_embedding_pipeline=text_emb,
            ...     cache_manager=cache,
            ...     checkpoint_interval=100,
            ...     experiment_id="long_eval_run",
            ...     resume_from_checkpoint=True
            ... )
    """
    # Determine mode
    if audio_embedding_pipeline is not None and text_embedding_pipeline is not None:
        mode = "audio_text_retrieval"
        logger.info("Evaluation mode: Audio-to-Text Retrieval (Phased)")
    elif audio_embedding_pipeline is not None:
        mode = "audio_emb_retrieval"
        logger.info("Evaluation mode: Audio Embedding Retrieval (Phased)")
    elif asr_pipeline is not None and text_embedding_pipeline is not None:
        if retrieval_pipeline is not None:
            mode = "asr_text_retrieval"
            logger.info("Evaluation mode: ASR + Text Retrieval (Phased)")
        else:
            mode = "asr_only"
            logger.info("Evaluation mode: ASR Only (Phased)")
    elif asr_pipeline is not None:
        mode = "asr_only"
        logger.info("Evaluation mode: ASR Only (Phased)")
    else:
        raise ValueError("Must provide either audio_embedding_pipeline OR asr_pipeline")

    stage_graph = build_stage_graph(
        mode,
        embedding_fusion_enabled=bool(
            embedding_fusion_config is not None and embedding_fusion_config.enabled
        ),
    )
    stage_levels = [[node.id for node in level] for level in stage_graph.topological_levels()]
    logger.info("Execution DAG mode=%s levels=%s", mode, stage_levels)
    
    logger.info(f"Dataset size: {len(dataset)}, Batch size: {batch_size}, k: {k}")
    
    # Initialize storage
    all_hypotheses = []
    all_ground_truth = []
    all_embeddings = []
    all_relevance = []
    _asr_hypotheses_for_wer: list = []  # raw ASR output, before any query optimization
    all_query_ids = []
    all_results_with_scores = []
    
    # ========== PHASE 1: Audio Embedding or ASR ==========
    if mode == "audio_text_retrieval":
        logger.info("=" * 50)
        logger.info("PHASE 1: Audio Embedding")
        logger.info("=" * 50)
        
        # Process audio files in batches
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        with TimingContext("Audio Embedding Phase", logger):
            for batch in tqdm(dataloader, desc="Audio embedding"):
                audio_list = [item["audio_array"] for item in batch]
                sampling_rates = [item["sampling_rate"] for item in batch]
                transcriptions = [item["transcription"] for item in batch]
                relevance_batch = [_build_relevant_from_item(item) for item in batch]
                query_ids_batch = [
                    str(item.get("question_id", len(all_query_ids) + idx))
                    for idx, item in enumerate(batch)
                ]
                
                # Embed audio
                batch_embeddings = audio_embedding_pipeline.process_batch(audio_list, sampling_rates)
                all_embeddings.extend(batch_embeddings)
                all_ground_truth.extend(transcriptions)
                all_relevance.extend(relevance_batch)
                all_query_ids.extend(query_ids_batch)
        
        logger.info(f"Audio Embedding Phase complete: {len(all_embeddings)} audio embeddings")
        
        # Debug: Check embedding properties
        if len(all_embeddings) > 0:
            emb_array = np.array(all_embeddings)
            logger.info(f"Audio embedding shape: {emb_array.shape}")
            logger.info(f"Audio embedding stats - mean: {emb_array.mean():.4f}, std: {emb_array.std():.4f}")
            logger.info(f"Audio embedding norms - mean: {np.linalg.norm(emb_array, axis=1).mean():.4f}")
        
        # PHASE 1.5: Generate text embeddings and fuse if enabled
        if embedding_fusion_config and embedding_fusion_config.enabled:
            if text_embedding_pipeline is None:
                logger.warning(
                    "Embedding fusion is enabled but text_embedding_pipeline is not available. "
                    "Skipping fusion and using audio embeddings only."
                )
            else:
                logger.info("=" * 50)
                logger.info("PHASE 1.5: Text Embedding & Fusion")
                logger.info("=" * 50)
                
                from ..models.retrieval.embedding_fusion import (
                    fuse_embeddings,
                    validate_fusion_config,
                )
                
                # Generate text embeddings from ground truth transcriptions
                with TimingContext("Text Embedding for Fusion", logger):
                    text_embeddings = text_embedding_pipeline.process_batch(
                        all_ground_truth,
                        show_progress=True,
                        desc="Embedding reference texts (fusion)",
                    )
                    text_embeddings = np.array(text_embeddings)
                
                logger.info(f"Text embedding shape: {text_embeddings.shape}")
                logger.info(f"Text embedding stats - mean: {text_embeddings.mean():.4f}, std: {text_embeddings.std():.4f}")
                
                # Validate fusion configuration
                audio_emb_array = np.array(all_embeddings)
                validate_fusion_config(
                    embedding_fusion_config,
                    audio_dim=audio_emb_array.shape[1],
                    text_dim=text_embeddings.shape[1]
                )
                
                # Fuse embeddings
                with TimingContext("Embedding Fusion", logger):
                    fused_embeddings, _ = fuse_embeddings(
                        audio_emb_array,
                        text_embeddings,
                        embedding_fusion_config
                    )
                
                logger.info(f"Fused embedding shape: {fused_embeddings.shape}")
                logger.info(f"Fused embedding stats - mean: {fused_embeddings.mean():.4f}, std: {fused_embeddings.std():.4f}")
                logger.info(f"Fused embedding norms - mean: {np.linalg.norm(fused_embeddings, axis=1).mean():.4f}")
                
                # Replace audio embeddings with fused embeddings
                all_embeddings = fused_embeddings
                logger.info(f"Using fused embeddings for retrieval (method: {embedding_fusion_config.fusion_method})")
        
    elif mode == "audio_emb_retrieval":
        logger.info("=" * 50)
        logger.info("PHASE 1: Audio Embedding")
        logger.info("=" * 50)
        
        # Process audio files in batches
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        with TimingContext("Audio Embedding Phase", logger):
            for batch in tqdm(dataloader, desc="Audio embedding"):
                audio_list = [item["audio_array"] for item in batch]
                sampling_rates = [item["sampling_rate"] for item in batch]
                transcriptions = [item["transcription"] for item in batch]
                relevance_batch = [_build_relevant_from_item(item) for item in batch]
                query_ids_batch = [
                    str(item.get("question_id", len(all_query_ids) + idx))
                    for idx, item in enumerate(batch)
                ]
                
                # Embed audio
                batch_embeddings = audio_embedding_pipeline.process_batch(audio_list, sampling_rates)
                all_embeddings.extend(batch_embeddings)
                all_ground_truth.extend(transcriptions)
                all_relevance.extend(relevance_batch)
                all_query_ids.extend(query_ids_batch)
        
        logger.info(f"Audio Embedding Phase complete: {len(all_embeddings)} audio embeddings")
        
        # Debug: Check embedding properties
        if len(all_embeddings) > 0:
            emb_array = np.array(all_embeddings)
            logger.info(f"Audio embedding shape: {emb_array.shape}")
            logger.info(f"Audio embedding stats - mean: {emb_array.mean():.4f}, std: {emb_array.std():.4f}")
            logger.info(f"Audio embedding norms - mean: {np.linalg.norm(emb_array, axis=1).mean():.4f}")
        
    elif mode in ["asr_text_retrieval", "asr_only"]:
        logger.info("=" * 50)
        logger.info("PHASE 1: ASR Transcription")
        logger.info("=" * 50)
        
        all_hypotheses, all_ground_truth = asr_pipeline.process_dataset(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            language=None,
            checkpoint_interval=checkpoint_interval,
            experiment_id=experiment_id,
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        logger.info(f"ASR Phase complete: {len(all_hypotheses)} transcriptions")
        # Snapshot raw ASR output — WER/CER must compare against this, not
        # any query-optimized version that may expand the text significantly.
        _asr_hypotheses_for_wer = list(all_hypotheses)

        if mode == "asr_text_retrieval":
            # Keep relevance aligned with query order for IR metrics.
            all_relevance = [_build_relevant_from_item(dataset[i]) for i in range(len(dataset))]
            all_query_ids = [str(dataset[i].get("question_id", i)) for i in range(len(dataset))]
    
    # ========== PHASE 1.5: Query Optimization ==========
    _query_opt_bypassed = False
    if query_opt_config is not None and getattr(query_opt_config, "enabled", False):
        logger.info("=" * 50)
        logger.info("PHASE 1.5: Query Optimization (%s)", query_opt_config.method)
        logger.info("=" * 50)
        from ..models.retrieval.query.optimization import (
            rewrite_query, generate_hypothetical_document,
            decompose_query, generate_multi_queries, combine_retrieval_results,
        )
        method = query_opt_config.method
        query_texts_src = all_hypotheses if mode == "asr_text_retrieval" else all_ground_truth

        if method in ("rewrite", "hyde"):
            fn = rewrite_query if method == "rewrite" else generate_hypothetical_document
            optimized = []
            for q in tqdm(query_texts_src, desc=f"Query optimization ({method})"):
                try:
                    optimized.append(fn(q, query_opt_config))
                except Exception as exc:
                    logger.warning("Query optimization failed for %r: %s", q[:80], exc)
                    optimized.append(q)
            if mode == "asr_text_retrieval":
                all_hypotheses = optimized
            else:
                all_ground_truth = optimized
            logger.info("Query optimization complete: %d queries transformed", len(optimized))

        elif method in ("decompose", "multi_query"):
            if text_embedding_pipeline is None or retrieval_pipeline is None:
                logger.warning(
                    "Query optimization %s requires text_embedding + retrieval pipeline — skipping",
                    method,
                )
            else:
                fn = decompose_query if method == "decompose" else generate_multi_queries
                all_results_with_scores = []
                all_retrieved = []
                for q in tqdm(query_texts_src, desc=f"Query optimization ({method})"):
                    try:
                        sub_qs = fn(q, query_opt_config)
                    except Exception as exc:
                        logger.warning("Query expansion failed for %r: %s", q[:80], exc)
                        sub_qs = [q]
                    sub_embs = np.array(text_embedding_pipeline.process_batch(sub_qs))
                    sub_results = retrieval_pipeline.search_batch(sub_embs, k, query_texts=sub_qs)
                    merged = combine_retrieval_results(
                        sub_results,
                        strategy=query_opt_config.combine_strategy,
                        k=k,
                    )
                    all_results_with_scores.append(merged)
                    all_retrieved.append(_search_results_to_keys(merged))
                _query_opt_bypassed = True
                logger.info(
                    "Query optimization (%s) complete: %d queries expanded and retrieved",
                    method, len(all_results_with_scores),
                )

    # ========== PHASE 2: Text Embedding ==========
    if mode == "asr_text_retrieval" and not _query_opt_bypassed:
        logger.info("=" * 50)
        logger.info("PHASE 2: Text Embedding (from ASR)")
        logger.info("=" * 50)

        with TimingContext("Embedding Phase", logger):
            # Process all hypotheses in batches
            all_embeddings = text_embedding_pipeline.process_batch(
                all_hypotheses,
                show_progress=True,
                desc="Embedding query transcriptions (ASR)",
            )

        logger.info(f"Text Embedding Phase complete: {len(all_embeddings)} embeddings")

        # Save checkpoint after embedding
        if cache_manager and experiment_id:
            checkpoint_data = {
                'phase': 'embedding',
                'hypotheses': all_hypotheses,
                'ground_truth': all_ground_truth
            }
            cache_manager.set_checkpoint(f"{experiment_id}_phased", checkpoint_data)

    # Convert embeddings list to numpy array if needed
    if isinstance(all_embeddings, list) and len(all_embeddings) > 0:
        all_embeddings = np.array(all_embeddings)

    # ========== PHASE 3: Retrieval ==========
    can_debug_with_search = False
    if not _query_opt_bypassed:
        all_retrieved = []
    if mode in ["asr_text_retrieval", "audio_emb_retrieval", "audio_text_retrieval"] and not _query_opt_bypassed:
        logger.info("=" * 50)
        logger.info("PHASE 3: Retrieval")
        logger.info("=" * 50)
        
        with TimingContext("Retrieval Phase", logger):
            query_texts = None
            if mode == "asr_text_retrieval":
                query_texts = all_hypotheses
            elif mode == "audio_text_retrieval":
                query_texts = all_ground_truth
            elif mode == "audio_emb_retrieval":
                retrieval_mode = retrieval_pipeline.strategy_config.core.mode
                if retrieval_mode != "dense":
                    raise ValueError(
                        "audio_emb_retrieval supports only retrieval_mode='dense'. "
                        "Sparse/hybrid requires query text unavailable in this path."
                    )

            results_list = retrieval_pipeline.search_batch(all_embeddings, k, query_texts=query_texts)
            all_results_with_scores = results_list

            for results in results_list:
                all_retrieved.append(_search_results_to_keys(results))
        
        logger.info(f"Retrieval Phase complete: {len(all_retrieved)} queries")
        
        # Debug: Check vector store properties
        if hasattr(retrieval_pipeline.vector_store, 'vectors') and retrieval_pipeline.vector_store.vectors is not None:
            db_vectors = retrieval_pipeline.vector_store.vectors
            logger.info(f"DB vectors shape: {db_vectors.shape}")
            logger.info(f"DB vectors stats - mean: {db_vectors.mean():.4f}, std: {db_vectors.std():.4f}")
            logger.info(f"DB vectors norms - mean: {np.linalg.norm(db_vectors, axis=1).mean():.4f}")
            logger.info(f"DB payload count: {len(getattr(retrieval_pipeline.vector_store, 'payloads', []))}")
        
        # Debug: Check first few retrievals with scores
        if len(all_retrieved) > 0 and len(all_ground_truth) > 0:
            can_debug_with_search = (
                retrieval_pipeline.strategy_config.core.mode == "dense"
                and retrieval_pipeline.strategy_config.reranking.mode == "none"
            )
            if not can_debug_with_search:
                logger.info("Skipping dense-only debug re-search for current retrieval mode/reranker")
                
        if len(all_retrieved) > 0 and len(all_ground_truth) > 0 and can_debug_with_search:
            logger.info("=" * 50)
            logger.info("RETRIEVAL DEBUG SAMPLE (first 3 queries):")
            logger.info("=" * 50)

            # Build a doc_id → text lookup from vector store payloads for ground-truth display
            _db_payloads = getattr(retrieval_pipeline.vector_store, 'payloads', [])
            _doc_text_lookup: dict = {}
            for _p in _db_payloads:
                if isinstance(_p, dict):
                    _did = str(_p.get("doc_id", ""))
                    _dtxt = _p.get("text") or _p.get("abstract") or _p.get("title") or ""
                    if _did:
                        _doc_text_lookup[_did] = str(_dtxt)

            emb_array = np.array(all_embeddings) if isinstance(all_embeddings, list) else all_embeddings
            for i in range(min(3, len(all_embeddings))):
                results_with_scores = retrieval_pipeline.search(emb_array[i], k=10)

                # gt_doc_id: the expected document id
                if i < len(all_relevance) and len(all_relevance[i]) > 0:
                    gt_doc_id = next(iter(all_relevance[i].keys()))
                else:
                    gt_doc_id = str(all_ground_truth[i])

                # query_text: what was actually searched (ASR output or reference)
                if mode == "asr_text_retrieval" and i < len(all_hypotheses):
                    query_text = all_hypotheses[i]
                elif i < len(all_ground_truth):
                    query_text = str(all_ground_truth[i])
                else:
                    query_text = all_query_ids[i] if i < len(all_query_ids) else "?"

                # gt_doc_text: the text of the expected document (from corpus)
                gt_doc_text = _doc_text_lookup.get(gt_doc_id, "")

                logger.info(f"\nQuery {i+1}:")
                logger.info(f"  ASR query text:        '{query_text[:120]}'")
                if mode == "asr_text_retrieval" and i < len(all_ground_truth):
                    ref = str(all_ground_truth[i])
                    if ref != query_text:
                        logger.info(f"  Ground truth question: '{ref[:120]}'")
                logger.info(f"  Expected doc:  [{gt_doc_id}] '{gt_doc_text[:100]}'")
                logger.info(f"  Query emb norm: {np.linalg.norm(emb_array[i]):.4f}")
                logger.info(f"  Top 5 retrieved:")

                for j, (payload, score) in enumerate(results_with_scores[:5], 1):
                    doc_id = payload.get("doc_id", "") if isinstance(payload, dict) else str(payload)
                    match = "✓" if doc_id == gt_doc_id else "✗"
                    preview = (payload.get("text", str(payload)) if isinstance(payload, dict) else str(payload))[:80]
                    logger.info(f"    {j}. [{match}] Score: {score:.4f} | [{doc_id}] '{preview}'")

                gt_rank = None
                for rank, (payload, score) in enumerate(results_with_scores, 1):
                    doc_id = payload.get("doc_id", "") if isinstance(payload, dict) else str(payload)
                    if doc_id == gt_doc_id:
                        gt_rank = rank
                        logger.info(f"  Expected doc found at rank: {gt_rank} (score: {score:.4f})")
                        break

                if gt_rank is None:
                    logger.info(f"  Expected doc NOT found in top-{k}")
                    payload_keys = [_payload_to_key(p) for p in _db_payloads]
                    if gt_doc_id in payload_keys:
                        logger.info(f"  (Doc exists in DB but outside top-{k})")
                    else:
                        logger.info(f"  (Doc NOT in DB at all!)")
    
    # ========== PHASE 4: IR Metrics ==========
    logger.info("=" * 50)
    logger.info("PHASE 4: IR Metrics")
    logger.info("=" * 50)
    
    results = {}
    results["pipeline_mode"] = mode
    results["phased"] = True
    
    # ASR Metrics
    if mode in ["asr_text_retrieval", "asr_only"]:
        results["asr"] = asr_pipeline.model.name()
        
        wer_scores = []
        cer_scores = []
        for gt_text, hyp_text in zip(all_ground_truth, _asr_hypotheses_for_wer):
            wer_scores.append(word_error_rate(gt_text, hyp_text))
            cer_scores.append(character_error_rate(gt_text, hyp_text))
        
        results["WER"] = sum(wer_scores) / len(wer_scores)
        results["CER"] = sum(cer_scores) / len(cer_scores)
        logger.info(f"ASR Metrics - WER: {results['WER']:.4f}, CER: {results['CER']:.4f}")
    
    # Embedding model info
    if mode == "asr_text_retrieval":
        results["embedder"] = text_embedding_pipeline.model.name()
    elif mode == "audio_text_retrieval":
        results["audio_embedder"] = audio_embedding_pipeline.model.name()
        results["text_embedder"] = text_embedding_pipeline.model.name()
    
    # IR Metrics
    if mode != "asr_only":
        if all_relevance:
            all_relevant = all_relevance
        else:
            all_relevant = [{str(gt): 1} for gt in all_ground_truth]
        
        ir_metrics = compute_ir_metrics(all_retrieved, all_relevant, k_values=[1, 5, 10])
        results.update(ir_metrics)
        log_ir_metrics(results, logger, k_values=[1, 5, 10])

        # ========== PHASE 5: Answer Generation ==========
        _ans_detail_by_qid: dict = {}
        if answer_gen_config is not None and getattr(answer_gen_config, "enabled", False):
            logger.info("=" * 50)
            logger.info("PHASE 5: Answer Generation")
            logger.info("=" * 50)
            from ..evaluation.answer_gen import generate_answers
            query_texts = (
                all_hypotheses if mode == "asr_text_retrieval" else all_ground_truth
            )
            # Build corpus_lookup from the full corpus so reference answers
            # can be found even for retrieval misses (not just retrieved docs).
            corpus_lookup: dict = {}
            if hasattr(dataset, "get_corpus"):
                for doc in dataset.get_corpus():
                    corpus_lookup[str(doc.get("doc_id", ""))] = doc
            # Overlay with retrieved payloads (may carry richer runtime metadata)
            for idx, result in enumerate(all_results_with_scores):
                for payload, _ in result:
                    corpus_lookup[str(payload.get("doc_id", payload.get("id", idx)))] = payload
            answer_results = generate_answers(
                traces_data=(all_query_ids, all_relevant, all_results_with_scores),
                all_query_texts=query_texts,
                corpus_lookup=corpus_lookup,
                config=answer_gen_config,
            )
            results["answer_generation"] = answer_results
            _ans_detail_by_qid = {
                d["query_id"]: d for d in answer_results["details"]
            }
            logger.info(
                "Answer generation complete — %d cases, mean ROUGE-L: %s",
                answer_results["cases"],
                f"{answer_results['mean_rougeL']:.4f}" if answer_results["mean_rougeL"] is not None else "n/a",
            )

        if trace_limit > 0 and all_results_with_scores:
            limit = min(trace_limit, len(all_results_with_scores))
            traces = []
            for i in range(limit):
                scored = all_results_with_scores[i]
                retrieved = [
                    {
                        "doc_key": _payload_to_key(payload),
                        "score": float(score),
                    }
                    for payload, score in scored
                ]
                query_id = all_query_ids[i] if i < len(all_query_ids) else str(i)
                ans_detail = _ans_detail_by_qid.get(query_id, {})
                traces.append(
                    {
                        "query_id": query_id,
                        "relevant": all_relevant[i] if i < len(all_relevant) else {},
                        "retrieved": retrieved,
                        "question": (all_ground_truth[i] if i < len(all_ground_truth) else ans_detail.get("question", "")),
                        "generated_answer": ans_detail.get("generated_answer", ""),
                        "reference_answer": ans_detail.get("reference_answer", ""),
                    }
                )
            results["query_traces"] = traces

        if judge_config is not None and getattr(judge_config, "enabled", False):
            if "query_traces" not in results:
                raise RuntimeError("LLM judge requires query traces; set trace_limit > 0")
            logger.info("=" * 50)
            logger.info("PHASE 6: Judge")
            logger.info("=" * 50)
            judge_mode = getattr(judge_config, "judge_mode", "retrieval")
            logger.info(
                "Running LLM judge — mode=%s model=%s cases=%d",
                judge_mode, judge_config.model, judge_config.max_cases,
            )
            judge_results = run_llm_judging(
                results["query_traces"],
                judge_config,
                judge_mode=judge_mode,
            )
            results["llm_judge"] = judge_results
            logger.info(
                "Judge complete — cases=%d mean_score=%.4f pass_rate=%.1f%%",
                judge_results["cases"],
                judge_results["mean_score"],
                100.0 * sum(
                    1 for r in judge_results["details"]
                    if r.get("judge", {}).get("verdict") == "PASS"
                ) / max(len(judge_results["details"]), 1),
            )
    
    if cache_manager and experiment_id:
        try:
            import os
            checkpoint_path = cache_manager._get_cache_path("checkpoints", f"{experiment_id}_phased", ".json")
            if checkpoint_path.exists():
                os.remove(checkpoint_path)
        except OSError as exc:
            logger.warning("Failed to clean up checkpoint file: %s", exc)

    if cache_manager:
        log_cache_stats(cache_manager, logger)
    
    return results


def evaluate_from_bundle(
    dataset: QueryDataset,
    bundle: "PipelineBundle",
    config: "EvaluationConfig",
    *,
    cache_manager: Optional[CacheManager] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: extract params from bundle + config.

    Reduces a 16-param call to 4 args.
    """
    return evaluate_phased(
        dataset=dataset,
        retrieval_pipeline=bundle.retrieval_pipeline,
        asr_pipeline=bundle.asr_pipeline,
        text_embedding_pipeline=bundle.text_embedding_pipeline,
        audio_embedding_pipeline=bundle.audio_embedding_pipeline,
        cache_manager=cache_manager,
        k=config.vector_db.k,
        batch_size=config.data.batch_size,
        trace_limit=config.data.trace_limit,
        judge_config=config.judge if config.judge.enabled else None,
        answer_gen_config=config.answer_generation if config.answer_generation.enabled else None,
        query_opt_config=config.query_optimization if config.query_optimization.enabled else None,
        num_workers=config.data.num_workers,
        checkpoint_interval=config.checkpoint_interval if config.checkpoint_enabled else 0,
        experiment_id=config.experiment_name,
        resume_from_checkpoint=getattr(config, 'resume_from_checkpoint', True),
    )


__all__ = ["evaluate_phased", "evaluate_from_bundle"]
