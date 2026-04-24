"""Main evaluation command logic."""

import os
import json
import argparse
from typing import Any, List

from evaluator.config import EvaluationConfig
from evaluator.storage.cache import CacheManager
from evaluator.logging_config import setup_logging, get_logger, log_cache_stats
from evaluator.pipeline import (
    create_pipeline_from_config,
    RetrievalPipeline,
    TextEmbeddingPipeline,
)
from evaluator.evaluation import evaluate_phased
from evaluator.evaluation.validation import validate_pubmed_dataset
from evaluator.datasets import (
    AdmedQueryDataset,
    QueryDataset,
    load_admed_voice_corpus,
    load_pubmed_qa_dataset,
)

from .parser import parse_args, apply_args_to_config
from .utils import generate_output_filename, generate_model_description


def load_dataset(config: EvaluationConfig) -> QueryDataset:
    """Load dataset by name.
    
    Args:
        config: Evaluation configuration.
        
    Returns:
        Loaded dataset implementing QueryDataset interface.
        
    Raises:
        ValueError: If dataset name is unsupported or required paths missing.
    """
    dataset_name = config.data.dataset_name

    if dataset_name == "admed_voice":
        train_set, _ = load_admed_voice_corpus(test_size=config.data.test_size)
        return AdmedQueryDataset(train_set)
    if dataset_name == "pubmed_qa":
        if not config.data.questions_path or not config.data.corpus_path:
            raise ValueError(
                "pubmed_qa requires data.questions_path and data.corpus_path in config"
            )
        return load_pubmed_qa_dataset(
            questions_path=config.data.questions_path,
            corpus_path=config.data.corpus_path,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def populate_db(
    retrieval_pipeline: RetrievalPipeline,
    text_emb_pipeline: TextEmbeddingPipeline,
    corpus_entries: List[Any],
    cache_manager: CacheManager,
    model_name: str
) -> None:
    """Populate vector database with text embeddings.
    
    Args:
        retrieval_pipeline: Retrieval pipeline with vector store.
        text_emb_pipeline: Text embedding pipeline.
        corpus_entries: List of corpus entries (dicts or strings).
        cache_manager: Cache manager for vector DB caching.
        model_name: Model name for cache key generation.
    """
    logger = get_logger(__name__)

    corpus_texts = [
        entry["text"] if isinstance(entry, dict) else entry 
        for entry in corpus_entries
    ]
    
    # Check if we have cached vector DB
    db_key = cache_manager._compute_hash(model_name, tuple(sorted(corpus_texts)))
    logger.info(f"Vector DB cache key: {db_key}")
    cached_db = cache_manager.get_vector_db(db_key)
    
    if cached_db is not None:
        vectors, cached_texts = cached_db
        logger.info(f"Loaded cached vector DB with {len(vectors)} vectors")
        retrieval_pipeline.build_index(vectors, cached_texts)
        return
    
    logger.info(f"No cached vector DB found. Encoding {len(corpus_texts)} texts...")
    
    # Encode texts using pipeline (which has its own caching)
    vectors = text_emb_pipeline.process_batch(corpus_texts, show_progress=True)
    
    logger.info("Building vector database...")
    retrieval_pipeline.build_index(vectors, corpus_entries)
    
    # Cache the vector DB
    cache_manager.set_vector_db(db_key, vectors, corpus_entries)
    logger.info("Vector database cached")


def run_evaluation(args: argparse.Namespace) -> None:
    """Run the evaluation with given arguments.
    
    Args:
        args: Parsed command-line arguments.
    """
    if args.list_models:
        from evaluator.services import ModelServiceProvider

        provider = ModelServiceProvider()
        print(json.dumps(provider.list_available_models(), indent=2))
        provider.shutdown()
        return

    # Load configuration
    if args.config:
        config = EvaluationConfig.from_yaml(args.config)
    else:
        config = EvaluationConfig()
    
    # Override config with command-line arguments
    apply_args_to_config(args, config)
    
    # Setup logging
    logger = setup_logging(
        experiment_name=config.experiment_name,
        log_dir=config.logging.log_dir,
        console_level=config.logging.get_console_level(),
        file_level=config.logging.get_file_level()
    )
    
    logger.info("=" * 60)
    logger.info("Starting Evaluation")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Initialize cache manager
    cache_manager = CacheManager(
        cache_dir=config.cache.cache_dir,
        enabled=config.cache.enabled
    )
    
    if args.clear_cache:
        logger.info("Clearing all caches...")
        cache_manager.clear_all()
    
    if cache_manager.enabled:
        log_cache_stats(cache_manager, logger)
    
    # Create pipelines
    logger.info("Creating pipelines...")
    bundle = create_pipeline_from_config(config, cache_manager)
    asr_pipeline = bundle.asr_pipeline
    text_emb_pipeline = bundle.text_embedding_pipeline
    audio_emb_pipeline = bundle.audio_embedding_pipeline
    retrieval_pipeline = bundle.retrieval_pipeline
    
    # Generate output path
    output_filename = generate_output_filename(config)
    output_path = os.path.join(config.output_dir, output_filename)
    
    # Check if results already exist
    if os.path.exists(output_path):
        logger.warning(f"Results file {output_path} already exists. Skipping evaluation.")
        return
    
    # Load dataset
    logger.info(f"Loading dataset: {config.data.dataset_name}...")

    if config.data.dataset_name == "pubmed_qa" and config.data.strict_validation:
        logger.info("Running PubMed dataset validation...")
        q_count, c_count = validate_pubmed_dataset(
            questions_path=config.data.questions_path,
            corpus_path=config.data.corpus_path,
            verify_audio=True,
        )
        logger.info(
            f"PubMed dataset validation passed (questions={q_count}, corpus={c_count})"
        )

    dataset = load_dataset(config)
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Build corpus entries for vector DB
    if hasattr(dataset, "get_corpus_entries"):
        logger.info("Using explicit corpus entries from dataset...")
        corpus_entries = dataset.get_corpus_entries()
        logger.info(f"Corpus entries: {len(corpus_entries)}")
    else:
        logger.info("Getting unique transcriptions for vector DB...")
        cached_texts = cache_manager.get_unique_texts(
            config.data.dataset_name, len(dataset)
        )
        if cached_texts is not None:
            logger.info(f"Loaded cached unique transcriptions: {len(cached_texts)}")
            corpus_entries = cached_texts
        else:
            logger.info("Extracting unique transcriptions from dataset...")
            cached_texts = list({item["transcription"] for item in dataset})
            logger.info(f"Unique transcriptions: {len(cached_texts)}")
            cache_manager.set_unique_texts(
                config.data.dataset_name, len(dataset), cached_texts
            )
            corpus_entries = cached_texts
    
    # Populate vector database if retrieval is needed
    if config.model.pipeline_mode in [
        "asr_text_retrieval", "audio_emb_retrieval", "audio_text_retrieval"
    ]:
        if config.model.pipeline_mode == "asr_text_retrieval":
            logger.info("Populating vector database with text embeddings...")
            model_name = text_emb_pipeline.model.name()
            populate_db(
                retrieval_pipeline, text_emb_pipeline, corpus_entries, 
                cache_manager, model_name
            )
        elif config.model.pipeline_mode == "audio_text_retrieval":
            logger.info(
                "Populating vector database with text embeddings for audio-text retrieval..."
            )
            if text_emb_pipeline is None:
                logger.error(
                    "Audio-text retrieval requires text embedding model to populate DB!"
                )
                logger.error(
                    "Please provide text_emb_model_type in config for DB population"
                )
                return
            model_name = text_emb_pipeline.model.name()
            populate_db(
                retrieval_pipeline, text_emb_pipeline, corpus_entries, 
                cache_manager, model_name
            )
        else:  # audio_emb_retrieval
            logger.info("Populating vector database for audio embedding retrieval...")
            if text_emb_pipeline is None:
                logger.error(
                    "Audio embedding retrieval requires text embedding model to populate DB!"
                )
                logger.error(
                    "Please provide text_emb_model_type in config for DB population"
                )
                return
            model_name = text_emb_pipeline.model.name()
            populate_db(
                retrieval_pipeline, text_emb_pipeline, corpus_entries, 
                cache_manager, model_name
            )
    
    # Run evaluation
    model_desc = generate_model_description(config)
    experiment_id = f"{config.experiment_name}_{model_desc}"
    
    logger.info("Starting evaluation...")
    results = evaluate_phased(
        dataset=dataset,
        asr_pipeline=asr_pipeline,
        text_embedding_pipeline=text_emb_pipeline,
        audio_embedding_pipeline=audio_emb_pipeline,
        retrieval_pipeline=retrieval_pipeline,
        cache_manager=cache_manager if config.checkpoint_enabled else None,
        k=config.vector_db.k,
        batch_size=config.data.batch_size,
        trace_limit=config.data.trace_limit,
        judge_config=config.judge,
        checkpoint_interval=config.checkpoint_interval,
        experiment_id=experiment_id,
        resume_from_checkpoint=config.resume_from_checkpoint
    )
    
    # Save results
    logger.info(f"Saving results to {output_path}...")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")
    
    if cache_manager.enabled:
        log_cache_stats(cache_manager, logger)


def main() -> None:
    """Main entry point for CLI."""
    args = parse_args()
    run_evaluation(args)
