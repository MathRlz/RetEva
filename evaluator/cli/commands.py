"""Main evaluation command logic."""

import os
import json
import argparse
from typing import Any, List

from evaluator.config import EvaluationConfig
from evaluator.storage.cache import CacheManager
from evaluator.services.evaluation_service import _vector_db_cache_key
from evaluator.pipeline.audio.prepare import synthesize_missing_query_audio
from evaluator.logging_config import setup_logging, get_logger, log_cache_stats
from evaluator.pipeline import (
    create_pipeline_from_config,
    RetrievalPipeline,
    TextEmbeddingPipeline,
)
from evaluator.evaluation import evaluate_from_bundle
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
    config: EvaluationConfig,
) -> None:
    """Populate vector database with text embeddings.

    Args:
        retrieval_pipeline: Retrieval pipeline with vector store.
        text_emb_pipeline: Text embedding pipeline.
        corpus_entries: List of corpus entries (dicts or strings).
        cache_manager: Cache manager for vector DB caching.
        config: Evaluation config (for the vector-DB cache key).
    """
    logger = get_logger(__name__)

    corpus_texts = [
        entry["text"] if isinstance(entry, dict) else entry
        for entry in corpus_entries
    ]

    # Vector-DB cache key — shared with the webapi service path so caches match.
    db_key = _vector_db_cache_key(config, retrieval_pipeline, corpus_entries)
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


def synthesize_missing_audio(dataset: Any, synth_config: Any, logger: Any) -> None:
    """Synthesize query audio for a dataset whose questions lack ``audio_path``.

    Thin wrapper over the shared ``synthesize_missing_query_audio`` helper so the
    CLI and webapi service synthesize identically. No-op when the dataset has no
    ``questions`` attribute.
    """
    questions = getattr(dataset, "questions", None)
    if not questions:
        return
    synthesize_missing_query_audio(questions, synth_config, log=logger)


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

    # Generate output path
    output_filename = generate_output_filename(config)
    output_path = os.path.join(config.output_dir, output_filename)

    # Check if results already exist
    if os.path.exists(output_path):
        logger.warning(f"Results file {output_path} already exists. Skipping evaluation.")
        return

    dataset = _load_dataset_with_audio(config, logger)

    corpus_entries = _build_corpus_entries(dataset, cache_manager, config, logger)
    if not _populate_vector_db(config, bundle, corpus_entries, cache_manager, logger):
        return

    # Run evaluation
    model_desc = generate_model_description(config)
    config.experiment_name = f"{config.experiment_name}_{model_desc}"

    logger.info("Starting evaluation...")
    results = evaluate_from_bundle(
        dataset,
        bundle,
        config,
        cache_manager=cache_manager,
    )

    _save_results(results, output_path, config, logger)

    if cache_manager.enabled:
        log_cache_stats(cache_manager, logger)


def _load_dataset_with_audio(config, logger):
    """Load the dataset, running PubMed validation and audio synthesis as needed."""
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

    if config.audio_synthesis.enabled:
        synthesize_missing_audio(dataset, config.audio_synthesis, logger)
    return dataset


def _build_corpus_entries(dataset, cache_manager, config, logger):
    """Return corpus entries: explicit from the dataset, or unique transcriptions."""
    if hasattr(dataset, "get_corpus_entries"):
        logger.info("Using explicit corpus entries from dataset...")
        corpus_entries = dataset.get_corpus_entries()
        logger.info(f"Corpus entries: {len(corpus_entries)}")
        return corpus_entries

    logger.info("Getting unique transcriptions for vector DB...")
    cached_texts = cache_manager.get_unique_texts(config.data.dataset_name, len(dataset))
    if cached_texts is not None:
        logger.info(f"Loaded cached unique transcriptions: {len(cached_texts)}")
        return cached_texts

    logger.info("Extracting unique transcriptions from dataset...")
    cached_texts = list({item["transcription"] for item in dataset})
    logger.info(f"Unique transcriptions: {len(cached_texts)}")
    cache_manager.set_unique_texts(config.data.dataset_name, len(dataset), cached_texts)
    return cached_texts


# Per-mode log line for DB population (all three embed a text corpus on the CLI).
_DB_POPULATE_MESSAGES = {
    "asr_text_retrieval": "Populating vector database with text embeddings...",
    "audio_text_retrieval": (
        "Populating vector database with text embeddings for audio-text retrieval..."
    ),
    "audio_emb_retrieval": (
        "Populating vector database for audio embedding retrieval "
        "(CLI embeds a TEXT corpus — cross-modal query=audio vs corpus=text; "
        "audio-corpus embedding is webapi-only)..."
    ),
}


def _populate_vector_db(config, bundle, corpus_entries, cache_manager, logger) -> bool:
    """Populate the vector DB for retrieval modes. Returns False to abort the run."""
    mode = config.model.pipeline_mode
    if mode not in _DB_POPULATE_MESSAGES:
        return True

    logger.info(_DB_POPULATE_MESSAGES[mode])
    if bundle.text_embedding_pipeline is None:
        logger.error(f"{mode} requires text embedding model to populate DB!")
        logger.error("Please provide text_emb_model_type in config for DB population")
        return False

    populate_db(
        bundle.retrieval_pipeline, bundle.text_embedding_pipeline, corpus_entries,
        cache_manager, config,
    )
    return True


def _save_results(results, output_path, config, logger) -> None:
    """Write results JSON to ``output_path`` and log completion."""
    logger.info(f"Saving results to {output_path}...")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")


def main() -> None:
    """Main entry point for CLI."""
    args = parse_args()
    run_evaluation(args)
