"""CLI evaluation command: parse args -> config -> shared execution core -> save."""

import os
import json
import argparse

from evaluator.config import EvaluationConfig
from evaluator.storage.cache import CacheManager
from evaluator.logging_config import setup_logging, log_cache_stats
from evaluator.evaluation.runner import run_evaluation_from_config

from .parser import parse_args, apply_args_to_config
from .utils import generate_output_filename, generate_model_description

__all__ = ["run_evaluation", "main"]


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

    if args.print_graph:
        _print_graph(config)
        return

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
    # Surface the interpreter + native threading stack (diagnoses env-specific native
    # crashes, e.g. a webapi subprocess running a different env than the CLI).
    from evaluator.diagnostics import log_runtime_summary
    log_runtime_summary(logger)
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
    
    # Output path (skip if results already exist) — computed before the experiment-name
    # is decorated with the model description, so the filename stays stable.
    output_filename = generate_output_filename(config)
    output_path = os.path.join(config.output_dir, output_filename)
    if os.path.exists(output_path):
        logger.warning(f"Results file {output_path} already exists. Skipping evaluation.")
        return

    config.experiment_name = f"{config.experiment_name}_{generate_model_description(config)}"

    # Single shared execution core (same path the webapi/API uses).
    results = run_evaluation_from_config(config, cache_manager=cache_manager)

    _save_results(results, output_path, config, logger)

    if cache_manager.enabled:
        log_cache_stats(cache_manager, logger)


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


def _print_graph(config: EvaluationConfig) -> None:
    """Print the execution DAG for *config* without loading any models."""
    from evaluator.pipeline import build_stage_graph

    graph = build_stage_graph(
        str(config.model.pipeline_mode),
        embedding_fusion_enabled=bool(config.embedding_fusion.enabled),
    )
    print(f"Execution DAG (mode={graph.mode}):")
    if config.audio_synthesis.enabled:
        print("  [pre-graph] synthesis: TTS query audio (in prepare_dataset)")
    for i, level in enumerate(graph.topological_levels()):
        print(f"  Level {i}: {', '.join(node.stage for node in level)}")
    print("  edges:")
    for node in graph.nodes:
        if node.depends_on:
            print(f"    {node.stage} <- {', '.join(node.depends_on)}")


def main() -> None:
    """Main entry point for CLI."""
    args = parse_args()
    run_evaluation(args)
