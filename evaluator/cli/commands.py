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

    # Load configuration. --print_graph is a structural preview, so skip validation
    # (it must work without the dataset/corpus files present).
    if args.config:
        config = EvaluationConfig.from_yaml(args.config, validate=not args.print_graph)
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
        file_level=config.logging.get_file_level(),
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
        cache_dir=config.cache.cache_dir, enabled=config.cache.enabled
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
        logger.warning(
            f"Results file {output_path} already exists. Skipping evaluation."
        )
        return

    config.experiment_name = (
        f"{config.experiment_name}_{generate_model_description(config)}"
    )

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
    from evaluator.pipeline import build_graph_for_config, get_stage_node_def
    from evaluator.pipeline.graph.display import display_label
    from evaluator.pipeline.stage_graph import (
        _effective_inputs,
        _effective_outputs,
        dataset_columns,
    )

    graph = build_graph_for_config(config)
    print(f"Execution DAG (mode={graph.mode}):")
    for i, level in enumerate(graph.topological_levels()):
        print(f"  Level {i}: {', '.join(node.id for node in level)}")
    print("  nodes (inputs -> outputs) [deps]:")
    for node in graph.nodes:
        d = get_stage_node_def(node.stage)
        ins = ", ".join(_effective_inputs(node.stage, node.params)) or "-"
        outs = ", ".join(_effective_outputs(node.stage, node.params)) or "-"
        deps = f"  [after {', '.join(node.depends_on)}]" if node.depends_on else ""
        shown_params = {k: v for k, v in (node.params or {}).items() if k != "fields"}
        params = f"  params={shown_params}" if shown_params else ""
        # technical id prefix (stable/parseable) + the friendly operator label (display.py)
        friendly = display_label(node.stage, node.params)
        suffix = f"  «{friendly}»" if friendly != node.id else ""
        print(f"    {node.id}: ({ins}) -> ({outs}){deps}{params}{suffix}")
        columns = dataset_columns(node.params)
        if columns:
            cols = ", ".join(f"{c['name']}:{c['type']}" for c in columns)
            print(f"      columns: {cols}")


def main() -> None:
    """Main entry point for CLI."""
    args = parse_args()
    run_evaluation(args)
