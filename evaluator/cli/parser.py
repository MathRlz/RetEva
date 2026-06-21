"""Command-line argument parsing for the evaluation script.

The CLI carries only *operational* concerns — where to save, verbosity, devices, cache policy, and
machine-specific execution knobs. The experiment itself (models, dataset, retrieval, judge, …) is
described entirely by the YAML ``--config``; it is never overridden from the command line, so a
given config always describes the same experiment.
"""

import argparse
from typing import List, Optional

from ..config.types import SERVICE_STARTUP_MODES, SERVICE_OFFLOAD_POLICIES


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments (from *argv*, default ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a speech-retrieval evaluation. The experiment is defined by --config; "
            "CLI flags are operational only (output, verbosity, devices, cache, execution)."
        ),
    )

    # Config file + utilities
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file (the experiment definition)")
    parser.add_argument("--list_models", action="store_true",
                        help="List available models by family and exit")
    parser.add_argument("--print_graph", action="store_true",
                        help="Print the execution DAG for the config and exit (no models loaded)")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to write results into")

    # Devices (GPU pool)
    parser.add_argument("--devices", type=str, default=None,
                        help="Comma-separated devices for the GPU pool (e.g., 'cuda:0,cuda:1')")
    parser.add_argument("--allocation_strategy", type=str, default=None,
                        choices=["memory_aware", "round_robin", "packing"],
                        help="GPU allocation strategy for distributing models")

    # Execution knobs — machine-specific; do not change what is measured
    parser.add_argument("--service_startup_mode", type=str, default=None,
                        choices=list(SERVICE_STARTUP_MODES),
                        help="Service startup strategy for model runtime")
    parser.add_argument("--service_offload_policy", type=str, default=None,
                        choices=list(SERVICE_OFFLOAD_POLICIES),
                        help="Service offload policy: free after last use | keep resident | "
                             "park warm on CPU")
    parser.add_argument("--streaming_window_size", type=int, default=None,
                        help="Run the query side in windows of this size (corpus-scale streaming)")
    parser.add_argument("--cpu_stage_executor", type=str, default=None,
                        choices=["sync", "thread", "process"],
                        help="Parallelism for CPU-bound per-item stages (default sync)")
    parser.add_argument("--cpu_stage_workers", type=int, default=None,
                        help="Worker count for --cpu_stage_executor (0 = auto)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override the dataset batch size for this run")

    # Cache policy
    parser.add_argument("--no_cache", action="store_true", help="Disable caching")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--clear_cache", action="store_true",
                        help="Clear all caches before running")

    # Checkpointing — run mechanics (resumability)
    parser.add_argument("--no_checkpoint", action="store_true", help="Disable checkpointing")
    parser.add_argument("--checkpoint_interval", type=int, default=None,
                        help="Checkpoint frequency")

    # Dataset-validation escape hatch
    parser.add_argument("--skip_dataset_validation", action="store_true",
                        help="Skip dataset integrity checks (not recommended for benchmark runs)")

    # Logging / verbosity
    parser.add_argument("--log_level", type=str, default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--verbosity", choices=["default", "verbose", "debug"], default=None,
        help="Console verbosity profile (default: quiet; env EVALUATOR_VERBOSITY)",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="-v = verbose, -vv = debug (overrides --verbosity)",
    )

    return parser.parse_args(argv)


# Mapping from CLI argument name to config nested path (tuple of attr names). Operational flags
# only — the experiment is described by the config and is never overridden from the CLI.
_CLI_ARG_MAP = {
    # Service runtime (service_runtime.*)
    "service_startup_mode": ("service_runtime", "startup_mode"),
    "service_offload_policy": ("service_runtime", "offload_policy"),
    # Streaming + CPU-stage parallelism
    "streaming_window_size": ("streaming", "window_size"),
    "cpu_stage_executor": ("cpu_stage_executor",),
    "cpu_stage_workers": ("cpu_stage_workers",),
    # Dataset batch size (data.*)
    "batch_size": ("data", "batch_size"),
    # Cache (cache.*)
    "cache_dir": ("cache", "cache_dir"),
    # Logging (logging.*)
    "log_level": ("logging", "console_level"),
    # Output (top-level config)
    "output_dir": ("output_dir",),
    # Checkpoint (top-level config)
    "checkpoint_interval": ("checkpoint_interval",),
}


def _set_nested_attr(obj, path: tuple, value) -> None:
    """Set a nested attribute via tuple path (e.g., ('cache', 'cache_dir')).

    Raises a clear AttributeError naming the full path if any segment is missing,
    rather than failing opaquely deep in the traversal.
    """
    *prefix, final = path
    for attr in prefix:
        if not hasattr(obj, attr):
            raise AttributeError(
                f"_CLI_ARG_MAP path {path!r} is invalid: '{attr}' not found on "
                f"{type(obj).__name__}."
            )
        obj = getattr(obj, attr)
    if not hasattr(obj, final):
        raise AttributeError(
            f"_CLI_ARG_MAP path {path!r} is invalid: '{final}' not found on "
            f"{type(obj).__name__}."
        )
    setattr(obj, final, value)


def apply_args_to_config(args: argparse.Namespace, config) -> None:
    """Apply the operational command-line arguments to the configuration object in place.

    Only operational concerns are applied — the experiment (models, dataset, retrieval, judge) is
    defined entirely by the config and is never modified from the CLI.
    """
    import os

    from ..config import DevicePoolConfig

    # Console verbosity: -vv > -v > --verbosity > EVALUATOR_VERBOSITY env > config.
    verbose_count = getattr(args, "verbose", 0) or 0
    verbosity = (
        "debug" if verbose_count >= 2
        else "verbose" if verbose_count == 1
        else getattr(args, "verbosity", None) or os.environ.get("EVALUATOR_VERBOSITY")
    )
    if verbosity:
        config.logging.verbosity = verbosity

    # Apply all mapped (operational) CLI arguments
    for arg_name, config_path in _CLI_ARG_MAP.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            _set_nested_attr(config, config_path, value)

    # Device pool configuration
    if args.devices is not None or args.allocation_strategy is not None:
        if config.device_pool is None:
            config.device_pool = DevicePoolConfig()

        if args.devices is not None:
            config.device_pool.available_devices = [
                d.strip() for d in args.devices.split(",")
            ]
        if args.allocation_strategy is not None:
            config.device_pool.allocation_strategy = args.allocation_strategy

    # Dataset validation flag (inverted logic)
    if args.skip_dataset_validation:
        config.data.strict_validation = False

    # Cache disabled flag (inverted logic)
    if args.no_cache:
        config.cache.enabled = False

    # Checkpoint disabled flag (inverted logic)
    if args.no_checkpoint:
        config.checkpoint_enabled = False
