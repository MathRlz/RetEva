"""Command-line argument parsing for evaluation script."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ASR and Text Embedding Models")
    
    # Config file
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML configuration file")
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available models by family and exit",
    )
    parser.add_argument(
        "--print_graph",
        action="store_true",
        help="Print the execution DAG for the config and exit (no models loaded)",
    )

    # Model arguments (override config)
    parser.add_argument("--asr_model_type", type=str, default=None,
                       choices=["whisper", "wav2vec2"])
    parser.add_argument("--asr_model_name", type=str, default=None)
    parser.add_argument("--asr_adapter_path", type=str, default=None,
                       help="Path to PEFT/LoRA adapter weights for ASR model")
    parser.add_argument("--text_emb_model_type", type=str, default=None,
                       choices=["labse", "jina_v4", "clip", "nemotron"])
    parser.add_argument("--text_emb_model_name", type=str, default=None)
    parser.add_argument("--text_emb_adapter_path", type=str, default=None,
                       help="Path to PEFT/LoRA adapter weights for text embedding model")
    parser.add_argument("--audio_emb_model_type", type=str, default=None)
    parser.add_argument("--audio_emb_model_name", type=str, default=None)
    parser.add_argument("--audio_emb_adapter_path", type=str, default=None,
                       help="Path to PEFT/LoRA adapter weights for audio embedding model")
    
    # Pipeline mode
    parser.add_argument("--pipeline_mode", type=str, default=None,
                       choices=["asr_text_retrieval", "audio_emb_retrieval", "asr_only"])
    
    # Device arguments
    parser.add_argument("--asr_device", type=str, default=None)
    parser.add_argument("--text_emb_device", type=str, default=None)
    parser.add_argument("--audio_emb_device", type=str, default=None)
    
    # GPU pool arguments
    parser.add_argument("--devices", type=str, default=None,
                       help="Comma-separated list of devices for GPU pool (e.g., 'cuda:0,cuda:1')")
    parser.add_argument("--allocation_strategy", type=str, default=None,
                       choices=["memory_aware", "round_robin", "packing"],
                       help="GPU allocation strategy for distributing models")
    parser.add_argument(
        "--service_startup_mode",
        type=str,
        default=None,
        choices=["lazy", "eager"],
        help="Service startup strategy for model runtime",
    )
    parser.add_argument(
        "--service_offload_policy",
        type=str,
        default=None,
        choices=["on_finish", "never"],
        help="Service offload policy on evaluation shutdown",
    )
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--questions_path", type=str, default=None,
                       help="Path to benchmark questions JSON/JSONL")
    parser.add_argument("--corpus_path", type=str, default=None,
                       help="Path to corpus documents JSON/JSONL")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--trace_limit", type=int, default=None,
                       help="Include up to N query-level traces in results (0 disables)")
    parser.add_argument("--skip_dataset_validation", action="store_true",
                       help="Skip dataset integrity checks (not recommended for benchmark runs)")
    
    # Vector DB arguments
    parser.add_argument("--db_type", type=str, default=None,
                       choices=["in_memory", "faiss", "faiss_gpu"])
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--retrieval_mode", type=str, default=None,
                       choices=["dense", "sparse", "hybrid"])
    parser.add_argument("--hybrid_dense_weight", type=float, default=None)
    parser.add_argument("--reranker_mode", type=str, default=None,
                       choices=["none", "token_overlap"])
    parser.add_argument("--reranker_top_k", type=int, default=None)
    parser.add_argument("--reranker_weight", type=float, default=None)
    
    # Cache arguments
    parser.add_argument("--no_cache", action="store_true",
                       help="Disable caching")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--clear_cache", action="store_true",
                       help="Clear all caches before running")
    
    # Logging arguments
    parser.add_argument("--log_level", type=str, default=None,
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # Output arguments
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    
    # Checkpoint arguments
    parser.add_argument("--no_checkpoint", action="store_true",
                       help="Disable checkpointing")
    parser.add_argument("--checkpoint_interval", type=int, default=None)

    # LLM judge arguments
    parser.add_argument("--judge_enabled", action="store_true",
                       help="Enable LLM-as-judge over query traces")
    parser.add_argument("--judge_model", type=str, default=None)
    parser.add_argument("--judge_api_base", type=str, default=None)
    parser.add_argument("--judge_api_key_env", type=str, default=None)
    parser.add_argument("--judge_max_cases", type=int, default=None)
    parser.add_argument("--judge_timeout_s", type=int, default=None)
    parser.add_argument("--judge_temperature", type=float, default=None)
    
    return parser.parse_args()


# Mapping from CLI argument name to config nested path (tuple of attr names)
_CLI_ARG_MAP = {
    # Model overrides (model.*)
    "asr_model_type": ("model", "asr_model_type"),
    "asr_model_name": ("model", "asr_model_name"),
    "asr_adapter_path": ("model", "asr_adapter_path"),
    "text_emb_model_type": ("model", "text_emb_model_type"),
    "text_emb_model_name": ("model", "text_emb_model_name"),
    "text_emb_adapter_path": ("model", "text_emb_adapter_path"),
    "audio_emb_model_type": ("model", "audio_emb_model_type"),
    "audio_emb_model_name": ("model", "audio_emb_model_name"),
    "audio_emb_adapter_path": ("model", "audio_emb_adapter_path"),
    "pipeline_mode": ("model", "pipeline_mode"),
    # Device overrides (model.*)
    "asr_device": ("model", "asr_device"),
    "text_emb_device": ("model", "text_emb_device"),
    "audio_emb_device": ("model", "audio_emb_device"),
    # Service runtime (service_runtime.*)
    "service_startup_mode": ("service_runtime", "startup_mode"),
    "service_offload_policy": ("service_runtime", "offload_policy"),
    # Dataset overrides (data.*)
    "dataset_name": ("data", "dataset_name"),
    "batch_size": ("data", "batch_size"),
    "trace_limit": ("data", "trace_limit"),
    "questions_path": ("data", "questions_path"),
    "corpus_path": ("data", "corpus_path"),
    # Vector DB overrides (vector_db.*)
    "db_type": ("vector_db", "type"),
    "k": ("vector_db", "k"),
    "retrieval_mode": ("vector_db", "retrieval_mode"),
    "hybrid_dense_weight": ("vector_db", "hybrid_dense_weight"),
    "reranker_mode": ("vector_db", "reranker_mode"),
    "reranker_top_k": ("vector_db", "reranker_top_k"),
    "reranker_weight": ("vector_db", "reranker_weight"),
    # Cache overrides (cache.*)
    "cache_dir": ("cache", "cache_dir"),
    # Logging overrides (logging.*)
    "log_level": ("logging", "console_level"),
    # Output overrides (top-level config)
    "experiment_name": ("experiment_name",),
    "output_dir": ("output_dir",),
    # Checkpoint overrides (top-level config)
    "checkpoint_interval": ("checkpoint_interval",),
    # Judge overrides (judge.*)
    "judge_model": ("judge", "model"),
    "judge_api_base": ("judge", "api_base"),
    "judge_api_key_env": ("judge", "api_key_env"),
    "judge_max_cases": ("judge", "max_cases"),
    "judge_timeout_s": ("judge", "timeout_s"),
    "judge_temperature": ("judge", "temperature"),
}


def _set_nested_attr(obj, path: tuple, value) -> None:
    """Set a nested attribute via tuple path (e.g., ('model', 'asr_model_type')).

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
    """Apply command-line arguments to configuration object using mapping.
    
    Args:
        args: Parsed command-line arguments.
        config: EvaluationConfig object to modify in-place.
    """
    from ..config import DevicePoolConfig
    
    # Apply all mapped CLI arguments
    for arg_name, config_path in _CLI_ARG_MAP.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            _set_nested_attr(config, config_path, value)
    
    # Handle special cases with custom logic
    
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
    
    # Judge enabled flag (explicit)
    if args.judge_enabled:
        config.judge.enabled = True
