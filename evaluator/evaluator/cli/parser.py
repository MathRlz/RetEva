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


def apply_args_to_config(args: argparse.Namespace, config) -> None:
    """Apply command-line arguments to configuration object.
    
    Args:
        args: Parsed command-line arguments.
        config: EvaluationConfig object to modify in-place.
    """
    from ..config import DevicePoolConfig
    
    # Model overrides
    if args.asr_model_type is not None:
        config.model.asr_model_type = args.asr_model_type
    if args.asr_model_name is not None:
        config.model.asr_model_name = args.asr_model_name
    if args.asr_adapter_path is not None:
        config.model.asr_adapter_path = args.asr_adapter_path
    if args.text_emb_model_type is not None:
        config.model.text_emb_model_type = args.text_emb_model_type
    if args.text_emb_model_name is not None:
        config.model.text_emb_model_name = args.text_emb_model_name
    if args.text_emb_adapter_path is not None:
        config.model.text_emb_adapter_path = args.text_emb_adapter_path
    if args.audio_emb_model_type is not None:
        config.model.audio_emb_model_type = args.audio_emb_model_type
    if args.audio_emb_model_name is not None:
        config.model.audio_emb_model_name = args.audio_emb_model_name
    if args.audio_emb_adapter_path is not None:
        config.model.audio_emb_adapter_path = args.audio_emb_adapter_path
    if args.pipeline_mode is not None:
        config.model.pipeline_mode = args.pipeline_mode
    
    # Device overrides
    if args.asr_device is not None:
        config.model.asr_device = args.asr_device
    if args.text_emb_device is not None:
        config.model.text_emb_device = args.text_emb_device
    if args.audio_emb_device is not None:
        config.model.audio_emb_device = args.audio_emb_device
    
    # GPU pool configuration from CLI
    if args.devices is not None or args.allocation_strategy is not None:
        # Create or update device pool config
        if config.device_pool is None:
            config.device_pool = DevicePoolConfig()
        
        if args.devices is not None:
            config.device_pool.available_devices = [d.strip() for d in args.devices.split(",")]
        
    if args.allocation_strategy is not None:
        config.device_pool.allocation_strategy = args.allocation_strategy

    service_startup_mode = getattr(args, "service_startup_mode", None)
    service_offload_policy = getattr(args, "service_offload_policy", None)
    if service_startup_mode is not None:
        config.service_runtime.startup_mode = service_startup_mode
    if service_offload_policy is not None:
        config.service_runtime.offload_policy = service_offload_policy
    
    # Dataset overrides
    if args.dataset_name is not None:
        config.data.dataset_name = args.dataset_name
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.trace_limit is not None:
        config.data.trace_limit = args.trace_limit
    if args.questions_path is not None:
        config.data.questions_path = args.questions_path
    if args.corpus_path is not None:
        config.data.corpus_path = args.corpus_path
    if args.skip_dataset_validation:
        config.data.strict_validation = False
    
    # Vector DB overrides
    if args.db_type is not None:
        config.vector_db.type = args.db_type
    if args.k is not None:
        config.vector_db.k = args.k
    if args.retrieval_mode is not None:
        config.vector_db.retrieval_mode = args.retrieval_mode
    if args.hybrid_dense_weight is not None:
        config.vector_db.hybrid_dense_weight = args.hybrid_dense_weight
    if args.reranker_mode is not None:
        config.vector_db.reranker_mode = args.reranker_mode
    if args.reranker_top_k is not None:
        config.vector_db.reranker_top_k = args.reranker_top_k
    if args.reranker_weight is not None:
        config.vector_db.reranker_weight = args.reranker_weight
    
    # Cache overrides
    if args.no_cache:
        config.cache.enabled = False
    if args.cache_dir is not None:
        config.cache.cache_dir = args.cache_dir
    
    # Logging overrides
    if args.log_level is not None:
        config.logging.console_level = args.log_level
    
    # Output overrides
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    
    # Checkpoint overrides
    if args.no_checkpoint:
        config.checkpoint_enabled = False
    if args.checkpoint_interval is not None:
        config.checkpoint_interval = args.checkpoint_interval
    
    # Judge overrides
    if args.judge_enabled:
        config.judge.enabled = True
    if args.judge_model is not None:
        config.judge.model = args.judge_model
    if args.judge_api_base is not None:
        config.judge.api_base = args.judge_api_base
    if args.judge_api_key_env is not None:
        config.judge.api_key_env = args.judge_api_key_env
    if args.judge_max_cases is not None:
        config.judge.max_cases = args.judge_max_cases
    if args.judge_timeout_s is not None:
        config.judge.timeout_s = args.judge_timeout_s
    if args.judge_temperature is not None:
        config.judge.temperature = args.judge_temperature
