from typing import Optional, TYPE_CHECKING
from ..storage.cache import CacheManager
from ..models import MultimodalClapStyleModel
from ..models.retrieval import RetrievalStrategyConfig
from .asr_pipeline import ASRPipeline
from .text_embedding_pipeline import TextEmbeddingPipeline
from .audio_embedding_pipeline import AudioEmbeddingPipeline
from .retrieval_pipeline import RetrievalPipeline
from .types import PipelineBundle
from .stage_graph import list_pipeline_mode_specs, resolve_pipeline_mode_spec

if TYPE_CHECKING:
    from ..devices import GPUPool
    from ..services import ModelServiceProvider


def create_asr_model(model_type: str, model_name: Optional[str], adapter_path: Optional[str],
                     device: str, *, size: Optional[str] = None, **extra_params):
    from ..models import create_asr_model as factory_create_asr_model
    return factory_create_asr_model(model_type, model_name=model_name, size=size,
                                    adapter_path=adapter_path, device=device, **extra_params)


def create_text_embedding_model(model_type: str, model_name: Optional[str], device: str,
                                *, size: Optional[str] = None, **extra_params):
    from ..models import create_text_embedding_model as factory_create_text_embedding_model
    return factory_create_text_embedding_model(model_type, model_name=model_name, size=size,
                                               device=device, **extra_params)


def create_audio_embedding_model(model_type: str, model_name: Optional[str], model_path: Optional[str],
                                  emb_dim: int, dropout: float, device: str,
                                  *, size: Optional[str] = None, **extra_params):
    from ..models import create_audio_embedding_model as factory_create_audio_embedding_model
    return factory_create_audio_embedding_model(model_type, model_name=model_name, size=size,
                                                model_path=model_path, emb_dim=emb_dim,
                                                dropout=dropout, device=device, **extra_params)


def create_reranker_from_config(vector_db_config, service_provider: Optional["ModelServiceProvider"] = None):
    """Create a reranker model from configuration.
    
    Args:
        vector_db_config: VectorDBConfig with reranker settings
        
    Returns:
        BaseReranker instance or None if reranking is disabled
    """
    if not vector_db_config.reranker_enabled:
        return None
    
    if service_provider is not None:
        return service_provider.get_reranker(
            model_type="cross_encoder",
            model_name=vector_db_config.reranker_model,
            device=vector_db_config.reranker_device,
        )
    from ..models import create_reranker
    return create_reranker(
        model_type="cross_encoder",
        model_name=vector_db_config.reranker_model,
        device=vector_db_config.reranker_device,
    )


def check_backend_dependencies(vector_db_config) -> None:
    """Validate that required optional backends are installed.

    Raises ``ImportError`` with an install hint when a configured backend
    package is missing.  Call this at config-validation time so the user
    gets immediate feedback instead of a mid-run crash.
    """
    import importlib
    store_type = str(vector_db_config.type) if hasattr(vector_db_config.type, 'value') else str(vector_db_config.type)
    if store_type == "chromadb":
        if importlib.util.find_spec("chromadb") is None:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install evaluator[chromadb]"
            )
    elif store_type == "qdrant":
        if importlib.util.find_spec("qdrant_client") is None:
            raise ImportError(
                "Qdrant client is not installed. Install with: pip install evaluator[qdrant]"
            )


def create_vector_store_from_config(vector_db_config):
    from ..storage.vector_store import InMemoryVectorStore, FaissVectorStore, FaissGpuVectorStore
    
    store_type = vector_db_config.type
    
    # Handle ChromaDB separately (optional dependency)
    if store_type == "chromadb":
        try:
            from ..storage.backends.chromadb_store import ChromaDBVectorStore
        except ImportError:
            raise ImportError(
                "ChromaDB is not installed. Install it with: pip install chromadb"
            )
        
        return ChromaDBVectorStore(
            collection_name=vector_db_config.chromadb_collection_name,
            persist_path=vector_db_config.chromadb_path,
            distance_fn=vector_db_config.distance_metric,
        )
    
    # Handle Qdrant separately (optional dependency)
    if store_type == "qdrant":
        try:
            from ..storage.backends.qdrant_store import QdrantVectorStore
        except ImportError:
            raise ImportError(
                "Qdrant client is not installed. Install it with: pip install qdrant-client"
            )
        
        return QdrantVectorStore(
            collection_name=vector_db_config.qdrant_collection_name,
            url=vector_db_config.qdrant_url,
            path=vector_db_config.qdrant_path,
            distance_fn=vector_db_config.distance_metric,
            api_key=vector_db_config.qdrant_api_key,
        )
    
    store_types = {
        "inmemory": InMemoryVectorStore,
        "faiss": FaissVectorStore,
        "faiss_gpu": FaissGpuVectorStore,
    }
    
    store_class = store_types.get(store_type)
    if store_class is None:
        available = ", ".join(sorted(list(store_types.keys()) + ["chromadb", "qdrant"]))
        raise ValueError(
            f"Unknown vector store type: '{store_type}'.\n"
            f"Available types: {available}"
        )
    
    return store_class()


def create_gpu_pool_from_config(config) -> Optional["GPUPool"]:
    """Create a GPUPool from configuration if device_pool is configured.
    
    Args:
        config: EvaluationConfig with optional device_pool settings.
        
    Returns:
        GPUPool instance or None if device_pool is not configured.
    """
    if config.device_pool is None:
        return None
    
    from ..devices import GPUPool
    from ..devices.strategy import create_strategy
    
    pool = GPUPool(
        devices=list(config.device_pool.available_devices),
        memory_buffer_percent=config.device_pool.memory_buffer_percent,
        allow_cpu_fallback=config.device_pool.allow_cpu_fallback,
    )
    
    # Set allocation strategy
    if config.device_pool.model_device_overrides:
        strategy = create_strategy("manual", overrides=config.device_pool.model_device_overrides)
    else:
        strategy = create_strategy(config.device_pool.allocation_strategy)
    pool.set_strategy(strategy)
    
    return pool


def _create_retrieval_pipeline(config, cache_manager, reranker):
    """
    Helper function to create RetrievalPipeline with common configuration.
    
    Extracts duplicate RetrievalPipeline creation logic to reduce repetition.
    """
    strategy_config = RetrievalStrategyConfig.from_vector_db_config(config.vector_db)
    return RetrievalPipeline(
        create_vector_store_from_config(config.vector_db),
        cache_manager,
        strategy_config=strategy_config,
        reranker=reranker,
    )


def create_pipeline_from_config(
    config,
    cache_manager: CacheManager,
    service_provider: Optional["ModelServiceProvider"] = None,
) -> PipelineBundle:
    """Create pipelines based on configuration.
    
    Args:
        config: Evaluation configuration
        cache_manager: Cache manager for pipeline caching
        
    Returns:
        PipelineBundle containing the created pipelines
    """
    asr_pipeline = None
    text_emb_pipeline = None
    audio_emb_pipeline = None
    retrieval_pipeline = None
    
    mode = config.model.pipeline_mode
    mode_spec = resolve_pipeline_mode_spec(str(mode))

    missing_required = []
    for field_path in mode_spec.required_model_fields:
        section_name, field_name = field_path.split(".", 1)
        section = getattr(config, section_name)
        value = getattr(section, field_name)
        if value in (None, ""):
            missing_required.append(field_path)
    if missing_required:
        raise ValueError(
            f"Pipeline mode '{mode}' missing required config fields: {', '.join(missing_required)}"
        )
    
    # Create GPU pool if configured
    device_pool = create_gpu_pool_from_config(config)
    
    # Helper to get device (from pool or config)
    def get_device(model_category: str, model_type: str, config_device: str) -> str:
        if device_pool is not None and model_type is not None:
            from ..config import estimate_model_memory_gb
            memory_gb = estimate_model_memory_gb(model_category, model_type)
            return device_pool.allocate(model_category, memory_gb)
        return config_device
    
    # Create reranker if enabled
    reranker = create_reranker_from_config(config.vector_db, service_provider=service_provider)
    
    # Shorthand helpers
    mcfg = config.model

    def _make_asr():
        dev = get_device("asr", mcfg.asr_model_type, mcfg.asr_device)
        if service_provider is not None:
            return service_provider.get_asr_model(
                mcfg.asr_model_type, mcfg.asr_model_name,
                mcfg.asr_adapter_path, dev,
            )
        return create_asr_model(
            mcfg.asr_model_type, mcfg.asr_model_name,
            mcfg.asr_adapter_path, dev,
            size=mcfg.asr_size, **mcfg.asr_params,
        )

    def _make_text_emb():
        dev = get_device("text_embedding", mcfg.text_emb_model_type, mcfg.text_emb_device)
        if service_provider is not None:
            return service_provider.get_text_embedding_model(
                mcfg.text_emb_model_type, mcfg.text_emb_model_name, dev,
            )
        return create_text_embedding_model(
            mcfg.text_emb_model_type, mcfg.text_emb_model_name, dev,
            size=mcfg.text_emb_size, **mcfg.text_emb_params,
        )

    def _make_audio_emb():
        dev = get_device("audio_embedding", mcfg.audio_emb_model_type, mcfg.audio_emb_device)
        if service_provider is not None:
            return service_provider.get_audio_embedding_model(
                mcfg.audio_emb_model_type, mcfg.audio_emb_model_name,
                mcfg.audio_emb_model_path, mcfg.audio_emb_dim,
                mcfg.audio_emb_dropout, dev,
            )
        return create_audio_embedding_model(
            mcfg.audio_emb_model_type, mcfg.audio_emb_model_name,
            mcfg.audio_emb_model_path, mcfg.audio_emb_dim,
            mcfg.audio_emb_dropout, dev,
            size=mcfg.audio_emb_size, **mcfg.audio_emb_params,
        )

    # Create models based on mode
    if mode == "audio_emb_retrieval":
        audio_emb_pipeline = AudioEmbeddingPipeline(_make_audio_emb(), cache_manager)
        retrieval_pipeline = _create_retrieval_pipeline(config, cache_manager, reranker)

    elif mode == "audio_text_retrieval":
        audio_emb_model = _make_audio_emb()

        is_multimodal = (
            mcfg.text_emb_model_type == "clap_text" and
            isinstance(audio_emb_model, MultimodalClapStyleModel)
        )
        text_emb_model = audio_emb_model if is_multimodal else _make_text_emb()

        audio_emb_pipeline = AudioEmbeddingPipeline(audio_emb_model, cache_manager)
        text_emb_pipeline = TextEmbeddingPipeline(text_emb_model, cache_manager)
        retrieval_pipeline = _create_retrieval_pipeline(config, cache_manager, reranker)

    elif mode == "asr_text_retrieval":
        asr_pipeline = ASRPipeline(_make_asr(), cache_manager)
        text_emb_pipeline = TextEmbeddingPipeline(_make_text_emb(), cache_manager)
        retrieval_pipeline = _create_retrieval_pipeline(config, cache_manager, reranker)

    elif mode == "asr_only":
        asr_pipeline = ASRPipeline(_make_asr(), cache_manager)
        
    else:
        available_modes = [spec.mode for spec in list_pipeline_mode_specs()]
        raise ValueError(
            f"Unknown pipeline mode: '{mode}'.\n"
            f"Available modes: {', '.join(available_modes)}\n"
            f"Set 'pipeline_mode' in config.model to one of the above."
        )
    
    return PipelineBundle(
        asr_pipeline=asr_pipeline,
        text_embedding_pipeline=text_emb_pipeline,
        audio_embedding_pipeline=audio_emb_pipeline,
        retrieval_pipeline=retrieval_pipeline,
        mode=mode,
        device_pool=device_pool,
        service_provider=service_provider,
    )
