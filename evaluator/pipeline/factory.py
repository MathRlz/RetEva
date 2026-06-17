from typing import Optional, TYPE_CHECKING
from ..storage.cache import CacheManager
from ..models import (
    create_asr_model,
    create_text_embedding_model,
    create_audio_embedding_model,
    create_reranker,
)
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
    return create_reranker(
        model_type="cross_encoder",
        model_name=vector_db_config.reranker_model,
        device=vector_db_config.reranker_device,
    )


# vector_db node param → VectorDBConfig field (per-node store override surface;
# `collection` fans out to both backends' fields — only the active one is read).
_VECTOR_DB_PARAM_FIELDS = {
    "store": ("type",),
    "gpu_id": ("gpu_id",),
    "path": ("chromadb_path",),
    "url": ("qdrant_url",),
    "collection": ("chromadb_collection_name", "qdrant_collection_name"),
}


def effective_vector_db_config(vector_db_config, params):
    """Global ``vector_db`` config overlaid with a ``vector_db`` node's params.

    Transient — the global config is never mutated (precedent: ``_node_reranking``).
    Empty/absent params fall back to the global field.
    """
    from dataclasses import replace

    overrides = {}
    for param, fields in _VECTOR_DB_PARAM_FIELDS.items():
        value = (params or {}).get(param)
        if value in (None, ""):
            continue
        if param == "gpu_id":
            value = int(value)
        for field in fields:
            overrides[field] = value
    return replace(vector_db_config, **overrides) if overrides else vector_db_config


def check_graph_backend_dependencies(config) -> None:
    """Pre-flight: validate optional store backends for every ``vector_db`` node.

    A per-node ``store`` override can name chromadb/qdrant even when the global
    ``vector_db.type`` doesn't — fail before any model loads, not mid-run.
    """
    from .stage_graph import build_graph_for_config

    try:
        graph = build_graph_for_config(config)
    except Exception:
        return  # graph problems surface in their own validation path
    from .graph.operators import node_kind

    for node in graph.nodes:
        if node_kind(node.stage, node.params) != "vector_db":
            continue
        check_backend_dependencies(
            effective_vector_db_config(config.vector_db, node.params or {})
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


def create_vector_store_from_config(vector_db_config, embedding_dim: Optional[int] = None):
    from ..storage.vector_store import (
        InMemoryVectorStore,
        FaissVectorStore,
        FaissGpuVectorStore,
        FaissMmapVectorStore,
    )

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

    if store_type == "inmemory":
        return InMemoryVectorStore()

    if store_type in ("faiss", "faiss_gpu", "faiss_mmap"):
        if embedding_dim is None:
            raise ValueError(
                f"vector_db.type='{store_type}' requires embedding_dim. "
                "Set model.audio_emb_dim or model.text_emb_model_type so the dimension can be resolved."
            )
        if store_type == "faiss":
            return FaissVectorStore(embedding_dim)
        if store_type == "faiss_mmap":
            # Off-RAM corpus/index (3b): mmap index + Parquet payloads.
            return FaissMmapVectorStore(embedding_dim)
        res = getattr(vector_db_config, "gpu_id", 0)
        return FaissGpuVectorStore(embedding_dim, gpu_id=res if isinstance(res, int) else 0)

    available = "inmemory, faiss, faiss_gpu, faiss_mmap, chromadb, qdrant"
    raise ValueError(f"Unknown vector store type: '{store_type}'.\nAvailable types: {available}")


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


def _resolve_embedding_dim(config) -> Optional[int]:
    """Derive embedding dim from model config for FAISS index construction."""
    mode = config.model.pipeline_mode
    if mode in ("audio_emb_retrieval", "audio_text_retrieval"):
        return config.model.audio_emb_dim or None
    if mode in ("asr_text_retrieval", "text_retrieval"):
        try:
            from ..config.evaluation import get_text_embedding_dim
            return get_text_embedding_dim(config.model.text_emb_model_type)
        except Exception:
            return None
    return None


def _create_retrieval_pipeline(config, cache_manager, reranker):
    """
    Helper function to create RetrievalPipeline with common configuration.

    Extracts duplicate RetrievalPipeline creation logic to reduce repetition.
    """
    strategy_config = RetrievalStrategyConfig.from_vector_db_config(config.vector_db)
    embedding_dim = _resolve_embedding_dim(config)
    return RetrievalPipeline(
        create_vector_store_from_config(config.vector_db, embedding_dim=embedding_dim),
        cache_manager,
        strategy_config=strategy_config,
        reranker=reranker,
    )


class _ModelBuilders:
    """Builds each model family for a run, hoisted out of ``create_pipeline_from_config``
    (F10/D4) so that function is a flat per-mode dispatch rather than a nest of closures.

    One place holds the device resolution (GPU pool vs config device) and the
    provider-vs-standalone-factory fallback; the per-model argument lists stay explicit.
    """

    def __init__(self, config, service_provider, device_pool):
        self.mcfg = config.model
        self.service_provider = service_provider
        self.device_pool = device_pool

    def _get_device(self, model_category: str, model_type: str, config_device: str) -> str:
        if self.device_pool is not None and model_type is not None:
            from ..config import estimate_model_memory_gb

            memory_gb = estimate_model_memory_gb(model_category, model_type)
            return self.device_pool.allocate(model_category, memory_gb)
        return config_device

    def _build(self, model_category, model_type, config_device, from_provider, from_factory):
        """Resolve device then build via the shared service provider when present, else the
        standalone factory."""
        dev = self._get_device(model_category, model_type, config_device)
        if self.service_provider is not None:
            return from_provider(dev)
        return from_factory(dev)

    def asr(self):
        mcfg = self.mcfg
        return self._build(
            "asr", mcfg.asr_model_type, mcfg.asr_device,
            lambda dev: self.service_provider.get_asr_model(
                mcfg.asr_model_type, mcfg.asr_model_name, mcfg.asr_adapter_path, dev,
            ),
            lambda dev: create_asr_model(
                mcfg.asr_model_type,
                model_name=mcfg.asr_model_name,
                adapter_path=mcfg.asr_adapter_path,
                device=dev,
                size=mcfg.asr_size,
                quantization=mcfg.quantization_for("asr"),
                **mcfg.asr_params,
            ),
        )

    def text_emb(self):
        mcfg = self.mcfg
        return self._build(
            "text_embedding", mcfg.text_emb_model_type, mcfg.text_emb_device,
            lambda dev: self.service_provider.get_text_embedding_model(
                mcfg.text_emb_model_type, mcfg.text_emb_model_name, dev,
            ),
            lambda dev: create_text_embedding_model(
                mcfg.text_emb_model_type,
                model_name=mcfg.text_emb_model_name,
                device=dev,
                size=mcfg.text_emb_size,
                quantization=mcfg.quantization_for("text_emb"),
                **mcfg.text_emb_params,
            ),
        )

    def audio_emb(self):
        mcfg = self.mcfg
        return self._build(
            "audio_embedding", mcfg.audio_emb_model_type, mcfg.audio_emb_device,
            lambda dev: self.service_provider.get_audio_embedding_model(
                mcfg.audio_emb_model_type, mcfg.audio_emb_model_name,
                mcfg.audio_emb_model_path, mcfg.audio_emb_dim,
                mcfg.audio_emb_dropout, dev,
            ),
            lambda dev: create_audio_embedding_model(
                mcfg.audio_emb_model_type,
                model_name=mcfg.audio_emb_model_name,
                model_path=mcfg.audio_emb_model_path,
                emb_dim=mcfg.audio_emb_dim,
                dropout=mcfg.audio_emb_dropout,
                device=dev,
                size=mcfg.audio_emb_size,
                quantization=mcfg.quantization_for("audio_emb"),
                **mcfg.audio_emb_params,
            ),
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

    # Create reranker if enabled
    reranker = create_reranker_from_config(config.vector_db, service_provider=service_provider)

    mcfg = config.model
    builders = _ModelBuilders(config, service_provider, device_pool)

    # Create models based on mode
    if mode == "audio_emb_retrieval":
        audio_emb_pipeline = AudioEmbeddingPipeline(builders.audio_emb(), cache_manager)
        # Cross-modal corpus: with a text embedder configured, the corpus is text-embedded
        # into the shared space (audio query vs text corpus — the APM self-retrieval setup).
        # Without one, corpus_embedding falls back to the audio-corpus path (TTS + audio-embed).
        if mcfg.text_emb_model_type:
            text_emb_pipeline = TextEmbeddingPipeline(builders.text_emb(), cache_manager)
        retrieval_pipeline = _create_retrieval_pipeline(config, cache_manager, reranker)

    elif mode == "audio_text_retrieval":
        # Local import: keeps the a2e family (torch/transformers-heavy) lazy
        # for every other pipeline mode.
        from ..models.a2e import MultimodalClapStyleModel

        audio_emb_model = builders.audio_emb()

        is_multimodal = (
            mcfg.text_emb_model_type == "clap_text" and
            isinstance(audio_emb_model, MultimodalClapStyleModel)
        )
        text_emb_model = audio_emb_model if is_multimodal else builders.text_emb()

        audio_emb_pipeline = AudioEmbeddingPipeline(audio_emb_model, cache_manager)
        text_emb_pipeline = TextEmbeddingPipeline(text_emb_model, cache_manager)
        retrieval_pipeline = _create_retrieval_pipeline(config, cache_manager, reranker)

    elif mode == "asr_text_retrieval":
        asr_pipeline = ASRPipeline(builders.asr(), cache_manager)
        text_emb_pipeline = TextEmbeddingPipeline(builders.text_emb(), cache_manager)
        retrieval_pipeline = _create_retrieval_pipeline(config, cache_manager, reranker)

    elif mode == "asr_only":
        asr_pipeline = ASRPipeline(builders.asr(), cache_manager)
        
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
