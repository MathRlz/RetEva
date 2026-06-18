from dataclasses import dataclass
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
from .stage_graph import resolve_graph_template

from ..logging_config import get_logger

if TYPE_CHECKING:
    from ..devices import GPUPool
    from ..services import ModelServiceProvider

logger = get_logger(__name__)


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
    from ..storage.registry import check_store_dependency

    check_store_dependency(vector_db_config)


def create_vector_store_from_config(vector_db_config, embedding_dim: Optional[int] = None):
    """Build the configured vector store. Dispatch lives in the storage registry (OCP) —
    a backend registers a ``(config, dim) -> VectorStore`` factory under its type name."""
    from ..storage.registry import create_vector_store

    return create_vector_store(vector_db_config, embedding_dim)


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


@dataclass(frozen=True)
class _GraphBuildPlan:
    """Which family pipelines a run builds + the FAISS index dim — all derived from the
    execution graph's nodes, not the ``pipeline_mode`` string (graph-first Phase 2)."""

    build_asr: bool
    build_text: bool
    build_audio: bool
    build_retrieval: bool
    embedding_dim: Optional[int]


def _graph_build_plan(graph, mcfg) -> "_GraphBuildPlan":
    """Decide what to build from ``graph``'s nodes (single source of truth).

    A family pipeline is built when a node references its model field (``node_model_field``);
    retrieval when a ``search`` node exists. The corpus embedder is *not* a distinct model
    field — it shares the query embedder, or text-embeds the corpus cross-modally — so a text
    embedder is *also* built when a corpus-embed node is present and a text embedder is
    configured (the ``audio_emb_retrieval`` cross-modal corpus case). The index dim follows the
    query embedder the graph actually has: an audio query node → ``audio_emb_dim``, a text query
    node → the text model's registered dim."""
    from .graph.registry import node_model_field

    fields = set()
    has_corpus_embed = False
    has_search = False
    for n in graph.nodes:
        field = node_model_field(n.stage, n.params)
        if field:
            fields.add(field)
        if n.stage == "search":
            has_search = True
        if n.stage == "embed" and (n.params or {}).get("axis") == "corpus":
            has_corpus_embed = True

    build_audio = "model.audio_emb_model_type" in fields
    build_text = (
        "model.text_emb_model_type" in fields
        or (has_corpus_embed and bool(mcfg.text_emb_model_type))
    )

    if build_audio:
        embedding_dim = mcfg.audio_emb_dim or None
    elif "model.text_emb_model_type" in fields:
        try:
            from ..config.evaluation import get_text_embedding_dim
            embedding_dim = get_text_embedding_dim(mcfg.text_emb_model_type)
        except Exception:
            embedding_dim = None
    else:
        embedding_dim = None

    return _GraphBuildPlan(
        build_asr="model.asr_model_type" in fields,
        build_text=build_text,
        build_audio=build_audio,
        build_retrieval=has_search,
        embedding_dim=embedding_dim,
    )


def _create_retrieval_pipeline(config, cache_manager, reranker, embedding_dim=None):
    """
    Helper function to create RetrievalPipeline with common configuration.

    Extracts duplicate RetrievalPipeline creation logic to reduce repetition. ``embedding_dim``
    is graph-derived by the caller (the query embedder's dim).
    """
    strategy_config = RetrievalStrategyConfig.from_vector_db_config(config.vector_db)
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
        logger.info(
            "build %s model: type=%s device=%s via=%s",
            model_category, model_type, dev,
            "provider" if self.service_provider is not None else "factory",
        )
        if self.service_provider is not None:
            return from_provider(dev)
        return from_factory(dev)

    def asr(self):
        mcfg = self.mcfg
        return self._build(
            "asr", mcfg.asr_model_type, mcfg.asr_device,
            lambda dev: self.service_provider.get_asr_model(
                mcfg.asr_model_type, mcfg.asr_model_name, mcfg.asr_adapter_path, dev,
                size=mcfg.asr_size, quantization=mcfg.quantization_for("asr"),
                **mcfg.asr_params,
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
                size=mcfg.text_emb_size, quantization=mcfg.quantization_for("text_emb"),
                **mcfg.text_emb_params,
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
                size=mcfg.audio_emb_size, quantization=mcfg.quantization_for("audio_emb"),
                **mcfg.audio_emb_params,
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

    from .graph.modes import _config_template

    mode = _config_template(config)
    # A template declares its required model fields up front; an explicit-graph config (no template)
    # has no such list — the nodes carry their own models, validated structurally by the graph. So
    # the field check is template-only.
    if mode is not None:
        mode_spec = resolve_graph_template(str(mode))
        missing_required = [
            field_path
            for field_path in mode_spec.required_model_fields
            if getattr(getattr(config, field_path.split(".", 1)[0]), field_path.split(".", 1)[1])
            in (None, "")
        ]
        if missing_required:
            raise ValueError(
                f"Pipeline mode '{mode}' missing required config fields: "
                f"{', '.join(missing_required)}"
            )

    # Which family pipelines to build is derived from the execution GRAPH's nodes, not the
    # mode string (graph-first Phase 2). The mode still selects the default graph + labels the
    # bundle; the build *decision* is graph-authoritative, so an explicit graph (or one with no
    # ASR node) never builds a model it doesn't use.
    from .graph.modes import build_graph_for_config

    plan = _graph_build_plan(build_graph_for_config(config), config.model)
    logger.info(
        "pipeline build plan (template=%s): asr=%s text_emb=%s audio_emb=%s retrieval=%s",
        mode, plan.build_asr, plan.build_text, plan.build_audio, plan.build_retrieval,
    )

    # Create GPU pool if configured
    device_pool = create_gpu_pool_from_config(config)

    # Create reranker if enabled
    reranker = create_reranker_from_config(config.vector_db, service_provider=service_provider)

    mcfg = config.model
    builders = _ModelBuilders(config, service_provider, device_pool)

    audio_emb_model = None
    if plan.build_audio:
        audio_emb_model = builders.audio_emb()
        audio_emb_pipeline = AudioEmbeddingPipeline(audio_emb_model, cache_manager)
    if plan.build_asr:
        asr_pipeline = ASRPipeline(builders.asr(), cache_manager)
    if plan.build_text:
        # Cross-modal multimodal CLAP reuses the audio model instance as the text embedder;
        # otherwise build a standalone text embedder. (Local import keeps the a2e family —
        # torch/transformers-heavy — lazy for every non-audio run.)
        from ..models.a2e import MultimodalClapStyleModel

        is_multimodal = (
            audio_emb_model is not None
            and mcfg.text_emb_model_type == "clap_text"
            and isinstance(audio_emb_model, MultimodalClapStyleModel)
        )
        text_emb_model = audio_emb_model if is_multimodal else builders.text_emb()
        text_emb_pipeline = TextEmbeddingPipeline(text_emb_model, cache_manager)
    if plan.build_retrieval:
        retrieval_pipeline = _create_retrieval_pipeline(
            config, cache_manager, reranker, plan.embedding_dim
        )

    built = [
        name for name, pipe in (
            ("asr", asr_pipeline), ("text_emb", text_emb_pipeline),
            ("audio_emb", audio_emb_pipeline), ("retrieval", retrieval_pipeline),
        ) if pipe is not None
    ]
    logger.info("pipelines ready: %s", ", ".join(built) or "(none)")
    return PipelineBundle(
        asr_pipeline=asr_pipeline,
        text_embedding_pipeline=text_emb_pipeline,
        audio_embedding_pipeline=audio_emb_pipeline,
        retrieval_pipeline=retrieval_pipeline,
        mode=mode,
        device_pool=device_pool,
        service_provider=service_provider,
    )
