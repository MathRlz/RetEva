from .asr_pipeline import ASRPipeline
from .audio import AudioSynthesizer, AudioAugmenter
from .audio_embedding_pipeline import AudioEmbeddingPipeline
from .base import BasePipelineABC, EmbeddingPipelineABC, TranscriptionPipelineABC
from .factory import check_backend_dependencies, create_pipeline_from_config, create_vector_store_from_config
from .protocols import (
    ASRPipelineProtocol,
    AudioEmbeddingPipelineProtocol,
    BatchSearchResults,
    BasePipeline,
    CacheStats,
    CacheStatsProvider,
    EmbeddingPipeline,
    RetrievalPayload,
    SearchResult,
    SearchResults,
    RetrievalPipelineProtocol,
    TextEmbeddingPipelineProtocol,
)
from .retrieval_pipeline import RetrievalPipeline
from .stage_graph import (
    PipelineModeSpec,
    StageNode,
    StageGraph,
    build_stage_graph,
    list_pipeline_mode_specs,
    resolve_pipeline_mode_spec,
)
from .text_embedding_pipeline import TextEmbeddingPipeline
from .types import PipelineBundle
from ..models.retrieval.contracts import ScoredRetrievalResult

__all__ = [
    # Pipeline implementations
    "ASRPipeline",
    "TextEmbeddingPipeline",
    "AudioEmbeddingPipeline",
    "RetrievalPipeline",
    "StageNode",
    "StageGraph",
    "PipelineModeSpec",
    "build_stage_graph",
    "list_pipeline_mode_specs",
    "resolve_pipeline_mode_spec",
    "AudioSynthesizer",
    "AudioAugmenter",
    # Base classes
    "BasePipelineABC",
    "EmbeddingPipelineABC",
    "TranscriptionPipelineABC",
    # Protocols
    "BasePipeline",
    "EmbeddingPipeline",
    "CacheStatsProvider",
    "TextEmbeddingPipelineProtocol",
    "AudioEmbeddingPipelineProtocol",
    "ASRPipelineProtocol",
    "RetrievalPipelineProtocol",
    "RetrievalPayload",
    "SearchResult",
    "SearchResults",
    "BatchSearchResults",
    "ScoredRetrievalResult",
    # Type aliases
    "CacheStats",
    # Factory functions
    "check_backend_dependencies",
    "create_pipeline_from_config",
    "create_vector_store_from_config",
    # Data structures
    "PipelineBundle",
]
