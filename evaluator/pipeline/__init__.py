from .asr_pipeline import ASRPipeline
from .audio import AudioSynthesizer, AudioAugmenter
from .audio_embedding_pipeline import AudioEmbeddingPipeline
from .factory import (
    check_backend_dependencies,
    create_pipeline_from_config,
    create_vector_store_from_config,
)
from .protocols import (
    ASRPipelineProtocol,
    AudioEmbeddingPipelineProtocol,
    BatchSearchResults,
    CacheStats,
    EmbeddingPipeline,
    RetrievalPayload,
    SearchResult,
    SearchResults,
    RetrievalPipelineProtocol,
    TextEmbeddingPipelineProtocol,
)
from .retrieval_pipeline import RetrievalPipeline
from .stage_graph import (
    DATASET_ROLE_BOTH,
    DATASET_ROLE_CORPUS,
    DATASET_ROLE_QUESTIONS,
    GraphTemplateSpec,
    StageNode,
    StageNodeDef,
    StageGraph,
    build_branched_graph,
    build_graph_for_config,
    build_graph_from_spec,
    build_stage_graph,
    collapse_common_subexpressions,
    expand_branches,
    get_stage_node_def,
    register_stage_node,
    resolve_graph_template,
    validate_graph_artifacts,
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
    "StageNodeDef",
    "StageGraph",
    "GraphTemplateSpec",
    "build_branched_graph",
    "build_graph_for_config",
    "build_graph_from_spec",
    "build_stage_graph",
    "collapse_common_subexpressions",
    "expand_branches",
    "get_stage_node_def",
    "register_stage_node",
    "resolve_graph_template",
    "validate_graph_artifacts",
    "DATASET_ROLE_BOTH",
    "DATASET_ROLE_CORPUS",
    "DATASET_ROLE_QUESTIONS",
    "AudioSynthesizer",
    "AudioAugmenter",
    # Protocols
    "EmbeddingPipeline",
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
