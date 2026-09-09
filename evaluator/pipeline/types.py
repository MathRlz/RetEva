"""Pipeline types and data structures."""
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .asr_pipeline import ASRPipeline
    from .text_embedding_pipeline import TextEmbeddingPipeline
    from .audio_embedding_pipeline import AudioEmbeddingPipeline
    from .retrieval_pipeline import RetrievalPipeline
    from ..devices import GPUPool
    from .model_provider import ModelServiceProvider


@dataclass
class PipelineBundle:
    """Bundle of pipelines created from configuration.

    A cleaner return type for create_pipeline_from_config(), with named attributes
    instead of tuple unpacking.
    """
    asr_pipeline: Optional["ASRPipeline"] = None
    text_embedding_pipeline: Optional["TextEmbeddingPipeline"] = None
    audio_embedding_pipeline: Optional["AudioEmbeddingPipeline"] = None
    retrieval_pipeline: Optional["RetrievalPipeline"] = None
    mode: str = ""
    device_pool: Optional["GPUPool"] = None
    service_provider: Optional["ModelServiceProvider"] = None
