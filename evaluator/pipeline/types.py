"""Pipeline types and data structures."""
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .asr_pipeline import ASRPipeline
    from .text_embedding_pipeline import TextEmbeddingPipeline
    from .audio_embedding_pipeline import AudioEmbeddingPipeline
    from .retrieval_pipeline import RetrievalPipeline
    from ..devices import GPUPool
    from ..services import ModelServiceProvider


@dataclass
class PipelineBundle:
    """Bundle of pipelines created from configuration.

    This dataclass provides a cleaner return type for create_pipeline_from_config(),
    with named attributes instead of tuple unpacking.

    Attributes:
        asr_pipeline: ASR pipeline for speech-to-text conversion
        text_embedding_pipeline: Text embedding pipeline for encoding text
        audio_embedding_pipeline: Audio embedding pipeline for encoding audio
        retrieval_pipeline: Retrieval pipeline for vector search
        mode: The evaluation mode used (e.g., 'asr_text_retrieval', 'audio_emb_retrieval')
        device_pool: Optional GPU pool used for device allocation
        service_provider: Optional model service provider for lifecycle management
    """
    asr_pipeline: Optional["ASRPipeline"] = None
    text_embedding_pipeline: Optional["TextEmbeddingPipeline"] = None
    audio_embedding_pipeline: Optional["AudioEmbeddingPipeline"] = None
    retrieval_pipeline: Optional["RetrievalPipeline"] = None
    mode: str = ""
    device_pool: Optional["GPUPool"] = None
    service_provider: Optional["ModelServiceProvider"] = None
