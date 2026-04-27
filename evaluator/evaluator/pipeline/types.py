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
    
    def validate(self, mode: str) -> None:
        """Validate that required pipelines for the given mode are present.
        
        Args:
            mode: The pipeline mode to validate against
            
        Raises:
            ValueError: If required pipelines for the mode are missing
        """
        required = self._get_required_pipelines(mode)
        missing = []
        
        if "asr" in required and self.asr_pipeline is None:
            missing.append("asr_pipeline")
        if "text_embedding" in required and self.text_embedding_pipeline is None:
            missing.append("text_embedding_pipeline")
        if "audio_embedding" in required and self.audio_embedding_pipeline is None:
            missing.append("audio_embedding_pipeline")
        if "retrieval" in required and self.retrieval_pipeline is None:
            missing.append("retrieval_pipeline")
        
        if missing:
            raise ValueError(
                f"Mode '{mode}' requires the following pipelines which are missing: "
                f"{', '.join(missing)}"
            )
    
    @staticmethod
    def _get_required_pipelines(mode: str) -> set:
        """Get the set of required pipeline types for a given mode."""
        requirements = {
            "audio_emb_retrieval": {"audio_embedding", "retrieval"},
            "audio_text_retrieval": {"audio_embedding", "text_embedding", "retrieval"},
            "asr_text_retrieval": {"asr", "text_embedding", "retrieval"},
            "asr_only": {"asr"},
        }
        if mode not in requirements:
            raise ValueError(f"Unknown pipeline mode: {mode}")
        return requirements[mode]
    
    @property
    def has_asr(self) -> bool:
        """Check if ASR pipeline is available."""
        return self.asr_pipeline is not None
    
    @property
    def has_text_embedding(self) -> bool:
        """Check if text embedding pipeline is available."""
        return self.text_embedding_pipeline is not None
    
    @property
    def has_audio_embedding(self) -> bool:
        """Check if audio embedding pipeline is available."""
        return self.audio_embedding_pipeline is not None
    
    @property
    def has_retrieval(self) -> bool:
        """Check if retrieval pipeline is available."""
        return self.retrieval_pipeline is not None
    
    @property
    def can_fuse_embeddings(self) -> bool:
        """Check if both audio and text embedding pipelines are available for fusion."""
        return self.has_audio_embedding and self.has_text_embedding
