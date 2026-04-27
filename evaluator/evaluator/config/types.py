"""Type definitions and enums for configuration management."""
from enum import Enum
from typing import Any, Union


class PipelineMode(str, Enum):
    """Pipeline execution modes for evaluation."""
    ASR_TEXT_RETRIEVAL = "asr_text_retrieval"
    AUDIO_EMB_RETRIEVAL = "audio_emb_retrieval"
    ASR_ONLY = "asr_only"
    AUDIO_TEXT_RETRIEVAL = "audio_text_retrieval"
    
    def __str__(self) -> str:
        return self.value


class VectorDBType(str, Enum):
    """Vector database backend types."""
    INMEMORY = "inmemory"
    FAISS = "faiss"
    FAISS_GPU = "faiss_gpu"
    CHROMADB = "chromadb"
    QDRANT = "qdrant"
    
    def __str__(self) -> str:
        return self.value


class AllocationStrategy(str, Enum):
    """GPU allocation strategies for device pool."""
    ROUND_ROBIN = "round_robin"
    MEMORY_AWARE = "memory_aware"
    PACKING = "packing"
    MANUAL = "manual"
    
    def __str__(self) -> str:
        return self.value


class DatasetType(str, Enum):
    """Type of evaluation dataset."""
    AUDIO_QUERY_RETRIEVAL = "audio_query_retrieval"
    TEXT_QUERY_RETRIEVAL = "text_query_retrieval"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    QUESTION_ANSWERING = "question_answering"
    MULTIMODAL_QA = "multimodal_qa"
    PASSAGE_RANKING = "passage_ranking"
    
    def __str__(self) -> str:
        return self.value


def to_enum(value: Any, enum_class: type[Enum]) -> Enum:
    """Convert a value to an enum, accepting both strings and enum instances.
    
    Args:
        value: String value or enum instance to convert.
        enum_class: Target enum class.
        
    Returns:
        Enum instance.
        
    Raises:
        ValueError: If value is not a valid enum member.
    """
    if isinstance(value, enum_class):
        return value
    if isinstance(value, str):
        try:
            return enum_class(value)
        except ValueError:
            valid_values = [e.value for e in enum_class]
            raise ValueError(
                f"Invalid {enum_class.__name__}: '{value}'. "
                f"Valid values: {', '.join(valid_values)}"
            )
    raise TypeError(f"Expected str or {enum_class.__name__}, got {type(value).__name__}")


def enum_to_str(value: Union[str, Enum]) -> str:
    """Convert enum to string for serialization.
    
    Args:
        value: String or enum instance.
        
    Returns:
        String value.
    """
    if isinstance(value, Enum):
        return value.value
    return value
