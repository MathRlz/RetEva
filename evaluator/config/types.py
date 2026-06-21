"""Type definitions and enums for configuration management."""
from enum import Enum
from typing import Any, Union


class VectorDBType(str, Enum):
    """Vector database backend types."""
    INMEMORY = "inmemory"
    FAISS = "faiss"
    FAISS_GPU = "faiss_gpu"
    FAISS_MMAP = "faiss_mmap"  # on-disk mmap index + off-RAM payloads (Roadmap 3b)
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


# ── Closed option sets validated as plain strings ─────────────────────────────────────
# The canonical source for each: config validation, the builder/form selects, the
# operators-catalog node forms, the CLI, and the /api/introspection/schema endpoint all
# reference these tuples, so a value can never drift between the UI and the core.
RETRIEVAL_MODES = ("dense", "sparse", "hybrid")
RERANKER_MODES = ("none", "token_overlap", "cross_encoder")
SERVICE_STARTUP_MODES = ("lazy", "eager")
SERVICE_OFFLOAD_POLICIES = ("on_finish", "never", "on_finish_soft_cpu")
DATASET_SOURCES = ("local", "huggingface", "custom")

# Coarse modality per dataset type (UI picker grouping + the introspection schema). Evaluator-owned
# so a new DatasetType is grouped from one place rather than a webapi-local map.
DATASET_TYPE_MODALITY = {
    DatasetType.AUDIO_TRANSCRIPTION.value: "audio",
    DatasetType.AUDIO_QUERY_RETRIEVAL.value: "audio",
    DatasetType.TEXT_QUERY_RETRIEVAL.value: "text",
    DatasetType.QUESTION_ANSWERING.value: "text",
    DatasetType.PASSAGE_RANKING.value: "text",
    DatasetType.MULTIMODAL_QA.value: "multimodal",
}


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
