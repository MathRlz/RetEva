"""Dataset capability profiles for pipeline planning."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..config.types import DatasetType
from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DatasetCapabilityProfile:
    """Capabilities and recommended execution modes for a dataset."""

    name: str
    dataset_type: DatasetType
    requires_audio: bool
    requires_text: bool
    supports_generation: bool
    evaluation_mode: str
    recommended_pipeline_modes: Sequence[str]

    def supports_pipeline_mode(self, mode: str) -> bool:
        return mode in set(self.recommended_pipeline_modes)


# Capability metadata for the built-in datasets lives on their DatasetDescriptors
# (the single source of truth) — resolve_dataset_profile derives a profile from there.
# This registry only holds profiles added at runtime via register_dataset_profile.
_PROFILE_REGISTRY: Dict[str, DatasetCapabilityProfile] = {}


_DATASET_TYPE_DEFAULTS: Dict[DatasetType, DatasetCapabilityProfile] = {
    DatasetType.AUDIO_TRANSCRIPTION: DatasetCapabilityProfile(
        name="generic_audio_transcription",
        dataset_type=DatasetType.AUDIO_TRANSCRIPTION,
        requires_audio=True,
        requires_text=False,
        supports_generation=False,
        evaluation_mode="transcription",
        recommended_pipeline_modes=("asr_only", "asr_text_retrieval"),
    ),
    DatasetType.AUDIO_QUERY_RETRIEVAL: DatasetCapabilityProfile(
        name="generic_audio_query_retrieval",
        dataset_type=DatasetType.AUDIO_QUERY_RETRIEVAL,
        requires_audio=True,
        requires_text=False,
        supports_generation=False,
        evaluation_mode="retrieval",
        recommended_pipeline_modes=("audio_emb_retrieval", "audio_text_retrieval"),
    ),
    DatasetType.TEXT_QUERY_RETRIEVAL: DatasetCapabilityProfile(
        name="generic_text_query_retrieval",
        dataset_type=DatasetType.TEXT_QUERY_RETRIEVAL,
        requires_audio=False,
        requires_text=True,
        supports_generation=False,
        evaluation_mode="retrieval",
        recommended_pipeline_modes=("asr_text_retrieval", "audio_text_retrieval"),
    ),
    DatasetType.QUESTION_ANSWERING: DatasetCapabilityProfile(
        name="generic_qa",
        dataset_type=DatasetType.QUESTION_ANSWERING,
        requires_audio=False,
        requires_text=True,
        supports_generation=True,
        evaluation_mode="qa",
        recommended_pipeline_modes=("asr_text_retrieval", "audio_text_retrieval"),
    ),
    DatasetType.MULTIMODAL_QA: DatasetCapabilityProfile(
        name="generic_multimodal_qa",
        dataset_type=DatasetType.MULTIMODAL_QA,
        requires_audio=True,
        requires_text=True,
        supports_generation=True,
        evaluation_mode="qa",
        recommended_pipeline_modes=("asr_text_retrieval", "audio_text_retrieval"),
    ),
    DatasetType.PASSAGE_RANKING: DatasetCapabilityProfile(
        name="generic_passage_ranking",
        dataset_type=DatasetType.PASSAGE_RANKING,
        requires_audio=False,
        requires_text=True,
        supports_generation=False,
        evaluation_mode="ranking",
        recommended_pipeline_modes=("asr_text_retrieval",),
    ),
}


def _descriptor_to_profile(desc: Any) -> DatasetCapabilityProfile:
    """Convert a DatasetDescriptor to a DatasetCapabilityProfile."""
    return DatasetCapabilityProfile(
        name=desc.id,
        dataset_type=desc.dataset_type,
        requires_audio=desc.requires_audio,
        requires_text=desc.requires_text,
        supports_generation=desc.supports_generation,
        evaluation_mode=desc.evaluation_mode,
        recommended_pipeline_modes=tuple(desc.compatible_pipeline_modes),
    )


def resolve_dataset_profile(
    dataset_name: str,
    dataset_type: Optional[DatasetType] = None,
) -> DatasetCapabilityProfile:
    """Resolve dataset profile by name with fallback to dataset_type.

    Checks the descriptor registry first (so descriptors registered via
    :func:`register_dataset` are visible here too), then the local profile
    registry, then type-based defaults, then a generic fallback.
    """
    # Descriptor registry has priority (single source of truth for known datasets)
    try:
        from .descriptor import get_descriptor
        desc = get_descriptor(dataset_name)
        if desc is not None:
            return _descriptor_to_profile(desc)
    except Exception as exc:
        logger.debug("descriptor lookup failed for %r: %s", dataset_name, exc)

    # Local profile registry (populated via register_dataset_profile)
    if dataset_name in _PROFILE_REGISTRY:
        return _PROFILE_REGISTRY[dataset_name]

    # Type-based fallback
    if dataset_type is not None and dataset_type in _DATASET_TYPE_DEFAULTS:
        return _DATASET_TYPE_DEFAULTS[dataset_type]

    # Generic fallback — preserves "generic:name" for unknown datasets
    return DatasetCapabilityProfile(
        name=f"generic:{dataset_name}",
        dataset_type=dataset_type or DatasetType.AUDIO_QUERY_RETRIEVAL,
        requires_audio=True,
        requires_text=False,
        supports_generation=False,
        evaluation_mode="retrieval",
        recommended_pipeline_modes=("audio_emb_retrieval", "asr_text_retrieval"),
    )


def list_known_dataset_names() -> List[str]:
    """Return known dataset names from both registries."""
    try:
        from .descriptor import list_registered_datasets
        names = set(list_registered_datasets())
    except Exception:
        names = set()
    names.update(_PROFILE_REGISTRY.keys())
    return sorted(names)
