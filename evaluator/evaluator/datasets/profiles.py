"""Dataset capability profiles for pipeline planning."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..config.types import DatasetType


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


_PROFILE_REGISTRY: Dict[str, DatasetCapabilityProfile] = {
    "admed_voice": DatasetCapabilityProfile(
        name="admed_voice",
        dataset_type=DatasetType.AUDIO_TRANSCRIPTION,
        requires_audio=True,
        requires_text=False,
        supports_generation=False,
        evaluation_mode="transcription",
        recommended_pipeline_modes=("asr_only", "asr_text_retrieval"),
    ),
    "pubmed_qa": DatasetCapabilityProfile(
        name="pubmed_qa",
        dataset_type=DatasetType.MULTIMODAL_QA,
        requires_audio=True,
        requires_text=True,
        supports_generation=True,
        evaluation_mode="qa_retrieval",
        recommended_pipeline_modes=("asr_text_retrieval", "audio_text_retrieval"),
    ),
}


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


def resolve_dataset_profile(
    dataset_name: str,
    dataset_type: Optional[DatasetType] = None,
) -> DatasetCapabilityProfile:
    """Resolve dataset profile by concrete dataset name, fallback to dataset_type."""
    if dataset_name in _PROFILE_REGISTRY:
        return _PROFILE_REGISTRY[dataset_name]
    if dataset_type is not None and dataset_type in _DATASET_TYPE_DEFAULTS:
        return _DATASET_TYPE_DEFAULTS[dataset_type]
    return DatasetCapabilityProfile(
        name=f"generic:{dataset_name}",
        dataset_type=dataset_type or DatasetType.AUDIO_QUERY_RETRIEVAL,
        requires_audio=True,
        requires_text=False,
        supports_generation=False,
        evaluation_mode="retrieval",
        recommended_pipeline_modes=("audio_emb_retrieval", "asr_text_retrieval"),
    )


def register_dataset_profile(profile: DatasetCapabilityProfile) -> None:
    """Register a new dataset capability profile."""
    _PROFILE_REGISTRY[profile.name] = profile


def list_known_dataset_names() -> List[str]:
    """Return known concrete dataset names registered in capability profiles."""
    return sorted(_PROFILE_REGISTRY.keys())
