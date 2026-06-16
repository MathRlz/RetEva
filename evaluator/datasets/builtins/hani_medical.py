"""Hani89 medical ASR recordings (HuggingFace) — audio transcription, self-retrieval."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import AudioTranscriptionDataset, register_eval_dataset

if TYPE_CHECKING:
    from ...config.data import DataConfig
    from ..core import QueryDataset


@register_eval_dataset(
    id="hani_medical",
    description="Hani89 medical ASR recordings (HF) — audio + medical transcriptions",
)
class HaniMedicalDataset(AudioTranscriptionDataset):
    """HF-backed medical ASR recordings (audio + ``sentence`` transcription). The retrieval
    corpus is derived from the transcriptions (self-retrieval), like admed_voice."""

    domain = "medical"
    splits = ("train", "test")
    default_split = "train"

    @classmethod
    def from_config(cls, data: "DataConfig") -> "QueryDataset":
        from ..loaders.factory import create_dataset_loader
        from ..runtime import AudioSamplesQueryDataset

        loader = create_dataset_loader(
            source="huggingface",
            huggingface_dataset="Hani89/medical_asr_recording_dataset",
            huggingface_split=getattr(data, "huggingface_split", None) or cls.default_split,
            # pin the dataset's columns (audio dict + `sentence` text); config may override
            column_mapping=getattr(data, "column_mapping", None)
            or {"audio": "audio", "transcription": "sentence"},
            max_samples=getattr(data, "max_samples", None),
            default_language=getattr(data, "default_language", "en"),
        )
        return AudioSamplesQueryDataset(
            loader.load(), trace_limit=getattr(data, "trace_limit", 0)
        )
