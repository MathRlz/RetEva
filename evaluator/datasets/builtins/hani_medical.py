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
        from ..descriptor import resolve_split
        from ..loaders.factory import create_dataset_loader
        from ..runtime import AudioSamplesQueryDataset

        split = (resolve_split(data, cls.default_split) or cls.default_split).lower()
        # "both" = the union of every declared HF split; a named split loads just that one.
        hf_splits = cls.splits if split == "both" else (split,)
        column_mapping = getattr(data, "column_mapping", None) or {
            "audio": "audio",
            "transcription": "sentence",
        }
        samples = []
        for hf_split in hf_splits:
            loader = create_dataset_loader(
                source="huggingface",
                huggingface_dataset="Hani89/medical_asr_recording_dataset",
                huggingface_split=hf_split,
                column_mapping=column_mapping,
                max_samples=getattr(data, "max_samples", None),
                default_language=getattr(data, "default_language", "en"),
            )
            split_samples = loader.load()
            if len(hf_splits) > 1:
                # The HF loader restarts its positional sample_id at 0 per split, so a
                # multi-split union ("both") would collide ids — and duplicate question_ids
                # make dataset_source drop ALL ground truth. Namespace by split.
                for sm in split_samples:
                    sm.sample_id = f"{hf_split}:{sm.sample_id}"
            samples.extend(split_samples)
        return AudioSamplesQueryDataset(
            samples, trace_limit=getattr(data, "trace_limit", 0)
        )
