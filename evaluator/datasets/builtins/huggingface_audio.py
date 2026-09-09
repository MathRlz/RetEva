"""Generic HuggingFace Hub audio dataset (audio-query retrieval; corpus optional)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import AudioRetrievalDataset, register_eval_dataset
from ...config.dataset_descriptors import resolve_split
from ..loaders import factory
from ._common import audio_query_dataset

if TYPE_CHECKING:
    from ...config.data import DataConfig
    from ..core import QueryDataset


@register_eval_dataset(
    id="huggingface",
    description="HuggingFace Hub audio dataset",
)
class HuggingFaceAudioDataset(AudioRetrievalDataset):
    """Audio query from a HuggingFace Hub dataset; corpus optional."""

    requires_text = False
    required_data_fields = ("huggingface_dataset",)
    splits = ("train", "validation", "test")
    default_split = "test"

    @classmethod
    def from_config(cls, data: "DataConfig") -> "QueryDataset":
        loader = factory.create_dataset_loader(
            source="huggingface",
            huggingface_dataset=data.huggingface_dataset,
            huggingface_subset=getattr(data, "huggingface_subset", None),
            huggingface_split=resolve_split(data, cls.default_split),
            column_mapping=getattr(data, "column_mapping", None),
            max_samples=getattr(data, "max_samples", None),
            default_language=getattr(data, "default_language", "en"),
        )
        return audio_query_dataset(loader, data)
