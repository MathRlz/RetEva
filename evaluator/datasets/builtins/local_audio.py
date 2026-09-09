"""Local audio directory dataset (audio-query retrieval; corpus optional)."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import AudioRetrievalDataset, register_eval_dataset
from ..loaders import factory
from ._common import audio_query_dataset

if TYPE_CHECKING:
    from ...config.data import DataConfig
    from ..core import QueryDataset


@register_eval_dataset(
    id="local",
    description="Local audio directory with optional corpus",
)
class LocalAudioDataset(AudioRetrievalDataset):
    """Audio query from a local directory; corpus optional, so text not required."""

    requires_text = False
    required_data_fields = ("audio_dir",)

    @classmethod
    def from_config(cls, data: "DataConfig") -> "QueryDataset":
        loader = factory.create_dataset_loader(
            source="local",
            audio_dir=data.audio_dir,
            transcripts_file=getattr(data, "transcripts_file", None),
            column_mapping=getattr(data, "column_mapping", None),
            max_samples=getattr(data, "max_samples", None),
            default_language=getattr(data, "default_language", "en"),
            sample_rate=getattr(data, "sample_rate", None),
        )
        return audio_query_dataset(loader, data)
