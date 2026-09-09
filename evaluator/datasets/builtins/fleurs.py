"""Google FLEURS — 102-language ASR benchmark (audio transcription)."""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from ..types import AudioTranscriptionDataset, register_eval_dataset
from ...config.dataset_descriptors import resolve_split
from ..loaders import factory
from ._common import audio_query_dataset

if TYPE_CHECKING:
    from ...config.data import DataConfig
    from ..core import QueryDataset


@register_eval_dataset(
    id="fleurs",
    description="Google FLEURS: 102-language ASR benchmark (requires huggingface_subset)",
)
class FleursDataset(AudioTranscriptionDataset):
    """FLEURS streams all 102 languages without a subset, so require one."""

    domain = "multilingual"
    required_data_fields = ("huggingface_subset",)
    splits = ("train", "validation", "test")
    default_split = "test"

    @classmethod
    def validate(cls, data: "DataConfig") -> List[str]:
        errors: List[str] = []
        if not getattr(data, "huggingface_subset", None):
            errors.append(
                "FLEURS requires huggingface_subset (e.g. 'pl_pl'). "
                "Without it the loader would attempt to stream all 102 languages."
            )
        return errors

    @classmethod
    def from_config(cls, data: "DataConfig") -> "QueryDataset":
        subset = getattr(data, "huggingface_subset", None) or ""
        # Derive Whisper language code from subset tag (e.g. "pl_pl" → "pl")
        derived_lang = subset.split("_")[0] if subset else "en"
        default_language = getattr(data, "default_language", None) or derived_lang

        loader = factory.create_dataset_loader(
            source="huggingface",
            huggingface_dataset="google/fleurs",
            huggingface_subset=subset or None,
            huggingface_split=resolve_split(data, cls.default_split),
            column_mapping=getattr(data, "column_mapping", None),
            max_samples=getattr(data, "max_samples", None),
            default_language=default_language,
        )
        return audio_query_dataset(loader, data)
