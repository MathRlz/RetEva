"""Data configuration."""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Union

from ..config.types import DatasetType, to_enum


@dataclass
class DataConfig:
    """Configuration for dataset loading and processing: sources, paths, processing
    options, and batch settings. Field comments below carry per-field semantics."""

    dataset_name: str = "admed_voice"
    questions_path: Optional[str] = None
    corpus_path: Optional[str] = None
    prepared_dataset_dir: Optional[str] = None
    # Unset: datasets that read it fall back to their own default (env var / data dir).
    data_path: Optional[str] = None
    strict_validation: bool = True
    trace_limit: int = 0  # 0 = no limit
    batch_size: int = 32
    # Estimate the ASR batch from a one-sample GPU warm-up instead of using `batch_size`
    # statically. Opt-in; no-op on CPU; doesn't change outputs.
    warmup_batch_sizing: bool = False
    num_workers: int = 0
    test_size: float = 0.0  # 0.0 = no split
    # Dataset-dependent (train/test/both/validation/...); None = the descriptor's
    # default_split, "both" = union of all declared splits.
    split: Optional[str] = None

    # Dataset loader configuration
    dataset_source: str = "local"  # local, huggingface, custom
    huggingface_dataset: Optional[str] = None
    huggingface_subset: Optional[str] = None
    huggingface_split: str = "test"
    audio_dir: Optional[str] = None
    transcripts_file: Optional[str] = None
    column_mapping: Optional[Dict[str, str]] = None
    default_language: str = "en"
    max_samples: Optional[int] = None
    sample_rate: Optional[int] = None

    # Dataset type for automatic metric selection
    dataset_type: Optional[Union[str, DatasetType]] = None

    # Multi-dataset graphs: id -> per-source spec (DataConfig field names + optional
    # `role` corpus/questions/both). A dataset_source node references an entry via its
    # `dataset` param. None/empty = single-dataset mode (the scalar fields above are used).
    datasets: Optional[Dict[str, Dict[str, Any]]] = field(default=None)

    def __post_init__(self):
        """Normalize dataset_type to enum if provided."""
        if self.dataset_type is not None and isinstance(self.dataset_type, str):
            self.dataset_type = to_enum(self.dataset_type, DatasetType)
