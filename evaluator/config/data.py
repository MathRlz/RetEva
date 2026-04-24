"""Data configuration."""
from dataclasses import dataclass
from typing import Optional, Dict, Union

from ..config.types import DatasetType, to_enum


@dataclass
class DataConfig:
    """
    Configuration for dataset loading and processing.
    
    Specifies dataset sources, paths, processing options, and batch settings.
    Supports multiple dataset sources including local files, HuggingFace datasets,
    and custom loaders.
    
    Attributes:
        dataset_name: Name/identifier of the dataset. Default: "admed_voice".
        questions_path: Path to questions/queries file. Optional.
        corpus_path: Path to corpus/documents file. Optional.
        prepared_dataset_dir: Path to pre-prepared dataset directory. Optional.
        strict_validation: Enforce strict dataset validation. Default: True.
        trace_limit: Limit number of samples (0 = no limit). Default: 0.
        batch_size: Batch size for processing. Default: 32.
        num_workers: Number of data loading workers. Default: 0.
        test_size: Fraction of data for test split (0.0 = no split). Default: 0.0.
        
        dataset_source: Source type. Default: "local".
            Options: "local", "huggingface", "custom".
        huggingface_dataset: HuggingFace dataset identifier. Optional.
        huggingface_subset: HuggingFace dataset subset/config. Optional.
        huggingface_split: HuggingFace dataset split. Default: "test".
        audio_dir: Directory containing audio files. Optional.
        transcripts_file: Path to transcripts file. Optional.
        column_mapping: Custom column name mapping. Optional.
        default_language: Default language code. Default: "en".
        max_samples: Maximum samples to load. Optional.
        sample_rate: Target sample rate for resampling. Optional.
    
    Examples:
        >>> config = DataConfig(
        ...     dataset_name="pubmed",
        ...     batch_size=16,
        ...     trace_limit=100
        ... )
        >>> # Load from HuggingFace
        >>> config = DataConfig(
        ...     dataset_source="huggingface",
        ...     huggingface_dataset="mozilla-foundation/common_voice_13_0",
        ...     huggingface_subset="en"
        ... )
    """
    dataset_name: str = "admed_voice"
    questions_path: Optional[str] = None
    corpus_path: Optional[str] = None
    prepared_dataset_dir: Optional[str] = None
    strict_validation: bool = True
    trace_limit: int = 0
    batch_size: int = 32
    num_workers: int = 0
    test_size: float = 0.0
    
    # Dataset loader configuration
    dataset_source: str = "local"  # local, huggingface, custom
    huggingface_dataset: Optional[str] = None  # HF dataset identifier
    huggingface_subset: Optional[str] = None  # HF dataset subset/config
    huggingface_split: str = "test"  # HF dataset split
    audio_dir: Optional[str] = None  # Local audio directory path
    transcripts_file: Optional[str] = None  # Path to transcripts file
    column_mapping: Optional[Dict[str, str]] = None  # Custom column mapping
    default_language: str = "en"  # Default language code
    max_samples: Optional[int] = None  # Maximum samples to load
    sample_rate: Optional[int] = None  # Target sample rate for resampling
    
    # Dataset type for automatic metric selection
    dataset_type: Optional[Union[str, DatasetType]] = None
    
    def __post_init__(self):
        """Normalize dataset_type to enum if provided."""
        if self.dataset_type is not None and isinstance(self.dataset_type, str):
            self.dataset_type = to_enum(self.dataset_type, DatasetType)
