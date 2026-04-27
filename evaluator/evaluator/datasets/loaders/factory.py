"""Factory function for creating dataset loaders."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import AudioSample, DatasetLoaderProtocol, GenericDatasetLoader
from .huggingface import HuggingFaceDatasetLoader
from .local import LocalAudioDatasetLoader


def create_dataset_loader(
    source: str = "local",
    *,
    # Local loader options
    audio_dir: Optional[Union[str, Path]] = None,
    transcripts_file: Optional[Union[str, Path]] = None,
    recursive: bool = False,
    file_pattern: Optional[str] = None,
    # HuggingFace loader options
    huggingface_dataset: Optional[str] = None,
    huggingface_subset: Optional[str] = None,
    huggingface_split: str = "test",
    column_mapping: Optional[Dict[str, str]] = None,
    max_samples: Optional[int] = None,
    streaming: bool = False,
    trust_remote_code: bool = False,
    cache_dir: Optional[str] = None,
    # Common options
    default_language: str = "en",
    sample_rate: Optional[int] = None,
    **kwargs: Any,
) -> DatasetLoaderProtocol:
    """Create a dataset loader based on the specified source type.
    
    This factory function creates the appropriate loader based on the
    source parameter and provided options.
    
    Args:
        source: Data source type. One of:
            - "local": Load from local audio directory
            - "huggingface": Load from HuggingFace Hub
            - "custom": Use GenericDatasetLoader with provided data
        
        # Local loader options:
        audio_dir: Path to directory containing audio files.
        transcripts_file: Optional path to transcripts file.
        recursive: Whether to search subdirectories.
        file_pattern: Optional glob pattern to filter files.
        
        # HuggingFace loader options:
        huggingface_dataset: HuggingFace dataset identifier.
        huggingface_subset: Dataset subset/configuration name.
        huggingface_split: Dataset split to load.
        column_mapping: Custom column name mapping.
        max_samples: Maximum number of samples to load.
        streaming: Whether to use streaming mode.
        trust_remote_code: Whether to trust remote code.
        cache_dir: Cache directory for HuggingFace datasets.
        
        # Common options:
        default_language: Default language code.
        sample_rate: Target sample rate for resampling.
        **kwargs: Additional keyword arguments.
        
    Returns:
        A dataset loader implementing DatasetLoaderProtocol.
        
    Raises:
        ValueError: If required options are missing for the source type.
        
    Example:
        >>> # Load from local directory
        >>> loader = create_dataset_loader(
        ...     source="local",
        ...     audio_dir="./data/audio",
        ...     default_language="en"
        ... )
        
        >>> # Load from HuggingFace
        >>> loader = create_dataset_loader(
        ...     source="huggingface",
        ...     huggingface_dataset="mozilla-foundation/common_voice_11_0",
        ...     huggingface_subset="en",
        ...     huggingface_split="test",
        ...     max_samples=100
        ... )
    """
    source = source.lower()
    
    if source == "local":
        if audio_dir is None:
            raise ValueError(
                "audio_dir is required for local dataset source. "
                "Provide the path to the directory containing audio files."
            )
        
        return LocalAudioDatasetLoader(
            audio_dir=audio_dir,
            transcripts_file=transcripts_file,
            default_language=default_language,
            sample_rate=sample_rate,
            recursive=recursive,
            file_pattern=file_pattern,
        )
    
    elif source == "huggingface":
        if huggingface_dataset is None:
            raise ValueError(
                "huggingface_dataset is required for HuggingFace source. "
                "Provide the dataset identifier (e.g., 'mozilla-foundation/common_voice_11_0')."
            )
        
        return HuggingFaceDatasetLoader(
            dataset_name=huggingface_dataset,
            subset=huggingface_subset,
            split=huggingface_split,
            column_mapping=column_mapping,
            default_language=default_language,
            max_samples=max_samples,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
    
    elif source == "custom":
        # For custom source, expect data and map_fn in kwargs
        data = kwargs.get("data")
        map_fn = kwargs.get("map_fn")
        filter_fn = kwargs.get("filter_fn")
        
        if data is None or map_fn is None:
            raise ValueError(
                "For custom dataset source, 'data' and 'map_fn' must be provided. "
                "data: iterable of items, map_fn: function to convert items to AudioSample."
            )
        
        return GenericDatasetLoader(
            data=data,
            map_fn=map_fn,
            filter_fn=filter_fn,
        )
    
    else:
        raise ValueError(
            f"Unknown dataset source: '{source}'. "
            f"Supported sources: 'local', 'huggingface', 'custom'."
        )
