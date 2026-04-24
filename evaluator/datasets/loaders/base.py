"""Base classes and protocols for dataset loaders."""

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Union

import numpy as np


@dataclass
class AudioSample:
    """Standardized audio sample returned by all dataset loaders.
    
    Attributes:
        audio_array: Float32 waveform numpy array.
        sampling_rate: Audio sampling rate in Hz.
        transcription: Ground truth text transcription.
        sample_id: Unique identifier for this sample.
        language: Language code (e.g., "en", "pl").
        metadata: Additional sample-specific metadata.
    """
    audio_array: np.ndarray
    sampling_rate: int
    transcription: str
    sample_id: str = ""
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and convert audio array to float32."""
        if not isinstance(self.audio_array, np.ndarray):
            self.audio_array = np.array(self.audio_array, dtype=np.float32)
        elif self.audio_array.dtype != np.float32:
            self.audio_array = self.audio_array.astype(np.float32)


class DatasetLoaderProtocol(Protocol):
    """Protocol defining the interface for dataset loaders.
    
    All dataset loaders must implement this interface to ensure
    consistent behavior across different data sources.
    """
    
    def load(self) -> List[AudioSample]:
        """Load all samples from the dataset.
        
        Returns:
            List of AudioSample objects.
        """
        ...
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...
    
    def __iter__(self) -> Iterator[AudioSample]:
        """Iterate over samples in the dataset."""
        ...


class GenericDatasetLoader:
    """Generic dataset loader for custom dataset formats.
    
    This loader wraps any iterable data source and applies
    a mapping function to convert items to AudioSample format.
    
    Example:
        >>> def map_fn(item):
        ...     return AudioSample(
        ...         audio_array=item["audio"],
        ...         sampling_rate=16000,
        ...         transcription=item["text"],
        ...         sample_id=item["id"]
        ...     )
        >>> loader = GenericDatasetLoader(my_data, map_fn)
        >>> for sample in loader:
        ...     print(sample.transcription)
    """
    
    def __init__(
        self,
        data: Union[List[Any], Iterator[Any]],
        map_fn: Callable[[Any], AudioSample],
        filter_fn: Optional[Callable[[Any], bool]] = None,
    ):
        """Initialize the generic loader.
        
        Args:
            data: Source data (list or iterator of items).
            map_fn: Function to convert each item to AudioSample.
            filter_fn: Optional function to filter items before mapping.
        """
        self._data = list(data) if not isinstance(data, list) else data
        self._map_fn = map_fn
        self._filter_fn = filter_fn
        self._samples: Optional[List[AudioSample]] = None
    
    def _load_samples(self) -> List[AudioSample]:
        """Load and cache all samples."""
        if self._samples is not None:
            return self._samples
        
        items = self._data
        if self._filter_fn is not None:
            items = [item for item in items if self._filter_fn(item)]
        
        self._samples = [self._map_fn(item) for item in items]
        return self._samples
    
    def load(self) -> List[AudioSample]:
        """Load all samples from the dataset.
        
        Returns:
            List of AudioSample objects.
        """
        return self._load_samples()
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._load_samples())
    
    def __iter__(self) -> Iterator[AudioSample]:
        """Iterate over samples."""
        return iter(self._load_samples())
    
    def __getitem__(self, idx: int) -> AudioSample:
        """Get a sample by index."""
        return self._load_samples()[idx]
