"""Base classes and protocols for dataset loaders."""

from dataclasses import dataclass, field
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
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[AudioSample]:
        ...


class GenericDatasetLoader:
    """Generic dataset loader for custom dataset formats.

    This loader wraps any iterable data source and applies
    a mapping function to convert items to AudioSample format.
    """

    def __init__(
        self,
        data: Union[List[Any], Iterator[Any]],
        map_fn: Callable[[Any], AudioSample],
        filter_fn: Optional[Callable[[Any], bool]] = None,
    ):
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
        return self._load_samples()

    def __len__(self) -> int:
        return len(self._load_samples())

    def __iter__(self) -> Iterator[AudioSample]:
        return iter(self._load_samples())

    def __getitem__(self, idx: int) -> AudioSample:
        return self._load_samples()[idx]
