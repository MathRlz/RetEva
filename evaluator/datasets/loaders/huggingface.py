"""HuggingFace Hub dataset loader for speech datasets."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .base import AudioSample


# Well-known HuggingFace speech dataset column mappings
KNOWN_DATASET_MAPPINGS: Dict[str, Dict[str, str]] = {
    "mozilla-foundation/common_voice_11_0": {
        "audio": "audio",
        "transcription": "sentence",
        "sample_id": "path",
        "language": "locale",
    },
    "mozilla-foundation/common_voice_13_0": {
        "audio": "audio",
        "transcription": "sentence",
        "sample_id": "path",
        "language": "locale",
    },
    "mozilla-foundation/common_voice_16_1": {
        "audio": "audio",
        "transcription": "sentence",
        "sample_id": "path",
        "language": "locale",
    },
    "google/fleurs": {
        "audio": "audio",
        "transcription": "transcription",
        "sample_id": "id",
        "language": "language",
    },
    "facebook/multilingual_librispeech": {
        "audio": "audio",
        "transcription": "text",
        "sample_id": "id",
        "language": None,  # Language determined by subset
    },
    "librispeech_asr": {
        "audio": "audio",
        "transcription": "text",
        "sample_id": "id",
        "language": None,  # English only
    },
    "superb": {
        "audio": "audio",
        "transcription": "text",
        "sample_id": "id",
        "language": None,
    },
    "openslr/librispeech_asr": {
        "audio": "audio",
        "transcription": "text",
        "sample_id": "id",
        "language": None,
    },
    # Medical speech datasets
    "speech-recognition-community-v2/medical_speech": {
        "audio": "audio",
        "transcription": "text",
        "sample_id": "id",
        "language": None,
    },
}

# Default mapping for unknown datasets
DEFAULT_COLUMN_MAPPING = {
    "audio": "audio",
    "transcription": "text",
    "sample_id": "id",
    "language": None,
}


@dataclass
class HuggingFaceDatasetLoader:
    """Dataset loader for HuggingFace Hub audio datasets.
    
    Supports common speech recognition datasets and automatically
    maps HuggingFace dataset columns to the expected AudioSample format.
    
    Example:
        >>> loader = HuggingFaceDatasetLoader(
        ...     dataset_name="mozilla-foundation/common_voice_11_0",
        ...     subset="en",
        ...     split="test",
        ... )
        >>> samples = loader.load()
        >>> print(len(samples))
    
    Attributes:
        dataset_name: HuggingFace dataset identifier.
        subset: Dataset configuration/subset name (e.g., language code).
        split: Dataset split to load (train, validation, test).
        column_mapping: Custom column name mapping (overrides auto-detection).
        default_language: Default language code if not in dataset.
        max_samples: Maximum number of samples to load (None for all).
        streaming: Whether to use streaming mode (memory efficient).
        trust_remote_code: Whether to trust remote code when loading.
    """
    dataset_name: str
    subset: Optional[str] = None
    split: str = "test"
    column_mapping: Optional[Dict[str, str]] = None
    default_language: str = "en"
    max_samples: Optional[int] = None
    streaming: bool = False
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None
    
    _dataset: Any = field(default=None, repr=False, init=False)
    _samples: Optional[List[AudioSample]] = field(default=None, repr=False, init=False)
    
    def _get_column_mapping(self) -> Dict[str, Optional[str]]:
        """Get column mapping for the dataset.
        
        Returns:
            Dictionary mapping standard names to dataset column names.
        """
        if self.column_mapping is not None:
            mapping = DEFAULT_COLUMN_MAPPING.copy()
            mapping.update(self.column_mapping)
            return mapping
        
        # Check known datasets
        for known_name, known_mapping in KNOWN_DATASET_MAPPINGS.items():
            if self.dataset_name.startswith(known_name) or known_name in self.dataset_name:
                return known_mapping
        
        return DEFAULT_COLUMN_MAPPING
    
    def _load_hf_dataset(self) -> Any:
        """Load the HuggingFace dataset."""
        if self._dataset is not None:
            return self._dataset
        
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "HuggingFace datasets library is required for HuggingFaceDatasetLoader. "
                "Install it with: pip install datasets"
            ) from e
        
        load_kwargs: Dict[str, Any] = {
            "path": self.dataset_name,
            "split": self.split,
            "streaming": self.streaming,
            "trust_remote_code": self.trust_remote_code,
        }
        
        if self.subset is not None:
            load_kwargs["name"] = self.subset
        
        if self.cache_dir is not None:
            load_kwargs["cache_dir"] = self.cache_dir
        
        self._dataset = load_dataset(**load_kwargs)
        return self._dataset
    
    def _extract_audio(self, item: Dict[str, Any], audio_col: str) -> tuple:
        """Extract audio array and sample rate from dataset item.
        
        Returns:
            Tuple of (audio_array, sampling_rate).
        """
        audio_data = item.get(audio_col)
        
        if audio_data is None:
            raise ValueError(f"Audio column '{audio_col}' not found in item")
        
        # HuggingFace audio format: {"array": np.ndarray, "sampling_rate": int}
        if isinstance(audio_data, dict):
            audio_array = audio_data.get("array")
            sampling_rate = audio_data.get("sampling_rate", 16000)
            
            if audio_array is None:
                raise ValueError(f"Audio data missing 'array' key")
            
            return np.array(audio_array, dtype=np.float32), sampling_rate
        
        # Fallback: assume raw numpy array with default sample rate
        return np.array(audio_data, dtype=np.float32), 16000
    
    def _map_item(self, item: Dict[str, Any], idx: int) -> AudioSample:
        """Map a HuggingFace dataset item to AudioSample.
        
        Args:
            item: Dataset item dictionary.
            idx: Item index (used as fallback sample_id).
            
        Returns:
            AudioSample instance.
        """
        mapping = self._get_column_mapping()
        
        # Extract audio
        audio_col = mapping.get("audio", "audio")
        audio_array, sampling_rate = self._extract_audio(item, audio_col)
        
        # Extract transcription
        trans_col = mapping.get("transcription", "text")
        transcription = str(item.get(trans_col, ""))
        
        # Extract sample ID
        id_col = mapping.get("sample_id")
        if id_col and id_col in item:
            sample_id = str(item[id_col])
        else:
            sample_id = f"{self.dataset_name}_{idx}"
        
        # Extract language
        lang_col = mapping.get("language")
        if lang_col and lang_col in item:
            language = str(item[lang_col])
        else:
            language = self.subset if self.subset else self.default_language
        
        # Collect remaining metadata
        metadata = {
            k: v for k, v in item.items() 
            if k not in [audio_col, trans_col, id_col, lang_col]
            and not isinstance(v, (np.ndarray, bytes))
        }
        
        return AudioSample(
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            transcription=transcription,
            sample_id=sample_id,
            language=language,
            metadata=metadata,
        )
    
    def load(self) -> List[AudioSample]:
        """Load all samples from the HuggingFace dataset.
        
        Returns:
            List of AudioSample objects.
        """
        if self._samples is not None:
            return self._samples
        
        dataset = self._load_hf_dataset()
        samples: List[AudioSample] = []
        
        if self.streaming:
            for idx, item in enumerate(dataset):
                if self.max_samples is not None and idx >= self.max_samples:
                    break
                samples.append(self._map_item(item, idx))
        else:
            n_samples = len(dataset) if self.max_samples is None else min(len(dataset), self.max_samples)
            for idx in range(n_samples):
                samples.append(self._map_item(dataset[idx], idx))
        
        self._samples = samples
        return self._samples
    
    def __len__(self) -> int:
        """Return the number of samples."""
        if self._samples is not None:
            return len(self._samples)
        
        dataset = self._load_hf_dataset()
        
        if self.streaming:
            # For streaming datasets, we need to load to know the count
            return len(self.load())
        
        total = len(dataset)
        if self.max_samples is not None:
            return min(total, self.max_samples)
        return total
    
    def __iter__(self) -> Iterator[AudioSample]:
        """Iterate over samples."""
        return iter(self.load())
    
    def __getitem__(self, idx: int) -> AudioSample:
        """Get a sample by index."""
        return self.load()[idx]
    
    @classmethod
    def list_known_datasets(cls) -> List[str]:
        """List known HuggingFace speech datasets with built-in mappings.
        
        Returns:
            List of dataset identifiers.
        """
        return list(KNOWN_DATASET_MAPPINGS.keys())
