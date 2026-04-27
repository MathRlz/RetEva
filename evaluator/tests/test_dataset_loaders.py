"""Tests for dataset loaders module."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evaluator.datasets.loaders.base import AudioSample, GenericDatasetLoader
from evaluator.datasets.loaders.huggingface import (
    HuggingFaceDatasetLoader,
    KNOWN_DATASET_MAPPINGS,
    DEFAULT_COLUMN_MAPPING,
)
from evaluator.datasets.loaders.local import LocalAudioDatasetLoader, SUPPORTED_AUDIO_FORMATS
from evaluator.datasets.loaders.factory import create_dataset_loader


class TestAudioSample:
    """Tests for AudioSample dataclass."""

    def test_basic_creation(self):
        """Test basic AudioSample creation."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        sample = AudioSample(
            audio_array=audio,
            sampling_rate=16000,
            transcription="test text",
            sample_id="sample_001",
            language="en",
        )
        
        assert sample.transcription == "test text"
        assert sample.sampling_rate == 16000
        assert sample.sample_id == "sample_001"
        assert sample.language == "en"
        assert sample.audio_array.dtype == np.float32

    def test_auto_convert_to_float32(self):
        """Test automatic conversion to float32."""
        audio = np.array([1, 2, 3], dtype=np.int16)
        sample = AudioSample(
            audio_array=audio,
            sampling_rate=16000,
            transcription="test",
        )
        
        assert sample.audio_array.dtype == np.float32

    def test_list_to_numpy(self):
        """Test conversion from list to numpy array."""
        sample = AudioSample(
            audio_array=[0.1, 0.2, 0.3],
            sampling_rate=16000,
            transcription="test",
        )
        
        assert isinstance(sample.audio_array, np.ndarray)
        assert sample.audio_array.dtype == np.float32

    def test_default_values(self):
        """Test default values."""
        sample = AudioSample(
            audio_array=np.zeros(100, dtype=np.float32),
            sampling_rate=16000,
            transcription="test",
        )
        
        assert sample.sample_id == ""
        assert sample.language == "en"
        assert sample.metadata == {}


class TestGenericDatasetLoader:
    """Tests for GenericDatasetLoader."""

    def test_basic_loading(self):
        """Test basic data loading with map function."""
        data = [
            {"audio": [0.1, 0.2], "text": "hello"},
            {"audio": [0.3, 0.4], "text": "world"},
        ]
        
        def map_fn(item):
            return AudioSample(
                audio_array=np.array(item["audio"], dtype=np.float32),
                sampling_rate=16000,
                transcription=item["text"],
            )
        
        loader = GenericDatasetLoader(data, map_fn)
        samples = loader.load()
        
        assert len(samples) == 2
        assert samples[0].transcription == "hello"
        assert samples[1].transcription == "world"

    def test_with_filter(self):
        """Test loading with filter function."""
        data = [
            {"audio": [0.1], "text": "short"},
            {"audio": [0.1, 0.2, 0.3, 0.4, 0.5], "text": "longer"},
        ]
        
        def map_fn(item):
            return AudioSample(
                audio_array=np.array(item["audio"], dtype=np.float32),
                sampling_rate=16000,
                transcription=item["text"],
            )
        
        def filter_fn(item):
            return len(item["audio"]) > 2
        
        loader = GenericDatasetLoader(data, map_fn, filter_fn)
        samples = loader.load()
        
        assert len(samples) == 1
        assert samples[0].transcription == "longer"

    def test_len(self):
        """Test __len__ method."""
        data = [{"a": 1}, {"a": 2}, {"a": 3}]
        map_fn = lambda x: AudioSample(np.zeros(10), 16000, str(x["a"]))
        
        loader = GenericDatasetLoader(data, map_fn)
        assert len(loader) == 3

    def test_iter(self):
        """Test __iter__ method."""
        data = [{"text": "one"}, {"text": "two"}]
        map_fn = lambda x: AudioSample(np.zeros(10), 16000, x["text"])
        
        loader = GenericDatasetLoader(data, map_fn)
        texts = [s.transcription for s in loader]
        
        assert texts == ["one", "two"]

    def test_getitem(self):
        """Test __getitem__ method."""
        data = [{"text": "one"}, {"text": "two"}, {"text": "three"}]
        map_fn = lambda x: AudioSample(np.zeros(10), 16000, x["text"])
        
        loader = GenericDatasetLoader(data, map_fn)
        
        assert loader[0].transcription == "one"
        assert loader[2].transcription == "three"

    def test_caching(self):
        """Test that samples are cached after first load."""
        data = [{"text": "test"}]
        call_count = 0
        
        def map_fn(x):
            nonlocal call_count
            call_count += 1
            return AudioSample(np.zeros(10), 16000, x["text"])
        
        loader = GenericDatasetLoader(data, map_fn)
        loader.load()
        loader.load()
        
        assert call_count == 1


class TestHuggingFaceDatasetLoader:
    """Tests for HuggingFaceDatasetLoader."""

    def test_known_dataset_mappings(self):
        """Test that known datasets have correct mappings."""
        assert "mozilla-foundation/common_voice_11_0" in KNOWN_DATASET_MAPPINGS
        assert "google/fleurs" in KNOWN_DATASET_MAPPINGS
        
        cv_mapping = KNOWN_DATASET_MAPPINGS["mozilla-foundation/common_voice_11_0"]
        assert cv_mapping["audio"] == "audio"
        assert cv_mapping["transcription"] == "sentence"

    def test_list_known_datasets(self):
        """Test listing known datasets."""
        known = HuggingFaceDatasetLoader.list_known_datasets()
        assert isinstance(known, list)
        assert len(known) > 0
        assert "mozilla-foundation/common_voice_11_0" in known

    def test_column_mapping_override(self):
        """Test custom column mapping override."""
        loader = HuggingFaceDatasetLoader(
            dataset_name="test/dataset",
            column_mapping={"audio": "sound", "transcription": "label"},
        )
        
        mapping = loader._get_column_mapping()
        assert mapping["audio"] == "sound"
        assert mapping["transcription"] == "label"

    def test_extract_audio_dict_format(self):
        """Test extracting audio from HuggingFace dict format."""
        loader = HuggingFaceDatasetLoader(dataset_name="test/dataset")
        
        item = {
            "audio": {
                "array": np.array([0.1, 0.2, 0.3]),
                "sampling_rate": 22050,
            }
        }
        
        audio, sr = loader._extract_audio(item, "audio")
        
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == 22050

    def test_extract_audio_default_sample_rate(self):
        """Test default sample rate when not specified."""
        loader = HuggingFaceDatasetLoader(dataset_name="test/dataset")
        
        item = {"audio": {"array": np.array([0.1, 0.2])}}
        
        audio, sr = loader._extract_audio(item, "audio")
        assert sr == 16000

    def test_extract_audio_missing_column(self):
        """Test error when audio column is missing."""
        loader = HuggingFaceDatasetLoader(dataset_name="test/dataset")
        
        with pytest.raises(ValueError, match="Audio column"):
            loader._extract_audio({}, "audio")

    def test_map_item(self):
        """Test mapping a single item to AudioSample."""
        loader = HuggingFaceDatasetLoader(
            dataset_name="test/dataset",
            default_language="de",
        )
        
        item = {
            "audio": {"array": np.array([0.1, 0.2]), "sampling_rate": 16000},
            "text": "hello world",
            "id": "sample_123",
        }
        
        sample = loader._map_item(item, 0)
        
        assert isinstance(sample, AudioSample)
        assert sample.transcription == "hello world"
        assert sample.sample_id == "sample_123"
        assert sample.language == "de"

    @patch("datasets.load_dataset")
    def test_load_calls_hf_load_dataset(self, mock_load_dataset):
        """Test that load() calls HuggingFace load_dataset."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))
        mock_load_dataset.return_value = mock_dataset
        
        loader = HuggingFaceDatasetLoader(
            dataset_name="test/dataset",
            subset="en",
            split="train",
        )
        loader.load()
        
        mock_load_dataset.assert_called_once()
        call_kwargs = mock_load_dataset.call_args.kwargs
        assert call_kwargs["path"] == "test/dataset"
        assert call_kwargs["name"] == "en"
        assert call_kwargs["split"] == "train"


class TestLocalAudioDatasetLoader:
    """Tests for LocalAudioDatasetLoader."""

    def test_supported_formats(self):
        """Test that common audio formats are supported."""
        assert ".wav" in SUPPORTED_AUDIO_FORMATS
        assert ".mp3" in SUPPORTED_AUDIO_FORMATS
        assert ".flac" in SUPPORTED_AUDIO_FORMATS

    def test_directory_not_found(self, tmp_path):
        """Test error when audio directory doesn't exist."""
        with pytest.raises(FileNotFoundError):
            LocalAudioDatasetLoader(audio_dir=tmp_path / "nonexistent")

    def test_find_transcripts_file(self, tmp_path):
        """Test finding transcripts file in directory."""
        transcripts = tmp_path / "transcripts.json"
        transcripts.write_text("[]")
        
        loader = LocalAudioDatasetLoader(audio_dir=tmp_path)
        found = loader._find_transcripts_file()
        
        assert found == transcripts

    def test_find_transcripts_file_jsonl(self, tmp_path):
        """Test finding JSONL transcripts file."""
        transcripts = tmp_path / "transcripts.jsonl"
        transcripts.write_text("")
        
        loader = LocalAudioDatasetLoader(audio_dir=tmp_path)
        found = loader._find_transcripts_file()
        
        assert found == transcripts

    def test_find_transcripts_file_custom(self, tmp_path):
        """Test using custom transcripts file path."""
        custom = tmp_path / "custom_transcripts.json"
        custom.write_text("[]")
        
        loader = LocalAudioDatasetLoader(
            audio_dir=tmp_path,
            transcripts_file=custom,
        )
        found = loader._find_transcripts_file()
        
        assert found == custom

    def test_load_transcripts_json_list(self, tmp_path):
        """Test loading transcripts from JSON list."""
        transcripts = tmp_path / "transcripts.json"
        data = [
            {"file": "audio1.wav", "text": "hello"},
            {"file": "audio2.wav", "text": "world"},
        ]
        transcripts.write_text(json.dumps(data))
        
        loader = LocalAudioDatasetLoader(audio_dir=tmp_path)
        result = loader._load_transcripts(transcripts)
        
        assert len(result) == 2
        assert result[0]["text"] == "hello"

    def test_load_transcripts_json_wrapped(self, tmp_path):
        """Test loading transcripts from wrapped JSON format."""
        transcripts = tmp_path / "transcripts.json"
        data = {
            "items": [
                {"file": "audio1.wav", "text": "hello"},
            ]
        }
        transcripts.write_text(json.dumps(data))
        
        loader = LocalAudioDatasetLoader(audio_dir=tmp_path)
        result = loader._load_transcripts(transcripts)
        
        assert len(result) == 1

    def test_load_transcripts_jsonl(self, tmp_path):
        """Test loading transcripts from JSONL file."""
        transcripts = tmp_path / "transcripts.jsonl"
        lines = [
            json.dumps({"file": "audio1.wav", "text": "hello"}),
            json.dumps({"file": "audio2.wav", "text": "world"}),
        ]
        transcripts.write_text("\n".join(lines))
        
        loader = LocalAudioDatasetLoader(audio_dir=tmp_path)
        result = loader._load_transcripts(transcripts)
        
        assert len(result) == 2

    def test_find_audio_files(self, tmp_path):
        """Test finding audio files in directory."""
        (tmp_path / "audio1.wav").touch()
        (tmp_path / "audio2.mp3").touch()
        (tmp_path / "audio3.flac").touch()
        (tmp_path / "readme.txt").touch()
        
        loader = LocalAudioDatasetLoader(audio_dir=tmp_path)
        files = loader._find_audio_files()
        
        assert len(files) == 3
        extensions = {f.suffix for f in files}
        assert extensions == {".wav", ".mp3", ".flac"}

    def test_find_audio_files_recursive(self, tmp_path):
        """Test recursive audio file search."""
        (tmp_path / "audio1.wav").touch()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "audio2.wav").touch()
        
        loader = LocalAudioDatasetLoader(audio_dir=tmp_path, recursive=True)
        files = loader._find_audio_files()
        
        assert len(files) == 2

    def test_find_audio_files_with_pattern(self, tmp_path):
        """Test audio file search with pattern filter."""
        (tmp_path / "train_001.wav").touch()
        (tmp_path / "train_002.wav").touch()
        (tmp_path / "test_001.wav").touch()
        
        loader = LocalAudioDatasetLoader(
            audio_dir=tmp_path,
            file_pattern="train_*.wav",
        )
        files = loader._find_audio_files()
        
        assert len(files) == 2
        assert all("train" in f.name for f in files)


class TestCreateDatasetLoader:
    """Tests for create_dataset_loader factory function."""

    def test_create_local_loader(self, tmp_path):
        """Test creating local dataset loader."""
        loader = create_dataset_loader(
            source="local",
            audio_dir=tmp_path,
            default_language="de",
        )
        
        assert isinstance(loader, LocalAudioDatasetLoader)
        assert loader.default_language == "de"

    def test_create_huggingface_loader(self):
        """Test creating HuggingFace dataset loader."""
        loader = create_dataset_loader(
            source="huggingface",
            huggingface_dataset="test/dataset",
            huggingface_subset="en",
            huggingface_split="train",
            max_samples=100,
        )
        
        assert isinstance(loader, HuggingFaceDatasetLoader)
        assert loader.dataset_name == "test/dataset"
        assert loader.subset == "en"
        assert loader.split == "train"
        assert loader.max_samples == 100

    def test_create_custom_loader(self):
        """Test creating custom dataset loader."""
        data = [{"audio": [0.1], "text": "test"}]
        map_fn = lambda x: AudioSample(
            np.array(x["audio"]), 16000, x["text"]
        )
        
        loader = create_dataset_loader(
            source="custom",
            data=data,
            map_fn=map_fn,
        )
        
        assert isinstance(loader, GenericDatasetLoader)
        assert len(loader) == 1

    def test_local_requires_audio_dir(self):
        """Test that local source requires audio_dir."""
        with pytest.raises(ValueError, match="audio_dir is required"):
            create_dataset_loader(source="local")

    def test_huggingface_requires_dataset(self):
        """Test that HuggingFace source requires dataset name."""
        with pytest.raises(ValueError, match="huggingface_dataset is required"):
            create_dataset_loader(source="huggingface")

    def test_custom_requires_data_and_map_fn(self):
        """Test that custom source requires data and map_fn."""
        with pytest.raises(ValueError, match="data.*map_fn"):
            create_dataset_loader(source="custom")
        
        with pytest.raises(ValueError, match="data.*map_fn"):
            create_dataset_loader(source="custom", data=[])

    def test_unknown_source(self):
        """Test error for unknown source type."""
        with pytest.raises(ValueError, match="Unknown dataset source"):
            create_dataset_loader(source="unknown")

    def test_case_insensitive_source(self, tmp_path):
        """Test that source is case-insensitive."""
        loader1 = create_dataset_loader(source="LOCAL", audio_dir=tmp_path)
        loader2 = create_dataset_loader(source="HuggingFace", huggingface_dataset="test/ds")
        
        assert isinstance(loader1, LocalAudioDatasetLoader)
        assert isinstance(loader2, HuggingFaceDatasetLoader)


class TestDataConfigIntegration:
    """Tests for DataConfig integration with dataset loaders."""

    def test_data_config_has_loader_fields(self):
        """Test that DataConfig has all required loader configuration fields."""
        from evaluator.config import DataConfig
        
        config = DataConfig()
        
        # Check new fields exist with defaults
        assert hasattr(config, "dataset_source")
        assert config.dataset_source == "local"
        
        assert hasattr(config, "huggingface_dataset")
        assert config.huggingface_dataset is None
        
        assert hasattr(config, "huggingface_split")
        assert config.huggingface_split == "test"
        
        assert hasattr(config, "audio_dir")
        assert config.audio_dir is None
        
        assert hasattr(config, "default_language")
        assert config.default_language == "en"

    def test_data_config_custom_values(self):
        """Test DataConfig with custom loader values."""
        from evaluator.config import DataConfig
        
        config = DataConfig(
            dataset_source="huggingface",
            huggingface_dataset="mozilla-foundation/common_voice_11_0",
            huggingface_subset="en",
            huggingface_split="validation",
            max_samples=500,
        )
        
        assert config.dataset_source == "huggingface"
        assert config.huggingface_dataset == "mozilla-foundation/common_voice_11_0"
        assert config.huggingface_subset == "en"
        assert config.huggingface_split == "validation"
        assert config.max_samples == 500
