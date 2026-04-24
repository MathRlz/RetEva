"""Local audio directory dataset loader."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import numpy as np

from .base import AudioSample


# Supported audio formats
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}


@dataclass
class LocalAudioDatasetLoader:
    """Dataset loader for local audio directories.
    
    Supports two directory structures:
    
    1. Transcripts file (JSON/JSONL):
       ```
       audio_dir/
         transcripts.json  # or transcripts.jsonl
         audio_001.wav
         audio_002.wav
       ```
       
       transcripts.json format:
       ```json
       [
         {"file": "audio_001.wav", "text": "transcription one", "language": "en"},
         {"file": "audio_002.wav", "text": "transcription two"}
       ]
       ```
    
    2. Paired text files:
       ```
       audio_dir/
         audio_001.wav
         audio_001.txt
         audio_002.wav
         audio_002.txt
       ```
    
    Attributes:
        audio_dir: Path to directory containing audio files.
        transcripts_file: Optional path to transcripts file (JSON or JSONL).
        file_column: Column name for audio filename in transcripts.
        text_column: Column name for transcription text in transcripts.
        language_column: Optional column name for language code.
        default_language: Default language code if not specified.
        sample_rate: Target sample rate for resampling (None to keep original).
        recursive: Whether to search subdirectories for audio files.
        file_pattern: Optional glob pattern to filter audio files.
    """
    audio_dir: Union[str, Path]
    transcripts_file: Optional[Union[str, Path]] = None
    file_column: str = "file"
    text_column: str = "text"
    language_column: Optional[str] = "language"
    default_language: str = "en"
    sample_rate: Optional[int] = None
    recursive: bool = False
    file_pattern: Optional[str] = None
    
    _samples: Optional[List[AudioSample]] = field(default=None, repr=False, init=False)
    
    def __post_init__(self):
        """Validate paths after initialization."""
        self.audio_dir = Path(self.audio_dir)
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")
        
        if self.transcripts_file is not None:
            self.transcripts_file = Path(self.transcripts_file)
    
    def _find_transcripts_file(self) -> Optional[Path]:
        """Find transcripts file in the audio directory.
        
        Returns:
            Path to transcripts file, or None if not found.
        """
        if self.transcripts_file is not None:
            return self.transcripts_file if self.transcripts_file.exists() else None
        
        # Look for common transcript filenames
        candidates = [
            "transcripts.json",
            "transcripts.jsonl",
            "manifest.json",
            "manifest.jsonl",
            "metadata.json",
            "metadata.jsonl",
            "data.json",
            "data.jsonl",
        ]
        
        for name in candidates:
            path = self.audio_dir / name
            if path.exists():
                return path
        
        return None
    
    def _load_transcripts(self, path: Path) -> List[Dict[str, Any]]:
        """Load transcripts from JSON or JSONL file.
        
        Args:
            path: Path to transcripts file.
            
        Returns:
            List of transcript dictionaries.
        """
        with path.open("r", encoding="utf-8") as f:
            if path.suffix == ".jsonl":
                transcripts = []
                for line in f:
                    line = line.strip()
                    if line:
                        transcripts.append(json.loads(line))
                return transcripts
            else:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                # Handle wrapped format {"items": [...]} or {"transcripts": [...]}
                for key in ["items", "transcripts", "data", "samples"]:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                raise ValueError(
                    f"Unsupported JSON structure in {path}. "
                    "Expected a list or dict with 'items'/'transcripts' key."
                )
    
    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in the directory.
        
        Returns:
            List of paths to audio files.
        """
        pattern = self.file_pattern or "*"
        
        if self.recursive:
            files = list(self.audio_dir.rglob(pattern))
        else:
            files = list(self.audio_dir.glob(pattern))
        
        # Filter by audio format
        audio_files = [
            f for f in files 
            if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_FORMATS
        ]
        
        return sorted(audio_files)
    
    def _load_audio(self, path: Path) -> tuple:
        """Load audio file and return array with sample rate.
        
        Args:
            path: Path to audio file.
            
        Returns:
            Tuple of (audio_array, sampling_rate).
        """
        try:
            import torchaudio
        except ImportError:
            try:
                import librosa
                audio_array, sr = librosa.load(path, sr=self.sample_rate)
                return audio_array.astype(np.float32), sr
            except ImportError as e:
                raise ImportError(
                    "Either torchaudio or librosa is required for loading audio. "
                    "Install with: pip install torchaudio or pip install librosa"
                ) from e
        
        waveform, sr = torchaudio.load(str(path))
        
        # Resample if needed
        if self.sample_rate is not None and sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            sr = self.sample_rate
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform.squeeze().numpy().astype(np.float32), sr
    
    def _load_from_transcripts(self, transcripts: List[Dict[str, Any]]) -> List[AudioSample]:
        """Load samples using transcripts file.
        
        Args:
            transcripts: List of transcript dictionaries.
            
        Returns:
            List of AudioSample objects.
        """
        samples: List[AudioSample] = []
        
        for idx, item in enumerate(transcripts):
            # Get audio filename
            filename = item.get(self.file_column)
            if filename is None:
                # Try alternative column names
                for alt in ["filename", "audio_file", "path", "audio", "audio_path"]:
                    if alt in item:
                        filename = item[alt]
                        break
            
            if filename is None:
                continue
            
            # Resolve audio path
            audio_path = self.audio_dir / filename
            if not audio_path.exists():
                # Try relative to transcripts file location
                if self.transcripts_file:
                    alt_path = self.transcripts_file.parent / filename
                    if alt_path.exists():
                        audio_path = alt_path
            
            if not audio_path.exists():
                continue
            
            # Get transcription
            text = item.get(self.text_column)
            if text is None:
                for alt in ["transcription", "sentence", "transcript", "label"]:
                    if alt in item:
                        text = item[alt]
                        break
            
            if text is None:
                text = ""
            
            # Get language
            language = self.default_language
            if self.language_column and self.language_column in item:
                language = str(item[self.language_column])
            
            # Get sample ID
            sample_id = item.get("id", item.get("sample_id", str(audio_path.stem)))
            
            # Load audio
            try:
                audio_array, sr = self._load_audio(audio_path)
            except (ImportError, OSError, RuntimeError, ValueError):
                # Skip files that can't be loaded
                continue
            
            # Collect metadata
            metadata = {
                k: v for k, v in item.items()
                if k not in [self.file_column, self.text_column, self.language_column, "id", "sample_id"]
            }
            metadata["source_file"] = str(audio_path)
            
            samples.append(AudioSample(
                audio_array=audio_array,
                sampling_rate=sr,
                transcription=str(text),
                sample_id=str(sample_id),
                language=language,
                metadata=metadata,
            ))
        
        return samples
    
    def _load_from_paired_files(self, audio_files: List[Path]) -> List[AudioSample]:
        """Load samples using paired audio/text files.
        
        Args:
            audio_files: List of audio file paths.
            
        Returns:
            List of AudioSample objects.
        """
        samples: List[AudioSample] = []
        
        for audio_path in audio_files:
            # Look for matching text file
            txt_path = audio_path.with_suffix(".txt")
            
            if txt_path.exists():
                with txt_path.open("r", encoding="utf-8") as f:
                    text = f.read().strip()
            else:
                text = ""
            
            # Load audio
            try:
                audio_array, sr = self._load_audio(audio_path)
            except (ImportError, OSError, RuntimeError, ValueError):
                continue
            
            samples.append(AudioSample(
                audio_array=audio_array,
                sampling_rate=sr,
                transcription=text,
                sample_id=audio_path.stem,
                language=self.default_language,
                metadata={"source_file": str(audio_path)},
            ))
        
        return samples
    
    def load(self) -> List[AudioSample]:
        """Load all samples from the local directory.
        
        Returns:
            List of AudioSample objects.
        """
        if self._samples is not None:
            return self._samples
        
        transcripts_path = self._find_transcripts_file()
        
        if transcripts_path is not None:
            transcripts = self._load_transcripts(transcripts_path)
            self._samples = self._load_from_transcripts(transcripts)
        else:
            audio_files = self._find_audio_files()
            self._samples = self._load_from_paired_files(audio_files)
        
        return self._samples
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.load())
    
    def __iter__(self) -> Iterator[AudioSample]:
        """Iterate over samples."""
        return iter(self.load())
    
    def __getitem__(self, idx: int) -> AudioSample:
        """Get a sample by index."""
        return self.load()[idx]
