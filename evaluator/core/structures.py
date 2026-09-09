from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class QuerySample:
    """Single query sample with audio and ground truth transcription."""
    audio_array: np.ndarray  # float32 waveform
    sampling_rate: int
    transcription: str
    query_id: str = field(default_factory=str)
    medical_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    language: str = field(default="en")

    def __post_init__(self):
        if not isinstance(self.audio_array, np.ndarray):
            self.audio_array = np.array(self.audio_array, dtype=np.float32)
        elif self.audio_array.dtype != np.float32:
            self.audio_array = self.audio_array.astype(np.float32)


@dataclass
class Document:
    """Single document in the retrieval corpus."""
    doc_id: str
    text: str
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptionResult:
    """Output from a Speech-to-Text model."""
    text: str
    confidence: Optional[float] = None
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    processing_time: Optional[float] = None
    raw_output: Optional[Any] = None


@dataclass
class RetrievalResult:
    """Single retrieval result for a query; score is higher-is-more-relevant."""
    doc_id: str
    score: float
    rank: int


@dataclass
class BenchmarkQuestion:
    """Single benchmark question with text/audio variants and relevance labels."""
    question_id: str
    question_text: str
    groundtruth_doc_ids: List[str] = field(default_factory=list)
    relevance_grades: Dict[str, int] = field(default_factory=dict)
    audio_path: Optional[str] = None
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusDocument:
    """Single corpus document used by the retrieval index."""
    doc_id: str
    text: str
    title: str = ""
    abstract: str = ""
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetManifest:
    """Metadata required to keep benchmark datasets reproducible."""
    dataset_id: str
    query_count: int
    corpus_size: int
    corpus_version_hash: str
    corpus_snapshot_id: str = ""
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)
