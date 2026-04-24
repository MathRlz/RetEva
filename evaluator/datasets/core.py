"""
Dataset loading utilities.

Environment Variables:
    EVALUATOR_DATA_DIR: Base directory for all datasets. Falls back to current
                        working directory if not set.
    ADMED_VOICE_PATH: Override path for the admed_voice dataset. If not set,
                      uses EVALUATOR_DATA_DIR/admed_voice.
"""

from ..core.structures import QuerySample, Document, TranscriptionResult, BenchmarkQuestion, CorpusDocument
from ..datasets.utils import DatasetLoader
from typing import List, Dict, Any, Optional
import os
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
import torchaudio
from pathlib import Path


def get_data_dir() -> Path:
    """Get base directory for datasets.
    
    Returns EVALUATOR_DATA_DIR env var if set, otherwise current directory.
    """
    env_path = os.environ.get("EVALUATOR_DATA_DIR")
    if env_path:
        return Path(env_path)
    return Path.cwd()


class QueryDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pass

class AdmedQueryDataset(QueryDataset, Dataset):
    def __init__(self, corpus_df: pd.DataFrame):
        self.corpus = corpus_df.reset_index(drop=True)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        item = self.corpus.iloc[idx]

        file_path = item["file_path"]
        waveform, sample_rate = torchaudio.load(file_path)

        transcription = item['phrase']

        return {
            "audio_array": waveform.squeeze().numpy(),
            "sampling_rate": sample_rate,
            "transcription": transcription,
            "language": "pl"
        }


class PubMedQADataset(QueryDataset, Dataset):
    """Benchmark dataset: text questions mapped to expected PubMed document ids."""

    def __init__(self, questions_path: str, corpus_path: str, trace_limit: int = 0):
        self.questions_path = Path(questions_path)
        self.corpus_path = Path(corpus_path)
        self.questions = self._load_questions(self.questions_path)
        self.corpus = self._load_corpus(self.corpus_path)
        if trace_limit and trace_limit > 0:
            self.questions = self.questions[:trace_limit]

    @staticmethod
    def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}\n"
                f"Searched path: {path.resolve()}\n"
                f"Tip: Set EVALUATOR_DATA_DIR environment variable to specify the base data directory."
            )

        loader = DatasetLoader(path)
        data = loader.load()

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ["questions", "documents", "corpus", "items"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            raise ValueError(f"Unsupported JSON structure in {path}")

        raise ValueError(f"Unsupported data type for {path}. Expected list or dict.")

    def _load_questions(self, path: Path) -> List[BenchmarkQuestion]:
        rows = self._load_json_or_jsonl(path)
        questions: List[BenchmarkQuestion] = []
        for row in rows:
            question_id = str(row.get("question_id") or row.get("id") or "")
            question_text = row.get("question_text") or row.get("question") or row.get("text")
            if not question_id or not question_text:
                raise ValueError(
                    f"Each question must contain 'question_id' and 'question_text'.\n"
                    f"Found fields: {list(row.keys())}\n"
                    f"Example: {{\"question_id\": \"q1\", \"question_text\": \"What is...\"}}"
                )

            gt_ids = row.get("groundtruth_doc_ids") or row.get("relevant_doc_ids") or []
            relevance = row.get("relevance_grades") or {}
            if not relevance and gt_ids:
                relevance = {str(doc_id): 1 for doc_id in gt_ids}

            questions.append(
                BenchmarkQuestion(
                    question_id=question_id,
                    question_text=str(question_text),
                    groundtruth_doc_ids=[str(doc_id) for doc_id in gt_ids],
                    relevance_grades={str(k): int(v) for k, v in relevance.items()},
                    audio_path=row.get("audio_path"),
                    language=row.get("language", "en"),
                    metadata=row.get("metadata", {}),
                )
            )
        return questions

    def _load_corpus(self, path: Path) -> List[CorpusDocument]:
        rows = self._load_json_or_jsonl(path)
        corpus: List[CorpusDocument] = []
        for row in rows:
            doc_id = str(row.get("doc_id") or row.get("pmid") or row.get("id") or "")
            text = row.get("text") or row.get("abstract") or row.get("content")
            if not doc_id or not text:
                raise ValueError(
                    f"Each corpus row must contain 'doc_id' and 'text'.\n"
                    f"Found fields: {list(row.keys())}\n"
                    f"Example: {{\"doc_id\": \"12345\", \"text\": \"Document content...\"}}"
                )

            corpus.append(
                CorpusDocument(
                    doc_id=doc_id,
                    text=str(text),
                    title=str(row.get("title", "")),
                    abstract=str(row.get("abstract", "")),
                    language=row.get("language", "en"),
                    metadata=row.get("metadata", {}),
                )
            )
        return corpus

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        question = self.questions[idx]

        if not question.audio_path:
            raise ValueError(
                f"Question {question.question_id} has no audio_path. "
                "Enable audio_synthesis in config or provide pre-recorded audio."
            )

        waveform, sample_rate = torchaudio.load(question.audio_path)

        return {
            "audio_array": waveform.squeeze().numpy(),
            "sampling_rate": sample_rate,
            "transcription": question.question_text,
            "question_text": question.question_text,
            "question_id": question.question_id,
            "groundtruth_doc_ids": question.groundtruth_doc_ids,
            "relevance_grades": question.relevance_grades,
            "language": question.language,
            "metadata": question.metadata,
        }

    def get_corpus_entries(self) -> List[Dict[str, Any]]:
        """Return corpus entries for index population."""
        return [
            {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "title": doc.title,
                "language": doc.language,
                "metadata": doc.metadata,
            }
            for doc in self.corpus
        ]

    def get_corpus(self) -> List[Dict[str, Any]]:
        """Return corpus entries (compat alias used by service path)."""
        return self.get_corpus_entries()


def load_pubmed_qa_dataset(
    questions_path: str,
    corpus_path: str,
    trace_limit: int = 0,
) -> PubMedQADataset:
    """Load benchmark dataset where each question maps to one or more PubMed document ids."""
    return PubMedQADataset(
        questions_path=questions_path,
        corpus_path=corpus_path,
        trace_limit=trace_limit,
    )

def get_admed_voice_path() -> Path:
    """Get path to admed_voice dataset.
    
    Returns ADMED_VOICE_PATH env var if set, otherwise EVALUATOR_DATA_DIR/admed_voice.
    """
    env_path = os.environ.get("ADMED_VOICE_PATH")
    if env_path:
        return Path(env_path)
    return get_data_dir() / "admed_voice"


def load_admed_voice_corpus(test_size: float = 0.3):
    def get_duration_in_seconds(duration_str):
        h, m, s = map(float, duration_str.split(":"))
        return h * 3600 + m * 60 + s

    admed_voice_base = get_admed_voice_path()
    
    def get_file_path(example):
        source = example["source"]
        if source == "natural":
            source = "human_voices/human/human"
        elif source == "anonymization":
            source = "anoni/anoni"
        elif source == "synthesis":
            source = "synth/synth"
        else:
            raise ValueError(f"Unknown source type: {source}")

        cat_code = example["cat_code"]
        rec_place = example["rec_place"]
        speaker_id = example["speaker_id"]
        filename = example["filename"]
        file_path = admed_voice_base / source / f"cat_{cat_code}" / f"{rec_place}-{speaker_id}" / filename
        return str(file_path)

    corpus_csv_path = admed_voice_base / "corpus_summary_all.csv"
    if not corpus_csv_path.exists():
        raise FileNotFoundError(
            f"admed_voice corpus not found.\n"
            f"Searched path: {corpus_csv_path.resolve()}\n"
            f"To fix this, either:\n"
            f"  - Set ADMED_VOICE_PATH to the dataset location\n"
            f"  - Set EVALUATOR_DATA_DIR to the parent directory containing 'admed_voice/'"
        )
    
    corpus = pd.read_csv(corpus_csv_path, sep=";")

    # filer out non human
    corpus = corpus[corpus["source"] == "natural"]

    # filter out longer than 30s
    corpus["duration_sec"] = corpus["file_duration"].apply(get_duration_in_seconds)
    corpus = corpus[corpus["duration_sec"] <= 30.0]

    corpus["file_path"] = corpus.apply(get_file_path, axis=1)

    if test_size == 0.0:
        return corpus, pd.DataFrame([])

    unique_speakers = corpus["speaker_id"].unique()
    train_speakers, test_speakers = train_test_split(
        unique_speakers, test_size=test_size, random_state=420
    )

    train_corpus = corpus[corpus["speaker_id"].isin(train_speakers)]
    test_corpus = corpus[corpus["speaker_id"].isin(test_speakers)]   

    return train_corpus, test_corpus
