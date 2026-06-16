"""PubMed QA — biomedical QA voiced via TTS (multimodal QA, document corpus)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, TYPE_CHECKING

from ...errors import ConfigurationError
from ..types import MultimodalQADataset, register_eval_dataset

if TYPE_CHECKING:
    from ...config.data import DataConfig
    from ..core import QueryDataset


@register_eval_dataset(
    id="pubmed_qa",
    description="PubMed QA: biomedical QA with audio queries and document corpus",
)
class PubMedQABuiltinDataset(MultimodalQADataset):
    """Text QA voiced via TTS. Audio is required at run time (query is spoken), and only
    the ASR-text and audio-text retrieval modes apply (no plain ASR, no audio-corpus)."""

    domain = "medical"
    requires_audio = True
    # Declared for discoverability (builder picker / API); validate() remains the
    # authority — prepared_dataset_dir is an accepted alternative to the pair.
    required_data_fields = ("questions_path", "corpus_path")

    @classmethod
    def compatible_pipeline_modes(cls) -> Sequence[str]:
        return ("asr_text_retrieval", "audio_text_retrieval")

    @classmethod
    def validate(cls, data: "DataConfig") -> List[str]:
        errors: List[str] = []
        if not getattr(data, "prepared_dataset_dir", None):
            if not getattr(data, "questions_path", None):
                errors.append(
                    "data.questions_path or data.prepared_dataset_dir required for pubmed_qa"
                )
            if not getattr(data, "corpus_path", None):
                errors.append(
                    "data.corpus_path or data.prepared_dataset_dir required for pubmed_qa"
                )
        return errors

    @classmethod
    def from_config(cls, data: "DataConfig") -> "QueryDataset":
        from ..core import load_corpus_documents, load_questions_file
        from ..runtime import LazyAudioQueryDataset

        if getattr(data, "prepared_dataset_dir", None):
            dataset_dir = Path(data.prepared_dataset_dir)
            if not dataset_dir.exists():
                raise ConfigurationError(
                    f"Prepared dataset directory not found: {data.prepared_dataset_dir}"
                )
            questions_path = dataset_dir / "questions.json"
            corpus_path = dataset_dir / "corpus.json"
            if not questions_path.exists() or not corpus_path.exists():
                raise ConfigurationError(
                    "Prepared dataset directory must contain questions.json and corpus.json"
                )
        else:
            questions_path = Path(data.questions_path)
            corpus_path = Path(data.corpus_path)

        questions = load_questions_file(questions_path)
        corpus = load_corpus_documents(corpus_path)
        return LazyAudioQueryDataset(
            questions,
            corpus,
            trace_limit=getattr(data, "trace_limit", 0),
        )
