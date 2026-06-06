"""Built-in datasets, written against the public per-type ABC extension API.

These are the canonical worked examples for adding a dataset: pick the type base
(:class:`~evaluator.datasets.types.AudioTranscriptionDataset` /
``AudioRetrievalDataset`` / ``TextRetrievalDataset`` / ``MultimodalQADataset``), set the
class attributes that differ from the type's defaults, implement ``from_config``, and
decorate with ``@register_eval_dataset``. The :class:`DatasetDescriptor` is derived from
the class — capability metadata has exactly one author.

Importing this module registers the built-ins (side-effect); ``datasets/__init__`` does so.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, TYPE_CHECKING

from ..errors import ConfigurationError
from .types import (
    AudioRetrievalDataset,
    AudioTranscriptionDataset,
    MultimodalQADataset,
    register_eval_dataset,
)

if TYPE_CHECKING:
    from ..config.data import DataConfig
    from .core import QueryDataset


# ── ASR transcription datasets ────────────────────────────────────────

@register_eval_dataset(
    id="admed_voice",
    description="ADMED Voice: Polish medical spoken-query dataset",
)
class AdmedVoiceDataset(AudioTranscriptionDataset):
    """Audio transcription; pure fit for the type defaults (no overrides)."""

    @classmethod
    def from_config(cls, data: "DataConfig") -> "QueryDataset":
        from .core import load_admed_voice_corpus, load_audio_file
        from .runtime import AudioSamplesQueryDataset
        from .loaders.base import AudioSample

        test_size = getattr(data, "test_size", 0.0)
        train_df, _ = load_admed_voice_corpus(test_size=test_size)

        samples = []
        for idx, row in train_df.reset_index(drop=True).iterrows():
            waveform, sr = load_audio_file(row["file_path"])
            samples.append(AudioSample(
                audio_array=waveform.squeeze().numpy(),
                sampling_rate=sr,
                transcription=str(row["phrase"]),
                sample_id=str(row.get("filename", str(idx))),
                language="pl",
                metadata={"speaker_id": str(row["speaker_id"])} if "speaker_id" in row else {},
            ))
        return AudioSamplesQueryDataset(
            samples,
            trace_limit=getattr(data, "trace_limit", 0),
        )


@register_eval_dataset(
    id="fleurs",
    description="Google FLEURS: 102-language ASR benchmark (requires huggingface_subset)",
)
class FleursDataset(AudioTranscriptionDataset):
    """FLEURS streams all 102 languages without a subset, so require one."""

    @classmethod
    def validate(cls, data: "DataConfig") -> List[str]:
        errors: List[str] = []
        if not getattr(data, "huggingface_subset", None):
            errors.append(
                "FLEURS requires huggingface_subset (e.g. 'pl_pl'). "
                "Without it the loader would attempt to stream all 102 languages."
            )
        return errors

    @classmethod
    def from_config(cls, data: "DataConfig") -> "QueryDataset":
        from .loaders.factory import create_dataset_loader
        from .runtime import AudioSamplesQueryDataset, _load_corpus_entries

        subset = getattr(data, "huggingface_subset", None) or ""
        # Derive Whisper language code from subset tag (e.g. "pl_pl" → "pl")
        derived_lang = subset.split("_")[0] if subset else "en"
        default_language = getattr(data, "default_language", None) or derived_lang

        loader = create_dataset_loader(
            source="huggingface",
            huggingface_dataset="google/fleurs",
            huggingface_subset=subset or None,
            huggingface_split=getattr(data, "huggingface_split", "test"),
            column_mapping=getattr(data, "column_mapping", None),
            max_samples=getattr(data, "max_samples", None),
            default_language=default_language,
        )
        samples = loader.load()
        corpus_entries = (
            _load_corpus_entries(data.corpus_path)
            if getattr(data, "corpus_path", None)
            else []
        )
        return AudioSamplesQueryDataset(
            samples,
            trace_limit=getattr(data, "trace_limit", 0),
            corpus_entries=corpus_entries,
        )


# ── Audio-query retrieval datasets ────────────────────────────────────

@register_eval_dataset(
    id="local",
    description="Local audio directory with optional corpus",
)
class LocalAudioDataset(AudioRetrievalDataset):
    """Audio query from a local directory; corpus optional, so text not required."""

    requires_text = False
    required_data_fields = ("audio_dir",)

    @classmethod
    def from_config(cls, data: "DataConfig") -> "QueryDataset":
        from .loaders.factory import create_dataset_loader
        from .runtime import AudioSamplesQueryDataset, _load_corpus_entries

        loader = create_dataset_loader(
            source="local",
            audio_dir=data.audio_dir,
            transcripts_file=getattr(data, "transcripts_file", None),
            column_mapping=getattr(data, "column_mapping", None),
            max_samples=getattr(data, "max_samples", None),
            default_language=getattr(data, "default_language", "en"),
            sample_rate=getattr(data, "sample_rate", None),
        )
        samples = loader.load()
        corpus_entries = (
            _load_corpus_entries(data.corpus_path)
            if getattr(data, "corpus_path", None)
            else []
        )
        return AudioSamplesQueryDataset(
            samples,
            trace_limit=getattr(data, "trace_limit", 0),
            corpus_entries=corpus_entries,
        )


@register_eval_dataset(
    id="huggingface",
    description="HuggingFace Hub audio dataset",
)
class HuggingFaceAudioDataset(AudioRetrievalDataset):
    """Audio query from a HuggingFace Hub dataset; corpus optional."""

    requires_text = False
    required_data_fields = ("huggingface_dataset",)

    @classmethod
    def from_config(cls, data: "DataConfig") -> "QueryDataset":
        from .loaders.factory import create_dataset_loader
        from .runtime import AudioSamplesQueryDataset, _load_corpus_entries

        loader = create_dataset_loader(
            source="huggingface",
            huggingface_dataset=data.huggingface_dataset,
            huggingface_subset=getattr(data, "huggingface_subset", None),
            huggingface_split=getattr(data, "huggingface_split", "test"),
            column_mapping=getattr(data, "column_mapping", None),
            max_samples=getattr(data, "max_samples", None),
            default_language=getattr(data, "default_language", "en"),
        )
        samples = loader.load()
        corpus_entries = (
            _load_corpus_entries(data.corpus_path)
            if getattr(data, "corpus_path", None)
            else []
        )
        return AudioSamplesQueryDataset(
            samples,
            trace_limit=getattr(data, "trace_limit", 0),
            corpus_entries=corpus_entries,
        )


# ── Multimodal QA (text + TTS audio) ──────────────────────────────────

@register_eval_dataset(
    id="pubmed_qa",
    description="PubMed QA: biomedical QA with audio queries and document corpus",
)
class PubMedQABuiltinDataset(MultimodalQADataset):
    """Text QA voiced via TTS. Audio is required at run time (query is spoken), and only
    the ASR-text and audio-text retrieval modes apply (no plain ASR, no audio-corpus)."""

    requires_audio = True

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
        from .core import PubMedQADataset
        from .runtime import LazyAudioQueryDataset

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

        questions = PubMedQADataset._load_questions(questions_path)
        corpus = PubMedQADataset._load_corpus(corpus_path)
        return LazyAudioQueryDataset(
            questions,
            corpus,
            trace_limit=getattr(data, "trace_limit", 0),
        )
