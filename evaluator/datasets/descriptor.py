"""Unified dataset descriptor registry.

Each DatasetDescriptor combines capability profile (what the dataset needs /
supports) with runtime spec (validation rules + loader).  New datasets register
once via :func:`register_dataset` — no other files need to be touched.

Built-in descriptors (admed_voice, pubmed_qa, local, huggingface) are
registered at module-import time so they are always available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

from ..config.types import DatasetType
from ..errors import ConfigurationError

if TYPE_CHECKING:
    from ..config.data import DataConfig
    from .core import QueryDataset


# ── Default metric sets per evaluation mode ───────────────────────────

METRICS_BY_MODE: Dict[str, Sequence[str]] = {
    "transcription": ("wer", "cer"),
    "retrieval":     ("mrr", "ndcg", "precision", "recall"),
    "qa_retrieval":  ("wer", "cer", "mrr", "ndcg", "precision", "recall"),
    "qa":            ("mrr", "ndcg", "llm_judge"),
    "ranking":       ("ndcg", "precision"),
}


# ── DatasetDescriptor ─────────────────────────────────────────────────

@dataclass
class DatasetDescriptor:
    """Unified descriptor for a dataset type.

    Combines what used to live in ``DatasetCapabilityProfile`` (capabilities)
    and ``DatasetRuntimeSpec`` (validation + loading) into a single object
    that registers once via :func:`register_dataset`.

    Attributes:
        id: Unique identifier; used as ``data.dataset_name`` in DataConfig.
        description: Human-readable description.
        dataset_type: Enum categorising the task type.
        requires_audio: Samples must have audio.
        requires_text: Samples must have text queries / corpus.
        supports_generation: TTS can synthesise audio from text queries.
        evaluation_mode: Key into METRICS_BY_MODE ("transcription" | "retrieval" |
            "qa_retrieval" | "qa" | "ranking").
        compatible_pipeline_modes: Pipeline modes valid for this dataset.
        default_metrics: Metrics to compute; auto-filled from evaluation_mode
            when left empty.
        required_data_fields: DataConfig attribute names that must be non-empty.
        load_fn: ``(DataConfig) -> QueryDataset`` called by :meth:`load`.
        validate_fn: Optional ``(DataConfig) -> List[str]`` for custom
            validation logic (overrides required_data_fields check when set).
    """

    id: str
    description: str
    dataset_type: DatasetType
    requires_audio: bool
    requires_text: bool
    supports_generation: bool
    evaluation_mode: str
    compatible_pipeline_modes: Sequence[str]
    default_metrics: Sequence[str] = field(default_factory=list)
    required_data_fields: Sequence[str] = field(default_factory=list)
    load_fn: Optional[Callable[["DataConfig"], "QueryDataset"]] = field(
        default=None, repr=False
    )
    validate_fn: Optional[Callable[["DataConfig"], List[str]]] = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        if not self.default_metrics:
            self.default_metrics = list(METRICS_BY_MODE.get(self.evaluation_mode, ()))

    # ── Public helpers ────────────────────────────────────────────────

    def supports_pipeline_mode(self, mode: str) -> bool:
        """Return True when *mode* is listed in compatible_pipeline_modes."""
        return mode in set(self.compatible_pipeline_modes)

    def validate_data_config(self, data: "DataConfig") -> List[str]:
        """Return validation error strings (empty list = valid).

        Uses validate_fn when provided, otherwise checks required_data_fields.
        """
        if self.validate_fn is not None:
            return self.validate_fn(data)
        errors: List[str] = []
        for fname in self.required_data_fields:
            val = getattr(data, fname, None)
            if val in (None, "", []):
                errors.append(
                    f"data.{fname} is required for dataset '{self.id}'"
                )
        return errors

    def load(self, data_config: "DataConfig") -> "QueryDataset":
        """Instantiate and return the dataset for *data_config*."""
        if self.load_fn is None:
            raise RuntimeError(
                f"No load_fn registered for dataset '{self.id}'"
            )
        return self.load_fn(data_config)


# ── Registry ──────────────────────────────────────────────────────────

_DESCRIPTOR_REGISTRY: Dict[str, DatasetDescriptor] = {}


def register_dataset(descriptor: DatasetDescriptor) -> DatasetDescriptor:
    """Register *descriptor* under its id.  Returns the descriptor unchanged."""
    _DESCRIPTOR_REGISTRY[descriptor.id] = descriptor
    return descriptor


def get_descriptor(dataset_id: str) -> Optional[DatasetDescriptor]:
    """Return descriptor by id, or None if not registered."""
    return _DESCRIPTOR_REGISTRY.get(dataset_id)


def list_registered_datasets() -> List[str]:
    """Return sorted list of registered dataset ids."""
    return sorted(_DESCRIPTOR_REGISTRY.keys())


def resolve_dataset_descriptor(data: "DataConfig") -> DatasetDescriptor:
    """Resolve the DatasetDescriptor for *data*.

    Resolution order (matches legacy ``resolve_dataset_runtime_spec`` priority):

    1. ``prepared_dataset_dir`` or ``questions_path`` set → ``pubmed_qa``
       (These explicit path fields take priority so configs that use the old
       default ``dataset_name="admed_voice"`` together with ``questions_path``
       still load the PubMed QA dataset, preserving backward compatibility.)
    2. ``data.dataset_name`` matches a registered id directly.
    3. ``dataset_source`` field: ``"local"`` → ``local``,
       ``"huggingface"`` → ``huggingface``.
    4. ``data.dataset_type`` matches any registered descriptor's dataset_type.

    Raises :class:`~evaluator.errors.ConfigurationError` when nothing matches.
    """
    # 1. Explicit file-path fields (highest priority — mirrors old runtime_spec logic)
    if getattr(data, "prepared_dataset_dir", None) or getattr(data, "questions_path", None):
        if "pubmed_qa" in _DESCRIPTOR_REGISTRY:
            return _DESCRIPTOR_REGISTRY["pubmed_qa"]

    # 2. Direct name lookup
    name = getattr(data, "dataset_name", None)
    if name and name in _DESCRIPTOR_REGISTRY:
        return _DESCRIPTOR_REGISTRY[name]

    # 3. Legacy dataset_source field dispatch
    source = getattr(data, "dataset_source", None)
    if source == "local" and "local" in _DESCRIPTOR_REGISTRY:
        return _DESCRIPTOR_REGISTRY["local"]
    if source == "huggingface" and "huggingface" in _DESCRIPTOR_REGISTRY:
        return _DESCRIPTOR_REGISTRY["huggingface"]

    # 4. Dataset-type fallback — first matching descriptor wins
    dtype = getattr(data, "dataset_type", None)
    if dtype is not None:
        for desc in _DESCRIPTOR_REGISTRY.values():
            if desc.dataset_type == dtype:
                return desc

    known = ", ".join(sorted(_DESCRIPTOR_REGISTRY.keys()))
    raise ConfigurationError(
        f"Cannot resolve dataset descriptor for dataset_name='{name}'. "
        f"Known datasets: {known}"
    )


# ── Built-in load / validate functions ───────────────────────────────

def _load_admed_voice(data: "DataConfig") -> "QueryDataset":
    from .core import load_admed_voice_corpus
    from .runtime import AudioSamplesQueryDataset
    from .loaders.base import AudioSample

    test_size = getattr(data, "test_size", 0.0)
    train_df, _ = load_admed_voice_corpus(test_size=test_size)
    from .core import load_audio_file

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


def _load_pubmed_qa(data: "DataConfig") -> "QueryDataset":
    from pathlib import Path
    from ..errors import ConfigurationError as _CE
    from .core import PubMedQADataset
    from .runtime import LazyAudioQueryDataset

    if getattr(data, "prepared_dataset_dir", None):
        dataset_dir = Path(data.prepared_dataset_dir)
        if not dataset_dir.exists():
            raise _CE(
                f"Prepared dataset directory not found: {data.prepared_dataset_dir}"
            )
        questions_path = dataset_dir / "questions.json"
        corpus_path = dataset_dir / "corpus.json"
        if not questions_path.exists() or not corpus_path.exists():
            raise _CE(
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


def _validate_pubmed_qa(data: "DataConfig") -> List[str]:
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


def _load_local(data: "DataConfig") -> "QueryDataset":
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


def _load_huggingface(data: "DataConfig") -> "QueryDataset":
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


# ── Built-in descriptor registrations ────────────────────────────────

register_dataset(DatasetDescriptor(
    id="admed_voice",
    description="ADMED Voice: Polish medical spoken-query dataset",
    dataset_type=DatasetType.AUDIO_TRANSCRIPTION,
    requires_audio=True,
    requires_text=False,
    supports_generation=False,
    evaluation_mode="transcription",
    compatible_pipeline_modes=("asr_only", "asr_text_retrieval"),
    load_fn=_load_admed_voice,
))

register_dataset(DatasetDescriptor(
    id="pubmed_qa",
    description="PubMed QA: biomedical QA with audio queries and document corpus",
    dataset_type=DatasetType.MULTIMODAL_QA,
    requires_audio=True,
    requires_text=True,
    supports_generation=True,
    evaluation_mode="qa_retrieval",
    compatible_pipeline_modes=("asr_text_retrieval", "audio_text_retrieval"),
    load_fn=_load_pubmed_qa,
    validate_fn=_validate_pubmed_qa,
))

register_dataset(DatasetDescriptor(
    id="local",
    description="Local audio directory with optional corpus",
    dataset_type=DatasetType.AUDIO_QUERY_RETRIEVAL,
    requires_audio=True,
    requires_text=False,
    supports_generation=False,
    evaluation_mode="retrieval",
    compatible_pipeline_modes=(
        "asr_only", "asr_text_retrieval",
        "audio_emb_retrieval", "audio_text_retrieval",
    ),
    required_data_fields=("audio_dir",),
    load_fn=_load_local,
))

register_dataset(DatasetDescriptor(
    id="huggingface",
    description="HuggingFace Hub audio dataset",
    dataset_type=DatasetType.AUDIO_QUERY_RETRIEVAL,
    requires_audio=True,
    requires_text=False,
    supports_generation=False,
    evaluation_mode="retrieval",
    compatible_pipeline_modes=(
        "asr_only", "asr_text_retrieval",
        "audio_emb_retrieval", "audio_text_retrieval",
    ),
    required_data_fields=("huggingface_dataset",),
    load_fn=_load_huggingface,
))


def _validate_fleurs(data: "DataConfig") -> List[str]:
    errors: List[str] = []
    if not getattr(data, "huggingface_subset", None):
        errors.append(
            "FLEURS requires huggingface_subset (e.g. 'pl_pl'). "
            "Without it the loader would attempt to stream all 102 languages."
        )
    return errors


def _load_fleurs(data: "DataConfig") -> "QueryDataset":
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


register_dataset(DatasetDescriptor(
    id="fleurs",
    description="Google FLEURS: 102-language ASR benchmark (requires huggingface_subset)",
    dataset_type=DatasetType.AUDIO_TRANSCRIPTION,
    requires_audio=True,
    requires_text=False,
    supports_generation=False,
    evaluation_mode="transcription",
    compatible_pipeline_modes=("asr_only", "asr_text_retrieval"),
    required_data_fields=(),
    load_fn=_load_fleurs,
    validate_fn=_validate_fleurs,
))
