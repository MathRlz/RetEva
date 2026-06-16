"""Unified dataset descriptor registry.

Each DatasetDescriptor combines capability profile (what the dataset needs /
supports) with runtime spec (validation rules + loader).  New datasets register
once via :func:`register_dataset` — no other files need to be touched.

This module is pure machinery (the descriptor, the registry, resolution). The
built-in datasets are defined as per-type ABC subclasses in :mod:`.builtins`
(imported for its registration side-effect by ``datasets/__init__``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING

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


# ── Default column schema per dataset type ────────────────────────────
# Column name → registered artifact (§2 "field → registered artifact"). A descriptor
# may declare its own `fields`; these defaults keep builtins terse. The dataset_source
# node advertises exactly these columns on the DAG (name + modality), so the diagram
# reads what data an experiment consumes.

FIELDS_BY_DATASET_TYPE: Dict[DatasetType, Dict[str, str]] = {
    DatasetType.AUDIO_TRANSCRIPTION: {
        "audio": "query_audio",
        "transcription": "query_text",
    },
    DatasetType.AUDIO_QUERY_RETRIEVAL: {
        "audio": "query_audio",
        "documents": "corpus",
        "relevance": "relevant_docs",
    },
    DatasetType.TEXT_QUERY_RETRIEVAL: {
        "question": "query_text",
        "documents": "corpus",
        "relevance": "relevant_docs",
    },
    DatasetType.MULTIMODAL_QA: {
        "question": "query_text",
        "audio": "query_audio",
        "documents": "corpus",
        "relevance": "relevant_docs",
        "answers": "short_answers",
    },
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
        fields: Column schema — ``{column_name: registered_artifact_name}``. The
            dataset_source node advertises these columns (name + modality) on the
            DAG; auto-filled from ``FIELDS_BY_DATASET_TYPE`` when left empty and
            validated against the artifact registry.
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
    fields: Mapping[str, str] = field(default_factory=dict)
    #: Free-form subject-area tag for grouping the dataset picker (e.g. "medical",
    #: "multilingual", "general"). Modality is derived from ``dataset_type``.
    domain: str = "general"
    #: Embedding-space id for precomputed-vector columns (§4.1 T4) — set when a
    #: column maps to query_vectors/corpus_vectors so retrieval's V[s] pairing holds.
    embedding_space: Optional[str] = None
    #: Splits the dataset offers (empty = no split concept; the picker stays hidden).
    splits: Sequence[str] = ()
    #: The split used when the config names none; first declared split if unset.
    default_split: Optional[str] = None
    load_fn: Optional[Callable[["DataConfig"], "QueryDataset"]] = field(
        default=None, repr=False
    )
    validate_fn: Optional[Callable[["DataConfig"], List[str]]] = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        if self.splits and not self.default_split:
            self.default_split = str(self.splits[0])
        if not self.default_metrics:
            self.default_metrics = list(METRICS_BY_MODE.get(self.evaluation_mode, ()))
        if not self.fields:
            self.fields = dict(FIELDS_BY_DATASET_TYPE.get(self.dataset_type, {}))
        if self.fields:
            # Fail loud at registration on a typo'd artifact name (§2 contract).
            from ..pipeline.artifacts import validate_field_mapping

            validate_field_mapping(self.fields)

    # ── Public helpers ────────────────────────────────────────────────

    def supports_pipeline_mode(self, mode: str) -> bool:
        """Return True when *mode* is listed in compatible_pipeline_modes."""
        return mode in set(self.compatible_pipeline_modes)

    def validate_data_config(self, data: "DataConfig") -> List[str]:
        """Return validation error strings (empty list = valid).

        Uses validate_fn when provided, otherwise checks required_data_fields.
        """
        split_errors = self._validate_split(data)
        if self.validate_fn is not None:
            return split_errors + self.validate_fn(data)
        errors: List[str] = list(split_errors)
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

    def _validate_split(self, data: "DataConfig") -> List[str]:
        """Reject a configured split the dataset doesn't declare (no-op without splits)."""
        if not self.splits:
            return []
        chosen = getattr(data, "huggingface_split", None)
        if chosen in (None, "") or chosen in set(self.splits):
            return []
        return [
            f"dataset '{self.id}' has no split '{chosen}' "
            f"(available: {', '.join(self.splits)})"
        ]


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
