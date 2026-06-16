"""Per-type abstract base classes for adding datasets.

Pick the base class that matches your data, set a few class attributes, implement
``from_config`` (and ``__len__``/``__getitem__``/``get_corpus`` from
:class:`~evaluator.datasets.core.QueryDataset`), then decorate with
``@register_eval_dataset(id=...)``. The :class:`DatasetDescriptor` (capabilities +
validation + loader) is derived from the class — you don't hand-fill it.

TTS bridge: the pipeline modes are audio-first. Set ``supports_generation = True`` on a
text dataset and it gains the audio pipeline modes — audio is synthesised from the text
at run time (query audio always; corpus audio for ``audio_emb_retrieval``). This is how a
text-retrieval dataset becomes an audio-retrieval dataset.

Example::

    @register_eval_dataset(id="my_qa")
    class MyQA(MultimodalQADataset):           # text QA, TTS-backed audio
        required_data_fields = ("questions_path", "corpus_path")

        @classmethod
        def from_config(cls, data):
            return cls(...)

        def get_corpus(self):
            return [{"doc_id": d.id, "text": d.text} for d in self._docs]
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Mapping, TYPE_CHECKING, List, Sequence

from ..config.types import DatasetType
from .core import QueryDataset
from .descriptor import METRICS_BY_MODE, DatasetDescriptor, register_dataset

if TYPE_CHECKING:
    from ..config.data import DataConfig

# Pipeline modes that consume audio input. Unlocked for text datasets via TTS generation.
_AUDIO_MODES: Sequence[str] = (
    "asr_only", "asr_text_retrieval", "audio_text_retrieval", "audio_emb_retrieval",
)


class EvalDataset(QueryDataset):
    """Base for user datasets. Declare capabilities as class attributes.

    Concrete subclasses implement ``from_config`` plus the
    :class:`~evaluator.datasets.core.QueryDataset` interface (``__len__``/``__getitem__``
    and ``get_corpus`` for retrieval datasets).
    """

    # Capability metadata — overridden by the type bases / concrete classes.
    dataset_type: DatasetType = DatasetType.AUDIO_TRANSCRIPTION
    requires_audio: bool = False
    requires_text: bool = True
    supports_generation: bool = False
    evaluation_mode: str = "retrieval"
    #: Subject-area tag for the categorized picker (e.g. "medical"); modality is derived
    #: from ``dataset_type``.
    domain: str = "general"
    native_pipeline_modes: Sequence[str] = ()
    required_data_fields: Sequence[str] = ()
    #: Column schema {name: registered artifact}; empty = per-type default
    #: (descriptor.FIELDS_BY_DATASET_TYPE). Shown on the DAG's dataset node.
    fields: Mapping[str, str] = {}
    #: Embedding-space id when a column carries precomputed vectors (§4.1 T4).
    embedding_space: str | None = None
    #: Splits the dataset offers (empty = none); default_split used when unset.
    splits: Sequence[str] = ()
    default_split: str | None = None
    description: str = ""

    @classmethod
    @abstractmethod
    def from_config(cls, data: "DataConfig") -> "EvalDataset":
        """Load and return a dataset instance for the given DataConfig."""

    @classmethod
    def validate(cls, data: "DataConfig") -> List[str]:
        """Return validation errors (empty list = valid)."""
        errors: List[str] = []
        for field_name in cls.required_data_fields:
            if getattr(data, field_name, None) in (None, "", []):
                errors.append(f"data.{field_name} is required for dataset '{cls._dataset_id()}'")
        return errors

    @classmethod
    def compatible_pipeline_modes(cls) -> Sequence[str]:
        """Native modes, plus the audio modes when TTS generation is supported."""
        modes = list(cls.native_pipeline_modes)
        if cls.supports_generation:
            modes += [m for m in _AUDIO_MODES if m not in modes]
        return tuple(modes)

    @classmethod
    def default_metrics(cls) -> Sequence[str]:
        return tuple(METRICS_BY_MODE.get(cls.evaluation_mode, ()))

    @classmethod
    def _dataset_id(cls) -> str:
        return getattr(cls, "_registered_id", cls.__name__)


# ── Type base classes ────────────────────────────────────────────────────────

class AudioTranscriptionDataset(EvalDataset):
    """Audio in, transcription out. Any retrieval corpus is derived from transcriptions."""

    dataset_type = DatasetType.AUDIO_TRANSCRIPTION
    requires_audio = True
    requires_text = False
    evaluation_mode = "transcription"
    native_pipeline_modes = ("asr_only", "asr_text_retrieval")


class AudioRetrievalDataset(EvalDataset):
    """Audio query evaluated against a corpus (audio or text)."""

    dataset_type = DatasetType.AUDIO_QUERY_RETRIEVAL
    requires_audio = True
    requires_text = True
    evaluation_mode = "retrieval"
    native_pipeline_modes = (
        "asr_only", "asr_text_retrieval", "audio_text_retrieval", "audio_emb_retrieval",
    )


class TextRetrievalDataset(EvalDataset):
    """Text query against a text corpus.

    The pipeline modes are audio-first, so set ``supports_generation = True`` to run via
    TTS (this unlocks the audio modes and synthesises audio from the text at run time).
    """

    dataset_type = DatasetType.TEXT_QUERY_RETRIEVAL
    requires_audio = False
    requires_text = True
    evaluation_mode = "retrieval"
    native_pipeline_modes = ()


class MultimodalQADataset(TextRetrievalDataset):
    """Text QA with a corpus, run as audio retrieval via TTS (generation on by default)."""

    dataset_type = DatasetType.MULTIMODAL_QA
    supports_generation = True
    evaluation_mode = "qa_retrieval"


# ── Class-based registration ─────────────────────────────────────────────────

def register_eval_dataset(*, id: str, description: str = ""):
    """Class decorator: register an :class:`EvalDataset` subclass as a dataset.

    Derives a :class:`DatasetDescriptor` from the class attributes (``load_fn`` =
    ``cls.from_config``, ``validate_fn`` = ``cls.validate``). The function-based
    :func:`~evaluator.datasets.descriptor.register_dataset` remains available for
    advanced cases.
    """
    def decorator(cls):
        cls._registered_id = id
        desc = description or cls.description or (cls.__doc__ or "").strip().split("\n", 1)[0] or id
        register_dataset(DatasetDescriptor(
            id=id,
            description=desc,
            dataset_type=cls.dataset_type,
            requires_audio=cls.requires_audio,
            requires_text=cls.requires_text,
            supports_generation=cls.supports_generation,
            evaluation_mode=cls.evaluation_mode,
            domain=cls.domain,
            compatible_pipeline_modes=cls.compatible_pipeline_modes(),
            default_metrics=list(cls.default_metrics()),
            required_data_fields=list(cls.required_data_fields),
            fields=dict(cls.fields),
            embedding_space=cls.embedding_space,
            splits=tuple(cls.splits),
            default_split=cls.default_split,
            load_fn=cls.from_config,
            validate_fn=cls.validate,
        ))
        return cls

    return decorator
