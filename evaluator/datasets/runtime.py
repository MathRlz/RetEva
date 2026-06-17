"""Runtime dataset contract and registry-backed loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..errors import ConfigurationError
from ..logging_config import get_logger
from .core import (
    QueryDataset,
    load_audio_file as _load_audio_waveform,
)

logger = get_logger(__name__)


class AudioSamplesQueryDataset(QueryDataset):
    """Adapter converting loader AudioSample rows into QueryDataset contract."""

    def __init__(
        self,
        samples: List[Any],
        *,
        trace_limit: int = 0,
        corpus_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.samples = (
            samples[:trace_limit] if trace_limit and trace_limit > 0 else samples
        )
        self.corpus_entries = corpus_entries or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        metadata = dict(getattr(sample, "metadata", {}) or {})
        sample_id = str(getattr(sample, "sample_id", "") or f"sample_{idx}")
        gt_ids = (
            metadata.get("groundtruth_doc_ids")
            or metadata.get("relevant_doc_ids")
            or []
        )
        if isinstance(gt_ids, str):
            gt_ids = [gt_ids]
        if not isinstance(gt_ids, list):
            gt_ids = []
        relevance = metadata.get("relevance_grades") or {}
        if not isinstance(relevance, dict):
            relevance = {}
        if not relevance and gt_ids:
            relevance = {str(doc_id): 1 for doc_id in gt_ids}
        question_text = str(getattr(sample, "transcription", ""))
        return {
            "audio_array": getattr(sample, "audio_array"),
            "sampling_rate": int(getattr(sample, "sampling_rate", 16000)),
            "transcription": question_text,
            "question_text": question_text,
            "question_id": sample_id,
            "groundtruth_doc_ids": [str(doc_id) for doc_id in gt_ids],
            "relevance_grades": {str(k): int(v) for k, v in relevance.items()},
            "language": str(getattr(sample, "language", "en")),
            "metadata": metadata,
        }

    def get_corpus(self) -> List[Dict[str, Any]]:
        if self.corpus_entries:
            return self.corpus_entries
        # CLI parity: derive a text corpus from the samples' unique transcriptions so
        # corpus-less audio datasets (e.g. admed_voice) support text retrieval. doc_id
        # is the transcription text, matching the ``{transcription: 1}`` relevance
        # fallback in ``_build_relevant_from_item`` (mirrors the CLI's transcription
        # corpus in ``cli/commands.py``).
        #
        # M5: deduplicating by text is CORRECT here, not under-counting — because the
        # relevance key is the transcription text (== doc_id), two samples that share a
        # transcription correctly point at the same single corpus doc (retrievable at rank 1
        # for both). We only log the collapse so a smaller-than-sample-count corpus is
        # explainable, not a surprise.
        derived: Dict[str, Dict[str, Any]] = {}
        for sample in self.samples:
            text = str(getattr(sample, "transcription", "") or "").strip()
            if text and text not in derived:
                derived[text] = {"doc_id": text, "text": text}
        n_with_text = sum(
            1 for s in self.samples if str(getattr(s, "transcription", "") or "").strip()
        )
        if len(derived) < n_with_text:
            logger.debug(
                "self-retrieval corpus: %d samples → %d unique transcription docs "
                "(%d duplicate transcriptions collapsed; relevance is text-keyed so this "
                "is exact, not lossy)",
                n_with_text, len(derived), n_with_text - len(derived),
            )
        return list(derived.values())


def _load_corpus_entries(path: str) -> List[Dict[str, Any]]:
    """Load a corpus file → row dicts with shared field-name fallbacks (B5/F19: reuses
    ``core.load_json_or_jsonl`` for the JSON/JSONL + corpus/documents/items unwrap)."""
    from .core import load_json_or_jsonl, parse_corpus_row

    corpus_path = Path(path)
    if not corpus_path.exists():
        raise ConfigurationError(f"Corpus file not found: {path}")
    try:
        rows = load_json_or_jsonl(corpus_path)
    except (ValueError, FileNotFoundError) as exc:
        raise ConfigurationError(f"Could not load corpus {path}: {exc}") from exc

    corpus: List[Dict[str, Any]] = []
    for row in rows:
        fields = parse_corpus_row(row)  # shared field-name fallbacks
        if fields is not None:  # rows missing doc_id/text are skipped
            corpus.append({**fields, "language": str(fields["language"])})
    return corpus


class LazyAudioQueryDataset(QueryDataset):
    """QueryDataset backed by BenchmarkQuestion + CorpusDocument lists.

    Audio is loaded lazily on ``__getitem__`` via torchaudio, so large datasets
    do not require loading all waveforms into memory at init time.  Used by the
    descriptor registry as the implementation for ``pubmed_qa`` and any other
    dataset whose questions have ``audio_path`` set.
    """

    def __init__(
        self,
        questions: List[Any],
        corpus: List[Any],
        *,
        trace_limit: int = 0,
    ) -> None:
        self.questions: List[Any] = (
            questions[:trace_limit] if trace_limit and trace_limit > 0 else questions
        )
        self._corpus: List[Any] = corpus

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        question = self.questions[idx]
        if not question.audio_path:
            raise ValueError(
                f"Question {question.question_id} has no audio_path. "
                "Enable audio_synthesis in config or provide pre-recorded audio."
            )
        waveform, sample_rate = _load_audio_waveform(question.audio_path)
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

    def get_corpus(self) -> List[Dict[str, Any]]:
        return [
            {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "title": doc.title,
                "language": doc.language,
                "metadata": doc.metadata,
            }
            for doc in self._corpus
        ]


def validate_dataset_runtime_config(config: Any, *, retrieval_required: bool = False):
    """Validate the dataset config via its descriptor (the single source of truth).

    Returns the resolved :class:`~evaluator.datasets.descriptor.DatasetDescriptor`.
    ``retrieval_required`` rejects datasets none of whose compatible pipeline modes
    retrieve (replaces the retired DatasetRuntimeSpec.supports_corpus check — every
    retired spec declared corpus support, so default-mode behavior is unchanged).
    """
    from .descriptor import resolve_dataset_descriptor

    descriptor = resolve_dataset_descriptor(config.data)
    errors = descriptor.validate_data_config(config.data)
    if errors:
        raise ConfigurationError(
            f"Dataset '{descriptor.id}' configuration invalid: " + "; ".join(errors)
        )
    if retrieval_required and not any(
        "retrieval" in str(mode) for mode in descriptor.compatible_pipeline_modes
    ):
        raise ConfigurationError(
            f"Dataset '{descriptor.id}' supports no retrieval pipeline mode."
        )
    return descriptor


def load_runtime_dataset(config: Any) -> QueryDataset:
    """Load the dataset described by *config* via the descriptor registry.

    Delegates to :func:`~evaluator.datasets.descriptor.resolve_dataset_descriptor`
    so that custom datasets registered via
    :func:`~evaluator.datasets.descriptor.register_dataset` are automatically
    supported without any changes here.
    """
    from .descriptor import resolve_dataset_descriptor

    descriptor = resolve_dataset_descriptor(config.data)
    errors = descriptor.validate_data_config(config.data)
    if errors:
        raise ConfigurationError("; ".join(errors))
    return descriptor.load(config.data)


def load_runtime_datasets(config: Any) -> "Dict[str, QueryDataset]":
    """Load every entry of a multi-source ``data.datasets`` map → ``{id: QueryDataset}`` (B1).

    Each entry already holds ``DataConfig`` field names (``questions_path``/``corpus_path``/…, from
    ``graph_config._DATASET_FIELDS``) plus an optional ``role``; a per-entry ``DataConfig`` is built
    by overlaying those fields onto ``config.data`` and loaded via :func:`load_runtime_dataset`.
    Returns ``{}`` in single-source mode (no ``datasets`` map) — the scalar ``config.data`` fields
    drive the single load as before, so single-source runs are unchanged."""
    from dataclasses import replace

    sources = getattr(getattr(config, "data", None), "datasets", None)
    if not sources:
        return {}
    out: "Dict[str, QueryDataset]" = {}
    for sid, entry in sources.items():
        fields = {
            k: v for k, v in (entry or {}).items() if k not in ("role", "datasets")
        }
        sub_data = replace(config.data, datasets=None, **fields)
        sub_config = replace(config, data=sub_data)
        out[str(sid)] = load_runtime_dataset(sub_config)
    return out


def _corpus_doc_ids(dataset: Any) -> set:
    if not hasattr(dataset, "get_corpus"):
        return set()  # legitimately corpus-less dataset
    try:
        return {str(d.get("doc_id", "")) for d in dataset.get_corpus()} - {""}
    except Exception as exc:
        # A raising get_corpus() (broken/malformed corpus) used to be swallowed silently,
        # which makes validate_dataset_join read an empty corpus and disable all IR metrics
        # with no trace (H1). Keep the run alive but surface it loudly.
        logger.warning(
            "could not read corpus doc ids (%s): %s; IR-metric join check will treat the "
            "corpus as empty", type(exc).__name__, exc,
        )
        return set()


def _relevant_doc_ids(dataset: Any) -> set:
    out: set = set()
    for q in getattr(dataset, "questions", None) or []:
        for d in getattr(q, "groundtruth_doc_ids", None) or []:
            out.add(str(d))
    return out


def validate_dataset_join(
    questions_dataset: Any, corpus_dataset: Any
) -> "Dict[str, Any]":
    """Check the cross-dataset join contract (B5): do the questions' relevant `doc_id`s overlap
    the corpus's `doc_id`s? IR metrics are only meaningful when they do — disjoint namespaces mean
    every relevant doc is absent from the corpus, so Recall/NDCG/… would be a misleading 0 rather
    than "not applicable". Returns ``{overlap, n_relevant, n_corpus, disjoint, warning}``; the
    caller disables IR metrics (QA/judge still run) when ``disjoint``."""
    relevant = _relevant_doc_ids(questions_dataset)
    corpus = _corpus_doc_ids(corpus_dataset)
    overlap = relevant & corpus
    disjoint = bool(relevant) and bool(corpus) and not overlap
    warning = ""
    if disjoint:
        warning = (
            f"Disjoint doc_id namespaces: {len(relevant)} relevant ids share 0 with "
            f"{len(corpus)} corpus ids — IR metrics disabled (QA/judge still run)."
        )
    return {
        "overlap": len(overlap),
        "n_relevant": len(relevant),
        "n_corpus": len(corpus),
        "disjoint": disjoint,
        "warning": warning,
    }


class _QueryIdSubset(QueryDataset):
    """A query-side slice of a dataset (Roadmap 2d, item replay): only the rows at
    ``indices`` are visible, but ``get_corpus`` passes through whole — the corpus is shared,
    so retrieval scores match a full run while the query set shrinks to the ids of interest."""

    def __init__(self, base: Any, indices: List[int]) -> None:
        self._base = base
        self._indices = list(indices)
        # Mirror the base's id-bearing list (sliced) so any reader sees the same subset.
        if hasattr(base, "questions"):
            self.questions = [base.questions[i] for i in indices]
        if hasattr(base, "samples"):
            self.samples = [base.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._base[self._indices[idx]]

    def get_corpus(self) -> List[Dict[str, Any]]:
        getter = getattr(self._base, "get_corpus", None)
        return getter() if callable(getter) else []

    def __getattr__(self, name: str) -> Any:
        # Delegate anything not overridden (e.g. corpus_entries) to the base dataset.
        return getattr(self._base, name)


def _row_id(base: Any, idx: int) -> str:
    """The query id for row ``idx`` via a light source (no audio decode) when available."""
    questions = getattr(base, "questions", None)
    if questions is not None:
        return str(getattr(questions[idx], "question_id", "") or "")
    samples = getattr(base, "samples", None)
    if samples is not None:
        return str(getattr(samples[idx], "sample_id", "") or "")
    row = base[idx]  # fallback: materialize the row (may decode audio)
    return str(row.get("question_id") or row.get("sample_id") or row.get("query_id") or "")


def slice_by_query_ids(dataset: Any, query_ids: Any) -> "_QueryIdSubset":
    """Slice ``dataset`` to the rows whose query id is in ``query_ids`` (corpus untouched).

    Used by ``evaluator replay`` to re-run a single item through the full graph. Returns a
    :class:`_QueryIdSubset` (possibly empty — the caller reports an unknown id)."""
    wanted = {str(q) for q in query_ids}
    indices = [i for i in range(len(dataset)) if _row_id(dataset, i) in wanted]
    return _QueryIdSubset(dataset, indices)
