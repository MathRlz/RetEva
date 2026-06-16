"""Runtime dataset contract and registry-backed loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..errors import ConfigurationError
from .core import (
    QueryDataset,
    load_audio_file as _load_audio_waveform,
)


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
        derived: Dict[str, Dict[str, Any]] = {}
        for sample in self.samples:
            text = str(getattr(sample, "transcription", "") or "").strip()
            if text and text not in derived:
                derived[text] = {"doc_id": text, "text": text}
        return list(derived.values())


def _load_corpus_entries(path: str) -> List[Dict[str, Any]]:
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise ConfigurationError(f"Corpus file not found: {path}")
    rows: List[Dict[str, Any]]
    if corpus_path.suffix == ".jsonl":
        rows = []
        with corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with corpus_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, list):
            rows = loaded
        elif isinstance(loaded, dict):
            if "corpus" in loaded and isinstance(loaded["corpus"], list):
                rows = loaded["corpus"]
            elif "documents" in loaded and isinstance(loaded["documents"], list):
                rows = loaded["documents"]
            elif "items" in loaded and isinstance(loaded["items"], list):
                rows = loaded["items"]
            else:
                raise ConfigurationError(
                    f"Unsupported corpus JSON structure in {path}; expected list or corpus/documents/items key"
                )
        else:
            raise ConfigurationError(f"Unsupported corpus file format in {path}")
    from .core import parse_corpus_row

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
        return set()
    try:
        return {str(d.get("doc_id", "")) for d in dataset.get_corpus()} - {""}
    except Exception:
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
    than "not applicable". Returns ``{overlap, n_relevant, n_corpus, disjoint, warning}``; the caller
    disables IR metrics (QA/judge still run) when ``disjoint``."""
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
