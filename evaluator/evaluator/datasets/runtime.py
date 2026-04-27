"""Runtime dataset contract and registry-backed loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..errors import ConfigurationError
from .core import AdmedQueryDataset, PubMedQADataset, QueryDataset, load_admed_voice_corpus, load_pubmed_qa_dataset
from .loaders.factory import create_dataset_loader


@dataclass(frozen=True)
class DatasetRuntimeSpec:
    """Declarative dataset runtime spec for evaluation entrypoints."""

    id: str
    source: str
    description: str
    required_fields: tuple[str, ...]
    supports_corpus: bool


def _get_path_value(config: Any, dotted_path: str) -> Any:
    current: Any = config
    for part in dotted_path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


class AudioSamplesQueryDataset(QueryDataset):
    """Adapter converting loader AudioSample rows into QueryDataset contract."""

    def __init__(
        self,
        samples: List[Any],
        *,
        trace_limit: int = 0,
        corpus_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.samples = samples[:trace_limit] if trace_limit and trace_limit > 0 else samples
        self.corpus_entries = corpus_entries or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        metadata = dict(getattr(sample, "metadata", {}) or {})
        sample_id = str(getattr(sample, "sample_id", "") or f"sample_{idx}")
        gt_ids = metadata.get("groundtruth_doc_ids") or metadata.get("relevant_doc_ids") or []
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
        return self.corpus_entries


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
    corpus: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        doc_id = str(row.get("doc_id") or row.get("id") or row.get("pmid") or "")
        text = row.get("text") or row.get("abstract") or row.get("content")
        if not doc_id or text is None:
            continue
        corpus.append(
            {
                "doc_id": doc_id,
                "text": str(text),
                "title": str(row.get("title", "")),
                "language": str(row.get("language", "en")),
                "metadata": row.get("metadata", {}),
            }
        )
    return corpus


def list_dataset_runtime_specs() -> List[DatasetRuntimeSpec]:
    return [
        DatasetRuntimeSpec(
            id="prepared_pubmed",
            source="prepared_dataset_dir",
            description="Prepared benchmark directory with questions.json/corpus.json",
            required_fields=("data.prepared_dataset_dir",),
            supports_corpus=True,
        ),
        DatasetRuntimeSpec(
            id="pubmed_paths",
            source="questions_path",
            description="PubMed QA style questions/corpus files",
            required_fields=("data.questions_path", "data.corpus_path"),
            supports_corpus=True,
        ),
        DatasetRuntimeSpec(
            id="admed_voice",
            source="dataset_name",
            description="Built-in admed_voice dataset",
            required_fields=("data.dataset_name",),
            supports_corpus=False,
        ),
        DatasetRuntimeSpec(
            id="loader_local",
            source="dataset_source",
            description="Local audio directory dataset loader",
            required_fields=("data.dataset_source", "data.audio_dir"),
            supports_corpus=True,
        ),
        DatasetRuntimeSpec(
            id="loader_huggingface",
            source="dataset_source",
            description="HuggingFace audio dataset loader",
            required_fields=("data.dataset_source", "data.huggingface_dataset"),
            supports_corpus=True,
        ),
    ]


def resolve_dataset_runtime_spec(config: Any) -> DatasetRuntimeSpec:
    if config.data.prepared_dataset_dir:
        return list_dataset_runtime_specs()[0]
    if config.data.questions_path:
        return list_dataset_runtime_specs()[1]
    if config.data.dataset_name == "admed_voice":
        return list_dataset_runtime_specs()[2]
    if config.data.dataset_source == "local":
        return list_dataset_runtime_specs()[3]
    if config.data.dataset_source == "huggingface":
        return list_dataset_runtime_specs()[4]
    raise ConfigurationError(
        "No supported data source specified. Use one of: "
        "prepared_dataset_dir/questions_path, dataset_name=admed_voice, "
        "or data.dataset_source in {local, huggingface} with matching fields."
    )


def validate_dataset_runtime_config(config: Any, *, retrieval_required: bool = False) -> DatasetRuntimeSpec:
    """Validate dataset runtime requirements and return resolved runtime spec."""
    spec = resolve_dataset_runtime_spec(config)
    missing = [
        field_name
        for field_name in spec.required_fields
        if _get_path_value(config, field_name) in (None, "", [])
    ]
    if missing:
        raise ConfigurationError(
            f"Dataset runtime '{spec.id}' missing required fields: {', '.join(missing)}"
        )
    if retrieval_required and not spec.supports_corpus:
        raise ConfigurationError(
            f"Dataset runtime '{spec.id}' does not provide corpus required for retrieval pipelines."
        )
    return spec


def load_runtime_dataset(config: Any) -> QueryDataset:
    spec = validate_dataset_runtime_config(config)

    if spec.id == "prepared_pubmed":
        dataset_dir = Path(config.data.prepared_dataset_dir)
        if not dataset_dir.exists():
            raise ConfigurationError(
                f"Prepared dataset directory not found: {config.data.prepared_dataset_dir}"
            )
        questions_path = dataset_dir / "questions.json"
        corpus_path = dataset_dir / "corpus.json"
        if not questions_path.exists() or not corpus_path.exists():
            raise ConfigurationError(
                "Prepared dataset directory must contain questions.json and corpus.json"
            )
        return load_pubmed_qa_dataset(
            str(questions_path),
            str(corpus_path),
            trace_limit=config.data.trace_limit,
        )

    if spec.id == "pubmed_paths":
        questions_path = Path(config.data.questions_path)
        if not questions_path.exists():
            raise ConfigurationError(
                f"Questions file not found: {config.data.questions_path}"
            )
        if not config.data.corpus_path:
            raise ConfigurationError(
                "Missing data.corpus_path for pubmed_qa dataset."
            )
        return PubMedQADataset(
            questions_path=str(questions_path),
            corpus_path=config.data.corpus_path,
            trace_limit=config.data.trace_limit,
        )

    if spec.id == "admed_voice":
        train_set, _ = load_admed_voice_corpus(test_size=config.data.test_size)
        return AdmedQueryDataset(train_set)

    loader = create_dataset_loader(
        source=config.data.dataset_source,
        audio_dir=config.data.audio_dir,
        transcripts_file=config.data.transcripts_file,
        huggingface_dataset=config.data.huggingface_dataset,
        huggingface_subset=config.data.huggingface_subset,
        huggingface_split=config.data.huggingface_split,
        column_mapping=config.data.column_mapping,
        max_samples=config.data.max_samples,
        default_language=config.data.default_language,
        sample_rate=config.data.sample_rate,
    )
    samples = loader.load()
    corpus_entries = _load_corpus_entries(config.data.corpus_path) if config.data.corpus_path else []
    return AudioSamplesQueryDataset(
        samples,
        trace_limit=config.data.trace_limit,
        corpus_entries=corpus_entries,
    )
