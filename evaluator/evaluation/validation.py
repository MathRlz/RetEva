"""Validation helpers for benchmark datasets and configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..datasets.utils import DatasetLoader


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"File does not exist: {path}")

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


def validate_pubmed_dataset(questions_path: str, corpus_path: str, verify_audio: bool = True) -> Tuple[int, int]:
    """Validate PubMed QA dataset files and return (question_count, corpus_count).

    Raises ValueError with actionable errors when validation fails.
    """
    q_path = Path(questions_path)
    c_path = Path(corpus_path)

    questions = _load_rows(q_path)
    corpus = _load_rows(c_path)

    if len(questions) == 0:
        raise ValueError(
            f"Questions file is empty: {q_path}\n"
            f"Expected format: JSONL with one question per line.\n"
            f"Example: {{\"question_id\": \"q1\", \"question_text\": \"What is...\", "
            f"\"groundtruth_doc_ids\": [\"doc1\"], \"audio_path\": \"/path/to/audio.wav\"}}"
        )
    if len(corpus) == 0:
        raise ValueError(
            f"Corpus file is empty: {c_path}\n"
            f"Expected format: JSONL with one document per line.\n"
            f"Example: {{\"doc_id\": \"doc1\", \"text\": \"Document content...\", \"title\": \"Title\"}}"
        )

    corpus_ids = set()
    for idx, row in enumerate(corpus, start=1):
        doc_id = row.get("doc_id") or row.get("pmid") or row.get("id")
        text = row.get("text") or row.get("abstract") or row.get("content")
        if not doc_id:
            raise ValueError(
                f"Corpus row {idx} missing required field: 'doc_id' (or 'pmid', 'id').\n"
                f"Found fields: {list(row.keys())}\n"
                f"Example: {{\"doc_id\": \"12345\", \"text\": \"Document content...\"}}"
            )
        if not text:
            raise ValueError(
                f"Corpus row {idx} missing required field: 'text' (or 'abstract', 'content').\n"
                f"Found fields: {list(row.keys())}\n"
                f"Example: {{\"doc_id\": \"12345\", \"text\": \"Document content...\"}}"
            )
        doc_id = str(doc_id)
        if doc_id in corpus_ids:
            raise ValueError(f"Duplicate corpus doc_id found: {doc_id}")
        corpus_ids.add(doc_id)

    for idx, row in enumerate(questions, start=1):
        question_id = row.get("question_id") or row.get("id")
        question_text = row.get("question_text") or row.get("question") or row.get("text")
        if not question_id:
            raise ValueError(
                f"Question row {idx} missing required field: 'question_id' (or 'id').\n"
                f"Found fields: {list(row.keys())}\n"
                f"Example: {{\"question_id\": \"q1\", \"question_text\": \"What is...\"}}"
            )
        if not question_text:
            raise ValueError(
                f"Question row {idx} missing required field: 'question_text' (or 'question', 'text').\n"
                f"Found fields: {list(row.keys())}\n"
                f"Example: {{\"question_id\": \"q1\", \"question_text\": \"What is...\"}}"
            )

        relevance = row.get("relevance_grades")
        gt_ids = row.get("groundtruth_doc_ids") or row.get("relevant_doc_ids") or []

        if relevance is not None:
            if not isinstance(relevance, dict) or len(relevance) == 0:
                raise ValueError(
                    f"Question {question_id}: field 'relevance_grades' must be a non-empty dict.\n"
                    f"Got: {type(relevance).__name__} = {relevance}\n"
                    f"Example: {{\"relevance_grades\": {{\"doc1\": 2, \"doc2\": 1}}}}"
                )
            for doc_id, grade in relevance.items():
                if str(doc_id) not in corpus_ids:
                    raise ValueError(f"Question {question_id}: relevance doc_id not in corpus: {doc_id}")
                if int(grade) < 0:
                    raise ValueError(f"Question {question_id}: relevance grade must be >= 0 for doc_id {doc_id}")
        else:
            if not gt_ids:
                raise ValueError(
                    f"Question {question_id}: missing ground truth information.\n"
                    f"Provide either 'relevance_grades' or 'groundtruth_doc_ids'.\n"
                    f"Example: {{\"groundtruth_doc_ids\": [\"doc1\", \"doc2\"]}} or "
                    f"{{\"relevance_grades\": {{\"doc1\": 2, \"doc2\": 1}}}}"
                )
            for doc_id in gt_ids:
                if str(doc_id) not in corpus_ids:
                    raise ValueError(f"Question {question_id}: groundtruth doc_id not in corpus: {doc_id}")

        if verify_audio:
            audio_path = row.get("audio_path")
            if not audio_path:
                raise ValueError(
                    f"Question {question_id} has no audio_path. "
                    "Run scripts/prepare_pubmed_audio.py before evaluation."
                )
            if not Path(audio_path).exists():
                raise ValueError(f"Question {question_id}: audio file not found: {audio_path}")

    return len(questions), len(corpus)
