import json

from evaluator.datasets.core import PubMedQADataset, load_pubmed_qa_dataset


def _write_pubmed_files(tmp_path, n=3):
    questions = []
    corpus = []
    for i in range(n):
        doc_id = f"d{i}"
        questions.append(
            {
                "question_id": f"q{i}",
                "question_text": f"question {i}",
                "groundtruth_doc_ids": [doc_id],
                "relevance_grades": {doc_id: 1},
            }
        )
        corpus.append(
            {
                "doc_id": doc_id,
                "text": f"document {i}",
            }
        )

    questions_path = tmp_path / "questions.json"
    corpus_path = tmp_path / "corpus.json"
    questions_path.write_text(json.dumps(questions), encoding="utf-8")
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    return questions_path, corpus_path


def test_pubmed_dataset_accepts_trace_limit(tmp_path):
    questions_path, corpus_path = _write_pubmed_files(tmp_path, n=5)

    dataset = PubMedQADataset(
        questions_path=str(questions_path),
        corpus_path=str(corpus_path),
        trace_limit=2,
    )

    assert len(dataset) == 2


def test_pubmed_loader_accepts_trace_limit_and_corpus_alias(tmp_path):
    questions_path, corpus_path = _write_pubmed_files(tmp_path, n=4)

    dataset = load_pubmed_qa_dataset(
        questions_path=str(questions_path),
        corpus_path=str(corpus_path),
        trace_limit=3,
    )

    assert len(dataset) == 3
    assert dataset.get_corpus() == dataset.get_corpus_entries()
