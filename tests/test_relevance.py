"""Self-retrieval relevance construction (_build_relevant_from_item)."""

from evaluator.evaluation.helpers import _build_relevant_from_item


def test_graded_relevance_wins():
    rel = _build_relevant_from_item(
        {"relevance_grades": {"d1": 2, "d2": 1}, "transcription": "x"}
    )
    assert rel == {"d1": 2, "d2": 1}


def test_groundtruth_doc_ids_binary():
    rel = _build_relevant_from_item(
        {"groundtruth_doc_ids": ["d1", "d2"], "transcription": "x"}
    )
    assert rel == {"d1": 1, "d2": 1}


def test_self_retrieval_falls_back_to_transcription():
    # admed/hani: no explicit relevance → the sentence is its own relevant doc.
    rel = _build_relevant_from_item(
        {
            "relevance_grades": {},
            "groundtruth_doc_ids": [],
            "transcription": "the patient is stable",
        }
    )
    assert rel == {"the patient is stable": 1}
