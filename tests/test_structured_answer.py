"""The `Short Answer:` / `Long Answer:` contract and the metrics that read it.

The decision line is exact-matched against the dataset label (PubMedQA `final_decision`); the
prose is what ROUGE scores. A reply that ignores the format must degrade the decision metric
only — never silently zero ROUGE.
"""

import pytest

from evaluator.evaluation.answer_gen import (
    DECISION_UNKNOWN,
    chunk_text,
    parse_structured_answer,
    score_answers,
    select_full_text_context,
)


@pytest.mark.parametrize("reply,decision,long_start", [
    ("Short Answer: yes\nLong Answer: Mitochondria do participate.", "yes", "Mitochondria"),
    ("**Short Answer:** No\n**Long Answer:** The evidence is against it.", "no", "The evidence"),
    ("Short answer - maybe\nLong answer: Findings are mixed.", "maybe", "Findings"),
    # the prompt ends with "Short Answer:", so a literal completion has no label of its own
    ("yes\nLong Answer: Because the study found an effect.", "yes", "Because"),
    ("YES. Long Answer: it works.", "yes", "it works."),
])
def test_parses_both_fields(reply, decision, long_start):
    got_decision, got_long = parse_structured_answer(reply)
    assert got_decision == decision
    assert got_long.startswith(long_start)


def test_format_miss_keeps_the_prose_for_rouge():
    decision, long_answer = parse_structured_answer(
        "The study suggests an association but the sample was small."
    )
    assert decision == DECISION_UNKNOWN
    assert long_answer.startswith("The study suggests")  # ROUGE still has text to score


def test_empty_reply():
    assert parse_structured_answer("") == (DECISION_UNKNOWN, "")


class _Cfg:
    compute_rouge = False
    reference_metadata_field = "long_answer"
    context_source = "retrieved_text"
    context_max_chars = 600
    context_chunk_chars = 40
    context_chunks = 2


def _results(replies):
    return {"details": [{"query_id": f"q{i}", "generated_answer": r}
                        for i, r in enumerate(replies)]}


def test_decision_accuracy_and_unknown_rate():
    results = _results([
        "Short Answer: yes\nLong Answer: a",     # correct
        "Short Answer: no\nLong Answer: b",      # wrong (gt yes)
        "no idea really",                        # unknown → wrong
    ])
    score_answers(
        results, traces_data=(["q0", "q1", "q2"], [{}, {}, {}], [[], [], []]),
        corpus_lookup={}, config=_Cfg(),
        decision_gt={"q0": "yes", "q1": "yes", "q2": "maybe"},
    )
    assert results["mean_decision_accuracy"] == pytest.approx(1 / 3)
    assert results["decision_unknown_rate"] == pytest.approx(1 / 3)
    assert [d["decision_pred"] for d in results["details"]] == ["yes", "no", DECISION_UNKNOWN]
    assert results["details"][0]["decision_correct"] is True


def test_decision_metric_absent_without_ground_truth():
    results = _results(["Short Answer: yes\nLong Answer: a"])
    score_answers(results, traces_data=(["q0"], [{}], [[]]),
                  corpus_lookup={}, config=_Cfg(), decision_gt=None)
    assert results["mean_decision_accuracy"] is None      # nothing to compare against
    assert results["decision_unknown_rate"] == 0.0        # the format WAS followed


def test_rouge_scores_the_long_answer_not_the_decision_line():
    """The reference is prose; scoring the whole reply would count "Short Answer: yes" as text."""
    class _RougeCfg(_Cfg):
        compute_rouge = True

    ref = "Mitochondria play a central role in remodelling lace plant leaves."
    results = _results([f"Short Answer: yes\nLong Answer: {ref}"])
    corpus = {"d1": {"doc_id": "d1", "metadata": {"long_answer": ref}}}
    score_answers(results, traces_data=(["q0"], [{"d1": 1}], [[]]),
                  corpus_lookup=corpus, config=_RougeCfg(), decision_gt={"q0": "yes"})
    # exact prose match → ROUGE-L is 1.0; it would be diluted if the decision line were included
    assert results["details"][0]["rougeL"] == pytest.approx(1.0)


def test_chunk_text_splits_on_boundaries_and_covers_everything():
    text = "\n\n".join(f"Paragraph {i} with some filler text about topic {i}." for i in range(8))
    chunks = chunk_text(text, size=80)
    assert len(chunks) > 1
    assert all(len(c) <= 160 for c in chunks)          # no runaway chunk
    joined = " ".join(chunks)
    for i in range(8):
        assert f"Paragraph {i}" in joined              # nothing dropped


def test_chunk_text_short_input_is_one_chunk():
    assert chunk_text("short", size=100) == ["short"]
    assert chunk_text("", size=100) == []


class _Embedder:
    """Scores a chunk by how many query words it contains (1-d 'embedding')."""

    def process_batch(self, texts, **kw):
        import numpy as np
        return np.array([[float(t.lower().count("aspirin"))] for t in texts], dtype="float32")


def test_full_text_selection_picks_query_relevant_chunks():
    article = (
        "## Introduction\n\nThis paper studies a cohort of patients over ten long years.\n\n"
        "## Results\n\nAspirin aspirin aspirin reduced the aspirin event rate markedly.\n\n"
        "## Methods\n\nWe enrolled participants from three unrelated regional centres.\n\n"
    )
    corpus = {"d1": {"doc_id": "d1", "metadata": {"full_text": article}}}

    class _FullCfg(_Cfg):
        context_source = "full_text"
        context_chunk_chars = 70
        context_chunks = 1

    picked = select_full_text_context(
        "does aspirin help?", {"doc_id": "d1"}, corpus, _FullCfg(), _Embedder()
    )
    assert "Aspirin" in picked
    assert "Methods" not in picked


def test_questions_loader_carries_the_decision_label(tmp_path):
    """The label has to survive the load, or `short_answers` is never published and the whole
    decision metric is silently unscored (it was)."""
    import json

    from evaluator.datasets.core import load_questions_file
    from evaluator.evaluation.handlers.source import _question_short_answer

    path = tmp_path / "q.json"
    path.write_text(json.dumps([
        {"question_id": "q1", "question_text": "Does it help?",
         "groundtruth_doc_ids": ["d1"], "short_answer": "maybe", "metadata": {"pubid": 1}},
        {"question_id": "q2", "question_text": "And this?", "answer": "no"},   # alias key
        {"question_id": "q3", "question_text": "No label here."},
    ]))
    qs = load_questions_file(path)
    assert [_question_short_answer(q) for q in qs] == ["maybe", "no", None]
    assert qs[0].metadata["pubid"] == 1        # existing metadata preserved


def test_full_text_selection_falls_back_when_absent():
    class _FullCfg(_Cfg):
        context_source = "full_text"

    # no metadata.full_text → None, so the caller keeps using the retrieved passage
    assert select_full_text_context(
        "q", {"doc_id": "d1"}, {"d1": {"doc_id": "d1"}}, _FullCfg(), _Embedder()
    ) is None
