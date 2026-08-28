"""Pipeline-mode detection — including the config-mode-wins regression."""

from unittest.mock import MagicMock

import pytest

from evaluator.evaluation.helpers import detect_graph_template


def test_audio_emb_retrieval_from_audio_pipeline_only():
    assert detect_graph_template(
        retrieval_pipeline=MagicMock(),
        asr_pipeline=None,
        text_embedding_pipeline=None,
        audio_embedding_pipeline=MagicMock(),
    ) == "audio_emb_retrieval"


def test_audio_text_retrieval_from_audio_plus_text():
    assert detect_graph_template(MagicMock(), None, MagicMock(), MagicMock()) == (
        "audio_text_retrieval"
    )


def test_asr_text_retrieval_from_asr_plus_text():
    assert detect_graph_template(MagicMock(), MagicMock(), MagicMock(), None) == (
        "asr_text_retrieval"
    )


def test_text_retrieval_from_text_plus_retrieval():
    # A text-RAG graph has no voice front-end: question text -> embed -> retrieve. It used to
    # raise here and die at run start.
    assert detect_graph_template(
        retrieval_pipeline=MagicMock(),
        asr_pipeline=None,
        text_embedding_pipeline=MagicMock(),
        audio_embedding_pipeline=None,
    ) == "text_retrieval"


def test_no_query_head_still_raises():
    with pytest.raises(ValueError, match="query head"):
        detect_graph_template(MagicMock(), None, None, None)


def test_configured_mode_wins_over_pipeline_detection():
    # Cross-modal audio_emb builds BOTH an audio and a text pipeline (the text corpus),
    # which presence-detection would misread as audio_text fusion — config mode must win.
    assert detect_graph_template(MagicMock(), None, MagicMock(), MagicMock()) == (
        "audio_text_retrieval"
    )
    assert detect_graph_template(
        MagicMock(), None, MagicMock(), MagicMock(),
        configured_mode="audio_emb_retrieval",
    ) == "audio_emb_retrieval"
