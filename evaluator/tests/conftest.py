"""Shared pytest fixtures for evaluator test suite."""

from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def sample_audio_batch() -> list:
    """Small synthetic batch used by pipeline tests."""
    return [
        {
            "audio_array": np.zeros(16000, dtype=np.float32),
            "sampling_rate": 16000,
            "transcription": "sample transcription",
            "question_id": "q1",
            "relevant_doc_ids": ["d1"],
        }
    ]


@pytest.fixture
def mock_asr_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.model.name.return_value = "mock-asr"
    pipeline.process.return_value = "hypothesis"
    pipeline.process_batch.return_value = ["hypothesis"]
    return pipeline


@pytest.fixture
def mock_text_embedding_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.model.name.return_value = "mock-text-embedder"
    pipeline.process.return_value = np.zeros(4, dtype=np.float32)
    pipeline.process_batch.return_value = np.zeros((1, 4), dtype=np.float32)
    return pipeline


@pytest.fixture
def mock_audio_embedding_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.model.name.return_value = "mock-audio-embedder"
    pipeline.process.return_value = np.zeros(4, dtype=np.float32)
    pipeline.process_batch.return_value = np.zeros((1, 4), dtype=np.float32)
    return pipeline


@pytest.fixture
def mock_retrieval_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.strategy_config.core.mode = "dense"
    pipeline.search_batch.return_value = [[({"id": "d1"}, 0.9)]]
    return pipeline

