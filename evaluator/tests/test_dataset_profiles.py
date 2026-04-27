"""Tests for dataset capability profiles."""

from evaluator.config.types import DatasetType
from evaluator.datasets import resolve_dataset_profile


def test_resolve_dataset_profile_by_name():
    profile = resolve_dataset_profile("pubmed_qa", None)
    assert profile.name == "pubmed_qa"
    assert profile.requires_audio is True
    assert profile.requires_text is True
    assert profile.supports_pipeline_mode("asr_text_retrieval")


def test_resolve_dataset_profile_by_dataset_type_fallback():
    profile = resolve_dataset_profile("custom_dataset", DatasetType.AUDIO_TRANSCRIPTION)
    assert profile.dataset_type == DatasetType.AUDIO_TRANSCRIPTION
    assert profile.requires_audio is True
    assert profile.supports_pipeline_mode("asr_only")


def test_unknown_dataset_gets_generic_profile():
    profile = resolve_dataset_profile("unknown_dataset", None)
    assert profile.name == "generic:unknown_dataset"
    assert profile.recommended_pipeline_modes
