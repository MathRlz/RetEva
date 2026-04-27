import pytest

from evaluator import EvaluationConfig
from evaluator.errors import ConfigurationError
from evaluator.datasets.runtime import list_dataset_runtime_specs, resolve_dataset_runtime_spec
from evaluator.datasets.runtime import validate_dataset_runtime_config


def test_runtime_specs_include_core_paths():
    spec_ids = {spec.id for spec in list_dataset_runtime_specs()}
    assert "prepared_pubmed" in spec_ids
    assert "pubmed_paths" in spec_ids
    assert "admed_voice" in spec_ids
    assert "loader_local" in spec_ids
    assert "loader_huggingface" in spec_ids


def test_runtime_spec_resolution_prefers_prepared_dir():
    config = EvaluationConfig()
    config.data.prepared_dataset_dir = "/tmp/prepared"
    spec = resolve_dataset_runtime_spec(config)
    assert spec.id == "prepared_pubmed"


def test_runtime_spec_resolution_local_source():
    config = EvaluationConfig()
    config.data.prepared_dataset_dir = None
    config.data.questions_path = None
    config.data.dataset_name = "generic_audio_transcription"
    config.data.dataset_source = "local"
    spec = resolve_dataset_runtime_spec(config)
    assert spec.id == "loader_local"


def test_validate_dataset_runtime_config_missing_required_field():
    config = EvaluationConfig()
    config.data.prepared_dataset_dir = None
    config.data.questions_path = None
    config.data.dataset_name = "generic_audio_transcription"
    config.data.dataset_source = "local"
    config.data.audio_dir = None
    with pytest.raises(ConfigurationError, match="missing required fields"):
        validate_dataset_runtime_config(config)
