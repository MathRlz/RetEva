"""Dataset descriptor registry: builtins register with the right type/domain/capabilities."""

from evaluator.config.types import DatasetType
from evaluator.datasets.descriptor import get_descriptor


def test_admed_voice_registered_medical_audio_transcription():
    d = get_descriptor("admed_voice")
    assert d is not None
    assert d.dataset_type == DatasetType.AUDIO_TRANSCRIPTION
    assert d.domain == "medical"
    assert d.evaluation_mode == "transcription"
    assert d.requires_audio is True


def test_derived_outputs_advertise_self_retrieval_corpus():
    """A retrieval-capable dataset with no corpus column derives a self-retrieval corpus; one
    that already declares a corpus column does not. Keeps builder picker ≡ config preview."""
    assert get_descriptor("admed_voice").derived_outputs == ("corpus",)   # no corpus column
    assert get_descriptor("hani_medical").derived_outputs == ("corpus",)
    assert get_descriptor("pubmed_qa").derived_outputs == ()              # has documents->corpus


def test_hani_medical_registered_with_domain_and_splits():
    d = get_descriptor("hani_medical")
    assert d is not None
    assert d.dataset_type == DatasetType.AUDIO_TRANSCRIPTION
    assert d.domain == "medical"
    assert list(d.splits) == ["train", "test"]


def test_domain_defaults_to_general_for_untagged():
    assert get_descriptor("local").domain == "general"
