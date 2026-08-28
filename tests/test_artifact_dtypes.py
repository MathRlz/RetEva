"""ArtifactType.dtype: modality-derived default + explicit override."""

from evaluator.pipeline.artifacts import (
    Modality,
    artifact_dtype,
    artifact_modality,
    get_artifact_type,
    register_artifact,
)


def test_dtype_defaults_from_modality():
    register_artifact("test_text_artifact_default", Modality.TEXT)
    assert artifact_dtype("test_text_artifact_default") == "str"
    assert artifact_modality("test_text_artifact_default") == Modality.TEXT


def test_dtype_explicit_override_wins():
    register_artifact("test_results_artifact_override", Modality.RESULTS, dtype="List[List[Tuple[str, float]]]")
    assert artifact_dtype("test_results_artifact_override") == "List[List[Tuple[str, float]]]"


def test_builtin_corpus_dtype():
    assert artifact_dtype("corpus") == "List[CorpusDocument]"


def test_builtin_retrieved_dtype_override():
    assert artifact_dtype("retrieved") == "List[List[Tuple[str, float]]]"


def test_reregistering_same_definition_is_noop():
    a = register_artifact("test_text_artifact_default", Modality.TEXT)
    b = get_artifact_type("test_text_artifact_default")
    assert a == b
