"""B1 pins (CRITIQUE.md): the config `metrics:` allowlist computes exactly what it names.

Auto-injection stays the default (absent key = collect-all, byte-identical reports); when
the allowlist is set the run computes exactly those registry metrics — explicit naming
bypasses ``MetricSpec.opt_in`` — and validation rejects unknown names with the registered
set listed.
"""

import pytest

from evaluator.evaluation.item_set import ItemSet
from evaluator.evaluation.metric_registry import compute_metrics


def _asr_artifacts():
    return {
        "query_text": ItemSet(["q1"], ["hello world"]),
        "reference_transcription": ItemSet(["q1"], ["hello world"]),
        "corrected_query_text": ItemSet(["q1"], ["hello world"]),
    }


def test_allowlist_computes_exactly_the_named_set():
    scores = compute_metrics(_asr_artifacts(), only=["wer"])
    assert set(scores) == {"wer"}


def test_absent_allowlist_keeps_collect_all():
    scores = compute_metrics(_asr_artifacts())
    assert {"wer", "cer", "ceer"} <= set(scores)
    assert "corrected_wer" not in scores  # opt_in still gated by default


def test_explicit_naming_bypasses_opt_in():
    scores = compute_metrics(_asr_artifacts(), only=["corrected_wer"])
    assert set(scores) == {"corrected_wer"}


def test_unsatisfiable_name_is_skipped_not_an_error():
    scores = compute_metrics(_asr_artifacts(), only=["wer", "mrr"])  # no retrieved/relevant
    assert set(scores) == {"wer"}


def test_unknown_name_raises_with_registered_set():
    with pytest.raises(KeyError, match="unknown metric 'nope'"):
        compute_metrics(_asr_artifacts(), only=["nope"])


def test_config_validation_rejects_unknown_metric_names():
    from evaluator.config.validation import _validate_metric_allowlist

    class Cfg:
        metrics = ["wer", "nope"]

    errors = []
    _validate_metric_allowlist(Cfg(), errors)
    assert len(errors) == 1
    assert "nope" in errors[0] and "Registered:" in errors[0]

    errors = []
    _validate_metric_allowlist(type("C", (), {"metrics": None})(), errors)
    assert errors == []
