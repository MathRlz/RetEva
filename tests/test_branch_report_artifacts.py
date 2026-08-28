"""P1 pins: the thesis-artifact emitters (previously untested)."""

import pytest

from evaluator.analysis.branch_report import latex_branch_table, latex_provenance_table

_REPORT = {
    "branches": {"asr": {"recall@5": {"mean": 0.8, "ci": [0.7, 0.9], "n": 20}}},
    "provenance": {
        "config_hash": "abc123def456",
        "seed": 42,
        "git_commit": "deadbee",
        "versions": {"torch": "2.12.0"},
        "determinism": {"cudnn_deterministic": True},
        "dataset": {"corpus_docs": 20, "corpus_sha256": "cafe", "questions": 20},
        "models": {"asr": {"type": "whisper", "size": "base"}},
    },
}


def test_latex_provenance_table_renders_all_groups():
    tex = latex_provenance_table({"report": _REPORT})
    assert tex.startswith(r"\begin{table}")
    for needle in ("Config hash & abc123def456", "Seed & 42", "Git commit & deadbee",
                   "Version: torch & 2.12.0", "Determinism: cudnn\\_deterministic & True",
                   "Model: asr & type=whisper size=base", "20 / cafe"):
        assert needle in tex, needle
    assert tex.rstrip().endswith(r"\end{table}")


def test_latex_provenance_table_requires_provenance():
    with pytest.raises(ValueError, match="provenance"):
        latex_provenance_table({"report": {}})


def test_latex_branch_table_smoke():
    tex = latex_branch_table(_REPORT)
    assert r"\toprule" in tex and "recall{@}5" in tex and "0.8000 [0.700, 0.900]" in tex
