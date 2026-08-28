"""B1-remainder pin (CRITIQUE.md): `evaluator graph --emit-metrics` prints the satisfiable
metric set as a ready-to-paste `metrics:` allowlist block, opt-in metrics commented."""

import os

from evaluator.cli.main import _cmd_graph

_ROOT = os.path.dirname(os.path.dirname(__file__))


def test_emit_metrics_lists_satisfiable_set(capsys):
    cfg = os.path.join(_ROOT, "configs", "e2e_pubmed_qa_small.yaml")
    assert _cmd_graph(["--config", cfg, "--emit-metrics"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("metrics:")
    for name in ("- wer", "- cer", "- mrr", "- recall@5", "- ndcg@10"):
        assert name in out
    assert "# - ceer_rx" in out          # opt-in → commented, not listed
    assert "- corrected_wer" not in out  # no corrector in this graph


def test_emit_metrics_marks_corrected_opt_ins(capsys):
    cfg = os.path.join(_ROOT, "configs", "c7_correction_branches.yaml")
    assert _cmd_graph(["--config", cfg, "--emit-metrics"]) == 0
    out = capsys.readouterr().out
    for name in ("corrected_wer", "corrected_cer", "corrected_ceer", "ceer_rx"):
        assert f"# - {name}" in out
