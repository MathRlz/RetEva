"""`evaluator graph --emit-metrics` prints the satisfiable
metric set as a ready-to-paste `metrics:` allowlist block, opt-in metrics commented."""

from evaluator.cli.main import _cmd_graph
from tests.fixtures_paths import E2E_MINI


def test_emit_metrics_lists_satisfiable_set(capsys):
    assert _cmd_graph(["--config", E2E_MINI, "--emit-metrics"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("metrics:")
    for name in ("- wer", "- cer", "- mrr", "- recall@5", "- ndcg@10"):
        assert name in out
    assert "# - ceer_rx" in out          # opt-in → commented, not listed
    assert "- corrected_wer" not in out  # no corrector in this graph
