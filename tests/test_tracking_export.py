"""Roadmap 1d — MLflow/W&B bridge: pure payload extraction + import-guarded logging."""

import contextlib
import sys
import types

from evaluator.analysis.tracking_export import report_to_tracking_payload, to_mlflow

_REPORT = {
    "report": {
        "branches": {"asr_text_retrieval": {"mrr": {"mean": 0.8, "n": 50},
                                            "wer": {"mean": 0.12, "n": 50}}},
        "provenance": {
            "config_hash": "abc123", "seed": 42, "git_commit": "deadbeef",
            "models": {"asr": "whisper:base",
                       "text_embedding": {"type": "labse", "resolved": "LaBSE"}},
            "dataset": {"corpus_docs": 20, "questions": 5},
        },
    }
}


def test_tracking_payload_extraction():
    p = report_to_tracking_payload(_REPORT)
    assert p["metrics"]["asr_text_retrieval/mrr"] == 0.8
    assert p["params"]["config_hash"] == "abc123" and p["params"]["seed"] == 42
    assert p["params"]["model.asr"] == "whisper:base"
    assert p["params"]["model.text_embedding"] == "LaBSE"  # structured → resolved
    assert p["params"]["dataset.corpus_docs"] == 20
    assert p["tags"]["git_commit"] == "deadbeef"


class _FakeMlflow(types.ModuleType):
    def __init__(self):
        super().__init__("mlflow")
        self.params, self.metrics, self.tags, self.exp = {}, {}, {}, None

    def set_experiment(self, e):
        self.exp = e

    def start_run(self, run_name=None):
        self.run_name = run_name
        return contextlib.nullcontext()

    def log_params(self, p):
        self.params.update(p)

    def log_metrics(self, m):
        self.metrics.update(m)

    def set_tag(self, k, v):
        self.tags[k] = v


def test_to_mlflow_logs_payload(monkeypatch):
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    to_mlflow(_REPORT, run_name="r1", experiment="exp")
    assert fake.exp == "exp" and fake.run_name == "r1"
    assert fake.metrics["asr_text_retrieval/mrr"] == 0.8
    assert fake.params["config_hash"] == "abc123"
    assert fake.tags["git_commit"] == "deadbeef"
