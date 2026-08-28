"""R4/per-variant persistence: `save_run_artifacts` (report/resolved-config/run-log +
per-variant answers.jsonl/metrics.json) and its wiring into the execution paths that write
files (`evaluation_service.run_evaluation` — the webapi/API path that previously wrote
NOTHING to disk)."""

import json

import pytest

from evaluator.config.evaluation import EvaluationConfig
from evaluator.evaluation.results_io import save_run_artifacts


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


_TRACE = {
    "query_id": "q1", "generated_answer": "iron deficiency",
    "retrieved": [{"doc_key": "d1", "score": 0.9}],
}


def test_save_run_artifacts_writes_report_resolved_config_and_single_variant(tmp_path):
    cfg = EvaluationConfig()
    cfg.output_dir = str(tmp_path)
    cfg.experiment_name = "t"
    results = {
        "MRR": 0.5,
        "report": {"traces": {"query_traces": [_TRACE]}, "judge": {}},
    }

    path = save_run_artifacts(results, cfg, _Logger())

    assert path.startswith(str(tmp_path))
    with open(path) as fh:
        assert json.load(fh)["MRR"] == 0.5
    resolved = path[:-5] + ".config_resolved.yaml"
    assert (tmp_path / resolved.rsplit("/", 1)[-1]).exists()

    variant_dir = tmp_path / "variants" / "run"
    assert (variant_dir / "metrics.json").exists()
    with open(variant_dir / "metrics.json") as fh:
        metrics_json = json.load(fh)
    assert metrics_json["MRR"] == 0.5
    # the large per-query blob is stripped from metrics.json...
    assert "query_traces" not in metrics_json["report"]["traces"]

    answers = (variant_dir / "answers.jsonl").read_text().strip().splitlines()
    assert len(answers) == 1
    # ...but present, whole, in answers.jsonl
    assert json.loads(answers[0]) == _TRACE

    # retrieved.jsonl is opt-in, off by default
    assert not (variant_dir / "retrieved.jsonl").exists()


def test_save_run_artifacts_persist_intermediate_flag_writes_retrieved_jsonl(tmp_path):
    cfg = EvaluationConfig()
    cfg.output_dir = str(tmp_path)
    cfg.experiment_name = "t"
    cfg.persist_intermediate_artifacts = True
    results = {"MRR": 0.5, "report": {"traces": {"query_traces": [_TRACE]}}}

    save_run_artifacts(results, cfg, _Logger())

    retrieved = (tmp_path / "variants" / "run" / "retrieved.jsonl").read_text().strip()
    assert json.loads(retrieved) == {"query_id": "q1", "retrieved": _TRACE["retrieved"]}


def test_save_run_artifacts_writes_one_directory_per_variant(tmp_path):
    cfg = EvaluationConfig()
    cfg.output_dir = str(tmp_path)
    cfg.experiment_name = "t"
    results = {
        "variants": {
            "finalize_a": {"MRR": 0.9, "report": {"traces": {"query_traces": [_TRACE]}}},
            "finalize_b": {"MRR": 0.2, "report": {"traces": {"query_traces": []}}},
        },
    }

    save_run_artifacts(results, cfg, _Logger())

    variants_dir = tmp_path / "variants"
    assert set(p.name for p in variants_dir.iterdir()) == {"finalize_a", "finalize_b"}
    with open(variants_dir / "finalize_a" / "metrics.json") as fh:
        assert json.load(fh)["MRR"] == 0.9
    with open(variants_dir / "finalize_b" / "metrics.json") as fh:
        assert json.load(fh)["MRR"] == 0.2
    assert len((variants_dir / "finalize_a" / "answers.jsonl").read_text().strip().splitlines()) == 1
    assert (variants_dir / "finalize_b" / "answers.jsonl").read_text() == ""


def test_run_evaluation_service_writes_artifacts_to_output_dir(tmp_path, monkeypatch):
    # Wiring check for the previously-silent webapi/API path: `_run_core` stands in for the
    # (heavy, model-loading) real execution core — this only pins that `run_evaluation` now
    # actually calls `save_run_artifacts` with the run's real metrics + config.
    import evaluator.services.evaluation_service as svc

    canned_metrics = {"MRR": 0.42, "report": {"traces": {"query_traces": []}}}

    def _fake_run_core(config, **kwargs):
        return canned_metrics, None

    monkeypatch.setattr(svc, "_run_core", _fake_run_core)

    cfg = EvaluationConfig()
    cfg.output_dir = str(tmp_path)
    cfg.experiment_name = "svc_test"
    cfg.cache.enabled = False
    cfg.tracking.enabled = False

    results = svc.run_evaluation(cfg)

    assert results.metrics["MRR"] == 0.42
    written = list(tmp_path.glob("results_svc_test_*.json"))
    assert len(written) == 1
    assert (tmp_path / "variants" / "run" / "metrics.json").exists()


def test_webapi_subprocess_job_copies_artifacts_before_deleting_jobdir(tmp_path, monkeypatch):
    # The bug this pins: `_run_eval_subprocess` ran the CLI in a temp `jobdir`, then
    # `shutil.rmtree`'d it in a `finally` — silently discarding the report/resolved-config/
    # run-log/variants it just wrote; only the in-memory parsed JSON survived.
    from evaluator.webapi.jobs import JobManager

    real_output_dir = tmp_path / "real_output"

    def _fake_launch_cli(self, job_id, cfg_path, progress_path=None):
        # Stand-in for the real CLI subprocess: write exactly what `save_run_artifacts`
        # would, into the jobdir (`cfg_path.parent`) — no process actually spawned.
        jobdir = cfg_path.parent
        (jobdir / "results_t_run1.json").write_text(json.dumps({"MRR": 0.7}))
        (jobdir / "results_t_run1.config_resolved.yaml").write_text("nodes: []\n")
        (jobdir / "run_t.log").write_text("log line\n")
        variant_dir = jobdir / "variants" / "run"
        variant_dir.mkdir(parents=True)
        (variant_dir / "metrics.json").write_text(json.dumps({"MRR": 0.7}))
        (variant_dir / "answers.jsonl").write_text("")

    monkeypatch.setattr(JobManager, "_launch_cli", _fake_launch_cli)
    jm = JobManager(evaluation_runner=lambda *a, **k: {})

    cfg = EvaluationConfig()
    cfg.output_dir = str(real_output_dir)
    cfg.experiment_name = "t"

    result = jm._run_eval_subprocess("job1", cfg)

    assert result == {"MRR": 0.7}
    # The temp jobdir is gone...
    assert not any(tmp_path.glob("evaljob-*"))
    # ...but everything it wrote survived, copied into the REAL configured output_dir.
    assert (real_output_dir / "results_t_run1.json").exists()
    assert (real_output_dir / "results_t_run1.config_resolved.yaml").exists()
    assert (real_output_dir / "run_t.log").exists()
    assert (real_output_dir / "variants" / "run" / "metrics.json").exists()
