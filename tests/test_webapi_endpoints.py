"""M8: webapi request-surface coverage via FastAPI TestClient.

Covers the read endpoints (health/datasets) and the config-validation surface
(happy path + malformed + unknown-key + missing-field). Does NOT submit a real job — only
the *rejected* job-submit path is exercised, so no subprocess is launched.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _valid_config():
    # fixture-backed so dataset validation passes regardless of cwd
    return {
        "experiment_name": "webapi_test",
        "model": {
            "pipeline_mode": "asr_text_retrieval",
            "asr_model_type": "whisper",
            "text_emb_model_type": "labse",
        },
        "data": {
            "dataset_name": "pubmed_qa",
            "questions_path": str(_ROOT / "examples/data/pubmed_qa_small/questions.json"),
            "corpus_path": str(_ROOT / "examples/data/pubmed_qa_small/corpus.json"),
        },
        "vector_db": {"type": "inmemory", "k": 5},
    }


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200


def test_datasets_list_carries_metadata(client):
    r = client.get("/api/datasets")
    assert r.status_code == 200
    body = r.json()
    assert "datasets" in body or "known_datasets" in body


def test_config_validate_happy_path(client):
    r = client.post(
        "/api/config/validate",
        json={"config": _valid_config(), "auto_devices": False},
    )
    assert r.status_code == 200, r.text
    cfg = r.json()["config"]  # to_dict() is a flat dict with pipeline_mode at top level
    assert cfg["pipeline_mode"] == "asr_text_retrieval"


def test_config_validate_unknown_key_is_400(client):
    # a ConfigurationError (unknown/misspelled key) must be a clean 400, not a 500
    r = client.post(
        "/api/config/validate",
        json={"config": {"totally_unknown_key": 1}, "auto_devices": False},
    )
    assert r.status_code == 400
    assert "Unknown config key" in r.text


def test_config_validate_missing_config_field_is_422(client):
    # config defaults to {} (a valid empty body), but a wrong *type* must be a 422, not 500
    r = client.post("/api/config/validate", json={"config": "not-a-dict"})
    assert r.status_code == 422


def test_job_submit_with_invalid_config_rejected_before_launch(client):
    # invalid config → 400 from prepare_run_config, so no subprocess is ever launched
    r = client.post(
        "/api/jobs/evaluation",
        json={"config": {"totally_unknown_key": 1}, "auto_devices": False},
    )
    assert r.status_code == 400


def test_data_root_confinement_opt_in(monkeypatch, client, tmp_path):
    # L1: with EVALUATOR_DATA_ROOT set, a dataset path escaping it is rejected (400); unset,
    # absolute local paths are unrestricted (the default — see the happy-path test above).
    monkeypatch.setenv("EVALUATOR_DATA_ROOT", str(tmp_path))
    cfg = _valid_config()  # fixture paths live under the repo, not under tmp_path
    r = client.post("/api/config/validate", json={"config": cfg, "auto_devices": False})
    assert r.status_code == 400
    assert "allowed data root" in r.text


def test_report_metrics_table_endpoint(client):
    # §6 CLI/web parity: POST a report dict → tidy metrics table + trace count.
    report = {"report": {"branches": {"b": {"mrr": {"mean": 0.8, "n": 10}}},
                         "traces": {"query_traces": [{"query_id": "q1"}, {"query_id": "q2"}]}}}
    r = client.post("/api/report/metrics-table", json=report)
    assert r.status_code == 200
    body = r.json()
    assert body["metrics"][0]["metric"] == "mrr"
    assert body["n_traces"] == 2


def test_graph_template_endpoint_serves_dag_and_404s(client):
    body = client.get("/api/graph/template/asr_text_retrieval").json()
    stages = {n["type"] for n in body["nodes"]}
    assert {"asr", "embed", "search"} <= stages              # asr + embed + retrieval
    assert body["nodes"] and body["levels"]                  # a wired, layered DAG
    assert client.get("/api/graph/template/bogus_template").status_code == 404


def test_graph_preview_payload_keeps_ports_and_all_bindings():
    """The /ui/config DAG payload must carry input_ports (a OneOf chain collapses to ONE port)
    and optional_inputs (the GT side-channel edges) — dropping them exploded the chain into
    dangling ports and silently erased every optional-GT edge. Bindings carry EVERY producer
    (the client renders earlier producers as faded fallback edges, newest as primary)."""
    import yaml
    from pathlib import Path

    from evaluator.config.evaluation import EvaluationConfig
    from evaluator.config.graph_config import build_evaluation_config_kwargs
    from evaluator.webapi.form_builder import graph_preview
    from evaluator.webapi.ui.config import _graph_diagram_context
    import json

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs/evaluation_config_pubmed_qa.yaml").read_text())
    cfg = EvaluationConfig.from_dict(build_evaluation_config_kwargs(raw), validate=False)
    payload = json.loads(_graph_diagram_context(graph_preview(cfg))["preview_json"])
    nodes = {n["id"]: n for n in payload["nodes"]}

    te = nodes["text_embedding"]
    assert len(te["input_ports"]) == 1                      # the OneOf chain is ONE port
    assert "query_text" in te["input_ports"][0]["names"]
    assert [b for b in te["bindings"] if b == ["query_text", "dataset_source"]]  # fallback kept
    assert [b for b in te["bindings"] if b == ["query_text", "asr"]]             # primary kept

    asr = nodes["asr"]
    assert "reference_transcription" in asr["optional_inputs"]   # GT edge can draw again
    assert nodes["tts"]["bindings"] == [["query_text", "dataset_source"]]
