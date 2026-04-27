import time
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from evaluator import EvaluationConfig
from evaluator.storage import ExperimentStore
from evaluator.webapi import create_app


class _FakeResult:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self, include_config=True):
        return self._payload


class _FakeProvider:
    def list_available_models(self):
        return {
            "asr": [{"type": "whisper", "name": "openai/whisper-base"}],
            "tts": [
                {"type": "piper", "name": "rhasspy/piper-voices"},
                {"type": "mms", "name": "facebook/mms-tts"},
            ],
        }

    def shutdown(self, *args, **kwargs):
        return None


class _FakeProviderStringModels:
    def list_available_models(self):
        return {
            "asr": ["whisper", "wav2vec2"],
            "text_embedding": ["labse"],
            "audio_embedding": ["clap_style"],
            "tts": ["piper", "mms"],
        }

    def shutdown(self, *args, **kwargs):
        return None


def _asr_only_config_dict():
    """Return a config dict that passes dataset validation (asr_only needs no corpus)."""
    cfg = EvaluationConfig()
    cfg.model.pipeline_mode = "asr_only"
    return cfg.to_dict(include_config=True)


def _wait_for_completion(client, job_id: str, timeout: float = 2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()["status"]
        if status in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


def test_health_and_models_endpoints():
    app = create_app(
        evaluation_runner=lambda cfg: _FakeResult({"metrics": {"MRR": 1.0}}),
        matrix_runner=lambda cfg, setups: {"runs": []},
        provider_factory=_FakeProvider,
    )
    client = testclient.TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert root.json()["service"] == "evaluator-webapi"
    assert root.json()["health"] == "/api/health"
    assert client.get("/favicon.ico").status_code == 204

    assert client.get("/api/health").json() == {"status": "ok"}
    models = client.get("/api/models")
    assert models.status_code == 200
    assert "asr" in models.json()


def test_cors_preflight_allows_frontend_origin():
    app = create_app(
        evaluation_runner=lambda cfg: _FakeResult({"ok": True}),
        matrix_runner=lambda cfg, setups: {"runs": []},
        provider_factory=_FakeProvider,
    )
    client = testclient.TestClient(app)

    response = client.options(
        "/api/config/options",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"


def test_config_options_endpoint():
    app = create_app(
        evaluation_runner=lambda cfg: _FakeResult({"ok": True}),
        matrix_runner=lambda cfg, setups: {"runs": []},
        provider_factory=_FakeProvider,
    )
    client = testclient.TestClient(app)

    response = client.get("/api/config/options")
    assert response.status_code == 200
    payload = response.json()
    assert "presets" in payload
    assert "pipeline_modes" in payload
    assert "dataset_types" in payload
    assert "dataset_sources" in payload
    assert "dataset_names" in payload
    assert "models" in payload
    assert "tts_providers" in payload
    assert "defaults" in payload


def test_config_options_normalizes_string_model_entries():
    app = create_app(
        evaluation_runner=lambda cfg: _FakeResult({"ok": True}),
        matrix_runner=lambda cfg, setups: {"runs": []},
        provider_factory=_FakeProviderStringModels,
    )
    client = testclient.TestClient(app)

    response = client.get("/api/config/options")
    assert response.status_code == 200
    payload = response.json()
    assert any(entry["type"] == "whisper" for entry in payload["models"]["asr"])
    assert "piper" in payload["tts_providers"]


def test_graph_preview_and_service_status_endpoints():
    app = create_app(
        evaluation_runner=lambda cfg: _FakeResult({"ok": True}),
        matrix_runner=lambda cfg, setups: {"runs": []},
        provider_factory=_FakeProvider,
    )
    client = testclient.TestClient(app)

    cfg = EvaluationConfig().to_dict(include_config=True)
    graph = client.post("/api/graph/preview", json={"config": cfg, "auto_devices": False})
    assert graph.status_code == 200
    body = graph.json()
    assert "nodes" in body
    assert "levels" in body
    assert "dataset_profile" in body

    services = client.get("/api/services/status")
    assert services.status_code == 200
    assert services.json()["status"] == "ok"
    assert "available_models" in services.json()

    leaderboard = client.get("/api/leaderboard")
    assert leaderboard.status_code == 200
    assert "rows" in leaderboard.json()


def test_leaderboard_runs_endpoints(tmp_path):
    output_dir = tmp_path / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    store = ExperimentStore(db_path=str(output_dir / "leaderboard.sqlite"))
    run_id = store.upsert_run(
        experiment_name="exp",
        dataset_name="pubmed_qa",
        pipeline_mode="asr_text_retrieval",
        start_time="2025-01-01T00:00:00+00:00",
        end_time="2025-01-01T00:01:00+00:00",
        duration_seconds=60.0,
        output_dir=str(output_dir),
        metrics={"MRR": 0.7},
        config={"experiment_name": "exp"},
        metadata={"cache": {"enabled": True}},
    )

    app = create_app(
        evaluation_runner=lambda cfg: _FakeResult({"ok": True}),
        matrix_runner=lambda cfg, setups: {"runs": []},
        provider_factory=_FakeProvider,
    )
    client = testclient.TestClient(app)

    runs = client.get(f"/api/leaderboard/runs?output_dir={output_dir}")
    assert runs.status_code == 200
    assert any(int(row["run_id"]) == run_id for row in runs.json()["runs"])

    run = client.get(f"/api/leaderboard/runs/{run_id}?output_dir={output_dir}")
    assert run.status_code == 200
    payload = run.json()
    assert payload["run_id"] == run_id
    assert payload["metrics"]["MRR"] == 0.7


def test_config_create_endpoint_applies_patch():
    app = create_app(
        evaluation_runner=lambda cfg: _FakeResult({"ok": True}),
        matrix_runner=lambda cfg, setups: {"runs": []},
        provider_factory=_FakeProvider,
    )
    client = testclient.TestClient(app)

    patch = {
        "runtime": {
            "model": {
                "pipeline_mode": "asr_only",
            },
            "vector_db": {
                "k": 9,
            },
        },
        "experiment": {
            "experiment_name": "webui-created",
            "service_runtime": {
                "startup_mode": "eager",
                "offload_policy": "never",
            },
        },
    }
    response = client.post(
        "/api/config/create",
        json={"preset_name": "fast_dev", "config_patch": patch, "auto_devices": False},
    )
    assert response.status_code == 200
    created = response.json()["config"]
    assert created["runtime"]["model"]["pipeline_mode"] == "asr_only"
    assert created["runtime"]["vector_db"]["k"] == 9
    assert created["experiment"]["service_runtime"]["startup_mode"] == "eager"
    assert created["experiment"]["service_runtime"]["offload_policy"] == "never"

def test_config_validate_endpoint_returns_normalized_config():
    app = create_app(
        evaluation_runner=lambda cfg: _FakeResult({"ok": True}),
        matrix_runner=lambda cfg, setups: {"runs": []},
        provider_factory=_FakeProvider,
    )
    client = testclient.TestClient(app)
    config_dict = _asr_only_config_dict()

    response = client.post("/api/config/validate", json={"config": config_dict, "auto_devices": False})
    assert response.status_code == 200
    normalized = response.json()["config"]
    assert normalized["experiment_name"] == config_dict["experiment_name"]


def test_submit_evaluation_returns_400_for_invalid_config():
    app = create_app(
        evaluation_runner=lambda cfg: _FakeResult({"ok": True}),
        matrix_runner=lambda cfg, setups: {"runs": []},
        provider_factory=_FakeProvider,
    )
    client = testclient.TestClient(app)

    invalid_config = {
        "runtime": {
            "model": {
                "asr_device": "bad_device",
            }
        }
    }
    response = client.post(
        "/api/jobs/evaluation",
        json={"config": invalid_config, "auto_devices": False},
    )
    assert response.status_code == 400
    assert "Invalid device format" in response.json()["detail"]


def test_submit_evaluation_returns_400_for_missing_dataset_fields():
    app = create_app(
        evaluation_runner=lambda cfg: _FakeResult({"ok": True}),
        matrix_runner=lambda cfg, setups: {"runs": []},
        provider_factory=_FakeProvider,
    )
    client = testclient.TestClient(app)

    invalid_config = {
        "runtime": {
            "data": {
                "dataset_source": "local",
                "dataset_name": "generic_audio_transcription",
                "audio_dir": None,
                "questions_path": None,
                "prepared_dataset_dir": None,
            }
        }
    }
    response = client.post(
        "/api/jobs/evaluation",
        json={"config": invalid_config, "auto_devices": False},
    )
    assert response.status_code == 400
    assert "missing required fields" in response.json()["detail"]


def test_evaluation_job_lifecycle_and_result_endpoint():
    app = create_app(
        evaluation_runner=lambda cfg: _FakeResult({"metrics": {"MRR": 0.5}}),
        matrix_runner=lambda cfg, setups: {"runs": []},
        provider_factory=_FakeProvider,
    )
    client = testclient.TestClient(app)
    config_dict = _asr_only_config_dict()

    submit = client.post("/api/jobs/evaluation", json={"config": config_dict, "auto_devices": False})
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]

    status = _wait_for_completion(client, job_id)
    assert status == "completed"

    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.json()["result"]["metrics"]["MRR"] == 0.5

    metadata = client.get(f"/api/jobs/{job_id}/metadata")
    assert metadata.status_code == 200
    meta_payload = metadata.json()
    assert "config" in meta_payload
    assert "metadata" in meta_payload

    artifacts = client.get(f"/api/jobs/{job_id}/artifacts")
    assert artifacts.status_code == 200
    assert "artifacts" in artifacts.json()


def test_matrix_job_lifecycle():
    app = create_app(
        evaluation_runner=lambda cfg: _FakeResult({"metrics": {"MRR": 0.5}}),
        matrix_runner=lambda cfg, setups: {"num_setups": len(setups), "runs": []},
        provider_factory=_FakeProvider,
    )
    client = testclient.TestClient(app)
    config_dict = _asr_only_config_dict()
    setups = [{"setup_id": "s1", "overrides": {"data_trace_limit": 5}}]

    submit = client.post(
        "/api/jobs/matrix",
        json={"base_config": config_dict, "test_setups": setups, "auto_devices": False},
    )
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]

    status = _wait_for_completion(client, job_id)
    assert status == "completed"

    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.json()["result"]["num_setups"] == 1


def test_live_query_endpoint(monkeypatch):
    class _FakeTextEmbeddingPipeline:
        def process(self, text):
            return [0.1, 0.2]

    class _FakeRetrievalPipeline:
        def search_batch(self, embeddings, k=5, query_texts=None):
            return [[{"id": "doc1", "title": "Doc 1", "text": "Body", "score": 0.9}]]

    monkeypatch.setattr(
        "evaluator.webapi.app.create_pipeline_from_config",
        lambda config, cache_manager, service_provider=None: SimpleNamespace(
            retrieval_pipeline=_FakeRetrievalPipeline(),
            text_embedding_pipeline=_FakeTextEmbeddingPipeline(),
        ),
    )
    monkeypatch.setattr("evaluator.webapi.app.load_dataset", lambda *args, **kwargs: object())

    app = create_app(
        evaluation_runner=lambda cfg: _FakeResult({"ok": True}),
        matrix_runner=lambda cfg, setups: {"runs": []},
        provider_factory=_FakeProvider,
    )
    client = testclient.TestClient(app)
    config_dict = EvaluationConfig().to_dict(include_config=True)

    response = client.post(
        "/api/live/query",
        json={"config": config_dict, "query_text": "test query", "k": 3, "auto_devices": False},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["query_text"] == "test query"
    assert payload["results"][0]["id"] == "doc1"
