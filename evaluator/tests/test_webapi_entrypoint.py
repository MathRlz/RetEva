import pytest

uvicorn = pytest.importorskip("uvicorn")

from evaluator.webapi.__main__ import main


def test_webapi_main_invokes_uvicorn(monkeypatch):
    calls = {}

    def _fake_run(app, **kwargs):
        calls["app"] = app
        calls["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", _fake_run)

    exit_code = main(["--host", "0.0.0.0", "--port", "9000", "--reload", "--log-level", "debug"])
    assert exit_code == 0
    assert calls["app"] == "evaluator.webapi.app:create_app"
    assert calls["kwargs"]["factory"] is True
    assert calls["kwargs"]["host"] == "0.0.0.0"
    assert calls["kwargs"]["port"] == 9000
    assert calls["kwargs"]["reload"] is True
    assert calls["kwargs"]["log_level"] == "debug"

