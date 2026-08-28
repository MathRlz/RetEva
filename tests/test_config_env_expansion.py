"""P3 pins: ``${VAR}`` in a config YAML expands from the environment at load.

Campaign configs keep machine-specific dataset/checkpoint roots in env vars so one
committed config runs on any machine.
"""

from evaluator.config.loading import _expand_env


def test_expands_nested_strings(monkeypatch):
    monkeypatch.setenv("APM_CHECKPOINT_DIR", "/models/apm")
    tree = {
        "nodes": {"audio_embedding": {"model_path": "${APM_CHECKPOINT_DIR}/w.pt"}},
        "graph": {"nodes": [{"params": {"paths": ["$APM_CHECKPOINT_DIR/a.pt"]}}]},
        "k": 5,
    }
    out = _expand_env(tree)
    assert out["nodes"]["audio_embedding"]["model_path"] == "/models/apm/w.pt"
    assert out["graph"]["nodes"][0]["params"]["paths"] == ["/models/apm/a.pt"]
    assert out["k"] == 5  # non-strings untouched


def test_unset_variable_is_left_verbatim(monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNSET_VAR_XYZ", raising=False)
    out = _expand_env({"p": "${DEFINITELY_UNSET_VAR_XYZ}/x.pt"})
    # left literal so the missing-file error names the variable the user forgot
    assert out["p"] == "${DEFINITELY_UNSET_VAR_XYZ}/x.pt"


def test_yaml_load_expands(tmp_path, monkeypatch):
    from evaluator.config.evaluation import EvaluationConfig

    monkeypatch.setenv("CAMPAIGN_DATA", str(tmp_path))
    cfg_path = tmp_path / "c.yaml"
    # block style: a scalar may start with ${...} unquoted (flow style {…} would not parse)
    cfg_path.write_text(
        "experiment:\n  name: t\n"
        "dataset:\n  id: pubmed_qa\n"
        "  questions: ${CAMPAIGN_DATA}/q.json\n"
        "  corpus: ${CAMPAIGN_DATA}/c.json\n"
    )
    cfg = EvaluationConfig.from_yaml(str(cfg_path), validate=False)
    assert cfg.data.questions_path == f"{tmp_path}/q.json"
