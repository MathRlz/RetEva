"""Phase 3 — results depth: download a run's resolved config.yaml + surface provenance/cost
(cache reuse, LLM cost, model identity, stage timing) on the run-detail page.
"""

from evaluator.storage import ExperimentStore


def _seed_run(tmp_path, *, metrics=None, config=None):
    store = ExperimentStore(db_path=str(tmp_path / "leaderboard.sqlite"))
    return store.upsert_run(
        experiment_name="exp",
        dataset_name="pubmed_qa",
        pipeline_mode="asr_text_retrieval",
        duration_seconds=12.3,
        config=config or {"experiment_name": "exp", "model": {"asr_model_type": "whisper"}},
        metrics=metrics or {"MRR": 0.5},
    )


def test_download_run_config_yaml(client, tmp_path):
    rid = _seed_run(tmp_path)
    r = client.get(f"/ui/runs/{rid}/config.yaml", params={"output_dir": str(tmp_path)})
    assert r.status_code == 200
    assert "asr_model_type: whisper" in r.text
    assert "attachment" in r.headers["content-disposition"]
    assert f"run-{rid}-config.yaml" in r.headers["content-disposition"]


def test_download_run_config_missing_is_404(client, tmp_path):
    r = client.get("/ui/runs/999/config.yaml", params={"output_dir": str(tmp_path)})
    assert r.status_code == 404


def test_run_detail_renders_provenance_and_download(client, tmp_path):
    rid = _seed_run(tmp_path, metrics={
        "MRR": 0.5,
        "report": {"provenance": {
            "cost": {"total_usd": 0.012, "tokens": 3400},
            "cache_stats": {"asr": {"hits": 3, "misses": 1}},
            "models": {"resolved": {"asr": "openai/whisper-base", "embedder": "labse"}},
            "timing": {"asr": 1.2, "retrieval": 0.4},
        }},
    })
    r = client.get(f"/ui/runs/{rid}", params={"output_dir": str(tmp_path)})
    assert r.status_code == 200
    assert "Provenance" in r.text          # the section renders
    assert "config.yaml" in r.text         # the download link is present
    assert "openai/whisper-base" in r.text  # model identity surfaced
    assert "total_usd" in r.text            # LLM cost surfaced
