"""V1 pin: chart data is escaped for the <script> context it is interpolated into.

Chart labels carry `experiment_name`, which comes from a user-authored config, through
sqlite, into an inline `<script>` block. `json.dumps` does NOT escape `<`, so a name
containing `</script>` used to break out of the tag (stored injection). Jinja's `|tojson`
emits `\\u003c`; these tests fail if anyone reverts to a pre-dumped `|safe` string.
"""

from evaluator.storage import ExperimentStore

PAYLOAD = '</script><img src=x onerror=alert(1)>'


def test_leaderboard_chart_labels_escape_script_breakout(client, tmp_path):
    store = ExperimentStore(db_path=str(tmp_path / "leaderboard.sqlite"))
    store.upsert_run(
        experiment_name=PAYLOAD,
        dataset_name="pubmed_qa",
        pipeline_mode="asr_text_retrieval",
        duration_seconds=1.0,
        config={"experiment_name": PAYLOAD},
        metrics={"MRR": 0.5},
    )
    html = client.get(f"/ui/leaderboard?metric=MRR&output_dir={tmp_path}").text

    # the raw breakout must never reach the script block…
    assert "</script><img" not in html
    # …and the payload must still be PRESENT, escaped (not silently dropped)
    assert "\\u003c/script\\u003e" in html


def test_pareto_chart_text_escapes_script_breakout(client, tmp_path):
    store = ExperimentStore(db_path=str(tmp_path / "leaderboard.sqlite"))
    for name in (PAYLOAD, "safe-run"):
        store.upsert_run(
            experiment_name=name,
            dataset_name="pubmed_qa",
            pipeline_mode="asr_text_retrieval",
            duration_seconds=1.0,
            config={"experiment_name": name},
            metrics={"MRR": 0.5, "latency_ms": 10.0},
            experiment_group="g",
        )
    html = client.get(
        f"/ui/pareto?experiment_group=g&objectives=MRR:max,latency_ms:min"
        f"&output_dir={tmp_path}"
    ).text
    assert "</script><img" not in html
