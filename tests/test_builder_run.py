"""Phase 1 — the builder→run loop: a graph built in the visual builder can be submitted as a
job. Covers the spec→config translation helper and the POST /api/jobs/from-graph endpoint.

A fake evaluation_runner is injected so jobs run in-process (no subprocess, no model loading) —
the JobManager only uses the real subprocess path when the runner *is* ``run_evaluation``.
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from evaluator.config.graph_config import GraphConfigError  # noqa: E402
from evaluator.webapi.app import create_app  # noqa: E402
from evaluator.webapi.form_config import graph_spec_to_config_dict  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_Q = str(_ROOT / "examples/data/pubmed_qa_small/questions.json")
_C = str(_ROOT / "examples/data/pubmed_qa_small/corpus.json")


def _spec(dataset_params=None):
    """A minimal asr_text_retrieval canvas spec; dataset_source defaults to pubmed_qa."""
    ds = {"dataset": "pubmed_qa", "questions": _Q, "corpus": _C}
    if dataset_params is not None:
        ds = dataset_params
    nodes = [
        {"id": "dataset_source", "type": "dataset_source", "params": ds},
        {"id": "asr", "type": "asr", "params": {"model": "whisper"}},
        {"id": "text_embedding", "type": "text_embedding", "params": {"model": "labse"}},
        {"id": "corpus_embedding", "type": "corpus_embedding", "params": {}},
        {"id": "vector_db", "type": "vector_db", "params": {}},
        {"id": "retrieval", "type": "retrieval", "params": {}},
        {"id": "metrics", "type": "metrics", "params": {}},
        {"id": "finalize", "type": "finalize", "params": {}},
    ]
    edges = [
        {"from": "dataset_source", "to": "asr"}, {"from": "asr", "to": "text_embedding"},
        {"from": "dataset_source", "to": "corpus_embedding"},
        {"from": "corpus_embedding", "to": "vector_db"},
        {"from": "text_embedding", "to": "retrieval"},
        {"from": "vector_db", "to": "retrieval"},
        {"from": "retrieval", "to": "metrics"}, {"from": "metrics", "to": "finalize"},
    ]
    return {"mode": "asr_text_retrieval", "nodes": nodes, "edges": edges}


class _FakeResult:
    def to_dict(self, include_config=False):
        return {"MRR": 0.5}


@pytest.fixture()
def captured():
    return {}


@pytest.fixture()
def client(captured):
    def fake_runner(config, progress_callback=None):
        captured["config"] = config
        return _FakeResult()

    return TestClient(create_app(evaluation_runner=fake_runner))


# ── translation helper ──────────────────────────────────────────────────────────


def test_spec_to_config_synthesizes_dataset():
    cfg = graph_spec_to_config_dict(_spec(), experiment_name="t")
    assert cfg["experiment_name"] == "t"
    datasets = cfg.get("data", {}).get("datasets")
    assert datasets and datasets["dataset_source"]["dataset_name"] == "pubmed_qa"


def test_builder_derives_plumbing_from_meaningful_graph():
    """P3.2: the builder authors only meaningful operations; graph_spec_to_config_dict appends the
    structural plumbing (metric comparisons + report + finalize) so the canvas graph runs as the
    full DAG, auto-wired. Meaningful nodes (incl. per-node models) are preserved."""
    from evaluator import EvaluationConfig
    from evaluator.pipeline import build_graph_for_config
    from evaluator.pipeline.graph.operators import node_kind

    spec = {"mode": "asr_text_retrieval", "nodes": [
        {"id": "dataset_source", "type": "dataset_source",
         "params": {"dataset": "pubmed_qa", "questions": _Q, "corpus": _C}},
        {"id": "tts", "type": "tts", "params": {}},
        {"id": "asr", "type": "asr", "params": {"model": "whisper"}},
        {"id": "text_embedding", "type": "text_embedding", "params": {"model": "labse"}},
        {"id": "corpus_embedding", "type": "corpus_embedding", "params": {}},
        {"id": "vector_db", "type": "vector_db", "params": {}},
        {"id": "retrieval", "type": "retrieval", "params": {}},
    ], "edges": []}  # NO metrics / finalize drawn — they are derived
    cfg = EvaluationConfig.from_dict(graph_spec_to_config_dict(spec)).with_auto_devices()
    kinds = {node_kind(n.stage, n.params) for n in build_graph_for_config(cfg).nodes}
    # the structural plumbing was appended + auto-wired; tts inferred audio_synthesis
    assert {"transcription_metrics", "retrieval_metrics", "metrics", "finalize", "tts"} <= kinds


def test_spec_to_config_unknown_dataset_raises():
    with pytest.raises(GraphConfigError):
        graph_spec_to_config_dict(_spec({"dataset": "no_such_dataset"}))


def test_duplicate_model_role_keeps_per_node_models():
    """Node-centric config: two stage-typed embed nodes of the same role no longer clobber — EACH
    keeps its own model ON the node (a per-node override the executor reads via
    ``resolved_model_config``); neither is promoted into the flat ``model.*`` default (only a
    top-level ``nodes.<role>`` block sets that — absent here). Translation succeeds with BOTH
    models preserved (the old 'same model role' rejection is gone — two distinct same-role
    models are now expressible)."""
    spec = {"mode": "asr_text_retrieval", "nodes": [
        {"id": "dataset_source", "type": "source",
         "params": {"dataset": "pubmed_qa", "questions": _Q, "corpus": _C}},
        {"id": "q_emb", "type": "embed", "params": {"model": "labse"}},
        {"id": "c_emb", "type": "embed", "params": {"model": "jina_v4"}},
    ], "edges": []}
    legacy = graph_spec_to_config_dict(spec)  # no GraphConfigError
    assert "model" not in legacy or "text_emb_model_type" not in legacy["model"]
    by_id = {n["id"]: n for n in legacy["graph_override"]["nodes"] if isinstance(n, dict)}
    assert by_id["q_emb"]["params"]["model"] == "labse"        # first → stays on its own node
    assert by_id["c_emb"]["params"]["model"] == "jina_v4"      # second → per-node override on-node


def test_spec_to_config_does_not_mutate_caller_spec():
    """build_evaluation_config_kwargs mutates node dicts in place; the helper deep-copies so the caller's spec
    is untouched and a second call yields the same result (idempotent)."""
    import copy

    spec = _spec()
    snapshot = copy.deepcopy(spec)
    first = graph_spec_to_config_dict(spec)
    second = graph_spec_to_config_dict(spec)
    assert spec == snapshot
    assert first == second


# ── endpoint ────────────────────────────────────────────────────────────────────


def test_run_from_graph_submits_job(client, captured):
    r = client.post(
        "/api/jobs/from-graph",
        json={"spec": _spec(), "experiment_name": "builder_test"},
    )
    assert r.status_code == 200
    assert r.json()["job_id"]
    # The injected runner received the translated, dataset-bearing config.
    cfg = captured.get("config")
    assert cfg is not None and cfg.experiment_name == "builder_test"
    assert "dataset_source" in (cfg.data.datasets or {})


def test_drawing_llm_feature_node_enables_it():
    """A judge / answer-gen node drawn in the builder flips its config flag on (it would
    otherwise silently no-op), and the global `llm` block rides into the feature config — so a
    builder graph with a judge node is a runnable judge. A graph without them stays off."""
    spec = _spec()
    spec["nodes"] += [
        {"id": "answer_gen", "type": "answer_gen", "params": {}},
        {"id": "answer_judge", "type": "answer_judge", "params": {}},
    ]
    spec["edges"] += [{"from": "metrics", "to": "answer_gen"},
                      {"from": "metrics", "to": "answer_judge"}]
    spec["llm"] = {"use_local_server": True,
                   "local_server_url": "http://localhost:11434/v1/chat/completions",
                   "model": "mistral:7b-instruct"}
    cfg = graph_spec_to_config_dict(spec, experiment_name="t")
    assert cfg["judge"]["enabled"] is True
    assert cfg["answer_generation"]["enabled"] is True

    # No LLM-feature node → no feature block injected (the judge stays off by default).
    plain = graph_spec_to_config_dict(_spec(), experiment_name="t")
    assert "judge" not in plain and "answer_generation" not in plain


def test_run_graph_applies_per_node_edits(client, captured):
    """Config & Run inline forms: a preset → config_to_canvas_spec → edit a node's param →
    /ui/run-graph submits a config reflecting the edit (the round-trip the page relies on)."""
    from evaluator import EvaluationConfig, get_preset, list_presets
    from evaluator.webapi.form_builder import config_to_canvas_spec

    preset = next((p for p in list_presets() if "small" in p), list_presets()[0])
    canvas = config_to_canvas_spec(EvaluationConfig.from_dict(get_preset(preset)))
    nodes = [{"id": n["id"], "type": n["type"], "params": dict(n.get("params") or {})}
             for n in canvas["nodes"] if not n.get("structural")]
    asr = [n for n in nodes if n["id"] == "asr"]
    assert asr, "preset has an asr node to edit"
    asr[0]["params"]["size"] = "tiny"   # the user's per-node edit
    r = client.post(
        "/ui/run-graph", json={"spec": {"mode": canvas["mode"], "nodes": nodes}, "name": preset}
    )
    assert r.status_code == 200
    assert captured["config"].model.asr_size == "tiny"   # the edit reached the run config


def test_validate_graph_reports_bad_dataset_path(client):
    """Config & Run non-blocking validate: an edited bad dataset path is REPORTED as a problem
    (the page already loaded the preview/forms), not raised — so the user fixes + re-validates."""
    from evaluator import EvaluationConfig, get_preset, list_presets
    from evaluator.webapi.form_builder import config_to_canvas_spec

    preset = next((p for p in list_presets() if "small" in p), list_presets()[0])
    canvas = config_to_canvas_spec(EvaluationConfig.from_dict(get_preset(preset), validate=False))
    nodes = [{"id": n["id"], "type": n["type"], "params": dict(n.get("params") or {})}
             for n in canvas["nodes"] if not n.get("structural")]
    src = [n for n in nodes if n["id"] == "dataset_source"][0]
    ok = client.post("/ui/validate-graph", json={"spec": {"mode": canvas["mode"], "nodes": nodes}})
    assert ok.status_code == 200 and "error-box" not in ok.text   # valid → no problems
    src["params"]["questions"] = "/nonexistent/bad.json"
    bad = client.post("/ui/validate-graph", json={"spec": {"mode": canvas["mode"], "nodes": nodes}})
    assert bad.status_code == 200 and "does not exist" in bad.text   # reported, not blocked


def test_build_metrics_preview_includes_judge(client):
    """J5: a canvas graph containing the LLM-judge node previews its config-independent judge
    metrics (judge_overall + judge_pass_rate) — the node declares both judge_scores + judge_pass
    as outputs, so graph_advice's applicable-metrics advice surfaces them while editing."""
    from evaluator import EvaluationConfig
    from evaluator.webapi.form_builder import build_canvas_graph, graph_advice

    spec = _spec()
    spec["nodes"] += [
        {"id": "build_query_traces", "type": "build_query_traces", "params": {}},
        {"id": "answer_judge", "type": "answer_judge", "params": {}},
    ]
    spec["edges"] += [
        {"from": "retrieval", "to": "build_query_traces"},
        {"from": "build_query_traces", "to": "answer_judge"},
    ]
    graph = build_canvas_graph(spec)
    _, metric_list = graph_advice(graph, EvaluationConfig())
    metrics = set(metric_list)
    assert {"judge_overall", "judge_pass_rate"} <= metrics


def test_build_no_spurious_ground_truth_warning(client):
    """P3.1: the GT notice is gated to source-DECLARED ground truth (relevant_docs). pubmed_qa
    publishes it, so a retrieval graph raises no 'ground truth' warning — and the run-time-loaded
    transcription GT (reference_text, not a declared node output) never false-positives."""
    from evaluator import EvaluationConfig
    from evaluator.webapi.form_builder import build_canvas_graph, graph_advice

    graph = build_canvas_graph(_spec())
    warnings, _ = graph_advice(graph, EvaluationConfig())
    assert [w for w in warnings if "ground truth" in w] == []


def test_validate_builder_renders_shared_fragment(client):
    """The builder's Validate uses the SAME _validation.html module as Config & Run: a valid graph
    returns the styled 'Valid ✓ — N levels' summary + metric chips; a broken graph the same
    error-box — no bespoke builder markup."""
    html = client.post("/ui/validate-builder", json=_spec()).text
    assert "Valid ✓" in html and "levels" in html
    assert 'class="chip"' in html  # metrics rendered as chips via the shared fragment
    bad = client.post("/ui/validate-builder",
                      json={"mode": "asr_text_retrieval",
                            "nodes": [{"id": "x", "type": "not_a_real_node", "params": {}}],
                            "edges": []}).text
    assert "error-box" in bad  # the same styled error component


def test_validate_builder_shares_the_export_run_pipeline_not_just_topology(client):
    """Validate must catch anything Export/Run would catch, not just internal topology
    consistency — it now calls the SAME `build_validated_run_config` those two use, instead of
    a lighter topology-only check that let a meaningful node renamed to a reserved structural id
    (e.g. "metrics") pass as "Valid" and only fail later at Export. Meaningful-only canvas spec
    (no metrics/finalize authored) with explicit port edges, matching what the builder actually
    exports -- `_spec()` above hand-writes metrics/finalize and uses ordering-only edges, so it
    can't exercise `complete_structural_plumbing` at all."""
    ds = {"dataset": "pubmed_qa", "questions": _Q, "corpus": _C}
    spec = {
        "mode": "asr_text_retrieval",
        "nodes": [
            {"id": "dataset_source", "type": "dataset_source", "params": ds},
            {"id": "corpus_embedding", "type": "corpus_embedding", "params": {}},
            {"id": "vector_db", "type": "vector_db", "params": {}},
            {"id": "text_embedding", "type": "text_embedding", "params": {"model": "labse"}},
            # renamed from "retrieval" to "metrics" -- collides with the auto-derived report node
            {"id": "metrics", "type": "retrieval", "params": {}},
        ],
        "edges": [
            {"from": "dataset_source", "output": "corpus", "to": "corpus_embedding",
             "input": "corpus"},
            {"from": "corpus_embedding", "output": "corpus_vectors", "to": "vector_db",
             "input": "corpus_vectors"},
            {"from": "dataset_source", "output": "reference_text", "to": "text_embedding",
             "input": "query_text"},
            {"from": "text_embedding", "output": "text_query_vectors", "to": "metrics",
             "input": "query_vectors"},
            {"from": "vector_db", "output": "vector_index", "to": "metrics",
             "input": "vector_index"},
        ],
    }
    html = client.post("/ui/validate-builder", json=spec).text
    assert "error-box" in html
    assert "already used by a" in html and "retrieval" in html  # HTML-escapes the quoted kind


def test_validate_graph_shares_the_export_run_pipeline_and_shows_metrics(client):
    """Config & Run's own Validate (`/ui/validate-graph`) had the identical gap the builder's
    Validate had: it built a bare `EvaluationConfig` from `graph_spec_to_config_dict` directly
    and called `collect_problems`, never running `complete_structural_plumbing` or
    `graph_advice` -- so it neither caught an Export/Run-only failure nor ever showed applicable
    metrics. Fixed to share `build_validated_run_config` + `graph_advice`, same as the builder."""
    ds = {"dataset": "pubmed_qa", "questions": _Q, "corpus": _C}
    spec = {
        "mode": "asr_text_retrieval",
        "nodes": [
            {"id": "dataset_source", "type": "dataset_source", "params": ds},
            {"id": "corpus_embedding", "type": "corpus_embedding", "params": {}},
            {"id": "vector_db", "type": "vector_db", "params": {}},
            {"id": "text_embedding", "type": "text_embedding", "params": {"model": "labse"}},
            {"id": "retrieval", "type": "retrieval", "params": {}},
        ],
        "edges": [
            {"from": "dataset_source", "output": "corpus", "to": "corpus_embedding",
             "input": "corpus"},
            {"from": "corpus_embedding", "output": "corpus_vectors", "to": "vector_db",
             "input": "corpus_vectors"},
            {"from": "dataset_source", "output": "reference_text", "to": "text_embedding",
             "input": "query_text"},
            {"from": "text_embedding", "output": "text_query_vectors", "to": "retrieval",
             "input": "query_vectors"},
            {"from": "vector_db", "output": "vector_index", "to": "retrieval",
             "input": "vector_index"},
        ],
    }
    html = client.post("/ui/validate-graph", json={"spec": spec}).text
    assert "Valid" in html and "levels" in html
    assert 'class="chip"' in html  # applicable-metrics chips, previously never shown here

    # and the same rename-collision class of bug the builder's Validate now catches
    spec["nodes"][-1]["id"] = "metrics"  # renamed from "retrieval" -- collides with the
    spec["edges"] = [{**e, "to": "metrics"} if e["to"] == "retrieval" else e for e in spec["edges"]]
    bad_html = client.post("/ui/validate-graph", json={"spec": spec}).text
    assert "error-box" in bad_html
    assert "already used by a" in bad_html


def test_run_from_graph_unknown_dataset_is_400(client):
    bad = {"spec": {"mode": "asr_text_retrieval",
                    "nodes": [{"id": "dataset_source", "type": "dataset_source",
                               "params": {"dataset": "no_such_dataset"}}],
                    "edges": []}}
    r = client.post("/api/jobs/from-graph", json=bad)
    assert r.status_code == 400
    assert "no_such_dataset" in r.json()["detail"]


def test_run_from_graph_requires_dataset_choice(client):
    """A dataset_source with no dataset chosen is rejected at Run, not silently defaulted."""
    spec = {"mode": "asr_text_retrieval", "nodes": [
        {"id": "dataset_source", "type": "source", "params": {}},
        {"id": "asr", "type": "convert", "params": {"model": "whisper"}},
    ], "edges": []}
    r = client.post("/api/jobs/from-graph", json={"spec": spec})
    assert r.status_code == 400
    assert "Select a dataset" in r.json()["detail"]


def test_run_from_graph_unknown_node_is_400(client):
    bad = {"spec": {"mode": "asr_text_retrieval",
                    "nodes": [{"id": "x", "type": "not_a_real_node", "params": {}}],
                    "edges": []}}
    r = client.post("/api/jobs/from-graph", json=bad)
    assert r.status_code == 400


def test_builder_page_wires_run_button(client):
    """The builder renders a Run button that posts to the from-graph endpoint."""
    t = client.get("/ui/builder").text
    assert 'id="btn-run"' in t
    assert "/api/jobs/from-graph" in t
