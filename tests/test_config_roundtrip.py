"""Phase 1 — YAML config round-trip into the builder canvas.

A YAML config → canvas seed (`/api/graph/from-config` / `config_to_canvas_spec`) → edit → run
must preserve the experiment: the configured models, the dataset, and the topology. Also guards
the core `build_evaluation_config_kwargs` fix that lets the builder's *stage*-typed graph nodes (`convert`,
`embed`, `search` — what the palette emits) fold their models, not just *kind*-typed (`asr`).
"""

from pathlib import Path

from evaluator.webapi.form_config import graph_spec_to_config_dict
from tests.graph_test_helpers import explicit_graph

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_YAML = (_ROOT / "configs/e2e_pubmed_qa_small.yaml").read_text()
_Q = str(_ROOT / "examples/data/pubmed_qa_small/questions.json")
_C = str(_ROOT / "examples/data/pubmed_qa_small/corpus.json")


def _canvas_to_run_spec(canvas):
    """The builder's exportSpec shape from a from-config canvas seed: PORT-LEVEL edges from
    the rendered bindings (artifact + producer), the input key resolved via the node's ports —
    exactly the mapping the browser's exportSpec performs (E5)."""
    def _port_key(node, art):
        for port in node.get("input_ports") or []:
            if art in port["names"]:
                return port["label"]
        return art

    edges, seen = [], set()
    for n in canvas["nodes"]:
        for art, prod in n["bindings"]:
            tup = (prod, art, n["id"], _port_key(n, art))
            if tup in seen:
                continue
            seen.add(tup)
            edges.append({"from": prod, "output": art, "to": n["id"], "input": tup[3]})
    return {
        "mode": canvas["mode"],
        "nodes": [{"id": n["id"], "type": n["type"], "params": n["params"]}
                  for n in canvas["nodes"]],
        "edges": edges,
    }


# ── stage-typed fold (the build_evaluation_config_kwargs fix) ───────────────────────────────────


def test_stage_typed_graph_nodes_fold_models():
    """A palette-style stage-typed export (`convert`/`embed`) folds its models into the config —
    previously only kind-typed (`asr`) did, silently dropping models from real builder graphs."""
    spec = {"mode": "asr_text_retrieval", "nodes": [
        {"id": "dataset_source", "type": "source", "params": {
            "dataset": "pubmed_qa",
            "questions": str(_ROOT / "examples/data/pubmed_qa_small/questions.json"),
            "corpus": str(_ROOT / "examples/data/pubmed_qa_small/corpus.json")}},
        {"id": "asr", "type": "convert",
         "params": {"model": "whisper", "name": "openai/whisper-base"}},
        {"id": "text_embedding", "type": "embed", "params": {"model": "labse"}},
    ], "edges": []}
    legacy = graph_spec_to_config_dict(spec)
    assert legacy["model"]["asr_model_type"] == "whisper"
    assert legacy["model"]["asr_model_name"] == "openai/whisper-base"
    assert legacy["model"]["text_emb_model_type"] == "labse"
    assert "dataset_source" in (legacy.get("data", {}).get("datasets") or {})


def test_two_same_role_models_no_clobber():
    """Node-centric config: two graph nodes of the same model role no longer clobber — the FIRST
    sets the flat ``model.*`` default (one-per-role back-compat), each SUBSEQUENT keeps its model ON
    the node (a per-node override the executor reads via ``_node_pipeline``). So one graph can run
    two distinct text embedders (asymmetric encoders / an ablation)."""
    from evaluator.config.graph_config import build_evaluation_config_kwargs

    legacy = build_evaluation_config_kwargs({"graph": explicit_graph([
        "dataset_source",  # gives the embedders a query_text producer (edges derivable)
        {"id": "q_emb", "type": "text_embedding", "params": {"model": "labse"}},
        {"id": "q_emb2", "type": "text_embedding", "params": {"model": "jina"}},
    ])})  # no GraphConfigError
    assert legacy["model"]["text_emb_model_type"] == "labse"   # first → flat default
    by_id = {n["id"]: n for n in legacy["graph_override"]["nodes"] if isinstance(n, dict)}
    assert not by_id["q_emb"].get("params", {}).get("model")   # first: stripped → uses the global
    assert by_id["q_emb2"]["params"]["model"] == "jina"        # second: per-node override on-node


# ── endpoint ────────────────────────────────────────────────────────────────────


def test_from_config_returns_canvas_seed(client):
    r = client.post("/api/graph/from-config", json={"yaml": _CONFIG_YAML})
    assert r.status_code == 200
    canvas = r.json()
    assert canvas["mode"] == "asr_text_retrieval"
    by_id = {n["id"]: n for n in canvas["nodes"]}
    # configured models are folded onto the right nodes
    assert by_id["asr"]["params"]["model"] == "whisper"
    assert by_id["text_embedding"]["params"]["model"] == "labse"
    # the single-source dataset is pushed onto the dataset_source node
    assert by_id["dataset_source"]["params"]["dataset"] == "pubmed_qa"


def test_from_config_empty_is_400(client):
    assert client.post("/api/graph/from-config", json={"yaml": ""}).status_code == 400


def test_from_config_bad_yaml_is_400(client):
    r = client.post("/api/graph/from-config", json={"yaml": "%not: [valid"})
    assert r.status_code == 400


# ── full round-trip ──────────────────────────────────────────────────────────────


def test_config_round_trips_through_canvas(client):
    """YAML → canvas → run spec → config reproduces the models + dataset."""
    canvas = client.post("/api/graph/from-config", json={"yaml": _CONFIG_YAML}).json()
    legacy = graph_spec_to_config_dict(_canvas_to_run_spec(canvas), experiment_name="rt")
    assert legacy["model"]["asr_model_type"] == "whisper"
    assert legacy["model"]["text_emb_model_type"] == "labse"
    datasets = legacy.get("data", {}).get("datasets") or {}
    assert datasets and datasets["dataset_source"]["dataset_name"] == "pubmed_qa"


def test_builder_page_has_load_affordance(client):
    t = client.get("/ui/builder").text
    assert 'id="btn-load-config"' in t
    assert "/api/graph/from-config" in t


# ── builder → config (the reverse bridge) ────────────────────────────────────────


def test_to_config_emits_runnable_yaml(client):
    canvas = client.post("/api/graph/from-config", json={"yaml": _CONFIG_YAML}).json()
    spec = _canvas_to_run_spec(canvas)
    r = client.post("/api/graph/to-config", json={"spec": spec, "experiment_name": "exported"})
    assert r.status_code == 200
    text = r.json()["yaml"]
    assert "experiment:" in text and "graph:" in text
    # the exported YAML re-loads onto the canvas (full config↔builder loop) with models intact
    reloaded = client.post("/api/graph/from-config", json={"yaml": text})
    assert reloaded.status_code == 200
    asr = next(n for n in reloaded.json()["nodes"] if n["id"] == "asr")
    assert asr["params"]["model"] == "whisper"


def test_to_config_empty_graph_is_400(client):
    r = client.post("/api/graph/to-config",
                    json={"spec": {"mode": "asr_text_retrieval", "nodes": []}})
    assert r.status_code == 400


# ── 4.1: clean single-source export + idempotency + knob preservation ─────────────


def _export(client, yaml_text):
    canvas = client.post("/api/graph/from-config", json={"yaml": yaml_text}).json()
    spec = _canvas_to_run_spec(canvas)
    return client.post("/api/graph/to-config",
                       json={"spec": spec, "experiment_name": "exp"}).json()["yaml"]


def test_export_lifts_single_source_to_clean_dataset_block(client):
    y = _export(client, _CONFIG_YAML)
    # a top-level `dataset:` block, not a verbose `data.datasets` map keyed by node id
    assert "dataset:" in y and "datasets:" not in y
    assert "id: pubmed_qa" in y


def test_export_preserves_per_run_knobs(client):
    # the fixture sets trace_limit: 5 and batch_size: 4 — both must survive the round-trip
    y = _export(client, _CONFIG_YAML)
    assert "trace_limit: 5" in y
    assert "batch_size: 4" in y


def test_export_keeps_edges_without_accumulating(client):
    # E5: the canvas connections ARE the edges — the export keeps every one, and a second
    # round-trip reproduces the same edge set (no growth, no loss).
    y = _export(client, _CONFIG_YAML)
    assert "edges:" in y
    y2 = _export(client, y)
    assert y.count("- from:") == y2.count("- from:") > 0


def test_export_is_fixpoint_idempotent(client):
    # the mode label normalises once (template→custom); thereafter export is a fixpoint
    y2 = _export(client, _export(client, _CONFIG_YAML))
    y3 = _export(client, y2)
    assert y2 == y3


def test_builder_page_has_export_config(client):
    t = client.get("/ui/builder").text
    assert 'id="btn-export-config"' in t
    assert "/api/graph/to-config" in t


# ── live builder advice: metric-applicability + embedding-space warnings ──────────


def test_build_previews_applicable_metrics(client):
    from evaluator import EvaluationConfig
    from evaluator.webapi.form_builder import build_canvas_graph, graph_advice

    canvas = client.post("/api/graph/from-config", json={"yaml": _CONFIG_YAML}).json()
    spec = _canvas_to_run_spec(canvas)
    graph = build_canvas_graph(spec)
    warnings, metrics = graph_advice(graph, EvaluationConfig())
    # a retrieval graph lands the ranking metrics; warnings empty for a consistent graph
    assert "recall@5" in metrics and "mrr" in metrics
    assert warnings == []


def _mismatch_spec():
    """A graph whose query embedder (labse) and corpus embedder (jina_v4) live in different
    embedding spaces — the V[s] check fires."""
    return {"mode": "asr_text_retrieval", "nodes": [
        {"id": "dataset_source", "type": "source",
         "params": {"dataset": "pubmed_qa", "questions": _Q, "corpus": _C}},
        {"id": "asr", "type": "convert", "params": {"model": "whisper"}},
        {"id": "text_embedding", "type": "embed", "params": {"model": "labse"}},
        {"id": "corpus_embedding", "type": "embed",
         "params": {"axis": "corpus", "model": "jina_v4"}},
        {"id": "vector_db", "type": "index", "params": {}},
        {"id": "retrieval", "type": "search", "params": {}},
        {"id": "metrics", "type": "measure", "params": {}},
        {"id": "finalize", "type": "sink", "params": {}},
    ], "edges": [
        {"from": "dataset_source", "to": "asr"}, {"from": "asr", "to": "text_embedding"},
        {"from": "dataset_source", "to": "corpus_embedding"},
        {"from": "corpus_embedding", "to": "vector_db"},
        {"from": "text_embedding", "to": "retrieval"},
        {"from": "vector_db", "to": "retrieval"},
        {"from": "retrieval", "to": "metrics"}, {"from": "metrics", "to": "finalize"},
    ]}


def test_build_warns_on_embedding_space_mismatch(client):
    from evaluator import EvaluationConfig
    from evaluator.webapi.form_builder import build_canvas_graph, graph_advice

    graph = build_canvas_graph(_mismatch_spec())
    warnings, _ = graph_advice(graph, EvaluationConfig())
    assert warnings and "Embedding-space mismatch" in warnings[0]


def test_validate_warning_and_run_rejection_agree(client):
    """4.2: the edit-time warning and the Run rejection are the *same* check + message — so a
    mismatch surfaced at Validate is exactly what blocks Run (no silent divergence)."""
    from evaluator import EvaluationConfig
    from evaluator.webapi.form_builder import build_canvas_graph, graph_advice

    spec = _mismatch_spec()
    graph = build_canvas_graph(spec)
    warnings, _ = graph_advice(graph, EvaluationConfig())
    warning = warnings[0]
    run = client.post("/api/jobs/from-graph", json={"spec": spec})
    assert run.status_code == 400
    assert run.json()["detail"] == warning


def test_builder_validate_warns_on_mismatch(client):
    """The builder's Validate renders the shared ``_validation.html`` fragment (same module as
    Config & Run), surfacing the embedding-space mismatch — the same check that blocks Run."""
    html = client.post("/ui/validate-builder", json=_mismatch_spec()).text
    assert "Embedding-space mismatch" in html
    assert 'class="warn"' in html  # rendered through the shared styled fragment


# ── 4.5: branched config loads as the base graph + re-hydratable branches ─────────


def test_from_config_restores_branches_and_base_graph(client):
    branched = (_ROOT / "configs/e2e_pubmed_qa_branched.yaml").read_text()
    j = client.post("/api/graph/from-config", json={"yaml": branched}).json()
    # the branch definitions ride along (the panel re-hydrates), not a tangle of @branch nodes
    assert j.get("branches"), "branch definitions should be passed through"
    assert all("@" not in n["id"] for n in j["nodes"]), "canvas shows the base graph, un-expanded"


def test_feature_node_params_rehydrate_onto_canvas_and_round_trip():
    """Audit 2026-07 #2: the loader folds feature-node params into sub-configs; the canvas seed
    must rehydrate them onto the owning node (not show form defaults), and the canvas→config
    path must deliver them back — else a config→builder→config trip silently resets the
    feature (judge ran with gpt-4o-mini instead of the configured local model)."""
    import yaml

    from evaluator.config.evaluation import EvaluationConfig
    from evaluator.config.graph_config import build_evaluation_config_kwargs
    from evaluator.webapi.form_builder import config_to_canvas_spec
    from evaluator.webapi.form_config import graph_spec_to_config_dict

    raw = yaml.safe_load((_ROOT / "configs/local_llm_example.yaml").read_text())
    cfg = EvaluationConfig.from_dict(build_evaluation_config_kwargs(raw), validate=False)
    canvas = config_to_canvas_spec(cfg)

    judge = next(n for n in canvas["nodes"] if n["id"] == "answer_judge")
    assert judge["params"]["model"] == cfg.judge.model          # configured, not the default

    spec = {"nodes": [{"id": n["id"], "type": n["type"], "params": dict(n.get("params") or {})}
                      for n in canvas["nodes"] if not n.get("structural")],
            "edges": canvas.get("edges") or []}
    cfg2 = EvaluationConfig.from_dict(
        graph_spec_to_config_dict(spec, experiment_name="rt"), validate=False)
    assert cfg2.judge.model == cfg.judge.model
    assert cfg2.judge.score_aggregation == cfg.judge.score_aggregation
    assert cfg2.judge.judge_aspects == cfg.judge.judge_aspects
    assert cfg2.query_optimization.model == cfg.query_optimization.model
    assert cfg2.query_optimization.use_local_server is cfg.query_optimization.use_local_server
