"""YAML config round-trip into the builder canvas.

A YAML config → canvas seed (`/api/graph/from-config` / `config_to_canvas_spec`) → edit → run
must preserve the experiment: the configured models, the dataset, and the topology. Models live
on their own nodes — `build_evaluation_config_kwargs` carries them through for the builder's
*stage*-typed nodes (`convert`, `embed`, `search` — what the palette emits) as well as for
*kind*-typed ones (`asr`), and the executor resolves them via `resolved_model_config`.
"""

from pathlib import Path

from evaluator.webapi.form_config import graph_spec_to_config_dict
from tests.fixtures_paths import E2E_MINI, MINI_C, MINI_Q
from tests.graph_test_helpers import explicit_graph

_CONFIG_YAML = Path(E2E_MINI).read_text()
_Q = MINI_Q
_C = MINI_C


def _canvas_to_run_spec(canvas):
    """The builder's exportSpec shape from a from-config canvas seed: PORT-LEVEL edges from
    the rendered bindings (artifact + producer), the input key resolved via the node's ports —
    exactly the mapping the browser's exportSpec performs."""
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


def test_stage_typed_graph_nodes_keep_their_models():
    """A palette-style stage-typed export (`convert`/`embed`) carries its models through
    translation — kind-typed (`asr`) nodes are not the only ones whose model survives."""
    spec = {"mode": "asr_text_retrieval", "nodes": [
        {"id": "dataset_source", "type": "source", "params": {
            "dataset": "pubmed_qa",
            "questions": _Q,
            "corpus": _C}},
        {"id": "asr", "type": "convert",
         "params": {"model": "whisper", "name": "openai/whisper-base"}},
        {"id": "text_embedding", "type": "embed", "params": {"model": "labse"}},
    ], "edges": []}
    legacy = graph_spec_to_config_dict(spec)
    by_id = {n["id"]: n for n in legacy["graph_override"]["nodes"] if isinstance(n, dict)}
    assert by_id["asr"]["params"]["model"] == "whisper"
    assert by_id["asr"]["params"]["name"] == "openai/whisper-base"
    assert by_id["text_embedding"]["params"]["model"] == "labse"
    assert "dataset_source" in (legacy.get("data", {}).get("datasets") or {})


def test_two_same_role_models_no_clobber():
    """Node-centric config: two graph nodes of the same model role do not clobber each other —
    EACH keeps its model ON the node (a per-node override the executor reads via
    ``resolved_model_config``) and neither is promoted into the flat ``model.*`` default. So one
    graph can run two distinct text embedders (asymmetric encoders / an ablation)."""
    from evaluator.config.graph_config import build_evaluation_config_kwargs

    legacy = build_evaluation_config_kwargs({"graph": explicit_graph([
        "dataset_source",  # gives the embedders a query_text producer (edges derivable)
        {"id": "q_emb", "type": "text_embedding", "params": {"model": "labse"}},
        {"id": "q_emb2", "type": "text_embedding", "params": {"model": "jina"}},
    ])})  # no GraphConfigError
    assert not (legacy.get("model") or {}).get("text_emb_model_type")
    by_id = {n["id"]: n for n in legacy["graph_override"]["nodes"] if isinstance(n, dict)}
    assert by_id["q_emb"]["params"]["model"] == "labse"        # first → stays on its own node
    assert by_id["q_emb2"]["params"]["model"] == "jina"        # second → per-node override on-node


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


def test_config_round_trips_through_canvas(client):
    """YAML → canvas → run spec → config reproduces the models + dataset."""
    canvas = client.post("/api/graph/from-config", json={"yaml": _CONFIG_YAML}).json()
    legacy = graph_spec_to_config_dict(_canvas_to_run_spec(canvas), experiment_name="rt")
    by_id = {n["id"]: n for n in legacy["graph_override"]["nodes"] if isinstance(n, dict)}
    assert by_id["asr"]["params"]["model"] == "whisper"
    assert by_id["text_embedding"]["params"]["model"] == "labse"
    datasets = legacy.get("data", {}).get("datasets") or {}
    assert datasets and datasets["dataset_source"]["dataset_name"] == "pubmed_qa"


def test_builder_page_has_load_affordance(client):
    t = client.get("/ui/builder").text
    assert 'id="btn-import-config"' in t
    assert "/api/graph/from-config" in t


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
    # the fixture (E2E_MINI) sets trace_limit: 4 and batch_size: 2 — both must survive the
    # round-trip
    y = _export(client, _CONFIG_YAML)
    assert "trace_limit: 4" in y
    assert "batch_size: 2" in y


def test_export_keeps_edges_without_accumulating(client):
    # the canvas connections ARE the edges — the export keeps every one, and a second
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


def test_builder_validate_flags_mismatch_as_a_problem(client):
    """The builder's Validate renders the shared ``_validation.html`` fragment (same module as
    Config & Run). It runs the whole run pipeline, so the embedding-space mismatch — the same
    check that blocks Run with a 400 — surfaces as a problem to fix, not as advice."""
    html = client.post("/ui/validate-builder", json=_mismatch_spec()).text
    assert "Embedding-space mismatch" in html
    assert 'class="error-box"' in html


def test_feature_node_params_rehydrate_onto_canvas_and_round_trip():
    """The loader folds feature-node params into sub-configs; the canvas seed
    must rehydrate them onto the owning node (not show form defaults), and the canvas→config
    path must deliver them back — else a config→builder→config trip silently resets the
    feature (judge ran with gpt-4o-mini instead of the configured local model)."""
    import yaml

    from evaluator.config.evaluation import EvaluationConfig
    from evaluator.config.graph_config import build_evaluation_config_kwargs
    from evaluator.webapi.form_builder import config_to_canvas_spec
    from evaluator.webapi.form_config import graph_spec_to_config_dict

    raw = yaml.safe_load(Path("configs/pubmed_qa_rag_fulltext.yaml").read_text())
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
    # judge is the only feature node this graph draws; no query-optimization node means
    # query_optimization isn't a rehydrated feature here.

