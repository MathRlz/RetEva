"""Phase 0.5 drift-guard: the /api/introspection/schema endpoint is the single contract the
web UI builds itself from. These tests assert every list in the schema *equals* its backing
registry/enum, so a model / metric / strategy / fusion registered in the core auto-appears in
the UI and nothing can silently drift.
"""

import pytest
from tests.graph_test_helpers import explicit_graph


@pytest.fixture(scope="module")
def schema(client):
    r = client.get("/api/introspection/schema")
    assert r.status_code == 200
    return r.json()


def test_schema_has_all_keys(schema):
    expected = {
        "graph_templates", "model_families", "datasets", "metrics", "headline_metrics",
        "artifacts", "vector_stores", "fusion_methods", "combine_strategies", "correctors",
        "retrieval_modes", "reranker_modes", "dataset_sources", "startup_modes",
        "offload_policies", "distance_metrics", "node_catalogue",
    }
    assert expected <= set(schema)


def test_schema_lists_equal_registries(schema):
    """Each schema list is its registry/enum verbatim — the anti-drift invariant."""
    from evaluator.config.types import (
        RETRIEVAL_MODES, RERANKER_MODES, SERVICE_STARTUP_MODES,
        SERVICE_OFFLOAD_POLICIES, DATASET_SOURCES,
    )
    from evaluator.evaluation.metric_registry import list_metrics
    from evaluator.evaluation.query_correction import list_correctors
    from evaluator.evaluation.results import HEADLINE_METRICS
    from evaluator.models.retrieval import list_fusions
    from evaluator.models.retrieval.query.optimization import list_combine_strategies
    from evaluator.models.retrieval.rag.strategies import DistanceMetric
    from evaluator.pipeline.artifacts import list_artifacts
    from evaluator.pipeline.graph.templates import list_graph_templates
    from evaluator.storage.registry import list_vector_stores

    assert schema["fusion_methods"] == list_fusions()
    assert schema["combine_strategies"] == list_combine_strategies()
    assert schema["correctors"] == list_correctors()
    assert schema["vector_stores"] == list_vector_stores()
    assert schema["retrieval_modes"] == list(RETRIEVAL_MODES)
    assert schema["reranker_modes"] == list(RERANKER_MODES)
    assert schema["dataset_sources"] == list(DATASET_SOURCES)
    assert schema["startup_modes"] == list(SERVICE_STARTUP_MODES)
    assert schema["offload_policies"] == list(SERVICE_OFFLOAD_POLICIES)
    assert schema["distance_metrics"] == [d.value for d in DistanceMetric]
    assert schema["headline_metrics"] == list(HEADLINE_METRICS)
    assert schema["graph_templates"] == list_graph_templates()
    assert [m["name"] for m in schema["metrics"]] == [m.name for m in list_metrics()]
    assert [a["name"] for a in schema["artifacts"]] == [a.name for a in list_artifacts()]


def test_model_families_present(schema):
    fams = schema["model_families"]
    assert set(fams) == {"asr", "text_embedding", "audio_embedding", "reranker", "tts"}
    # Each entry carries the registry-declared sizes + metadata shape.
    for entry in fams["asr"]:
        assert "type" in entry and "sizes" in entry


def test_new_fusion_auto_appears(client):
    """Registering a throwaway fusion makes it show up in the schema with no UI edit."""
    from evaluator.models.retrieval.fusion_registry import (
        register_fusion, FUSION_REGISTRY,
    )

    name = "_drift_probe_fusion"
    register_fusion(name, object())
    try:
        r = client.get("/api/introspection/schema")
        assert name in r.json()["fusion_methods"]
    finally:
        FUSION_REGISTRY.pop(name, None)


def test_judge_node_and_metrics_surface(schema):
    """J5: the LLM-judge node + its metrics/artifacts auto-surface in the registry-driven
    schema with no UI hardcoding — the builder offers the node and the leaderboard discovers
    the judge_* metrics by virtue of registration alone."""
    metric_names = {m["name"] for m in schema["metrics"]}
    assert {"judge_overall", "judge_pass_rate", "judge_relevance",
            "judge_faithfulness"} <= metric_names
    # Every judge metric is reference-free (single input, no ground truth).
    judge_metrics = [m for m in schema["metrics"] if m["name"].startswith("judge")]
    assert judge_metrics and all(m["gt"] is None for m in judge_metrics)
    # The per-query judge score artifacts are registered + advertised.
    artifact_names = {a["name"] for a in schema["artifacts"]}
    assert {"judge_scores", "judge_pass", "judge_aspect_relevance"} <= artifact_names
    # The judge node is offered in the builder palette (the measure{family:judge} preset).
    presets = schema["node_catalogue"].get("presets", [])
    assert any(p.get("params", {}).get("family") == "judge" for p in presets), \
        "LLM judge node missing from the builder palette"


@pytest.mark.parametrize("path", ["/ui/config", "/ui/builder"])
def test_ui_pages_render(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert "<html" in r.text.lower()


def test_schema_carries_field_help(schema):
    from evaluator.webapi.field_help import FIELD_HELP

    assert schema["field_help"] == dict(FIELD_HELP)
    assert "retrieval_mode" in schema["field_help"]


def test_dataset_source_outputs_reflect_chosen_dataset():
    """P1.1: a dataset_source advertises the CHOSEN dataset's real fields (its output ports + the GT
    fields the user can wire), not the union of what every dataset can output."""
    from evaluator.webapi.form_builder import resolve_node_form

    # admed_voice publishes audio + its transcription GT — in audio modes query_text is the asr
    # node's hypothesis, so the dataset's transcription is reference_transcription, not the query.
    av = set(resolve_node_form("source", {"dataset": "admed_voice"})["outputs"])
    assert av == {"query_audio", "reference_transcription"}
    # pubmed_qa adds corpus + relevance + answers, but NOT reference_transcription (not its field).
    pq = set(resolve_node_form("source", {"dataset": "pubmed_qa"})["outputs"])
    assert {"corpus", "relevant_docs", "short_answers"} <= pq
    assert "reference_transcription" not in pq
    # No dataset chosen → the full union (unchanged) so a fresh node still shows every port.
    assert "corpus" in resolve_node_form("source", {})["outputs"]


def test_ground_truth_is_a_real_input_port():
    """Ground truth is a real (optional) input PORT (wireable from the source), not a decoration:
    retrieval's form exposes `relevant_docs` IN its input_ports (no separate `ground_truth` key)."""
    from evaluator.webapi.form_builder import resolve_node_form

    form = resolve_node_form("retrieval", None)
    port_names = {n for p in form["input_ports"] for n in p["names"]}
    assert "relevant_docs" in port_names          # GT is a real input port
    assert "ground_truth" not in form             # no GT-annotation channel


def test_structural_artifacts_match_js_filter():
    """The simplified DAG (dag_view.js) hides plumbing-bundle input ports whose producer is
    always-hidden reporting plumbing — the judge's `metrics` + `query_traces` — keyed on the same
    STRUCTURAL_ARTIFACTS the pipeline declares. Data-spine intermediates (corpus_vectors) are NOT
    here. Guard the two against drift (the JS inlines this small set)."""
    import pathlib

    import evaluator.webapi as w
    from evaluator.pipeline.introspection import STRUCTURAL_ARTIFACTS

    assert STRUCTURAL_ARTIFACTS == {"metrics", "query_traces"}
    js = (pathlib.Path(w.__file__).parent / "static" / "dag_view.js").read_text()
    for art in STRUCTURAL_ARTIFACTS:
        assert f"'{art}'" in js, f"{art} missing from dag_view STRUCTURAL_ARTIFACTS"


def test_render_node_marks_structural_for_preview():
    """P2.3: the read-only preview payload marks plumbing `structural`, so the config-page graph
    hides it by default (with a power-user 'view full DAG' toggle)."""
    from types import SimpleNamespace

    from evaluator.webapi.form_builder import render_node

    def mk(stage):
        return SimpleNamespace(id=stage, stage=stage, params={}, bindings=[])

    assert render_node(mk("metrics"), {})["structural"] is True
    assert render_node(mk("finalize"), {})["structural"] is True
    assert render_node(mk("asr"), {})["structural"] is False


def test_config_preview_payload_marks_structural():
    """The Config & Run preview renders via `graph_preview`/`_preview_node` (NOT render_node), then
    the `_graph_diagram_context` projection embeds it as `preview_json`. `structural` must survive
    BOTH steps — else `_graph.html`'s default-simplified filter + 'View full DAG' toggle no-op
    (Report/Finalize leak into the simple view). Guards the full rendering chain."""
    import json

    from evaluator import EvaluationConfig
    from evaluator.webapi.form_builder import graph_preview
    from evaluator.webapi.form_config import graph_spec_to_config_dict
    from evaluator.webapi.ui.config import _graph_diagram_context

    _Q = "examples/data/pubmed_qa_small/questions.json"
    _C = "examples/data/pubmed_qa_small/corpus.json"
    spec = {"mode": "asr_text_retrieval", "nodes": [
        {"id": "dataset_source", "type": "dataset_source",
         "params": {"dataset": "pubmed_qa", "questions": _Q, "corpus": _C}},
        {"id": "asr", "type": "asr", "params": {"model": "whisper"}},
        {"id": "text_embedding", "type": "text_embedding", "params": {"model": "labse"}},
        {"id": "corpus_embedding", "type": "corpus_embedding", "params": {}},
        {"id": "vector_db", "type": "vector_db", "params": {}},
        {"id": "retrieval", "type": "retrieval", "params": {}},
    ], "edges": []}
    cfg = EvaluationConfig.from_dict(graph_spec_to_config_dict(spec)).with_auto_devices()
    # step 1: graph_preview / _preview_node mark structural
    assert all("structural" in n for n in graph_preview(cfg)["nodes"])
    # step 2: the preview_json projection that actually feeds _graph.html keeps it — and the
    # report (`metrics`) + finalize plumbing is hidden by default, the meaningful ops shown.
    preview_json = _graph_diagram_context(graph_preview(cfg))["preview_json"]
    data = json.loads(preview_json.replace("<\\/", "</"))
    hidden = {n["id"] for n in data["nodes"] if n["structural"]}
    assert {"metrics", "finalize"} <= hidden
    assert any(not n["structural"] for n in data["nodes"])
    assert all(n["structural"] for n in data["nodes"] if n["stage"] in ("measure", "sink"))


def test_refine_input_port_count_varies_by_op(client):
    """The builder reconciles a node's Drawflow port COUNT when a discriminator change alters the
    spec (reconcilePorts). That only matters because the counts genuinely vary by op — rerank scores
    with query inputs, threshold has none. Guards the precondition for the reconcile fix."""
    def n_in(op):
        r = client.post("/api/graph/node-form", json={"type": "refine", "params": {"op": op}})
        return len(r.json()["input_ports"])

    assert n_in("rerank") > n_in("mmr") > n_in("threshold")


def test_open_in_builder_source_resolves_real_dataset():
    """Opening a config in the builder must seed the source with its REAL dataset name (+ fields),
    not the datasets-map role key — else the dataset picker falls to the blank '(from run config)'.
    graph_spec_to_config_dict emits a datasets map keyed by node id; the preview/seed must resolve
    the name back."""
    from evaluator import EvaluationConfig
    from evaluator.webapi.form_builder import graph_preview
    from evaluator.webapi.form_config import graph_spec_to_config_dict

    _Q = "examples/data/pubmed_qa_small/questions.json"
    _C = "examples/data/pubmed_qa_small/corpus.json"
    spec = {"mode": "asr_text_retrieval", "nodes": [
        {"id": "dataset_source", "type": "dataset_source",
         "params": {"dataset": "pubmed_qa", "questions": _Q, "corpus": _C}},
        {"id": "asr", "type": "asr", "params": {"model": "whisper"}},
        {"id": "text_embedding", "type": "text_embedding", "params": {"model": "labse"}},
        {"id": "corpus_embedding", "type": "corpus_embedding", "params": {}},
        {"id": "vector_db", "type": "vector_db", "params": {}},
        {"id": "retrieval", "type": "retrieval", "params": {}},
    ], "edges": []}
    cfg = EvaluationConfig.from_dict(graph_spec_to_config_dict(spec)).with_auto_devices()
    src = [n for n in graph_preview(cfg)["nodes"] if n["stage"] == "source"][0]
    assert src["inspect"].get("dataset") == "pubmed_qa"   # the name, not 'dataset_source'
    assert src["inspect"].get("fields")                   # column→artifact map carried for the seed


def test_palette_excludes_structural_and_groups_by_section():
    """P2.2: the builder palette offers only meaningful operations, grouped into the user-facing
    sections; the structural plumbing (measure/report + sink/finalize/aggregate/persistence) is
    filtered out, and tts + the LLM judge live under Models."""
    from evaluator.webapi.form_builder import PALETTE_SECTIONS, node_catalogue

    cat = node_catalogue()
    entries = cat["nodes"] + cat["presets"]
    meaningful = [n for n in entries if not n.get("structural")]
    # every meaningful entry lands in a known user section (nothing meaningful is ungrouped)
    assert {n.get("section") for n in meaningful} <= set(PALETTE_SECTIONS)
    # the measure + sink operators (all the report/finalize/persistence plumbing) are structural
    by_type = {n["type"]: n for n in cat["nodes"]}
    assert by_type["measure"]["structural"] and by_type["sink"]["structural"]
    # tts + the judge are user-facing Models, not plumbing
    models = {n.get("palette_id") or n["type"] for n in meaningful if n.get("section") == "Models"}
    assert {"tts", "answer_judge"} <= models


def test_builder_has_no_freeform_param_wildcard():
    """P1.3: the builder no longer offers a free-form '+ add param' row — every settable param is
    auto-surfaced (param_spec switches + the model section + the model's Params schema). The
    `setExtraParam` channel stays (renderSchemaFields uses it for the model Params)."""
    import pathlib

    import evaluator.webapi as w

    html = (pathlib.Path(w.__file__).parent / "templates" / "builder.html").read_text()
    assert "param name" not in html          # the wildcard key-input placeholder is gone
    assert "+ add'" not in html.replace("+ add branch", "")  # the wildcard button is gone
    assert "function setExtraParam" in html  # still used by renderSchemaFields (model Params)


def test_rerank_surfaces_node_settable_tuning():
    """P1.2: the reranker tuning the handler reads (`_node_reranking`: model/mode/top_k/weight/
    device) is all surfaced as builder fields — so the wildcard isn't needed to set it. mmr's
    lambda + threshold's cutoff are global config (not node params), so they're intentionally
    absent."""
    from evaluator.webapi.form_builder import resolve_node_form

    specs = {p["key"]: p for p in resolve_node_form("refine", {"op": "rerank"})["node_params"]}
    assert {"model", "mode", "top_k", "weight", "device"} <= set(specs)
    assert specs["weight"].get("show_if") == {"op": ["rerank"]}


@pytest.mark.parametrize("node_type,params,family", [
    # convert: asr uses an ASR model; tts is model-free (configured via audio_synthesis).
    ("convert", {"op": "asr"}, "asr"),
    ("convert", {"op": "tts"}, None),
    # refine: only rerank uses a reranker model; mmr/threshold are model-free refinements.
    ("refine", {"op": "rerank"}, "reranker"),
    ("refine", {"op": "mmr"}, None),
    ("refine", {"op": "threshold"}, None),
    ("refine", {}, "reranker"),  # bare refine defaults to rerank
    # embed: the model family follows the modality (query side).
    ("embed", {"axis": "query", "modality": "text"}, "text_embedding"),
    ("embed", {"axis": "query", "modality": "audio"}, "audio_embedding"),
])
def test_node_form_model_family_is_discriminator_aware(node_type, params, family):
    """The builder's model picker follows a node's discriminator fields, never the operator's
    bare default — so a model-free variant (convert{op:tts}, refine{op:mmr}) shows no model
    picker instead of leaking the default family (the asr / reranker pickers)."""
    from evaluator.webapi.form_builder import resolve_node_form

    assert resolve_node_form(node_type, params).get("family") == family


def test_text_embedding_tile_hides_axis_and_modality():
    """The bare 'embed' palette tile IS text_embedding by default (node_kind("embed", {})
    resolves to it) — its form must not expose axis/modality as editable, or a "Text
    embedding" node could be turned into an audio/corpus embedder via a field that looks
    like generic config."""
    from evaluator.webapi.form_builder import node_catalogue

    embed = next(n for n in node_catalogue()["nodes"] if n["type"] == "embed")
    keys = {p["key"] for p in embed["node_params"]}
    assert "axis" not in keys and "modality" not in keys
    assert embed["family"] == "text_embedding"


def test_builder_supports_node_copy_paste_and_selection_highlight():
    """The builder tracks the selected node and offers Ctrl+C/Ctrl+V to clone a node plus
    its incoming edges; the selected-node CSS highlight uses outline (not border-color,
    which the category classes set !important) so it's visible regardless of node type."""
    import pathlib

    import evaluator.webapi as w

    html = (pathlib.Path(w.__file__).parent / "templates" / "builder.html").read_text()
    assert "nodeUnselected" in html
    assert "function copySelectedNode" in html
    assert "function pasteClipboardNode" in html
    assert "addEventListener('keydown'" in html

    css = (pathlib.Path(w.__file__).parent / "static" / "app.css").read_text()
    assert "outline: 3px solid var(--accent)" in css


def test_node_param_switches_carry_help(schema):
    """Phase 3 polish: builder node-param switches get a tooltip from the FIELD_HELP glossary
    (the same source the config form uses), so the param panel explains non-obvious fields."""
    nodes = schema["node_catalogue"]["nodes"] + schema["node_catalogue"]["presets"]
    by_key = {p["key"]: p for n in nodes for p in n.get("node_params", [])}
    # 'store' (vector_db) + 'family' (measure) are FIELD_HELP-covered switches → help present.
    assert by_key["store"].get("help")
    assert by_key["family"].get("help")


def test_model_params_carry_help(client):
    """Model Params fields surface a tooltip: a model-author DESCRIPTIONS entry wins, else the
    shared glossary fills it in at the endpoint."""
    # attention_pool declares DESCRIPTIONS → its pooling help is the author's richer text.
    ap = client.get("/api/models/audio_embedding/attention_pool/params").json()
    assert "mean_abtt" in ap["params_schema"]["pooling"]["help"]   # author-declared
    # faster_whisper has no DESCRIPTIONS → the endpoint layers FIELD_HELP["compute_type"].
    fw = client.get("/api/models/asr/faster_whisper/params").json()
    assert "int8" in fw["params_schema"]["compute_type"]["help"]   # glossary fallback


def test_get_params_schema_honors_descriptions():
    """Registry unit: a Params.DESCRIPTIONS ClassVar populates each field's `help` (core-side,
    no webapi dependency); DESCRIPTIONS itself is not emitted as a field."""
    from evaluator.models.registry import FAMILY_REGISTRIES

    reg = FAMILY_REGISTRIES["audio_embedding"]
    sch = reg.get_params_schema("attention_pool")
    assert "DESCRIPTIONS" not in sch
    assert sch["emb_dim"].get("help") and sch["pooling"].get("help")


def test_node_form_names_the_default_model():
    """The form's empty model option must NAME the inherited model (no bare '(default)'):
    resolve_node_form carries the dataclass default; the Config & Run embed resolves the
    LOADED config's flat default instead."""
    import json

    from evaluator.webapi.form_builder import default_model_for, resolve_node_form

    assert resolve_node_form("asr")["default_model"] == "wav2vec2"      # ModelConfig default
    assert resolve_node_form("text_embedding")["default_model"] == "labse"
    assert resolve_node_form("tts")["default_model"] is None            # model-free node

    # Config & Run: the loaded config's model wins over the dataclass default.
    from evaluator.config.evaluation import EvaluationConfig
    from evaluator.config.graph_config import build_evaluation_config_kwargs
    from evaluator.webapi.form_builder import config_to_canvas_spec
    from evaluator.webapi.ui.config import _edit_nodes_json

    cfg = EvaluationConfig.from_dict(build_evaluation_config_kwargs({
        "dataset": {"id": "pubmed_qa",
                    "questions": "examples/data/pubmed_qa_small/questions.json",
                    "corpus": "examples/data/pubmed_qa_small/corpus.json"},
        "graph": explicit_graph(["dataset_source", "corpus_embedding", "vector_db", "tts",
                                 "asr", "text_embedding", "retrieval", "metrics", "finalize"]),
        "nodes": {"asr": {"model": "whisper"}, "text_embedding": {"model": "jina_v4"}},
    }), validate=False)
    payload = json.loads(_edit_nodes_json("t", config_to_canvas_spec(cfg), cfg))
    by_id = {n["id"]: n for n in payload["nodes"]}
    assert by_id["asr"]["default_model"] == "whisper"
    assert by_id["text_embedding"]["default_model"] == "jina_v4"
    assert default_model_for("vector_db.reranker_model")                # rerank picker resolves


def test_edit_nodes_json_carries_input_ports_for_config_and_run_buildspec():
    """Regression: Config & Run's client-side ``buildSpec()`` reconstructs each node's edges
    from its ``bindings`` ([artifact, producer] pairs) — for a OneOf-collapsed port (e.g.
    ``query_vectors`` accepting audio/text/fused query vectors) the bound artifact name
    differs from the port's own label, so ``buildSpec()`` needs ``input_ports`` to map one to
    the other. Missing this field made a REAL run through the webapi fail with exactly:
    "node 'retrieval' required input 'query_vectors' has no edge, but the graph produces
    ['audio_query_vectors']" — reproduced live via a running server before this fix."""
    import json

    from evaluator.config.evaluation import EvaluationConfig
    from evaluator.webapi.form_builder import config_to_canvas_spec
    from evaluator.webapi.ui.config import _edit_nodes_json

    cfg = EvaluationConfig()
    cfg.graph_override = {"template": "audio_emb_retrieval"}
    cfg.model.audio_emb_model_type = "attention_pool"
    payload = json.loads(_edit_nodes_json("t", config_to_canvas_spec(cfg), cfg))
    retrieval = next(n for n in payload["nodes"] if n["id"] == "retrieval")
    query_vectors_port = next(
        p for p in retrieval["input_ports"] if p["label"] == "query_vectors"
    )
    assert "audio_query_vectors" in query_vectors_port["names"]
    # the OneOf bound artifact name (from `bindings`) differs from the port label it maps to
    assert ("audio_query_vectors", "audio_embedding") in [tuple(b) for b in retrieval["bindings"]]


def test_graph_preview_overlays_routed_aliases_same_as_render_node():
    """`_preview_node` (feeds `graph_preview` -> `_edit_nodes_json` -> Config & Run's real
    `/ui/config-preview` route) must apply the SAME `_overlay_routed_aliases` step `render_node`
    (feeds `config_to_canvas_spec`) already does -- it didn't, so a B2 *routed* artifact (a
    type-open port fed something the node def never statically lists, e.g.
    `dataset_source.reference_text` -> an embedder's `query_text` port -- the real shape of
    `configs/pubmed_qa_rag_fulltext.yaml`, a pure-text-retrieval config) was invisible to
    `graph_preview`'s `input_ports`, so Config & Run's buildSpec() couldn't map the binding back
    to the real port and emitted an edge naming the wrong input -- caught live via Playwright
    against the running dev server (Validate on this exact preset), not by the ORIGINAL OneOf
    regression test below, which happens to construct its payload via `config_to_canvas_spec`
    (uses `render_node`) rather than the real production path (`graph_preview`, uses
    `_preview_node`) and so never exercised this divergence."""
    from evaluator import EvaluationConfig, get_preset
    from evaluator.webapi.form_builder import config_to_canvas_spec, graph_preview

    config = EvaluationConfig.from_dict(
        get_preset("pubmed_qa_rag_fulltext"), validate=False
    ).with_auto_devices()

    def names_for(payload):
        node = next(n for n in payload["nodes"] if n["id"] == "text_embedding")
        port = next(p for p in node["input_ports"] if p["label"] == "query_text")
        return set(port["names"])

    canvas_names = names_for(config_to_canvas_spec(config))
    preview_names = names_for(graph_preview(config))
    assert "reference_text" in canvas_names  # sanity: the routed alias really exists
    assert preview_names == canvas_names, (
        f"graph_preview names {preview_names} diverged from config_to_canvas_spec names "
        f"{canvas_names} -- _preview_node is missing an alias overlay render_node has."
    )


def test_config_and_run_buildspec_reconstructs_a_oneof_port_edge_correctly():
    """End-to-end: the FIXED `buildSpec()` logic (translated 1:1 from `_config_run.html`),
    run against the real `_edit_nodes_json` payload, must produce a graph that builds — not
    the pre-fix `{input: <bound artifact name>}` edge (no `output`), which named a port
    ("audio_query_vectors") the node doesn't have and broke every audio-embedding →
    retrieval wiring submitted through Config & Run."""
    import json

    from evaluator.config.evaluation import EvaluationConfig
    from evaluator.webapi.form_builder import config_to_canvas_spec
    from evaluator.webapi.form_config import graph_spec_to_config_dict, prepare_run_config
    from evaluator.webapi.ui.config import _edit_nodes_json
    from evaluator.pipeline import build_graph_for_config

    cfg = EvaluationConfig()
    cfg.graph_override = {"template": "audio_emb_retrieval"}
    cfg.model.audio_emb_model_type = "attention_pool"
    payload = json.loads(_edit_nodes_json("t", config_to_canvas_spec(cfg), cfg))

    def build_spec_js_equivalent(data):
        nodes = [{"id": n["id"], "type": n["type"], "params": n["params"]} for n in data["nodes"]]
        edges = []
        for n in data["nodes"]:
            ports = n.get("input_ports") or []
            for art, prod in n.get("bindings") or []:
                port = next((p for p in ports if art in (p.get("names") or [])), None)
                edges.append({
                    "from": prod, "output": art, "to": n["id"],
                    "input": port["label"] if port else art,
                })
        return {"mode": data["mode"], "nodes": nodes, "edges": edges}

    spec = build_spec_js_equivalent(payload)
    retrieval_edges = [e for e in spec["edges"] if e["to"] == "retrieval"]
    assert {"from": "audio_embedding", "output": "audio_query_vectors",
            "to": "retrieval", "input": "query_vectors"} in retrieval_edges

    # Wiring only (skips the unrelated embedding-space-alignment check
    # build_validated_run_config also runs) — this pins the query_vectors edge fix itself.
    legacy = graph_spec_to_config_dict(spec, experiment_name="t")
    config = prepare_run_config(legacy, auto_devices=False)
    graph = build_graph_for_config(config)
    retrieval_node = next(n for n in graph.nodes if n.id == "retrieval")
    assert ("audio_query_vectors", "audio_embedding") in retrieval_node.bindings
