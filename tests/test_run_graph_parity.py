"""With explicit port-level edges the two graph builders produce ONE graph (E4/E6).

`build_graph_for_config` (config/preview path) additionally attaches the dataset column schema
to `dataset_source` nodes as DISPLAY params; `build_run_graph` does not. Bindings, aliases and
deps are identical — the config's edges are the single wiring source. This pins the
unification (the pre-edges builders diverged on run-only fallback bindings).
"""

from types import SimpleNamespace

from evaluator.config.evaluation import EvaluationConfig
from evaluator.config.graph_config import build_evaluation_config_kwargs
from evaluator.pipeline.graph.modes import build_graph_for_config, build_run_graph
from tests.graph_test_helpers import mode_graph

_DATASET = {
    "id": "pubmed_qa",
    "questions": "examples/data/pubmed_qa_small/questions.json",
    "corpus": "examples/data/pubmed_qa_small/corpus.json",
}


def _cfg():
    return EvaluationConfig.from_dict(
        build_evaluation_config_kwargs({"dataset": _DATASET, "graph": mode_graph("asr_text_retrieval")}),
        validate=False,
    )


def _stub_retrieval_pipeline():
    # No rerank / mmr / threshold — so the run graph matches the plain config graph structurally.
    return SimpleNamespace(
        reranker=None,
        strategy_config=SimpleNamespace(
            reranking=SimpleNamespace(mode="none"),
            post_processing=SimpleNamespace(use_mmr=False, min_similarity_threshold=None),
        ),
    )


def _run_graph(cfg):
    return build_run_graph(
        "asr_text_retrieval",
        graph_override=cfg.graph_override,
        embedding_fusion_config=None,
        query_opt_config=None,
        retrieval_pipeline=_stub_retrieval_pipeline(),
        eval_config=cfg,
        trace_limit=0,
    )


def _bindings(graph, node_id):
    node = next(n for n in graph.nodes if n.id == node_id)
    return set(node.bindings)


def test_builders_agree_on_node_set():
    cfg = _cfg()
    cfg_ids = [n.id for n in build_graph_for_config(cfg).nodes]
    run_ids = [n.id for n in _run_graph(cfg).nodes]
    assert cfg_ids == run_ids


def test_builders_agree_on_bindings_and_gt_flows():
    cfg = _cfg()
    run = _run_graph(cfg)
    cfg_graph = build_graph_for_config(cfg)

    # One wiring source (the edges): the run's reference_transcription GT flow exists in BOTH
    # builders now — the preview no longer hides it (it renders faded instead).
    assert ("reference_transcription", "dataset_source") in _bindings(run, "metrics")
    assert ("reference_transcription", "dataset_source") in _bindings(cfg_graph, "metrics")
    for a, b in zip(cfg_graph.nodes, run.nodes):
        assert a.bindings == b.bindings, a.id
        assert a.input_aliases == b.input_aliases, a.id
        assert a.depends_on == b.depends_on, a.id


def test_run_graph_dataset_source_has_no_fields_param():
    run = _run_graph(_cfg())
    ds = next(n for n in run.nodes if n.id == "dataset_source")
    assert "fields" not in (ds.params or {})   # run wires from static outputs, not columns
