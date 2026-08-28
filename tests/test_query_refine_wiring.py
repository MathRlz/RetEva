"""Two-hop refine: the rewrite must reach hop 2 and nothing else.

The failure modes are silent, not loud: hop-1 vectors leaking into hop 2 (the rewrite does
nothing), the rewritten query reaching the generator (generation quality confounded with rewrite
quality), or a missing query-optimization config making every refine call fail into a pass-through.
"""

import logging

import pytest
import yaml

from evaluator.config.evaluation import EvaluationConfig
from evaluator.config.graph_config import build_evaluation_config_kwargs
from evaluator.evaluation.handlers.query import _stage_query_refine
from evaluator.evaluation.item_set import ItemSet
from evaluator.pipeline import build_graph_for_config

from tests.graph_test_helpers import make_state

_CONFIG = "configs/pubmed_qa_rag_fulltext_refine.yaml"


def _graph():
    raw = yaml.safe_load(open(_CONFIG))
    cfg = EvaluationConfig.from_dict(build_evaluation_config_kwargs(raw), validate=False)
    return {n.id: n for n in build_graph_for_config(cfg).nodes}, cfg


def test_refiner_sees_the_question_and_the_first_hop_docs():
    nodes, _ = _graph()
    bound = dict((art, prod) for art, prod in nodes["query_refine"].bindings)
    assert bound["retrieved"] == "retrieval"          # hop 1, not hop 2 (that would be a cycle)
    assert "reference_text" in bound                  # the question as asked


def test_second_hop_embeds_only_the_rewrite():
    nodes, _ = _graph()
    arts = [art for art, _ in nodes["text_embedding_refined"].bindings]
    assert arts == ["refined_query_text"], arts       # hop-1 vectors must not leak in
    vecs = [(art, prod) for art, prod in nodes["retrieval_refined"].bindings
            if art.endswith("query_vectors")]
    assert vecs == [("text_query_vectors", "text_embedding_refined")]


def test_downstream_reads_hop_two_but_the_prompt_reads_the_original_question():
    nodes, _ = _graph()
    for consumer in ("answer_gen", "retrieval_metrics", "answer_metrics",
                     "build_query_traces", "answer_judge", "finalize"):
        producers = {prod for art, prod in nodes[consumer].bindings if art == "retrieved"}
        assert producers == {"retrieval_refined"}, (consumer, producers)
    # the generator prompts with the question, never the rewrite
    gen = dict((art, prod) for art, prod in nodes["answer_gen"].bindings)
    assert "reference_text" in gen and "refined_query_text" not in gen


def test_query_optimization_is_enabled_in_the_config():
    """Without it the run carries no config and every refine silently passes through."""
    _, cfg = _graph()
    assert cfg.query_optimization.enabled is True


@pytest.fixture
def _capture(caplog):
    """caplog listens on the ROOT logger, but the repo's logging setup sets propagate=False on
    the `evaluator` logger once configured (logging_config.py:136) — so records never reach it
    and the assertions pass or fail depending on test order. Attach the capture handler to the
    emitting logger directly."""
    logger = logging.getLogger("evaluator.evaluation.handlers.query")
    logger.addHandler(caplog.handler)
    previous = logger.level
    logger.setLevel(logging.INFO)
    yield caplog
    logger.removeHandler(caplog.handler)
    logger.setLevel(previous)


def _refine_state(**kw):
    s = make_state(**kw)
    s.current_node = type("N", (), {"id": "query_refine", "params": {}, "stage": "transform"})()
    s.ctx.put("dataset_source", "query_text", ItemSet(["q1", "q2"], ["first?", "second?"]))
    s.ctx.put("retrieval", "retrieved", [[({"doc_id": "d1", "text": "ctx one"}, 1.0)],
                                         [({"doc_id": "d2", "text": "ctx two"}, 1.0)]])
    s.current_node.bindings = (("query_text", "dataset_source"), ("retrieved", "retrieval"))
    return s


def test_publishes_refined_text_keyed_and_leaves_the_original_alone(monkeypatch):
    monkeypatch.setattr("evaluator.models.retrieval.query.optimization.refine_query",
                        lambda q, docs, cfg, method="", context_top_k=3: f"{q} [{docs[0]}]")
    s = _refine_state()
    _stage_query_refine(s)
    items = s.ctx.get("query_refine", "refined_query_text")
    assert list(items.ids) == ["q1", "q2"]
    assert list(items.values) == ["first? [ctx one]", "second? [ctx two]"]
    # the original query_text artifact is untouched — the chain resolves by priority, not mutation
    assert list(s.ctx.get("dataset_source", "query_text").values) == ["first?", "second?"]


def test_missing_config_warns_instead_of_silently_degrading(monkeypatch, _capture):
    monkeypatch.setattr("evaluator.models.retrieval.query.optimization.refine_query",
                        lambda q, docs, cfg, method="", context_top_k=3: q)
    s = _refine_state(query_opt_config=None)
    _stage_query_refine(s)
    assert "query_optimization" in _capture.text          # names the fix
    assert "NO query was rewritten" in _capture.text      # and the no-op itself


@pytest.mark.parametrize("changed,expected", [(True, "1 rewritten"), (False, "0 rewritten")])
def test_logs_how_many_queries_actually_changed(monkeypatch, _capture, changed, expected):
    monkeypatch.setattr(
        "evaluator.models.retrieval.query.optimization.refine_query",
        lambda q, docs, cfg, method="", context_top_k=3: (
            (q + "!") if (changed and q == "first?") else q
        ),
    )
    s = _refine_state()
    _stage_query_refine(s)
    assert expected in _capture.text
