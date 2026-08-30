"""Executor mode-label resolution for explicit graphs."""


def test_headless_explicit_graph_gets_custom_mode():
    # No-retrieval RAG baseline: no query head, no pipelines — the executor must fall
    # back to the 'custom' label instead of raising (regression: cluster crash on
    # pubmed_qa_rag_fulltext_noretrieval).
    from evaluator.evaluation.executor.run import _resolve_mode_and_graph

    class _Ctx:
        retrieval_pipeline = None
        asr_pipeline = None
        text_embedding_pipeline = None
        audio_embedding_pipeline = None
        trace_limit = 0

    class _Features:
        embedding_fusion_config = None
        query_opt_config = None
        query_correction_config = None

    from evaluator.config import EvaluationConfig
    cfg = EvaluationConfig.from_yaml(
        "configs/pubmed_qa_rag_fulltext_noretrieval.yaml", validate=False
    )
    mode, graph = _resolve_mode_and_graph(_Ctx(), _Features(), cfg, cfg.graph_override)
    assert mode == "custom"
    assert any(n.id == "answer_gen" for n in graph.nodes)


def test_retrieved_or_text_ids_falls_back_to_query_text():
    # No-retrieval graph: per-query identity comes from the keyed query_text ItemSet,
    # with empty per-query result lists (empty context is the point of that baseline).
    from types import SimpleNamespace

    from evaluator.evaluation.handlers.rag import _retrieved_or_text_ids
    from evaluator.evaluation.item_set import ItemSet

    class _State:
        def get_artifact(self, name, default=None):
            return default

        def keyed_items(self, name, default=None):
            if name == "retrieved":
                return None
            if name == "query_text":
                return ItemSet(["q1", "q2"], ["what?", "why?"])
            return default

    results, keys, ids = _retrieved_or_text_ids(_State())
    assert ids == ["q1", "q2"]
    assert results == [[], []]
    assert keys == [[], []]
