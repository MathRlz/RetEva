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
