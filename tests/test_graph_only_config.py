"""A config specifies only a DAG (graph.nodes): the executed graph drives building + behavior,
and the run derives its display label from the graph's node kinds.
"""

from unittest.mock import MagicMock, patch

from evaluator.config.evaluation import EvaluationConfig
from evaluator.config.graph_config import build_evaluation_config_kwargs
from evaluator.pipeline.graph.build import build_graph_for_config, graph_label


_DATASET = {
    "id": "pubmed_qa",
    "questions": "examples/data/pubmed_qa_small/questions.json",
    "corpus": "examples/data/pubmed_qa_small/corpus.json",
}


def _graph_only_cfg(nodes):
    from tests.graph_test_helpers import explicit_graph

    legacy = build_evaluation_config_kwargs({"dataset": _DATASET, "graph": explicit_graph(nodes)})
    return EvaluationConfig.from_dict(legacy, validate=False)


_ASR_TEXT_NODES = [
    {"id": "dataset_source", "type": "dataset_source"},
    {"id": "asr", "type": "asr", "params": {"model": "whisper", "name": "openai/whisper-base"}},
    {"id": "text_embedding", "type": "text_embedding", "params": {"model": "labse"}},
    {"id": "corpus_embedding", "type": "corpus_embedding"},
    {"id": "vector_db", "type": "vector_db", "params": {"store": "inmemory"}},
    {"id": "retrieval", "type": "retrieval", "params": {"k": 5}},
]


def test_graph_only_config_has_no_mode_but_loads():
    cfg = _graph_only_cfg(_ASR_TEXT_NODES)
    # No graph.mode → the label is DERIVED from the graph's node kinds (the graph is the spec).
    # A graph node's own model params stay ON the node (a per-node override) — resolve them the
    # way the executor does, not off the flat default.
    assert cfg.graph_label == "asr_text_retrieval"   # asr + text_embedding + retrieval
    from evaluator.config.graph_config import resolved_model_config

    nodes = {n.id: n for n in cfg.graph.nodes}
    assert resolved_model_config(cfg, nodes["asr"]).asr_model_type == "whisper"
    assert resolved_model_config(cfg, nodes["text_embedding"]).text_emb_model_type == "labse"


def test_graph_only_config_builds_its_graph_with_derived_label():
    cfg = _graph_only_cfg(_ASR_TEXT_NODES)
    graph = build_graph_for_config(cfg)            # must not require a mode
    assert graph.mode == "asr_text_retrieval"      # label derived from node kinds, not "custom"
    stages = {n.stage for n in graph.nodes}
    assert {"asr", "embed", "search"} <= stages   # asr + embed + retrieval are present


def test_graph_only_factory_builds_pipelines_and_label_is_derived():
    cfg = _graph_only_cfg(_ASR_TEXT_NODES)
    with patch("evaluator.pipeline.factory.create_gpu_pool_from_config", return_value=None), \
         patch("evaluator.pipeline.factory.create_reranker_from_config", return_value=None), \
         patch("evaluator.pipeline.factory._create_retrieval_pipeline") as m_retr, \
         patch("evaluator.pipeline.factory.create_asr_model"), \
         patch("evaluator.pipeline.factory.create_text_embedding_model"), \
         patch("evaluator.pipeline.factory.ASRPipeline"), \
         patch("evaluator.pipeline.factory.TextEmbeddingPipeline"), \
         patch("evaluator.pipeline.factory.AudioEmbeddingPipeline"):
        from evaluator.pipeline.factory import create_pipeline_from_config

        bundle = create_pipeline_from_config(cfg, cache_manager=MagicMock())

    assert bundle.asr_pipeline is not None
    assert bundle.text_embedding_pipeline is not None
    assert bundle.audio_embedding_pipeline is None       # no audio node in this graph
    assert m_retr.called                                 # a search node → retrieval built
    # The label is derived from the graph's node kinds: asr + retrieval → asr_text_retrieval.
    assert bundle.mode == "asr_text_retrieval"
    assert graph_label(cfg.graph_override) == "asr_text_retrieval"
