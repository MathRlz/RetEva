"""`quick_evaluate` carries an explicit graph: `api._build_quick_eval_config_dict` emits
`graph_override.{nodes,edges}` (no mode flag), and the loader still tolerates a legacy
`model.pipeline_mode` key instead of rejecting it as unknown.
"""

from pathlib import Path

from evaluator.api import _build_quick_eval_config_dict
from evaluator.config import EvaluationConfig


def _quick_dict(tmp_path, *, trace_limit=0, **kwargs):
    return _build_quick_eval_config_dict(
        audio_path=Path(tmp_path), asr_type="whisper", emb_type="labse",
        model="whisper", embedding="labse", model_size=None, embedding_size=None,
        k=5, batch_size=1, trace_limit=trace_limit, corpus_path=None, kwargs=kwargs,
    )


def _node_ids(tmp_path, **kw):
    config = EvaluationConfig.from_dict(_quick_dict(tmp_path, **kw), validate=False)
    return {n.id for n in config.graph.nodes}


def test_quick_eval_dict_carries_an_explicit_graph(tmp_path):
    d = _quick_dict(tmp_path)
    assert "pipeline_mode" not in d["model"]
    assert d["graph_override"]["nodes"][:2] == ["dataset_source", "corpus_embedding"]
    assert d["graph_override"]["edges"]


def test_quick_eval_config_builds_and_derives_its_label(tmp_path):
    config = EvaluationConfig.from_dict(_quick_dict(tmp_path))
    assert config.graph_label == "asr_text_retrieval"
    assert {n.id for n in config.graph.nodes} >= {"asr", "retrieval", "finalize"}


def test_quick_eval_feature_kwargs_reshape_the_graph(tmp_path):
    # The knobs quick_evaluate exposes must still add their nodes, as the mode template did.
    plain = _node_ids(tmp_path)
    assert not plain & {"build_query_traces", "rerank", "mmr", "threshold", "retrieval_sparse"}

    tuned = _node_ids(tmp_path, trace_limit=5, vector_db_reranker_enabled=True)
    assert {"build_query_traces", "rerank"} <= tuned

    assert "mmr" in _node_ids(tmp_path, vector_db_use_mmr=True)
    assert "threshold" in _node_ids(tmp_path, vector_db_min_similarity_threshold=0.5)
    assert {"retrieval_sparse", "result_fusion"} <= _node_ids(
        tmp_path, vector_db_retrieval_mode="hybrid"
    )


def test_loader_tolerates_a_legacy_pipeline_mode_key():
    config = EvaluationConfig.from_dict(
        {"model": {"pipeline_mode": "asr_text_retrieval", "asr_model_type": "whisper"}},
        validate=False,
    )
    assert not getattr(config.model, "pipeline_mode", None)
    assert config.model.asr_model_type == "whisper"
