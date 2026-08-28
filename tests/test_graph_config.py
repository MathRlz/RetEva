"""Node-centric YAML → legacy config translation, incl. the node-list DAG form."""

from evaluator.config.evaluation import EvaluationConfig
from evaluator.config.graph_config import (
    _FUSION_KEYS,
    _MMR_KEYS,
    _RERANKER_KEYS,
    _RETRIEVAL_SCALARS,
    _VDB_TO_RETRIEVAL,
    legacy_yaml_to_graph_yaml,
    build_evaluation_config_kwargs,
)
from tests.graph_test_helpers import explicit_graph, mode_graph


def _build(new):
    return EvaluationConfig.from_dict(build_evaluation_config_kwargs(new), validate=False)


def test_dataset_and_node_map_translation():
    cfg = _build(
        {
            "graph": mode_graph("asr_text_retrieval"),
            "dataset": {"id": "admed_voice", "split": "test"},
            "nodes": {
                "asr": {"model": "whisper", "name": "openai/whisper-medium"},
                "text_embedding": {"model": "jina_v4"},
                "vector_db": {"store": "inmemory"},
                "retrieval": {"k": 5},
            },
        }
    )
    assert cfg.model.asr_model_type == "whisper"
    assert cfg.model.text_emb_model_type == "jina_v4"
    assert cfg.data.dataset_name == "admed_voice"
    assert cfg.data.split == "test"


def test_graph_nodes_list_folds_inline_model_config():
    # Self-contained node-list DAG: vector_db / retrieval config inline in node params folds
    # into vector_db (mirrors the nodes: map). An asr/*_embedding node's own model params stay
    # ON the node as a per-node override — never promoted into the flat ``model.*`` default
    # (only a top-level ``nodes.<role>`` block sets that).
    cfg = _build(
        {
            "graph": explicit_graph([
                    {"id": "dataset_source", "type": "dataset_source",
                     "params": {"dataset": "admed_voice", "split": "test"}},
                    {"id": "audio_embedding", "type": "audio_embedding",
                     "params": {"model": "attention_pool_m4t", "size": "v2-large", "dim": 2048,
                                "embedding_space": "jina_v4_space",
                                "params": {"pooling": "mean_abtt"}}},
                    {"id": "corpus_embedding", "type": "corpus_embedding"},
                    {"id": "vector_db", "type": "vector_db", "params": {"store": "inmemory"}},
                    {"id": "retrieval", "type": "retrieval", "params": {"k": 7}},
            ])
        }
    )
    from evaluator.config.types import enum_to_str

    assert cfg.model.audio_emb_model_type is None
    node = next(n for n in cfg.graph_override["nodes"] if n["id"] == "audio_embedding")
    assert node["params"]["model"] == "attention_pool_m4t"
    assert node["params"]["params"] == {"pooling": "mean_abtt"}
    assert node["params"]["embedding_space"] == "jina_v4_space"
    assert enum_to_str(cfg.vector_db.type) == "inmemory"
    assert cfg.vector_db.k == 7


def test_retrieval_node_keeps_mode_and_vectors_but_folds_tuning():
    # A per-arm retrieval node's `mode`/`vectors` are functional params the search handler reads
    # off the node — they stay on the node (so two arms can differ); only tuning keys fold to
    # vector_db. (Explicit multi-arm graphs; the nodes: map path is unchanged.)
    cfg = _build(
        {
            "graph": {
                "nodes": [
                    {"id": "r_sparse", "type": "retrieval",
                     "params": {"mode": "sparse", "k": 5}},
                    {"id": "r_audio", "type": "retrieval",
                     "params": {"vectors": "audio_query_vectors"}},
                ],
                # degenerate 2-node fixture: no derivable edges — pin one port edge so the
                # loader's explicitness cut is satisfied while the fold stays under test
                "edges": [{"from": "r_sparse", "output": "retrieved",
                           "to": "r_audio", "input": "retrieved"}],
            }
        }
    )
    # `complete_structural_plumbing` also appends the derived metrics/finalize plumbing this
    # degenerate fixture doesn't spell out (bare-string node specs, unrelated to the fold under
    # test here) — filter to the dict-shaped nodes this test actually cares about.
    nodes = {n["id"]: n for n in cfg.graph_override["nodes"] if isinstance(n, dict)}
    assert nodes["r_sparse"]["params"] == {"mode": "sparse"}      # mode kept, k folded away
    assert nodes["r_audio"]["params"] == {"vectors": "audio_query_vectors"}
    assert cfg.vector_db.k == 5                                   # tuning folded to vector_db


def test_two_vector_db_nodes_keep_distinct_params():
    # Two vector_db node instances (two corpora/indices) must each retain their own
    # store/path/collection after the fold — only then can effective_vector_db_config
    # diverge per node. The flat legacy["vector_db"] default still gets folded (last
    # node wins) for the few node-context-free call sites; it isn't authoritative here.
    cfg = _build(
        {
            "graph": explicit_graph([
                    {"id": "dataset_source", "type": "dataset_source",
                     "params": {"dataset": "admed_voice", "split": "test"}},
                    {"id": "corpus_embedding", "type": "corpus_embedding"},
                    {"id": "vdb_a", "type": "vector_db",
                     "params": {"store": "chromadb", "path": "./idx_a", "collection": "coll_a"}},
                    {"id": "vdb_b", "type": "vector_db",
                     "params": {"store": "chromadb", "path": "./idx_b", "collection": "coll_b"}},
            ])
        }
    )
    nodes = {n["id"]: n for n in cfg.graph_override["nodes"]}
    assert nodes["vdb_a"]["params"] == {
        "store": "chromadb", "path": "./idx_a", "collection": "coll_a",
    }
    assert nodes["vdb_b"]["params"] == {
        "store": "chromadb", "path": "./idx_b", "collection": "coll_b",
    }
    # Last-node-wins flat default (documented, informational-only fallback).
    assert cfg.vector_db.chromadb_collection_name == "coll_b"


def test_feature_node_params_fold_into_capability_subconfig():
    # A feature node carries its capability config as params; they fold into the sub-config
    # (so the built node stays structural) and the node's presence enables the capability.
    cfg = _build(
        {
            "graph": explicit_graph([
                    "dataset_source",
                    {"id": "tts", "type": "tts",
                     "params": {"provider": "mms", "voice": "en", "seed": 42}},
                    "asr",
                    {"id": "answer_judge", "type": "answer_judge",
                     "params": {"model": "gpt-4o-mini", "temperature": 0.0}},
            ])
        }
    )
    nodes = {n["id"]: n for n in cfg.graph_override["nodes"] if isinstance(n, dict)}
    # Params stay ON the node too (parity with model/vector_db nodes) — resolve_node_config
    # reads them at run time so two same-kind feature nodes (e.g. comparing TTS engines)
    # stay genuinely distinct, not just whichever one folded into the flat default last.
    assert nodes["tts"]["params"] == {"provider": "mms", "voice": "en", "seed": 42}
    assert cfg.audio_synthesis.enabled is True                   # presence enables
    assert cfg.audio_synthesis.provider == "mms"
    assert cfg.audio_synthesis.voice == "en"
    assert cfg.judge.enabled is True
    assert cfg.judge.model == "gpt-4o-mini"


def test_bare_feature_node_enables_capability_without_params():
    # A bare feature node (no params) still enables its capability from presence alone.
    cfg = _build({"graph": explicit_graph(["dataset_source", "audio_embedding",
                                           "text_embedding", "fusion", "retrieval"])})
    assert cfg.embedding_fusion.enabled is True


def test_graph_nodes_plain_strings_unchanged():
    # Back-compat: the plain-string list form (model config in nodes: map) is untouched — the
    # AUTHORED nodes stay bare strings, in order, as a prefix. `complete_structural_plumbing`
    # (pipeline/graph/modes.py) now appends the metrics/finalize plumbing this graph doesn't
    # spell out (asr_text_retrieval's structural chain) — the CLI-path counterpart of the
    # webapi behavior `tests/test_multi_variant_plumbing.py` already covers.
    legacy = build_evaluation_config_kwargs(
        {
            "graph": explicit_graph(["asr", "text_embedding", "retrieval"]),
            "nodes": {"asr": {"model": "whisper"}, "text_embedding": {"model": "labse"}},
        }
    )
    assert legacy["model"]["asr_model_type"] == "whisper"
    assert legacy["model"]["text_emb_model_type"] == "labse"
    nodes = legacy["graph_override"]["nodes"]
    assert nodes[:3] == ["asr", "text_embedding", "retrieval"]
    assert all(isinstance(n, str) for n in nodes[:3])
    assert "metrics" in nodes and "finalize" in nodes


def test_split_maps_to_general_field_hf_split_separate():
    legacy = build_evaluation_config_kwargs({"dataset": {"id": "huggingface", "split": "validation"}})
    assert legacy["data"]["split"] == "validation"
    legacy_hf = build_evaluation_config_kwargs({"dataset": {"id": "huggingface", "hf_split": "train"}})
    assert legacy_hf["data"]["huggingface_split"] == "train"


def test_vdb_to_retrieval_is_exact_inverse_of_forward_maps():
    """F11: the vector_db→retrieval inverse is DERIVED from the forward maps, so the two
    directions cannot drift. Reconstruct it independently and require an exact match."""
    expected = {}
    for node_key, vdb_field in _RETRIEVAL_SCALARS.items():
        expected[vdb_field] = (node_key,)
    for group, fwd in (("fusion", _FUSION_KEYS), ("reranker", _RERANKER_KEYS),
                       ("mmr", _MMR_KEYS)):
        for sub_key, vdb_field in fwd.items():
            expected[vdb_field] = (group, sub_key)
    assert _VDB_TO_RETRIEVAL == expected
    # every vdb field maps to a real node path, and back (1-tuple = scalar, 2-tuple = nested)
    for vdb_field, path in _VDB_TO_RETRIEVAL.items():
        assert 1 <= len(path) <= 2


def test_retrieval_config_round_trips_through_node_form():
    """legacy vector_db → node-centric retrieval block → legacy recovers the same fields."""
    legacy_in = {
        "model": {"pipeline_mode": "asr_text_retrieval"},
        "vector_db": {
            "type": "inmemory", "k": 7, "retrieval_mode": "hybrid",
            "hybrid_fusion_method": "rrf", "rrf_k": 42,
            "reranker_enabled": True, "reranker_top_k": 9, "use_mmr": True,
        },
    }
    graph_form = legacy_yaml_to_graph_yaml(legacy_in)
    rebuilt = build_evaluation_config_kwargs(graph_form)
    vdb = rebuilt["vector_db"]
    assert vdb["k"] == 7
    assert vdb["retrieval_mode"] == "hybrid"
    assert vdb["hybrid_fusion_method"] == "rrf"
    assert vdb["rrf_k"] == 42
    assert vdb["reranker_enabled"] is True
    assert vdb["reranker_top_k"] == 9
    assert vdb["use_mmr"] is True
    assert vdb["type"] == "inmemory"
