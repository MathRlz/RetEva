"""Multi-vector_db configs: distinct nodes must resolve to distinct backends/collections."""

import pytest

from evaluator.config.evaluation import EvaluationConfig
from evaluator.errors import ConfigurationError
from evaluator.evaluation.validation import validate_graph_vector_db_configs
from evaluator.pipeline.graph import StageGraph, StageNode


def _graph(*node_params):
    nodes = tuple(
        StageNode(id=f"vdb_{i}", stage="vector_db", params=params)
        for i, params in enumerate(node_params)
    )
    return StageGraph(mode="custom", nodes=nodes)


def test_distinct_collections_pass():
    graph = _graph(
        {"store": "chromadb", "path": "./idx_a", "collection": "coll_a"},
        {"store": "chromadb", "path": "./idx_b", "collection": "coll_b"},
    )
    validate_graph_vector_db_configs(graph, EvaluationConfig())


def test_colliding_chromadb_collection_raises():
    graph = _graph(
        {"store": "chromadb", "path": "./idx", "collection": "documents"},
        {"store": "chromadb", "path": "./idx", "collection": "documents"},
    )
    with pytest.raises(ConfigurationError, match="both resolve to the same"):
        validate_graph_vector_db_configs(graph, EvaluationConfig())


def test_inmemory_nodes_never_collide():
    graph = _graph({"store": "inmemory"}, {"store": "inmemory"})
    validate_graph_vector_db_configs(graph, EvaluationConfig())


def test_qdrant_collision_on_url_and_collection():
    graph = _graph(
        {"store": "qdrant", "url": "http://localhost:6333", "collection": "documents"},
        {"store": "qdrant", "url": "http://localhost:6333", "collection": "documents"},
    )
    with pytest.raises(ConfigurationError, match="both resolve to the same"):
        validate_graph_vector_db_configs(graph, EvaluationConfig())
