"""Tests for DAG-lite stage graph planning."""

import pytest

from evaluator.pipeline import (
    StageGraph,
    StageNode,
    build_stage_graph,
    list_pipeline_mode_specs,
    resolve_pipeline_mode_spec,
)


def test_build_stage_graph_asr_text_retrieval():
    graph = build_stage_graph("asr_text_retrieval")
    assert graph.mode == "asr_text_retrieval"
    assert graph.node_ids() == ["asr", "text_embedding", "retrieval", "metrics"]
    assert graph.topological_levels()[0][0].id == "asr"


def test_build_stage_graph_audio_text_with_fusion_has_parallel_level():
    graph = build_stage_graph("audio_text_retrieval", embedding_fusion_enabled=True)
    levels = [[node.id for node in level] for level in graph.topological_levels()]
    assert levels[0] == ["audio_embedding", "text_embedding"]
    assert levels[1] == ["fusion"]
    assert levels[2] == ["retrieval"]
    assert levels[3] == ["metrics"]


def test_build_stage_graph_audio_text_without_fusion_no_text_stage():
    graph = build_stage_graph("audio_text_retrieval", embedding_fusion_enabled=False)
    assert "text_embedding" not in graph.node_ids()
    assert graph.node_ids() == ["audio_embedding", "retrieval", "metrics"]


def test_build_stage_graph_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown pipeline mode"):
        build_stage_graph("unknown-mode")


def test_stage_graph_cycle_detection():
    graph = StageGraph(
        mode="invalid",
        nodes=(
            StageNode(id="a", stage="a", depends_on=("b",)),
            StageNode(id="b", stage="b", depends_on=("a",)),
        ),
    )
    with pytest.raises(ValueError, match="Cycle detected"):
        graph.topological_levels()


def test_pipeline_mode_registry_lists_modes():
    modes = [spec.mode for spec in list_pipeline_mode_specs()]
    assert "asr_only" in modes
    assert "asr_text_retrieval" in modes
    assert "audio_emb_retrieval" in modes
    assert "audio_text_retrieval" in modes


def test_resolve_pipeline_mode_spec_has_required_fields():
    spec = resolve_pipeline_mode_spec("asr_text_retrieval")
    assert "model.asr_model_type" in spec.required_model_fields
    assert "model.text_emb_model_type" in spec.required_model_fields
