"""Test helper: explicit graph.nodes for a pipeline mode.

Configs no longer accept graph.mode (it's rejected by the loader) — a config is an explicit graph.
Tests that only need "a config that builds mode X's graph" use this to generate the explicit node
list (the same builder skeleton assemble_specs produces), instead of the removed mode shorthand.
"""
from evaluator.pipeline.graph.assembly import FeatureSet, assemble_specs


def mode_nodes(mode, **features):
    """The explicit graph.nodes list for a pipeline mode (+ optional FeatureSet flags)."""
    return assemble_specs(mode, FeatureSet(**features))


def mode_graph(mode, **features):
    """A full explicit `graph:` block (nodes + generated port-level edges) for a mode —
    what a post-cut-over config carries; edges via the sanctioned authoring generator."""
    from evaluator.pipeline.graph.wiring import emit_edges

    nodes = mode_nodes(mode, **features)
    return {"nodes": nodes, "edges": emit_edges(nodes)}


def explicit_graph(nodes, extra_edges=()):
    """An explicit `graph:` block for a hand-written node list: edges generated + any
    extras appended (the test-side equivalent of `evaluator graph --emit-edges`)."""
    from evaluator.pipeline.graph.wiring import emit_edges

    return {"nodes": list(nodes), "edges": emit_edges(nodes) + list(extra_edges)}


def make_state(**overrides):
    """A minimal RunState for handler/bus unit tests; override only what the test needs."""
    from evaluator.evaluation.executor.state import RunState

    kwargs = dict(
        dataset=None, mode="custom", retrieval_pipeline=None, asr_pipeline=None,
        text_embedding_pipeline=None, audio_embedding_pipeline=None, cache_manager=None,
        config=None, load_info=None, k=5, batch_size=1, num_workers=0,
        checkpoint_interval=0, experiment_id=None, resume_from_checkpoint=False,
        oracle_mode=False, embedding_fusion_config=None, query_opt_config=None,
        query_correction_config=None, answer_gen_config=None, judge_config=None,
        trace_limit=0, term_weights=None, compute_confidence_intervals=False,
        total=0, cb=lambda *a: None,
    )
    kwargs.update(overrides)
    return RunState(**kwargs)
