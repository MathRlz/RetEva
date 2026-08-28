"""A3 pins (CRITIQUE.md): the resolved data flow is recorded, fallbacks flagged.

Which producer feeds a port is decided at run time by OneOf priority × newest-published ×
node-list order; ``RunState`` now records the winner per (consumer node, input key) in
``data_flow`` (surfaced as ``report.provenance.data_flow``), flagging a fired fallback —
a winner other than the newest producer of the highest-priority *bound* candidate.
"""

from evaluator.evaluation.executor.state import RunState
from evaluator.pipeline.graph.registry import StageNode

from tests.graph_test_helpers import make_state


def _state() -> RunState:
    return make_state()


def _embedding_node() -> StageNode:
    return StageNode(
        id="text_embedding",
        stage="embed",
        bindings=(("optimized_query_text", "query_optimization"), ("query_text", "asr")),
        input_aliases=(("query_text", ("optimized_query_text", "query_text")),),
    )


def test_oneof_highest_priority_winner_is_not_a_fallback():
    s = _state()
    s.current_node = _embedding_node()
    s.ctx.put("asr", "query_text", ["hi"])
    s.ctx.put("query_optimization", "optimized_query_text", ["hi rewritten"])
    assert s.input("query_text") == ["hi rewritten"]
    flow = s.data_flow["text_embedding"]["query_text"]
    assert flow == {"artifact": "optimized_query_text", "producer": "query_optimization"}


def test_oneof_fallback_to_base_hypothesis_is_flagged():
    s = _state()
    s.current_node = _embedding_node()
    s.ctx.put("asr", "query_text", ["hi"])  # optimization bailed: published nothing
    assert s.input("query_text") == ["hi"]
    flow = s.data_flow["text_embedding"]["query_text"]
    assert flow == {"artifact": "query_text", "producer": "asr", "fallback": True}


def test_unbound_alternatives_do_not_count_as_fallback():
    # A text-only config binds only query_text — reading it is the expected path, not a
    # fallback, even though the OneOf vocabulary lists higher-priority names.
    s = _state()
    s.current_node = StageNode(
        id="text_embedding",
        stage="embed",
        bindings=(("query_text", "dataset_source"),),
        input_aliases=(("query_text", ("optimized_query_text", "query_text")),),
    )
    s.ctx.put("dataset_source", "query_text", ["q"])
    assert s.input("query_text") == ["q"]
    assert "fallback" not in s.data_flow["text_embedding"]["query_text"]


def test_get_artifact_older_producer_is_flagged():
    s = _state()
    s.current_node = StageNode(
        id="retrieval",
        stage="search",
        bindings=(("vector_index", "vector_db_a"), ("vector_index", "vector_db_b")),
    )
    s.ctx.put("vector_db_a", "vector_index", "idx_a")  # newest (vector_db_b) skipped
    assert s.get_artifact("vector_index") == "idx_a"
    flow = s.data_flow["retrieval"]["vector_index"]
    assert flow == {"artifact": "vector_index", "producer": "vector_db_a", "fallback": True}


def test_resolved_producer_reads_back_the_data_flow_winner():
    # Used by the retrieval runtime space guard (resolve_query_space) to find the ACTUAL
    # bound embedder for the current branch instead of the flat config default.
    s = _state()
    s.current_node = _embedding_node()
    s.ctx.put("asr", "query_text", ["hi"])
    s.ctx.put("query_optimization", "optimized_query_text", ["hi rewritten"])
    s.input("query_text")  # resolves + records into data_flow
    assert s.resolved_producer("query_text") == ("optimized_query_text", "query_optimization")


def test_resolved_producer_is_none_before_the_key_is_resolved():
    s = _state()
    s.current_node = _embedding_node()
    assert s.resolved_producer("query_text") == (None, None)
