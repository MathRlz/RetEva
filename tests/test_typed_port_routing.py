"""B2 pins (CRITIQUE.md): type-open OneOf ports + derivation-ranked candidates.

A multi-name (OneOf) port accepts any registered artifact of the same modality via an
explicit edge — the declared chain is a default and tiebreak, not a closed gate — and a
port's bound candidates are ranked by derivation (a candidate produced from another bound
candidate supersedes it), so an inverted topology (optimize→correct) ranks corrected above
optimized where the chain order alone would not. Plain (single-name) ports accept the same
kind of same-modality routed extra too (e.g. dataset_source.query_text into asr's plain
reference_transcription port) — resolved through the same alias mechanism at run time
(RunState.get_artifact/input); only a cross-modality edge is rejected.
"""

import pytest

from evaluator.pipeline.graph.wiring import bind_explicit_edges, edges_for_graph


def _by_id(nodes):
    return {n.id: n for n in nodes}


def test_non_chain_text_artifact_routes_into_oneof_port():
    # reference_transcription is TEXT but NOT in QUERY_TEXT_CHAIN — accepted via an
    # explicit edge (an oracle-style experiment: embed the spoken ground truth).
    nodes = bind_explicit_edges(
        ["dataset_source", "text_embedding"],
        [{"from": "dataset_source", "output": "reference_transcription",
          "to": "text_embedding", "input": "query_text"}],
    )
    emb = _by_id(nodes)["text_embedding"]
    assert ("reference_transcription", "dataset_source") in emb.bindings
    assert dict(emb.input_aliases)["query_text"] == ("reference_transcription",)


def test_inverted_topology_ranks_corrected_above_optimized():
    # optimize → correct: correction DERIVES from optimization, so the embed port must
    # rank corrected first — the declared chain order (optimized > corrected) would not.
    nodes = bind_explicit_edges(
        ["asr", "query_optimization", "query_correction", "text_embedding"],
        [
            {"from": "asr", "output": "query_text",
             "to": "query_optimization", "input": "query_text"},
            {"from": "query_optimization", "output": "optimized_query_text",
             "to": "query_correction", "input": "query_text"},
            {"from": "query_correction", "output": "corrected_query_text",
             "to": "text_embedding", "input": "query_text"},
            {"from": "query_optimization", "output": "optimized_query_text",
             "to": "text_embedding", "input": "query_text"},  # fallback if corrector bails
        ],
    )
    by_id = _by_id(nodes)
    # correction reads the optimizer's output (correction-after-optimization).
    assert dict(by_id["query_correction"].input_aliases)["query_text"] == (
        "optimized_query_text",
    )
    # embed ranks the derived candidate first, chain order notwithstanding.
    assert dict(by_id["text_embedding"].input_aliases)["query_text"] == (
        "corrected_query_text",
        "optimized_query_text",
    )
    assert by_id["text_embedding"].bindings == (
        ("corrected_query_text", "query_correction"),
        ("optimized_query_text", "query_optimization"),
    )


def test_canonical_chain_order_is_unchanged():
    # asr → correction, embed binds corrected + base fallback: declared order preserved.
    nodes = bind_explicit_edges(
        ["asr", "query_correction", "text_embedding"],
        [
            {"from": "asr", "output": "query_text",
             "to": "query_correction", "input": "query_text"},
            {"from": "query_correction", "output": "corrected_query_text",
             "to": "text_embedding", "input": "query_text"},
            {"from": "asr", "output": "query_text",
             "to": "text_embedding", "input": "query_text"},
        ],
    )
    emb = _by_id(nodes)["text_embedding"]
    assert dict(emb.input_aliases)["query_text"] == (
        "corrected_query_text", "query_text",
    )


def test_cross_modality_edge_still_rejected():
    with pytest.raises(ValueError, match="accepts"):
        bind_explicit_edges(
            ["dataset_source", "text_embedding"],
            [{"from": "dataset_source", "output": "corpus",
              "to": "text_embedding", "input": "query_text"}],
        )


def test_dataset_query_text_routes_into_asr_reference_transcription():
    # The concrete real-world case: no ASR reference field on this dataset, but query_text
    # (Text) can stand in as the WER/CER ground truth via an explicit edge into asr's plain
    # reference_transcription port.
    nodes = bind_explicit_edges(
        ["dataset_source", "asr"],
        [
            {"from": "dataset_source", "output": "query_audio",
             "to": "asr", "input": "query_audio"},
            {"from": "dataset_source", "output": "query_text",
             "to": "asr", "input": "reference_transcription"},
        ],
    )
    asr = _by_id(nodes)["asr"]
    assert ("query_text", "dataset_source") in asr.bindings
    assert dict(asr.input_aliases)["reference_transcription"] == ("query_text",)


def test_plain_port_accepts_same_modality_routed_extra():
    # tts's query_text input is a plain (single-name) port, but B2 lets a same-modality
    # artifact route into it via an explicit edge — get_artifact/input resolve the alias
    # at run time (state.py:_input_candidates), so the handler needs no code change.
    nodes = bind_explicit_edges(
        ["dataset_source", "query_correction", "tts"],
        [
            {"from": "dataset_source", "output": "query_text",
             "to": "query_correction", "input": "query_text"},
            {"from": "query_correction", "output": "corrected_query_text",
             "to": "tts", "input": "query_text"},
        ],
    )
    tts_node = next(n for n in nodes if n.id == "tts")
    assert ("corrected_query_text", "query_correction") in tts_node.bindings
    assert ("query_text", tuple(["corrected_query_text"])) in tts_node.input_aliases


def test_plain_port_rejects_mismatched_modality():
    # A genuinely different-modality artifact still can't satisfy a plain text port.
    with pytest.raises(ValueError, match="accepts"):
        bind_explicit_edges(
            ["dataset_source", "audio_embedding", "tts"],
            [
                {"from": "dataset_source", "output": "query_text",
                 "to": "tts", "input": "query_text"},
                {"from": "dataset_source", "output": "query_audio",
                 "to": "audio_embedding", "input": "query_audio"},
                {"from": "audio_embedding", "output": "audio_query_vectors",
                 "to": "tts", "input": "query_text"},
            ],
        )


def test_routed_graph_edge_round_trip():
    node_ids = ["asr", "query_optimization", "query_correction", "text_embedding"]
    edges = [
        {"from": "asr", "output": "query_text",
         "to": "query_optimization", "input": "query_text"},
        {"from": "query_optimization", "output": "optimized_query_text",
         "to": "query_correction", "input": "query_text"},
        {"from": "query_correction", "output": "corrected_query_text",
         "to": "text_embedding", "input": "query_text"},
    ]
    first = bind_explicit_edges(node_ids, edges)
    rebuilt = bind_explicit_edges(node_ids, edges_for_graph(first))
    sig = lambda ns: [(n.id, n.bindings, n.input_aliases, n.depends_on) for n in ns]  # noqa: E731
    assert sig(rebuilt) == sig(first)


def test_runtime_read_resolves_the_routed_artifact():
    from tests.graph_test_helpers import make_state

    nodes = bind_explicit_edges(
        ["dataset_source", "text_embedding"],
        [{"from": "dataset_source", "output": "reference_transcription",
          "to": "text_embedding", "input": "query_text"}],
    )
    s = make_state()
    s.current_node = _by_id(nodes)["text_embedding"]
    s.ctx.put("dataset_source", "reference_transcription", ["the spoken text"])
    assert s.input("query_text") == ["the spoken text"]
    flow = s.data_flow["text_embedding"]["query_text"]
    assert flow["artifact"] == "reference_transcription"
    assert "fallback" not in flow  # the routed candidate IS the highest-priority binding
