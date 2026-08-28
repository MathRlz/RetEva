"""R5: a config/canvas author authors only *meaningful* nodes (retrieval/embedding/...) and
`complete_structural_plumbing` (`pipeline/graph/modes.py`, shared by the CLI YAML path and the
webapi builder) derives the *structural* plumbing (metrics/traces/finalize) around them. For a
graph comparing several paths (e.g. multiple ASR models or audio encoders — several meaningful
`retrieval` nodes in one flat graph), appending only ONE shared structural chain silently merged
every variant's `retrieved` into it. Each retrieval node must get its own private chain instead.
"""

import pytest

from evaluator.pipeline import build_graph_for_config
from evaluator.pipeline.graph.modes import (
    _terminal_variant_ids,
    _topo_sort_nodes,
    complete_structural_plumbing,
)
from evaluator.pipeline.graph.operators import node_kind
from evaluator.webapi.form_config import graph_spec_to_config_dict, prepare_run_config


def _two_variant_meaningful_spec():
    """Two retrieval variants (whisper vs m4t) sharing dataset_source/corpus_embedding/
    vector_db — exactly the "meaningful nodes only" shape the builder canvas exports (the
    structural retrieval_metrics/metrics/finalize nodes are absent; the server derives
    them)."""
    nodes = [
        "dataset_source",
        {"id": "corpus_embedding", "type": "corpus_embedding",
         "params": {"model": "jina_v4", "embedding_space": "shared_space"}},
        {"id": "vector_db", "type": "vector_db"},
        {"id": "audio_embedding_whisper", "type": "audio_embedding",
         "params": {"model": "attention_pool", "dim": 8, "embedding_space": "shared_space"}},
        {"id": "audio_embedding_m4t", "type": "audio_embedding",
         "params": {"model": "attention_pool", "dim": 8, "embedding_space": "shared_space"}},
        {"id": "retrieval_whisper", "type": "retrieval"},
        {"id": "retrieval_m4t", "type": "retrieval"},
    ]
    edges = [
        {"from": "dataset_source", "to": "audio_embedding_whisper", "input": "query_audio"},
        {"from": "dataset_source", "to": "audio_embedding_m4t", "input": "query_audio"},
        {"from": "dataset_source", "to": "corpus_embedding", "input": "corpus"},
        {"from": "corpus_embedding", "to": "vector_db", "input": "corpus_vectors"},
        {"from": "audio_embedding_whisper", "output": "audio_query_vectors",
         "to": "retrieval_whisper", "input": "query_vectors"},
        {"from": "vector_db", "to": "retrieval_whisper", "input": "vector_index"},
        {"from": "audio_embedding_m4t", "output": "audio_query_vectors",
         "to": "retrieval_m4t", "input": "query_vectors"},
        {"from": "vector_db", "to": "retrieval_m4t", "input": "vector_index"},
    ]
    return {"mode": "audio_emb_retrieval", "nodes": nodes, "edges": edges}


def _build(spec):
    legacy = graph_spec_to_config_dict(spec, experiment_name="t")
    config = prepare_run_config(legacy, auto_devices=False)
    return build_graph_for_config(config)


def test_each_retrieval_variant_gets_its_own_metrics_and_finalize_chain():
    graph = _build(_two_variant_meaningful_spec())
    finalize_ids = sorted(n.id for n in graph.nodes if node_kind(n.stage, n.params) == "finalize")
    metrics_ids = sorted(
        n.id for n in graph.nodes
        if node_kind(n.stage, n.params) == "metrics"
    )
    assert len(finalize_ids) == 2
    assert len(metrics_ids) == 2
    assert finalize_ids[0] != finalize_ids[1]


def test_each_chain_binds_only_to_its_own_retrieval_node():
    graph = _build(_two_variant_meaningful_spec())
    by_id = {n.id: n for n in graph.nodes}

    def metrics_for(retrieval_id):
        return next(
            n for n in graph.nodes
            if node_kind(n.stage, n.params) == "metrics"
            and ("retrieved", retrieval_id) in n.bindings
        )

    metrics_whisper = metrics_for("retrieval_whisper")
    metrics_m4t = metrics_for("retrieval_m4t")
    assert metrics_whisper.id != metrics_m4t.id
    # The bug this pins: before the fix, BOTH would resolve to the SAME single metrics node,
    # which bound retrieved from BOTH retrieval_whisper AND retrieval_m4t.
    retrieved_producers = {p for art, p in metrics_whisper.bindings if art == "retrieved"}
    assert retrieved_producers == {"retrieval_whisper"}


def _shared_subset_ancestor_spec():
    """Two ASR nodes, each shared by exactly TWO of four retrieval variants (asr_a feeds
    retrieval_a1/a2, asr_b feeds retrieval_b1/b2) — the shape that broke the OLD "not
    exclusively owned by a DIFFERENT variant" scope: a node reachable from exactly 2 of 4
    variant roots is never exclusively owned by any ONE of them, so it stayed in EVERY
    other variant's scope too (not just its real two owners) — transcription_metrics then
    auto-bound to BOTH asr nodes for every variant instead of just its own branch's."""
    nodes = [
        "dataset_source",
        {"id": "corpus_embedding", "type": "corpus_embedding"},
        {"id": "vector_db", "type": "vector_db"},
        {"id": "asr_a", "type": "asr"},
        {"id": "asr_b", "type": "asr"},
        {"id": "text_embedding_a1", "type": "text_embedding"},
        {"id": "text_embedding_a2", "type": "text_embedding"},
        {"id": "text_embedding_b1", "type": "text_embedding"},
        {"id": "text_embedding_b2", "type": "text_embedding"},
        {"id": "retrieval_a1", "type": "retrieval"},
        {"id": "retrieval_a2", "type": "retrieval"},
        {"id": "retrieval_b1", "type": "retrieval"},
        {"id": "retrieval_b2", "type": "retrieval"},
    ]
    edges = [
        {"from": "dataset_source", "to": "corpus_embedding", "input": "corpus"},
        {"from": "corpus_embedding", "to": "vector_db", "input": "corpus_vectors"},
        {"from": "dataset_source", "to": "asr_a", "input": "query_audio"},
        {"from": "dataset_source", "to": "asr_b", "input": "query_audio"},
    ]
    for asr_id, txt_ids in (("asr_a", ("a1", "a2")), ("asr_b", ("b1", "b2"))):
        for suffix in txt_ids:
            te_id = f"text_embedding_{suffix}"
            r_id = f"retrieval_{suffix}"
            edges += [
                {"from": asr_id, "to": te_id, "input": "query_text"},
                {"from": te_id, "output": "text_query_vectors", "to": r_id,
                 "input": "query_vectors"},
                {"from": "vector_db", "to": r_id, "input": "vector_index"},
                {"from": asr_id, "to": r_id, "input": "query_text"},
                {"from": "dataset_source", "to": r_id, "input": "reference_transcription"},
            ]
    return {"mode": "asr_text_retrieval", "nodes": nodes, "edges": edges}


def test_transcription_metrics_binds_only_to_its_own_variants_asr():
    """The bug this pins: asr_a is shared by retrieval_a1 AND retrieval_a2 (not all 4
    variants, not exclusively 1) — the old ownership model never excluded it from
    retrieval_b1/b2's scope either, so their transcription_metrics auto-bound to BOTH
    asr_a and asr_b instead of just their own asr_b."""
    graph = _build(_shared_subset_ancestor_spec())

    def transcription_metrics_for(retrieval_id):
        # transcription_metrics has no formal port linking it to a specific retrieval node
        # directly — find it via the metrics node that DOES bind to this retrieval's
        # `retrieved`, then follow that metrics node's own transcription_scores producer.
        metrics = next(
            n for n in graph.nodes
            if node_kind(n.stage, n.params) == "metrics"
            and ("retrieved", retrieval_id) in n.bindings
        )
        tm_id = next(p for art, p in metrics.bindings if art == "transcription_scores")
        return next(n for n in graph.nodes if n.id == tm_id)

    tm_a1 = transcription_metrics_for("retrieval_a1")
    tm_a2 = transcription_metrics_for("retrieval_a2")
    tm_b1 = transcription_metrics_for("retrieval_b1")

    asr_producers_a1 = {p for art, p in tm_a1.bindings if art == "query_text"}
    asr_producers_b1 = {p for art, p in tm_b1.bindings if art == "query_text"}
    # dataset_source's own native query_text is a legitimate shared-global candidate (B2)
    # and expected in every variant's set; the bug was picking up the SIBLING asr too.
    assert "asr_a" in asr_producers_a1 and "asr_b" not in asr_producers_a1
    assert "asr_b" in asr_producers_b1 and "asr_a" not in asr_producers_b1
    # a1 and a2 legitimately share asr_a (same real ancestor) — sharing itself is fine,
    # the bug was b1/b2 ALSO picking up asr_a despite having no relationship to it.
    asr_producers_a2 = {p for art, p in tm_a2.bindings if art == "query_text"}
    assert "asr_a" in asr_producers_a2 and "asr_b" not in asr_producers_a2


def test_single_retrieval_node_graph_keeps_unsuffixed_ids():
    """Byte-parity: the common single-variant case must be untouched by the R5 fix — plain
    "metrics"/"finalize" ids, not "metrics_retrieval"/"finalize_retrieval"."""
    spec = _two_variant_meaningful_spec()
    spec["nodes"] = [n for n in spec["nodes"] if n != "audio_embedding_m4t"
                      and (not isinstance(n, dict) or n["id"] not in
                           ("audio_embedding_m4t", "retrieval_m4t"))]
    spec["edges"] = [e for e in spec["edges"]
                      if "m4t" not in e.get("from", "") and "m4t" not in e.get("to", "")]
    graph = _build(spec)
    finalize_ids = [n.id for n in graph.nodes if node_kind(n.stage, n.params) == "finalize"]
    assert finalize_ids == ["finalize"]


# ── The webUI round-trip bugs found while verifying every repo config against the UI ──

def test_topo_sort_nodes_moves_producer_before_consumer():
    """A structural node appended after `complete_structural_plumbing` can land after a
    meaningful node that consumes it — the loader requires producer-before-consumer in
    graph.nodes."""
    nodes = ["a", "consumer", "producer"]  # "producer" wrongly listed AFTER "consumer"
    edges = [{"from": "producer", "to": "consumer", "input": "x"}]
    ordered = _topo_sort_nodes(nodes, edges)
    assert ordered.index("producer") < ordered.index("consumer")
    # untouched relative order for anything with no constraint
    assert ordered.index("a") < ordered.index("producer")


def test_topo_sort_nodes_is_a_noop_when_order_is_already_valid():
    nodes = ["producer", "consumer", "a"]
    edges = [{"from": "producer", "to": "consumer", "input": "x"}]
    assert _topo_sort_nodes(nodes, edges) == nodes


def test_terminal_variant_ids_excludes_an_upstream_hop_feeding_another_retrieval():
    """query-refine's 2-hop pipeline (retrieval -> query_refine -> retrieval_refined, ONE
    shared judge at the end) has 2 retrieval nodes too — but they're sequential stages of ONE
    pipeline, not independent variants. Only the terminal hop should get its own chain."""
    edges = [
        {"from": "retrieval", "to": "query_refine", "input": "retrieved"},
        {"from": "query_refine", "to": "retrieval_refined", "input": "refined_query_text"},
        {"from": "retrieval_refined", "to": "answer_judge", "input": "retrieved"},
    ]
    assert _terminal_variant_ids(["retrieval", "retrieval_refined"], edges) == [
        "retrieval_refined"
    ]


def test_terminal_variant_ids_keeps_independent_parallel_variants():
    edges = [
        {"from": "retrieval_a", "to": "metrics_a", "input": "retrieved"},
        {"from": "retrieval_b", "to": "metrics_b", "input": "retrieved"},
    ]
    assert _terminal_variant_ids(["retrieval_a", "retrieval_b"], edges) == [
        "retrieval_a", "retrieval_b"
    ]


def test_complete_with_plumbing_orders_build_query_traces_before_answer_judge():
    """The webUI bug this pins: a meaningful `answer_judge` node's OWN binding names
    `build_query_traces` as its query_traces producer — that node is structural (stripped from
    the canvas) and must be appended back BEFORE edge validation, and placed BEFORE
    answer_judge in graph.nodes (producer-before-consumer)."""
    graph = {
        "nodes": [
            "dataset_source",
            {"id": "corpus_embedding", "type": "corpus_embedding"},
            {"id": "vector_db", "type": "vector_db"},
            {"id": "text_embedding", "type": "text_embedding"},
            {"id": "retrieval", "type": "retrieval"},
            {"id": "answer_gen", "type": "answer_gen"},
            {"id": "answer_judge", "type": "answer_judge"},
        ],
        "edges": [
            {"from": "dataset_source", "to": "corpus_embedding", "input": "corpus"},
            {"from": "corpus_embedding", "to": "vector_db", "input": "corpus_vectors"},
            {"from": "dataset_source", "output": "reference_text", "to": "text_embedding",
             "input": "query_text"},
            {"from": "text_embedding", "to": "retrieval", "input": "query_vectors"},
            {"from": "vector_db", "to": "retrieval", "input": "vector_index"},
            {"from": "retrieval", "to": "answer_gen", "input": "retrieved"},
            {"from": "dataset_source", "output": "reference_text", "to": "answer_gen",
             "input": "query_text"},
            {"from": "dataset_source", "to": "answer_gen", "input": "relevant_docs"},
            # answer_judge's OWN existing binding to a NOT-YET-PRESENT structural producer:
            {"from": "build_query_traces", "to": "answer_judge", "input": "query_traces"},
            {"from": "answer_gen", "to": "answer_judge", "input": "generated_answers"},
            {"from": "retrieval", "to": "answer_judge", "input": "retrieved"},
            {"from": "dataset_source", "to": "answer_judge", "input": "relevant_docs"},
        ],
    }
    complete_structural_plumbing(graph)
    ids = [n if isinstance(n, str) else n["id"] for n in graph["nodes"]]
    assert "build_query_traces" in ids
    assert ids.index("build_query_traces") < ids.index("answer_judge")
    # unaffected: build_evaluation_config_kwargs's edge validation must now succeed
    from evaluator.config.graph_config import build_evaluation_config_kwargs

    build_evaluation_config_kwargs({"graph": graph})


def test_complete_with_plumbing_falls_back_to_generic_template_for_text_only_retrieval():
    """label_from_graph returns None for a graph with neither asr nor audio_embedding (a pure
    text-retrieval graph, e.g. configs/pubmed_qa_rag_fulltext.yaml) — plumbing completion must
    not silently no-op just because there's no NAMED template for that exact shape."""
    graph = {
        "nodes": [
            "dataset_source",
            {"id": "corpus_embedding", "type": "corpus_embedding"},
            {"id": "vector_db", "type": "vector_db"},
            {"id": "text_embedding", "type": "text_embedding"},
            {"id": "retrieval", "type": "retrieval"},
        ],
        "edges": [
            {"from": "dataset_source", "to": "corpus_embedding", "input": "corpus"},
            {"from": "corpus_embedding", "to": "vector_db", "input": "corpus_vectors"},
            {"from": "dataset_source", "output": "reference_text", "to": "text_embedding",
             "input": "query_text"},
            {"from": "text_embedding", "to": "retrieval", "input": "query_vectors"},
            {"from": "vector_db", "to": "retrieval", "input": "vector_index"},
        ],
    }
    from evaluator.pipeline.graph.modes import label_from_graph

    assert label_from_graph(graph) is None  # confirms the gap this fallback closes
    complete_structural_plumbing(graph)
    ids = [n if isinstance(n, str) else n["id"] for n in graph["nodes"]]
    assert "metrics" in ids and "finalize" in ids


def test_complete_with_plumbing_per_variant_skips_already_present_structural_nodes():
    """A full-DAG spec (structural nodes already present, e.g. a canvas round-trip that
    doesn't strip them) must not get a DUPLICATE structural node appended alongside the
    existing one."""
    graph = {
        "nodes": [
            "dataset_source",
            {"id": "vector_db", "type": "vector_db"},
            {"id": "retrieval_a", "type": "retrieval"},
            {"id": "retrieval_b", "type": "retrieval"},
            {"id": "metrics_a", "type": "metrics"},
            {"id": "metrics_b", "type": "metrics"},
        ],
        "edges": [
            {"from": "vector_db", "to": "retrieval_a", "input": "vector_index"},
            {"from": "vector_db", "to": "retrieval_b", "input": "vector_index"},
            {"from": "retrieval_a", "to": "metrics_a", "input": "retrieved"},
            {"from": "retrieval_b", "to": "metrics_b", "input": "retrieved"},
        ],
    }
    complete_structural_plumbing(graph)
    ids = [n if isinstance(n, str) else n["id"] for n in graph["nodes"]]
    assert ids.count("metrics_a") == 1
    assert ids.count("metrics_b") == 1


def test_complete_with_plumbing_per_variant_reuses_an_existing_dangling_reference_id():
    """A per-variant meaningful node's own binding already names the EXACT structural id the
    original (hand-authored) config used (e.g. `build_query_traces_dense`, not the
    `build_query_traces_retrieval_dense` this code would otherwise invent) — reuse it, don't
    mint a different name the rest of the graph doesn't expect."""
    graph = {
        "nodes": [
            "dataset_source",
            {"id": "vector_db", "type": "vector_db"},
            {"id": "retrieval_dense", "type": "retrieval"},
            {"id": "retrieval_sparse", "type": "retrieval"},
            {"id": "answer_gen_dense", "type": "answer_gen"},
            {"id": "answer_judge_dense", "type": "answer_judge"},
        ],
        "edges": [
            {"from": "vector_db", "to": "retrieval_dense", "input": "vector_index"},
            {"from": "vector_db", "to": "retrieval_sparse", "input": "vector_index"},
            {"from": "retrieval_dense", "to": "answer_gen_dense", "input": "retrieved"},
            {"from": "retrieval_dense", "to": "answer_judge_dense", "input": "retrieved"},
            {"from": "answer_gen_dense", "to": "answer_judge_dense", "input": "generated_answers"},
            # the existing (dangling) reference using the ORIGINAL config's own convention:
            {"from": "build_query_traces_dense", "to": "answer_judge_dense",
             "input": "query_traces"},
        ],
    }
    complete_structural_plumbing(graph)
    ids = [n if isinstance(n, str) else n["id"] for n in graph["nodes"]]
    assert "build_query_traces_dense" in ids
    assert "build_query_traces_retrieval_dense" not in ids


# ── Moving derivation into core (pipeline/graph/modes.py): the CLI YAML path gets it too ──

def test_cli_yaml_config_can_omit_structural_nodes_entirely():
    """The actual deliverable: `complete_structural_plumbing` is now wired into
    `config/graph_config.py:build_evaluation_config_kwargs` itself — the single chokepoint BOTH
    `config/loading.py:build_from_yaml` (CLI, `EvaluationConfig.from_yaml`) and the webapi
    builder already call — so a hand-authored CLI YAML config can omit the metrics/finalize
    plumbing entirely and still build, exactly like the webapi canvas path already could.
    Exercises the same chokepoint `from_yaml` uses (`build_evaluation_config_kwargs`), without
    needing a real dataset/YAML file on disk — proves the mechanism, not the I/O wrapper."""
    from evaluator.config.evaluation import EvaluationConfig
    from evaluator.config.graph_config import build_evaluation_config_kwargs

    raw = {
        "graph": {
            "nodes": [
                "dataset_source",
                {"id": "corpus_embedding", "type": "corpus_embedding"},
                {"id": "vector_db", "type": "vector_db"},
                {"id": "text_embedding", "type": "text_embedding"},
                {"id": "retrieval", "type": "retrieval"},
            ],
            "edges": [
                {"from": "dataset_source", "to": "corpus_embedding", "input": "corpus"},
                {"from": "corpus_embedding", "to": "vector_db", "input": "corpus_vectors"},
                {"from": "dataset_source", "output": "reference_text", "to": "text_embedding",
                 "input": "query_text"},
                {"from": "text_embedding", "output": "text_query_vectors", "to": "retrieval",
                 "input": "query_vectors"},
                {"from": "vector_db", "to": "retrieval", "input": "vector_index"},
            ],
            # no metrics/build_query_traces/finalize -- derived automatically
        }
    }
    legacy = build_evaluation_config_kwargs(raw)
    ids = [n if isinstance(n, str) else n["id"] for n in legacy["graph_override"]["nodes"]]
    assert "metrics" in ids and "finalize" in ids
    config = EvaluationConfig.from_dict(legacy, validate=False)
    graph = build_graph_for_config(config)
    kinds = {node_kind(n.stage, n.params) for n in graph.nodes}
    assert "metrics" in kinds and "finalize" in kinds


# ── Bugs found wiring completion into real CLI YAML configs (test_graph_golden.py caught these
#    against configs whose full hand-authored structural set already exists) ──

def test_transcription_metrics_excluded_when_no_asr_node_present():
    """The generic "asr_text_retrieval" stand-in template (used when label_from_graph returns
    None for a pure-text-retrieval graph) unconditionally includes `transcription_metrics` (an
    ASR-specific measure with no declared input port, so it can't be filtered by input
    satisfiability) -- wrong when there's no `asr` node at all. Caught by
    `test_graph_golden.py`'s byte-exact comparison against `configs/pubmed_qa_rag_fulltext.yaml`
    (a real config with its own correct, ASR-free structural set already written by hand)."""
    graph = {
        "nodes": [
            "dataset_source",
            {"id": "corpus_embedding", "type": "corpus_embedding"},
            {"id": "vector_db", "type": "vector_db"},
            {"id": "text_embedding", "type": "text_embedding"},
            {"id": "retrieval", "type": "retrieval"},
        ],
        "edges": [
            {"from": "dataset_source", "to": "corpus_embedding", "input": "corpus"},
            {"from": "corpus_embedding", "to": "vector_db", "input": "corpus_vectors"},
            {"from": "dataset_source", "output": "reference_text", "to": "text_embedding",
             "input": "query_text"},
            {"from": "text_embedding", "to": "retrieval", "input": "query_vectors"},
            {"from": "vector_db", "to": "retrieval", "input": "vector_index"},
        ],
    }
    complete_structural_plumbing(graph)
    ids = [n if isinstance(n, str) else n["id"] for n in graph["nodes"]]
    assert "transcription_metrics" not in ids
    assert "retrieval_metrics" in ids and "metrics" in ids and "finalize" in ids


def test_variant_root_suffix_strips_the_roots_own_kind_not_just_retrieval():
    """The per-variant id-reconciliation suffix must strip whichever :data:`VARIANT_ROOT_KINDS`
    kind the root actually is (`answer_gen_mms` -> "mms"), not just a hardcoded "retrieval_"
    prefix (which left `answer_gen_mms` un-stripped, minting `answer_metrics_answer_gen_mms`
    instead of reusing the real config's own `answer_metrics_mms`). Caught by
    `test_graph_golden.py` against `configs/pubmed_qa_fulltext_big_tts_cmp.yaml` (4 TTS-voice
    variants, each a query-refine chain terminating in its OWN `answer_gen_<voice>`)."""
    graph = {
        "nodes": [
            "dataset_source",
            {"id": "vector_db", "type": "vector_db"},
            {"id": "retrieval_mms", "type": "retrieval"},
            {"id": "retrieval_piper", "type": "retrieval"},
            {"id": "answer_gen_mms", "type": "answer_gen"},
            {"id": "answer_gen_piper", "type": "answer_gen"},
            # already-present structural nodes using the real convention (bare voice suffix):
            {"id": "answer_metrics_mms", "type": "answer_metrics"},
            {"id": "answer_metrics_piper", "type": "answer_metrics"},
        ],
        "edges": [
            {"from": "vector_db", "to": "retrieval_mms", "input": "vector_index"},
            {"from": "vector_db", "to": "retrieval_piper", "input": "vector_index"},
            {"from": "retrieval_mms", "to": "answer_gen_mms", "input": "retrieved"},
            {"from": "retrieval_piper", "to": "answer_gen_piper", "input": "retrieved"},
            {"from": "answer_gen_mms", "to": "answer_metrics_mms", "input": "generated_answers"},
            {"from": "answer_gen_piper", "to": "answer_metrics_piper",
             "input": "generated_answers"},
        ],
    }
    complete_structural_plumbing(graph)
    ids = [n if isinstance(n, str) else n["id"] for n in graph["nodes"]]
    assert ids.count("answer_metrics_mms") == 1
    assert ids.count("answer_metrics_piper") == 1
    assert "answer_metrics_answer_gen_mms" not in ids
    assert "answer_metrics_answer_gen_piper" not in ids


def test_converging_variant_roots_get_one_shared_chain_not_per_variant_ones():
    """Two variant-root candidates can both be individually "terminal" (`_terminal_variant_ids`:
    neither reaches the OTHER root) while still CONVERGING into a shared downstream node (a
    hybrid/fusion combine) before reaching any structural measure -- that's one variant with two
    sources, not two variants to compare, and must get a single shared chain. Caught by
    `test_graph_golden.py` against `configs/examples/hybrid_retrieval.yaml` (dense `retrieval` +
    `retrieval_sparse`, both feeding one shared `result_fusion`)."""
    graph = {
        "nodes": [
            "dataset_source",
            {"id": "vector_db", "type": "vector_db"},
            {"id": "retrieval", "type": "retrieval"},
            {"id": "retrieval_sparse", "type": "retrieval"},
            {"id": "result_fusion", "type": "result_fusion"},
        ],
        "edges": [
            {"from": "vector_db", "to": "retrieval", "input": "vector_index"},
            {"from": "vector_db", "to": "retrieval_sparse", "input": "vector_index"},
            {"from": "retrieval", "to": "result_fusion", "input": "dense_results"},
            {"from": "retrieval_sparse", "to": "result_fusion", "input": "sparse_results"},
        ],
    }
    complete_structural_plumbing(graph)
    ids = [n if isinstance(n, str) else n["id"] for n in graph["nodes"]]
    # single shared chain: plain "metrics"/"finalize", never suffixed per-root ids
    assert "metrics" in ids and "finalize" in ids
    assert not any(isinstance(i, str) and i.startswith("metrics_") for i in ids)
    assert not any(isinstance(i, str) and i.startswith("finalize_") for i in ids)


# ── Bug found while live-testing the builder's new rename feature: renaming a meaningful node
#    to a name a structural node also needs must error clearly, not silently corrupt ──

def test_renaming_a_node_to_a_reserved_structural_id_errors_clearly():
    """A user can freely rename a canvas node (structural ids aren't on the canvas, so the
    client-side uniqueness check can't see the future collision) -- if they pick e.g. "metrics"
    for a `retrieval` node, `complete_structural_plumbing` must refuse loudly instead of
    silently colliding. Before the fix: `_topo_sort_nodes`'s own id-keyed dict comprehensions
    silently collapsed the duplicate, dropping the renamed node's edges onto the WRONG
    (structural) node -- surfaced downstream as a confusing "node 'metrics' has no input port
    'vector_index'" error instead of naming the real cause."""
    graph = {
        "nodes": [
            "dataset_source",
            {"id": "corpus_embedding", "type": "corpus_embedding"},
            {"id": "vector_db", "type": "vector_db"},
            {"id": "text_embedding", "type": "text_embedding"},
            # renamed from "retrieval" to "metrics" -- still type=retrieval, just a colliding id
            {"id": "metrics", "type": "retrieval"},
        ],
        "edges": [
            {"from": "dataset_source", "to": "corpus_embedding", "input": "corpus"},
            {"from": "corpus_embedding", "to": "vector_db", "input": "corpus_vectors"},
            {"from": "dataset_source", "output": "reference_text", "to": "text_embedding",
             "input": "query_text"},
            {"from": "text_embedding", "output": "text_query_vectors", "to": "metrics",
             "input": "query_vectors"},
            {"from": "vector_db", "to": "metrics", "input": "vector_index"},
        ],
    }
    with pytest.raises(ValueError, match="already used by a 'retrieval' node"):
        complete_structural_plumbing(graph)


def test_renaming_a_node_to_a_reserved_per_variant_structural_id_errors_clearly():
    """Same bug, per-variant shape: a node renamed to collide with the id a variant's OWN
    structural chain would need (e.g. "metrics_dense") must error, not be silently treated as
    "already present" (the old `if new_id in all_ids: continue` trusted ANY existing node at
    that id, regardless of kind)."""
    graph = {
        "nodes": [
            "dataset_source",
            {"id": "vector_db", "type": "vector_db"},
            {"id": "retrieval_dense", "type": "retrieval"},
            {"id": "retrieval_sparse", "type": "retrieval"},
            # collides with the id retrieval_dense's own per-variant metrics chain would need
            # -- "rerank" so it isn't itself a VARIANT_ROOT_KINDS candidate root.
            {"id": "metrics_dense", "type": "rerank"},
        ],
        "edges": [
            {"from": "vector_db", "to": "retrieval_dense", "input": "vector_index"},
            {"from": "vector_db", "to": "retrieval_sparse", "input": "vector_index"},
            {"from": "retrieval_dense", "to": "metrics_dense", "input": "retrieved"},
        ],
    }
    with pytest.raises(ValueError, match="already used by a 'rerank' node"):
        complete_structural_plumbing(graph)
