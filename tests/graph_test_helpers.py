"""Test helper: explicit graph.nodes lists a config can carry.

A config is an explicit graph (graph.mode is rejected by the loader), so tests that only need
"a config shaped like X" build one from a literal node list here.
"""

#: The canonical node list behind each run label, for tests that assert on a labelled graph.
_LABEL_NODES = {
    "asr_only": ["dataset_source", "asr", "transcription_metrics", "metrics", "finalize"],
    "asr_text_retrieval": [
        "dataset_source", "corpus_embedding", "vector_db", "asr", "text_embedding",
        "retrieval", "transcription_metrics", "retrieval_metrics", "metrics", "finalize",
    ],
    "audio_emb_retrieval": [
        "dataset_source", "corpus_embedding", "vector_db", "audio_embedding", "retrieval",
        "retrieval_metrics", "metrics", "finalize",
    ],
    # cross-modal fusion: the text-query stream is asr(audio) -> text_embedding, fused with the
    # audio_embedding stream before one retrieval
    "audio_text_retrieval": [
        "dataset_source", "corpus_embedding", "vector_db", "asr", "audio_embedding",
        "text_embedding", "fusion", "retrieval", "embedding_alignment_metrics",
        "retrieval_metrics", "metrics", "finalize",
    ],
}


def mode_graph(label):
    """A full explicit `graph:` block (nodes + generated port-level edges) for a run label."""
    return explicit_graph(_LABEL_NODES[label])


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
