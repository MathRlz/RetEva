"""Build the execution DAG for a run from its mode + enabled features.

Moved out of the former ``evaluation/phased.py`` (Phase 1, X9). This belongs with the
pipeline's other graph builders: it mirrors ``build_graph_for_config`` but sources the
enabled stages from the run's features + pipelines rather than a single config object.
"""

from __future__ import annotations


def _build_run_graph(
    mode,
    *,
    graph_override,
    embedding_fusion_config,
    query_opt_config,
    retrieval_pipeline,
    eval_config,
    query_correction_config=None,
):
    """Build the execution DAG for a run: an explicit ``graph_override`` if given, else
    derived from the mode + the run's enabled features (fusion / query-opt / rerank / tts
    / sink). Mirrors ``build_graph_for_config`` but sourced from the run's features +
    pipelines rather than a single config object."""
    from . import build_graph_from_spec, build_stage_graph

    if graph_override and not graph_override.get("branches"):
        return build_graph_from_spec(
            graph_override["nodes"], mode=mode, edges=graph_override.get("edges")
        )

    def _enabled(cfg):
        return bool(cfg is not None and getattr(cfg, "enabled", False))

    rerank = bool(
        retrieval_pipeline is not None
        and getattr(retrieval_pipeline, "needs_refinement", False)
    )
    tts = _enabled(getattr(eval_config, "audio_synthesis", None))
    sink = _enabled(getattr(eval_config, "dataset_sink", None))
    if graph_override and graph_override.get("branches"):
        from . import build_branched_graph
        from .stage_graph import _mode_node_ids

        base = _mode_node_ids(
            mode,
            _enabled(embedding_fusion_config),
            _enabled(query_opt_config),
            rerank,
            tts,
            sink,
            _enabled(query_correction_config),
        )
        return build_branched_graph(base, graph_override["branches"], mode=mode)

    return build_stage_graph(
        mode,
        embedding_fusion_enabled=_enabled(embedding_fusion_config),
        query_opt_enabled=_enabled(query_opt_config),
        rerank_enabled=rerank,
        tts_enabled=tts,
        sink_enabled=sink,
        correction_enabled=_enabled(query_correction_config),
    )
