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
    trace_limit=0,
):
    """Build the execution DAG for a run: an explicit ``graph_override`` if given, else
    derived from the mode + the run's enabled features (fusion / query-opt / rerank / tts
    / sink). Mirrors ``build_graph_for_config`` but sourced from the run's features +
    pipelines rather than a single config object."""
    from . import build_graph_from_spec

    if graph_override and not graph_override.get("branches"):
        return build_graph_from_spec(
            graph_override["nodes"], mode=mode, edges=graph_override.get("edges")
        )

    from dataclasses import replace

    from . import build_branched_graph
    from .graph.assembly import assemble_specs
    from .graph.modes import _features_from_config

    def _enabled(cfg):
        return bool(cfg is not None and getattr(cfg, "enabled", False))

    # Base feature set from the run's config (judge / answer_gen / sink / tts / rag rounds);
    # then override the run-time-resolved features the caller passed explicitly (fusion /
    # query-opt / correction come from the RunFeatures, not eval_config). ``rerank`` is a
    # property of the built pipeline; ``trace_limit`` is the authoritative trace gate.
    fusion_on = _enabled(embedding_fusion_config)
    # The built retrieval pipeline is the authoritative source for the refine sub-steps
    # (rerank / mmr / threshold) — read them off its strategy config.
    sc = getattr(retrieval_pipeline, "strategy_config", None)
    rerank_on = bool(
        sc
        and (
            str(sc.reranking.mode) != "none"
            or getattr(retrieval_pipeline, "reranker", None) is not None
        )
    )
    mmr_on = bool(sc and sc.post_processing.use_mmr)
    threshold_on = bool(sc and sc.post_processing.min_similarity_threshold is not None)
    features = replace(
        _features_from_config(eval_config),
        embedding_fusion_enabled=fusion_on,
        result_fusion_enabled=fusion_on
        and getattr(embedding_fusion_config, "level", "embedding") == "result",
        query_opt_enabled=_enabled(query_opt_config),
        query_opt_method=str(getattr(query_opt_config, "method", "rewrite")),
        correction_enabled=_enabled(query_correction_config),
        rerank_enabled=rerank_on,
        mmr_enabled=mmr_on,
        threshold_enabled=threshold_on,
        trace_enabled=trace_limit > 0,
    )

    specs = assemble_specs(mode, features)
    if graph_override and graph_override.get("branches"):
        return build_branched_graph(specs, graph_override["branches"], mode=mode)
    return build_graph_from_spec(specs, mode=mode)
