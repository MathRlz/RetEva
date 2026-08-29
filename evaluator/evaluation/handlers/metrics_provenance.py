"""Provenance assembly for the metrics report (extracted from ``handlers/metrics.py``).

These builders turn the run's ``RunState`` (the pipelines that ran, their cache stats, the
LLM cost ledger, soft-CPU offload counters) into the report's provenance block + the
machine-readable model-identity record (F30/C6). The metrics core imports
``_run_provenance`` / ``_record_model_info``; the rest are its helpers. Pure read-from-state —
no scoring — so it lives apart from the report-assembly / typed-metric-node core.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ...logging_config import get_logger
from ._common import _ctx_first, is_asr_text_retrieval

logger = get_logger(__name__)


def _collect_cache_stats(s: "Any") -> Optional[Dict[str, Any]]:
    """Per-stage cache hit/miss counts for the provenance block (T3): which artifacts were
    reused vs recomputed. Best-effort — a pipeline without stats is just omitted."""
    out: Dict[str, Any] = {}
    for attr, label in (
        ("asr_pipeline", "asr"),
        ("text_embedding_pipeline", "text_embedding"),
        ("audio_embedding_pipeline", "audio_embedding"),
    ):
        pipe = getattr(s, attr, None)
        if pipe is not None and hasattr(pipe, "get_cache_stats"):
            try:
                stats = pipe.get_cache_stats()
                if stats:
                    out[label] = stats
            except Exception as exc:
                logger.debug("cache stats unavailable for %s: %s", label, exc)
    return out or None


def _llm_cost_summary() -> Optional[Dict[str, Any]]:
    """The run's accumulated LLM token/latency cost for the provenance block (T8)."""
    from ...llm_client.cost import COST

    return COST.summary()


def _offload_summary(s: "Any") -> Optional[Dict[str, int]]:
    """Soft-CPU offload event counters (2c), or None when nothing was parked warm — so the
    default full-free policy leaves the provenance block absent (parity-preserving)."""
    provider = getattr(s, "service_provider", None)
    stats = provider.offload_stats() if hasattr(provider, "offload_stats") else None
    return stats if stats and stats.get("soft_offloads") else None


def _run_provenance(s: "Any", dropped_by_branch: Optional[Dict] = None):
    """The run's provenance block (seed, dataset fingerprint, timing, drops, cache, LLM cost)."""
    from ..provenance import build_provenance, dataset_content_fingerprint

    seed = getattr(getattr(s.config, "audio_synthesis", None), "seed", None)
    prov = build_provenance(
        s.config,
        seed=seed,
        dataset=dataset_content_fingerprint(getattr(s, "dataset", None)),
        failure_analysis=s.drop_sink.failure_summary() or None,
        timing=dict(s.stage_times),
        dropped_by_branch=dropped_by_branch or None,
        dropped_by_node=dict(s.drop_sink.by_node) or None,
        cache_stats=_collect_cache_stats(s),
        cost=_llm_cost_summary(),
        offload=_offload_summary(s),
        models=_build_provenance(s),  # structured per-pipeline identity (F30/C6)
        optimization_fallbacks=getattr(s, "optimization_fallbacks", 0) or None,
        correction_fallbacks=getattr(s, "correction_fallbacks", 0) or None,
    )
    flow = getattr(s, "data_flow", None)
    if flow:
        # A3: which producer/alternative actually fed each input port — the OneOf-priority +
        # newest-published resolution is otherwise invisible in a saved run.
        prov["data_flow"] = {nid: dict(ports) for nid, ports in sorted(flow.items())}
        fired = [
            f"{nid}.{key} ← {e['producer']}:{e['artifact']}"
            for nid, ports in sorted(flow.items())
            for key, e in sorted(ports.items())
            if e.get("fallback")
        ]
        if fired:
            logger.warning(
                "data-flow fallbacks fired (a lower-priority producer served an input): %s",
                "; ".join(fired),
            )
    return prov


def _resolved_name(prov: Optional[Dict[str, Any]], fallback_pipeline: "Any") -> Optional[str]:
    """``resolved`` off a per-branch provenance artifact when one was published (a node that
    overrode its model for this branch), else the pipeline's own name (correct as-is for a
    node using the shared/global pipeline, which is never transiently swapped). Some configs
    run a stage with NO shared/global pipeline at all — every branch supplies its own
    override — so ``fallback_pipeline`` can be ``None`` here even though the stage genuinely
    ran; ``None`` back means the caller has nothing to record."""
    if prov and prov.get("resolved"):
        return prov["resolved"]
    return fallback_pipeline.model.name() if fallback_pipeline is not None else None


def _record_model_info(results: "Any", s: "Any") -> None:
    """Record model names (display) + audio<->text embedding alignment (metadata, not metrics).

    Driven by which pipelines actually ran, so every mode is covered — including
    ``audio_emb_retrieval`` (audio query + text corpus), which previously recorded nothing.

    A per-node model override (``params.model``/``params.name``) only lives on
    ``s.asr_pipeline`` / ``s.text_embedding_pipeline`` / ``s.audio_embedding_pipeline`` for
    the duration of that node's own handler (``_node_pipeline`` reverts it on exit) — this
    report node runs after every branch's nodes, so reading those attributes directly here
    always sees the last-restored value (the flat global default, or ``None`` for a stage
    with no global default at all — a config where every branch supplies its own override
    and the field would otherwise go missing entirely), not this branch's own.
    `sibling_artifact` recovers the branch-scoped identity that `_node_pipeline` published
    alongside its node's regular output; a node that never overrode its model has nothing
    published, so this falls back to the (already-correct, never-swapped) pipeline attribute
    unchanged — hence the lookup always runs first, the pipeline attribute only as fallback."""
    asr_prov = s.sibling_artifact("query_text", "asr_model_provenance")
    asr_name = _resolved_name(asr_prov, s.asr_pipeline)
    if asr_name:
        results["asr"] = asr_name

    audio_prov = s.sibling_artifact("retrieved", "query_audio_embedder_model_provenance")
    audio_name = _resolved_name(audio_prov, s.audio_embedding_pipeline)
    if audio_name:
        results["audio_embedder"] = audio_name

    # asr_text / plain text_retrieval: the text embedder IS the query embedder →
    # 'embedder'; audio_emb_retrieval: it embeds the corpus, audio is the query →
    # 'text_embedder' (back-compat key names). `retrieved`'s sibling is always safe to try
    # regardless of key: it resolves to whichever node actually fed retrieval's query
    # vectors, so in the corpus-role case it simply finds nothing (the corpus-embedding node
    # doesn't feed `retrieved`) and falls through to the old read.
    text_prov = s.sibling_artifact("retrieved", "query_text_embedder_model_provenance")
    text_name = _resolved_name(text_prov, s.text_embedding_pipeline)
    if text_name:
        results["embedder" if is_asr_text_retrieval(s) else "text_embedder"] = text_name
    # The embedding-alignment artifact is published only by the fusion node (audio_text);
    # its presence is the signal — no need to consult the mode.
    alignment = _ctx_first(s, "embedding_alignment")
    if alignment is not None:
        results["embedding_alignment"] = alignment
        logger.info(
            "Embedding alignment - cosine mean=%.4f std=%.4f",
            alignment["audio_text_cosine_mean"],
            alignment["audio_text_cosine_std"],
        )


def _build_provenance(s: "Any") -> Dict[str, Any]:
    """Machine-readable model identity for the report (F30/C6): the structured per-pipeline
    fields that define the experiment — type/size/name/dim/model_path/adapter/
    embedding_space/params (pooling) + retrieval knobs — so a saved result is reproducible and
    the leaderboard can group/filter by them. Driven by the pipelines that ran. The model's
    ``.name()`` rides along under ``resolved`` for display."""
    from ...config.types import enum_to_str

    m = getattr(s.config, "model", None) if s.config is not None else None
    prov: Dict[str, Any] = {}

    def _clean(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if v not in (None, "", {}, [])}

    def _flat_block(pipeline: "Any", fields: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """The run's flat global config block for a pipeline with no per-branch override.
        ``None`` when there's nothing to report: no run-level model config (``m``), or this
        stage has no shared/global pipeline at all (every branch supplies its own — see
        `_resolved_name`)."""
        if m is None or pipeline is None:
            return None
        d = {key: getattr(m, attr) for key, attr in fields.items() if key != "params"}
        d["params"] = dict(getattr(m, fields["params"]) or {})
        d["resolved"] = pipeline.model.name()
        return _clean(d)

    # A per-node override (`params.model`/`params.name`) publishes its OWN effective
    # type/size/name/params/resolved as a branch-scoped artifact (see `_node_pipeline` /
    # `_publish_node_model_provenance`) — reachable here via `sibling_artifact` — because
    # `m` above is the run's flat global config and can't see it; a node with no override
    # has nothing published, so this falls back to `m` unchanged (already correct there,
    # since no override means the branch used the flat default anyway — or, for a stage
    # with no global default at all, correctly reports nothing rather than crashing).
    asr_prov = s.sibling_artifact("query_text", "asr_model_provenance")
    asr_block = asr_prov or _flat_block(s.asr_pipeline, {
        "type": "asr_model_type", "size": "asr_size", "name": "asr_model_name",
        "adapter": "asr_adapter_path", "params": "asr_params",
    })
    if asr_block:
        prov["asr"] = asr_block

    # Safe regardless of role (see `_record_model_info`): resolves to nothing when
    # text_embedding_pipeline is acting as the corpus embedder, not retrieval's query.
    text_prov = s.sibling_artifact("retrieved", "query_text_embedder_model_provenance")
    text_block = text_prov or _flat_block(s.text_embedding_pipeline, {
        "type": "text_emb_model_type", "size": "text_emb_size", "name": "text_emb_model_name",
        "adapter": "text_emb_adapter_path", "model_path": "text_emb_model_path",
        "embedding_space": "text_emb_embedding_space", "params": "text_emb_params",
    })
    if text_block:
        prov["text_emb"] = text_block

    audio_prov = s.sibling_artifact("retrieved", "query_audio_embedder_model_provenance")
    audio_block = audio_prov or _flat_block(s.audio_embedding_pipeline, {
        "type": "audio_emb_model_type", "size": "audio_emb_size", "name": "audio_emb_model_name",
        "dim": "audio_emb_dim", "model_path": "audio_emb_model_path",
        "adapter": "audio_emb_adapter_path", "embedding_space": "audio_emb_embedding_space",
        "params": "audio_emb_params",
    })
    if audio_block:
        prov["audio_emb"] = audio_block
    vdb = getattr(s.config, "vector_db", None)
    if vdb is not None:
        reranker = (
            getattr(vdb, "reranker_model", None)
            if getattr(vdb, "reranker_enabled", False) else None
        )
        prov["retrieval"] = _clean({
            "store": enum_to_str(vdb.type) if getattr(vdb, "type", None) is not None else None,
            "k": getattr(vdb, "k", None),
            "mode": (enum_to_str(vdb.retrieval_mode)
                     if getattr(vdb, "retrieval_mode", None) is not None else None),
            "reranker": reranker,
        })
    return prov
