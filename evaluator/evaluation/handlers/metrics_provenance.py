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
    from ...llm.cost import COST

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
    return build_provenance(
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
    )


def _record_model_info(results: "Any", s: "Any") -> None:
    """Record model names (display) + audio<->text embedding alignment (metadata, not metrics).

    Driven by which pipelines actually ran, so every mode is covered — including
    ``audio_emb_retrieval`` (audio query + text corpus), which previously recorded nothing."""
    if s.asr_pipeline is not None:
        results["asr"] = s.asr_pipeline.model.name()
    if s.audio_embedding_pipeline is not None:
        results["audio_embedder"] = s.audio_embedding_pipeline.model.name()
    if s.text_embedding_pipeline is not None:
        # asr_text: the text embedder IS the query embedder → 'embedder'; otherwise it
        # embeds the corpus → 'text_embedder' (back-compat key names).
        key = "embedder" if is_asr_text_retrieval(s) else "text_embedder"
        results[key] = s.text_embedding_pipeline.model.name()
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
    fields that define the experiment — type/size/name/dim/dropout/model_path/adapter/
    embedding_space/params (pooling) + retrieval knobs — so a saved result is reproducible and
    the leaderboard can group/filter by them. Driven by the pipelines that ran. The model's
    ``.name()`` rides along under ``resolved`` for display."""
    from ...config.types import enum_to_str

    m = getattr(s.config, "model", None) if s.config is not None else None
    prov: Dict[str, Any] = {}
    if m is None:
        return prov

    def _clean(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if v not in (None, "", {}, [])}

    if s.asr_pipeline is not None:
        prov["asr"] = _clean({
            "type": m.asr_model_type, "size": m.asr_size, "name": m.asr_model_name,
            "adapter": m.asr_adapter_path, "params": dict(m.asr_params or {}),
            "resolved": s.asr_pipeline.model.name(),
        })
    if s.text_embedding_pipeline is not None:
        prov["text_emb"] = _clean({
            "type": m.text_emb_model_type, "size": m.text_emb_size,
            "name": m.text_emb_model_name,
            "adapter": m.text_emb_adapter_path, "embedding_space": m.text_emb_embedding_space,
            "params": dict(m.text_emb_params or {}),
            "resolved": s.text_embedding_pipeline.model.name(),
        })
    if s.audio_embedding_pipeline is not None:
        prov["audio_emb"] = _clean({
            "type": m.audio_emb_model_type, "size": m.audio_emb_size,
            "name": m.audio_emb_model_name,
            "dim": m.audio_emb_dim, "dropout": m.audio_emb_dropout,
            "model_path": m.audio_emb_model_path, "adapter": m.audio_emb_adapter_path,
            "embedding_space": m.audio_emb_embedding_space,
            "params": dict(m.audio_emb_params or {}),
            "resolved": s.audio_embedding_pipeline.model.name(),
        })
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
