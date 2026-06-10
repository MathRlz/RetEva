"""Per-node model-override context managers (R1/D3).

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X4). When a branch
node names a model in its params, these swap the relevant pipeline (or reranker config)
on the run state transiently and restore it on exit, so per-node/per-branch model
divergence works without disturbing the shared global pipelines.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

from ...logging_config import get_logger
from .state import RunState

logger = get_logger(__name__)

_NODE_PIPELINE_ATTR = {
    "asr": "asr_pipeline",
    "text_embedding": "text_embedding_pipeline",
    "audio_embedding": "audio_embedding_pipeline",
}


@contextmanager
def _node_pipeline(s: "RunState", stage: str, params):
    """Per-node model override (R1, generalizes `_node_reranking`): when a branch node names
    a model (`params.model`/`params.name`) for ``stage``, build it (via the service provider,
    else the factory) and swap the relevant pipeline on ``s`` transiently; restore on exit.
    No model params → no-op (the shared global pipeline is used)."""
    params = params or {}
    attr = _NODE_PIPELINE_ATTR.get(stage)
    if attr is None or not (params.get("model") or params.get("name")):
        yield
        return
    provider = getattr(s, "service_provider", None)
    cm = getattr(s.config, "model", None)
    cache = s.cache_manager
    saved = getattr(s, attr)
    try:
        if stage == "asr":
            mtype = params.get("model") or getattr(cm, "asr_model_type", None) or ""
            name = params.get("name") or getattr(cm, "asr_model_name", None)
            device = params.get("device") or getattr(cm, "asr_device", "cuda:0")
            adapter = params.get("adapter") or getattr(cm, "asr_adapter_path", None)
            from ...pipeline.asr_pipeline import ASRPipeline

            if provider is not None:
                model = provider.get_asr_model(mtype, name, adapter, device)
            else:
                from ...models import create_asr_model

                model = create_asr_model(
                    model_type=mtype,
                    model_name=name,
                    adapter_path=adapter,
                    device=device,
                )
            s.asr_pipeline = ASRPipeline(model, cache)
        elif stage == "text_embedding":
            mtype = (
                params.get("model") or getattr(cm, "text_emb_model_type", None) or ""
            )
            name = params.get("name") or getattr(cm, "text_emb_model_name", None)
            device = params.get("device") or getattr(cm, "text_emb_device", "cuda:0")
            from ...pipeline.text_embedding_pipeline import TextEmbeddingPipeline

            if provider is not None:
                model = provider.get_text_embedding_model(mtype, name, device)
            else:
                from ...models import create_text_embedding_model

                model = create_text_embedding_model(
                    model_type=mtype, model_name=name, device=device
                )
            s.text_embedding_pipeline = TextEmbeddingPipeline(model, cache)
        elif stage == "audio_embedding":
            mtype = (
                params.get("model") or getattr(cm, "audio_emb_model_type", None) or ""
            )
            name = params.get("name") or getattr(cm, "audio_emb_model_name", None)
            device = params.get("device") or getattr(cm, "audio_emb_device", "cuda:0")
            from ...pipeline.audio_embedding_pipeline import AudioEmbeddingPipeline

            if provider is not None:
                model = provider.get_audio_embedding_model(
                    mtype,
                    name,
                    getattr(cm, "audio_emb_model_path", None),
                    getattr(cm, "audio_emb_dim", 2048),
                    getattr(cm, "audio_emb_dropout", 0.1),
                    device,
                )
            else:
                from ...models import create_audio_embedding_model

                model = create_audio_embedding_model(
                    model_type=mtype, model_name=name, device=device
                )
            s.audio_embedding_pipeline = AudioEmbeddingPipeline(model, cache)
        logger.info(
            "node '%s' per-instance model: type=%s name=%s",
            getattr(s.current_node, "id", "?"),
            params.get("model"),
            params.get("name"),
        )
        yield
    finally:
        setattr(s, attr, saved)


@contextmanager
def _node_reranking(rp, params, provider):
    """Temporarily apply a rerank node's per-instance config to the retrieval pipeline
    (D3): swap in its reranker (built from ``params.model`` via the provider/factory) +
    reranking settings, restore on exit. No params → use the pipeline's global reranker.
    """
    if not params:
        yield
        return
    saved_reranker = rp.reranker
    saved_strategy = rp.strategy_config
    try:
        rk = saved_strategy.reranking
        mode = params.get("mode") or (
            "cross_encoder" if params.get("model") else rk.mode
        )
        rp.strategy_config = replace(
            saved_strategy,
            reranking=replace(
                rk,
                mode=mode,
                top_k=params.get("top_k", rk.top_k),
                weight=params.get("weight", rk.weight),
            ),
        )
        if mode == "cross_encoder" and params.get("model"):
            model_name = params["model"]
            device = params.get("device")
            if provider is not None:
                rp.reranker = provider.get_reranker(
                    model_name=model_name, device=device
                )
            else:
                from ...models import create_reranker

                rp.reranker = create_reranker(
                    model_type="cross_encoder", model_name=model_name, device=device
                )
        elif mode != "cross_encoder":
            rp.reranker = None  # token_overlap / none — lexical, no model
        logger.info(
            "rerank node using per-instance config: mode=%s model=%s",
            mode,
            params.get("model"),
        )
        yield
    finally:
        rp.reranker = saved_reranker
        rp.strategy_config = saved_strategy
