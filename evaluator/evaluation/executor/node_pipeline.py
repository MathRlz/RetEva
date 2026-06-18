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
    saved = getattr(s, attr)
    try:
        setattr(s, attr, _build_node_pipeline(s, stage, params))
        logger.info(
            "node '%s' per-instance model: type=%s name=%s",
            getattr(s.current_node, "id", "?"),
            params.get("model"),
            params.get("name"),
        )
        yield
    finally:
        setattr(s, attr, saved)


_BUILDER_METHOD = {"asr": "asr", "text_embedding": "text_emb", "audio_embedding": "audio_emb"}


def _build_node_pipeline(s: "RunState", stage: str, params: dict):
    """Build a per-node pipeline from the node's params overlaid on the global model config.

    Honors the FULL model param set (``model_path`` / ``dim`` / ``dropout`` / ``pooling`` (via
    ``params``) / ``embedding_space`` / ``quantization`` / ``size``) by overlaying the node's
    params onto a copy of ``config.model`` and building through the **factory branch** of
    ``_ModelBuilders`` (``service_provider=None``) — the provider's ``get_*`` signatures omit
    several of these, so a per-node override must build standalone (it builds once per run, so
    forgoing model-reuse is fine)."""
    from types import SimpleNamespace

    from ...config.graph_config import _MODEL_NODE_FIELDS
    from ...pipeline.factory import _ModelBuilders

    node_to_field = _MODEL_NODE_FIELDS.get(stage, {})
    overrides = {field: params[key] for key, field in node_to_field.items() if key in params}
    eff_model = replace(s.config.model, **overrides)
    model = getattr(
        _ModelBuilders(SimpleNamespace(model=eff_model), None, None), _BUILDER_METHOD[stage]
    )()
    cache = s.cache_manager
    if stage == "asr":
        from ...pipeline.asr_pipeline import ASRPipeline

        return ASRPipeline(model, cache)
    if stage == "text_embedding":
        from ...pipeline.text_embedding_pipeline import TextEmbeddingPipeline

        return TextEmbeddingPipeline(model, cache)
    from ...pipeline.audio_embedding_pipeline import AudioEmbeddingPipeline

    return AudioEmbeddingPipeline(model, cache)


@contextmanager
def _node_reranking(rp, params, provider):
    """Temporarily apply a rerank node's per-instance config to the retrieval pipeline
    (D3): swap in its reranker (built from ``params.model`` via the provider/factory) +
    reranking settings, restore on exit. No params → use the pipeline's global reranker.
    """
    # Only an actual per-node reranker override triggers the swap — not the operator
    # discriminator fields the alias injects (e.g. {op: rerank}), which would otherwise
    # fire a no-op rebuild and (in tests) touch a stub pipeline's missing reranker.
    if not (params and any(k in params for k in ("model", "mode", "top_k", "weight"))):
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
