"""Per-node model-override context managers.

When a branch
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

# The swap this context manager performs is transient — `s.<attr>` reverts the moment the
# node's own handler returns, so anything reading `s.asr_pipeline` / `s.text_embedding_pipeline`
# later in the run (the report/provenance assembler, which runs after every branch's nodes)
# only ever sees the last-restored value, not this branch's override. Publishing the resolved
# identity as a node artifact here — while the override is live — lets those later readers
# fetch it back through the bus (`sibling_artifact`), which stays branch-scoped.
_PROVENANCE_ARTIFACT = {
    "asr": "asr_model_provenance",
    "text_embedding": "text_embedding_model_provenance",
    "audio_embedding": "audio_embedding_model_provenance",
}
_PROVENANCE_FIELDS = {
    "asr": {
        "type": "asr_model_type", "size": "asr_size", "name": "asr_model_name",
        "adapter": "asr_adapter_path", "params": "asr_params",
    },
    "text_embedding": {
        "type": "text_emb_model_type", "size": "text_emb_size",
        "name": "text_emb_model_name", "adapter": "text_emb_adapter_path",
        "model_path": "text_emb_model_path", "embedding_space": "text_emb_embedding_space",
        "params": "text_emb_params",
    },
    "audio_embedding": {
        "type": "audio_emb_model_type", "size": "audio_emb_size",
        "name": "audio_emb_model_name", "dim": "audio_emb_dim",
        "model_path": "audio_emb_model_path", "adapter": "audio_emb_adapter_path",
        "embedding_space": "audio_emb_embedding_space", "params": "audio_emb_params",
    },
}


def _publish_node_model_provenance(s: "RunState", stage: str, pipeline, eff_model) -> None:
    """Record ``stage``'s effective per-node model identity as an artifact of the
    currently-running node, so a later branch-scoped reader (report/provenance) can recover
    it via `sibling_artifact` instead of the transient, already-reverted `s.<attr>`."""
    fields = _PROVENANCE_FIELDS.get(stage)
    if fields is None or pipeline is None:
        return
    prov = {
        key: getattr(eff_model, attr, None) for key, attr in fields.items() if key != "params"
    }
    prov["params"] = dict(getattr(eff_model, fields["params"], None) or {})
    prov = {k: v for k, v in prov.items() if v not in (None, "", {}, [])}
    prov["resolved"] = pipeline.model.name()
    s.put_artifact(_PROVENANCE_ARTIFACT[stage], prov)


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
    pipeline = None
    try:
        from ...config.graph_config import resolved_model_config

        eff_model = resolved_model_config(s.config, s.current_node)
        pipeline = _build_node_pipeline(s, stage, params)
        setattr(s, attr, pipeline)
        _publish_node_model_provenance(s, stage, pipeline, eff_model)
        logger.info(
            "node '%s' per-instance model: type=%s name=%s",
            getattr(s.current_node, "id", "?"),
            params.get("model"),
            params.get("name"),
        )
        yield
    finally:
        setattr(s, attr, saved)
        if pipeline is not None and getattr(s, "aggressive_offload", False):
            _release_node_model(s, pipeline)


def _release_node_model(s: "RunState", pipeline) -> None:
    """Aggressive lifecycle (`on_use_soft_cpu`): park the node's model on CPU the moment
    its node finishes and return its GPU-pool reservation, so the GPU only ever holds the
    actively-executing node's models. The provider keeps the instance warm (bounded LRU),
    so a later node using the same model reactivates it with a CPU→GPU move instead of a
    reload; the pool key travels on the model (``_gpu_pool_key``, set at build)."""
    model = getattr(pipeline, "model", None)
    if model is None:
        return
    try:
        if s.service_provider is not None:
            s.service_provider.release_model_instance(model, soft_cpu=True)
        pool = getattr(s, "device_pool", None)
        key = getattr(model, "_gpu_pool_key", None)
        if pool is not None and key is not None:
            pool.release(key)
        from ...devices.memory import get_memory_manager

        get_memory_manager().clear_gpu_cache()
        logger.info(
            "aggressive offload: node '%s' model parked on cpu",
            getattr(s.current_node, "id", "?"),
        )
    except Exception as exc:  # noqa: BLE001 - offload must never break the run
        logger.warning("aggressive offload failed: %s", exc)


_BUILDER_METHOD = {"asr": "asr", "text_embedding": "text_emb", "audio_embedding": "audio_emb"}


def _build_node_pipeline(s: "RunState", stage: str, params: dict):
    """Build a per-node pipeline from the node's params overlaid on the global model config.

    Honors the FULL model param set (``model_path`` / ``dim`` / ``pooling`` (via
    ``params``) / ``embedding_space`` / ``quantization`` / ``size``) by overlaying the node's
    params onto a copy of ``config.model`` and building through ``_ModelBuilders`` — routed via
    the run's shared ``service_provider`` (same as ``_node_reranking``) so two nodes that resolve
    to the SAME model (type/name/device/params) share one loaded instance instead of each paying
    for its own load; a node with a genuinely different override still gets its own cache key.
    Also threads the run's ``device_pool`` (was hardcoded ``None`` — a multi-variant graph's
    per-node model overrides never went through memory-aware allocation/eviction at all,
    regardless of a configured ``device_pool:``) so a per-node override is packed/evicted the
    same way the shared global pipeline already was."""
    from types import SimpleNamespace

    from ...config.graph_config import resolved_model_config
    from ...pipeline.factory import _ModelBuilders

    node = SimpleNamespace(stage=stage, params=params)
    eff_model = resolved_model_config(s.config, node)
    model = getattr(
        _ModelBuilders(
            SimpleNamespace(model=eff_model), s.service_provider, getattr(s, "device_pool", None),
        ),
        _BUILDER_METHOD[stage],
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
