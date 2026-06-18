"""Query-side stage handlers: correction, augmentation, LLM optimization.

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X7). Each handler
registers itself via ``@register_stage_handler`` at import time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm

from ...utils.progress import progress_disabled
from ..stage_registry import register_stage_handler
from ...logging_config import get_logger
from ..executor.state import RunState
from ...models.retrieval.query.optimization import (
    rewrite_query,
    generate_hypothetical_document,
    decompose_query,
    generate_multi_queries,
    combine_retrieval_results,
)

logger = get_logger(__name__)


def _augment_one_text(id_text, *, augmenter, base_seed, node_id):
    """One ``(id, text)`` → its deterministically-perturbed text. Module-level + picklable so it is
    the per-item unit for the 4b ``parallel_map`` (sync default → byte-identical to the serial
    comprehension; thread/process parallelize a heavy custom augmenter)."""
    from ..provenance import item_seed

    qid, text = id_text
    return augmenter.augment(text, seed=item_seed(base_seed, qid, node_id, 0))


def _node_correction_config(s: "RunState") -> Any:
    """The correction config for the current node: a transient `QueryCorrectionConfig` built
    from `current_node.params` (per-branch divergence, R2) when present, else the global one.
    """
    params = s.node_params
    # Only a real per-node override builds a transient config — not the operator
    # discriminator fields the alias injects ({op: correct, axis: query}), which would
    # otherwise mask the global config with defaults (parity break).
    _keys = ("enabled", "method", "replacements", "use_default_rules", "kb_terms",
             "kb_max_distance")
    if not (params and any(k in params for k in _keys)):
        return s.query_correction_config
    from ...config.query_correction import QueryCorrectionConfig

    return QueryCorrectionConfig(
        enabled=bool(params.get("enabled", True)),
        method=params.get("method", "rule"),
        replacements=params.get("replacements", {}) or {},
        use_default_rules=bool(params.get("use_default_rules", True)),
        kb_terms=params.get("kb_terms", []) or [],
        kb_max_distance=int(params.get("kb_max_distance", 1)),
    )


@register_stage_handler("transform", time_key="query_opt_s")
def _stage_transform(s: "RunState") -> None:
    """The ``transform`` operator (type-preserving X→X): dispatch by op to query
    correction / optimization / refine / text-augment / audio-augment. Bodies unchanged;
    augment_audio lives in the audio handlers."""
    from .audio import _stage_augment_audio
    from ._dispatch import dispatch_operator

    return dispatch_operator("transform", {
        "query_correction": _stage_query_correction,
        "query_optimization": _stage_query_optimization,
        "query_refine": _stage_query_refine,
        "augmenter": _stage_augmenter,
        "augment_audio": _stage_augment_audio,
    }, s)


def _stage_query_correction(s: "RunState") -> None:
    """Post-ASR query correction node (rule-based domain repair). Rewrites the query
    hypotheses in place and republishes ``query_text`` so the next consumer (query
    optimization / text embedding) reads the corrected text.

    Per-branch divergence (R2): a node's ``params`` build a transient ``QueryCorrectionConfig``
    so correction can be enabled on *one* branch only (the `corr` branch) while others no-op.
    """
    cfg = _node_correction_config(s)
    if cfg is None or not getattr(cfg, "enabled", False):
        return
    import functools

    from ..query_correction import (
        correct_one_text,
        resolve_correction_client,
        correction_diff,
    )
    from ..executor.cpu_parallel import parallel_map, resolve_cpu_backend
    from ..item_set import ItemSet

    items = s.input_items("query_text")
    texts = list(items.values) if items is not None else list(s.input("query_text"))
    # Per-item correction through the 4b parallel_map (sync default → byte-identical to the batch
    # call). The llm method's client is built once and shared across items; the process backend
    # can't pickle a live client, so llm correction runs sync/thread (it's I/O-bound anyway).
    client = resolve_correction_client(cfg)
    backend, workers = resolve_cpu_backend(s.config)
    corrected = parallel_map(
        functools.partial(correct_one_text, config=cfg, client=client),
        texts,
        backend=backend, workers=workers,
    )
    n_changed = sum(1 for a, b in zip(texts, corrected) if a != b)
    # Publish the corrected text under a DISTINCT name (no query_text mutation); downstream
    # reads it via QUERY_TEXT_CHAIN. Per-item identity rides the incoming ItemSet (M1d-2).
    ids = (
        [str(i) for i in items.ids]
        if items is not None
        else [str(i) for i in range(len(corrected))]
    )
    if len(ids) == len(corrected) and len(set(ids)) == len(ids):
        s.put_items("corrected_query_text", ItemSet(ids, corrected))
        # Correction-diff artifact (C1): what the corrector changed, per query — the evidence
        # behind the asr-vs-asr+correction comparison.
        s.put_items("correction_diff", ItemSet(ids, correction_diff(texts, corrected)))
    else:
        s.put_artifact("corrected_query_text", corrected)
        s.put_artifact("correction_diff", correction_diff(texts, corrected))
    logger.info(
        "Query correction complete: %d/%d texts changed", n_changed, len(corrected)
    )


def _stage_augmenter(s: "RunState") -> None:
    """Robustness perturbation node (C2): corrupt each query text deterministically (seeded per
    item) — ASR-confusion homophones + dangerous dose/unit swaps — then republish ``query_text``
    so downstream embeds the corrupted query. A branch-divergence source: one branch clean, one
    augmented, and the cross-branch delta quantifies robustness. Params (per node/branch) drive
    which perturbations apply."""
    from ...pipeline.text_augmentation import TextAugmentConfig, TextAugmenter
    from ..item_set import ItemSet
    from ..provenance import DEFAULT_SEED

    params = s.node_params
    node_id = getattr(s.current_node, "id", "augmenter")
    cfg = TextAugmentConfig(
        homophones=bool(params.get("homophones", True)),
        unit_corruption=bool(params.get("unit_corruption", True)),
        char_swap_prob=float(params.get("char_swap_prob", 0.0)),
        max_edits=int(params.get("max_edits", 2)),
    )
    augmenter = TextAugmenter(cfg)
    seed = getattr(getattr(s.config, "audio_synthesis", None), "seed", None)
    base_seed = int(seed) if seed is not None else DEFAULT_SEED

    if params.get("axis") == "docs":
        # Corpus axis (§4.1 T2): perturb each DOCUMENT's text, republish `corpus` —
        # downstream corpus_embedding reads the newest producer. Same per-item
        # determinism, seeded by doc_id.
        corpus_items = s.get_items("corpus", default=None)
        if not isinstance(corpus_items, ItemSet):
            logger.warning("augmenter '%s' (docs axis): no corpus on the bus", node_id)
            return
        import functools

        from ..executor.cpu_parallel import parallel_map, resolve_cpu_backend

        perturbed = [
            dict(d) if isinstance(d, dict) else {"text": str(d)}
            for d in corpus_items.values
        ]
        old_texts = [str(d.get("text", "")) for d in perturbed]
        backend, workers = resolve_cpu_backend(s.config)
        new_texts = parallel_map(
            functools.partial(
                _augment_one_text, augmenter=augmenter, base_seed=base_seed, node_id=node_id
            ),
            list(zip(corpus_items.ids, old_texts)),
            backend=backend, workers=workers,
        )
        n_changed = sum(1 for o, n in zip(old_texts, new_texts) if o != n)
        for d, new_text in zip(perturbed, new_texts):
            d["text"] = new_text
        s.put_items("corpus", ItemSet(list(corpus_items.ids), perturbed))
        logger.info(
            "augmenter '%s' (docs axis): %d/%d corpus docs perturbed",
            node_id,
            n_changed,
            len(perturbed),
        )
        return

    items = s.input_items("query_text")
    if isinstance(items, ItemSet):
        ids, texts = items.ids, items.values
    else:
        texts = list(s.input("query_text"))
        ids = [str(i) for i in range(len(texts))]

    import functools

    from ..executor.cpu_parallel import parallel_map, resolve_cpu_backend

    backend, workers = resolve_cpu_backend(s.config)
    augmented = parallel_map(
        functools.partial(
            _augment_one_text, augmenter=augmenter, base_seed=base_seed, node_id=node_id
        ),
        list(zip(ids, texts)),
        backend=backend, workers=workers,
    )
    n_changed = sum(1 for a, b in zip(texts, augmented) if a != b)
    # Distinct output (no query_text mutation); downstream reads QUERY_TEXT_CHAIN.
    if len(ids) == len(augmented) and len(set(ids)) == len(ids):
        s.put_items("augmented_query_text", ItemSet(ids, augmented))
    else:
        s.put_artifact("augmented_query_text", augmented)
    logger.info(
        "augmenter '%s': %d/%d query texts perturbed",
        node_id,
        n_changed,
        len(augmented),
    )


def _optimize_one_text(q, *, fn, cfg):
    """Per-item query optimization (rewrite / HyDE) — the 4b ``parallel_map`` unit. A bad query
    falls back to its original (one failure never kills the map). Top-level + picklable so the
    ``process`` backend can run it."""
    try:
        return fn(q, cfg)
    except Exception as exc:  # noqa: BLE001 — a bad query falls back to the original
        logger.warning("Query optimization failed for %r: %s", q[:80], exc)
        return q


def _stage_query_optimization(s: "RunState") -> None:
    """Pure pre-retrieval query optimization (rewrite / HyDE): query text → improved text
    (``optimized_query_text``). It does NOT retrieve — the fan-out methods (decompose /
    multi_query) live in the explicit ``multi_query_retrieval`` node."""
    cfg = _node_query_opt_config(s)
    if cfg is None or cfg.method not in ("rewrite", "hyde"):
        return
    fn = rewrite_query if cfg.method == "rewrite" else generate_hypothetical_document
    items = s.input_items("query_text")
    texts = list(items.values) if items is not None else list(s.input("query_text"))
    ids = (
        [str(i) for i in items.ids]
        if items is not None
        else [str(i) for i in range(len(texts))]
    )
    s.cb("phase_1_5_query_opt", 0, s.total, f"Phase 1.5: Query optimization ({cfg.method})")
    # Per-item optimization through the 4b parallel_map (sync default → byte-identical to the serial
    # loop; rewrite/HyDE are I/O-bound LLM calls, so thread is the right backend). The per-item unit
    # falls back to the original query on failure.
    import functools

    from ..executor.cpu_parallel import parallel_map, resolve_cpu_backend

    backend, workers = resolve_cpu_backend(s.config)
    optimized = parallel_map(
        functools.partial(_optimize_one_text, fn=fn, cfg=cfg),
        texts,
        backend=backend, workers=workers,
    )
    from .asr import _publish_keyed_or_plain

    _publish_keyed_or_plain(s, "optimized_query_text", optimized, ids)
    logger.info("Query optimization complete: %d queries transformed", len(optimized))


def _stage_multi_query_retrieval(s: "RunState") -> None:
    """Composite retrieval strategy (RAG-fusion): expand each query into sub-queries
    (decompose / multi_query), embed + retrieve each, and fuse the result sets. The fan-out
    count is runtime-variable per query, so it is an explicit composite node rather than
    static DAG instances — but it is honestly a ``retrieved`` producer (no bypass flag)."""
    from dataclasses import replace
    from ...config.query_optimization import QueryOptimizationConfig
    from .retrieval import _publish_retrieved

    method = s.node_params.get("method", "multi_query")
    base = s.query_opt_config or QueryOptimizationConfig(enabled=True, method=method)
    cfg = replace(base, enabled=True, method=method)
    rp = s.get_artifact("vector_index", default=s.retrieval_pipeline)
    tep = s.text_embedding_pipeline
    if tep is None or rp is None:
        logger.warning(
            "multi_query_retrieval (%s) needs text_embedding + retrieval — skipping", method
        )
        return
    expand = decompose_query if method == "decompose" else generate_multi_queries
    items = s.input_items("query_text")
    texts = list(items.values) if items is not None else list(s.input("query_text"))
    ids = (
        [str(i) for i in items.ids]
        if items is not None
        else [str(i) for i in range(len(texts))]
    )
    k = int(s.node_params.get("k", s.k))
    results_with_scores = []
    for q in tqdm(texts, desc=f"multi_query_retrieval ({method})", disable=progress_disabled()):
        try:
            sub_qs = expand(q, cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query expansion failed for %r: %s", q[:80], exc)
            sub_qs = [q]
        sub_embs = np.array(tep.process_batch(sub_qs))
        sub_results = rp.search_batch(sub_embs, k, query_texts=sub_qs)
        results_with_scores.append(
            combine_retrieval_results(sub_results, strategy=cfg.combine_strategy, k=k)
        )
    _publish_retrieved(s, results_with_scores, ids)
    logger.info(
        "multi_query_retrieval (%s): expanded + retrieved %d queries",
        method,
        len(results_with_scores),
    )


def _node_query_opt_config(s: "RunState") -> Any:
    """The optimization config for the current node: the global config with this
    node's params overlaid transiently (per-branch divergence — branch A rewrite vs
    branch B HyDE; precedent ``_node_correction_config``). No params → global."""
    base = s.query_opt_config
    params = s.node_params
    overlay = {
        k: v
        for k, v in params.items()
        if k in ("method", "temperature", "max_iterations", "combine_strategy",
                 "context_top_k", "use_initial_context")
        and v not in (None, "")
    }
    if base is None or not overlay:
        return base
    from dataclasses import replace

    if "temperature" in overlay:
        overlay["temperature"] = float(overlay["temperature"])
    for int_key in ("max_iterations", "context_top_k"):
        if int_key in overlay:
            overlay[int_key] = int(overlay[int_key])
    if "use_initial_context" in overlay:
        overlay["use_initial_context"] = bool(overlay["use_initial_context"])
    return replace(base, **overlay)


def _retrieved_doc_texts(results: Any, top_k: int) -> list:
    """Top-k document texts from a query's ``(payload, score)`` retrieval results — the
    context fed to query refinement."""
    texts = []
    for payload, _score in list(results or [])[:top_k]:
        if isinstance(payload, dict):
            texts.append(
                str(payload.get("text") or payload.get("content") or payload.get("doc_id") or "")
            )
        else:
            texts.append(str(payload))
    return [t for t in texts if t]


def _stage_query_refine(s: "RunState") -> None:
    """Post-retrieval query reformulation (the iterative-RAG refine step): read the current
    query + the retrieved docs and emit ``refined_query_text`` (top of the query-text chain),
    so a later hop's text_embedding embeds the improved query. Pure text transform — does NOT
    retrieve."""
    from ...models.retrieval.query.optimization import refine_query
    from ...config.query_optimization import QueryOptimizationConfig
    from .asr import _publish_keyed_or_plain

    params = s.node_params
    method = params.get("method", "rewrite_with_context")
    top_k = int(params.get("context_top_k", 3))
    cfg = s.query_opt_config or QueryOptimizationConfig(enabled=True)

    items = s.input_items("query_text")
    queries = list(items.values) if items is not None else list(s.input("query_text"))
    ids = (
        [str(i) for i in items.ids]
        if items is not None
        else [str(i) for i in range(len(queries))]
    )
    retrieved = list(s.get_artifact("retrieved", default=[]))
    refined = []
    for idx, q in enumerate(queries):
        results = retrieved[idx] if idx < len(retrieved) else []
        doc_texts = _retrieved_doc_texts(results, top_k)
        refined.append(
            refine_query(q, doc_texts, cfg, method=method, context_top_k=top_k)
        )
    _publish_keyed_or_plain(s, "refined_query_text", refined, ids)
    logger.info("query_refine (%s): reformulated %d queries", method, len(refined))
