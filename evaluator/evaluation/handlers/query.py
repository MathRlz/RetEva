"""Query-side stage handlers: correction, augmentation, LLM optimization.

Each handler
registers itself via ``@register_stage_handler`` at import time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm

from ...utils.progress import progress_disabled
from ..stage_registry import register_stage_handler
from ...logging_config import get_logger
from ...metrics.ir import report_depth
from ..executor.state import RunState
from ._common import publish_keyed_or_plain
from ...models.retrieval.query.optimization import (
    rewrite_query,
    generate_hypothetical_document,
    decompose_query,
    generate_multi_queries,
    combine_retrieval_results,
)

logger = get_logger(__name__)


def _augment_one_text(id_text, *, augmenter, base_seed, node_id):
    """One ``(id, text)`` → its deterministically-perturbed text. Top-level + picklable for the
    process backend."""
    from ..provenance import item_seed

    qid, text = id_text
    return augmenter.augment(text, seed=item_seed(base_seed, qid, node_id, 0))


def _node_correction_config(s: "RunState") -> Any:
    """The correction config for the current node — resolved before execution:
    the GLOBAL config with this node's params overlaid, so a branch that overrides only
    e.g. `method` keeps the global LLM backend. Every QueryCorrectionConfig field is
    overridable by construction (the allowlist is the dataclass), and the operator's
    discriminator fields ({op: correct, axis: query}) never overlay."""
    return s.resolved_config(default=s.query_correction_config)


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
    """Post-ASR query correction node (domain repair). Does NOT mutate ``query_text``:
    it publishes the corrected text under the distinct ``corrected_query_text`` artifact
    (plus a per-query ``correction_diff``), and downstream consumers (query optimization /
    text embedding) pick it up via QUERY_TEXT_CHAIN.

    Per-branch divergence (R2): a node's ``params`` build a transient ``QueryCorrectionConfig``
    so correction can be enabled on *one* branch only (the `corr` branch) while others no-op.
    """
    cfg = _node_correction_config(s)
    # Stash for the (separate) metrics node's `corrected_metrics` opt-in check — see
    # `evaluation/handlers/metrics.py:_corrected_metrics_enabled`.
    s.query_correction_resolved_config = cfg
    if cfg is None or not getattr(cfg, "enabled", False):
        return
    from ..query_correction import (
        correct_one_text_status,
        resolve_correction_client,
        correction_diff,
    )
    from ..executor.cpu_parallel import run_per_item

    items = s.input_items("query_text")
    texts = list(items.values) if items is not None else list(s.input("query_text"))
    # The llm client is built once and shared across items; the process backend can't pickle a
    # live client, so llm correction runs sync/thread.
    client = resolve_correction_client(cfg)
    results = run_per_item(s, correct_one_text_status, texts, config=cfg, client=client)
    corrected = [text for text, _n in results]
    fallbacks = sum(n for _text, n in results)
    if fallbacks:
        # Surfaced in report.provenance.correction_fallbacks — a dead LLM endpoint must
        # not read as "correction ran and changed nothing" (mirrors optimization_fallbacks).
        from ..executor.engine import _STAGE_TIMES_LOCK

        with _STAGE_TIMES_LOCK:
            s.correction_fallbacks += fallbacks
        logger.warning(
            "Query correction fell back to the ORIGINAL query for %d/%d items "
            "(the llm corrector failed — check the LLM endpoint)",
            fallbacks, len(results),
        )
    n_changed = sum(1 for a, b in zip(texts, corrected) if a != b)
    # Publish the corrected text under a DISTINCT name (no query_text mutation); downstream
    # reads it via QUERY_TEXT_CHAIN. Per-item identity rides the incoming ItemSet (M1d-2).
    ids = (
        [str(i) for i in items.ids]
        if items is not None
        else [str(i) for i in range(len(corrected))]
    )
    publish_keyed_or_plain(s, "corrected_query_text", corrected, ids)
    # Correction-diff artifact (C1): what the corrector changed, per query — the evidence
    # behind the asr-vs-asr+correction comparison.
    publish_keyed_or_plain(s, "correction_diff", correction_diff(texts, corrected), ids)
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
        corpus_items = s.keyed_items("corpus")  # corpus is always keyed
        if not isinstance(corpus_items, ItemSet):
            logger.warning("augmenter '%s' (docs axis): no corpus on the bus", node_id)
            return
        from ..executor.cpu_parallel import run_per_item

        perturbed = [
            dict(d) if isinstance(d, dict) else {"text": str(d)}
            for d in corpus_items.values
        ]
        old_texts = [str(d.get("text", "")) for d in perturbed]
        new_texts = run_per_item(
            s, _augment_one_text, list(zip(corpus_items.ids, old_texts)),
            augmenter=augmenter, base_seed=base_seed, node_id=node_id,
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

    from ..executor.cpu_parallel import run_per_item

    augmented = run_per_item(
        s, _augment_one_text, list(zip(ids, texts)),
        augmenter=augmenter, base_seed=base_seed, node_id=node_id,
    )
    n_changed = sum(1 for a, b in zip(texts, augmented) if a != b)
    # Distinct output (no query_text mutation); downstream reads QUERY_TEXT_CHAIN.
    publish_keyed_or_plain(s, "augmented_query_text", augmented, ids)
    logger.info(
        "augmenter '%s': %d/%d query texts perturbed",
        node_id,
        n_changed,
        len(augmented),
    )


def _optimize_one_text(q, *, fn, cfg):
    """Per-item query optimization (rewrite / HyDE) — the 4b ``parallel_map`` unit. A bad query
    falls back to its original (one failure never kills the map). Returns ``(text, ok)`` so the
    caller can count fallbacks — an unreachable LLM would otherwise yield a complete,
    plausible-looking report in which optimization silently did nothing. Top-level + picklable
    so the ``process`` backend can run it (which is also why the count is returned, not
    accumulated in shared state)."""
    try:
        return fn(q, cfg), True
    except Exception as exc:  # noqa: BLE001 — a bad query falls back to the original
        logger.warning("Query optimization failed for %r: %s", q[:80], exc)
        return q, False


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
    # rewrite/HyDE are I/O-bound LLM calls (thread is the right backend); the per-item unit falls
    # back to the original query on failure.
    from ..executor.cpu_parallel import run_per_item

    results = run_per_item(s, _optimize_one_text, texts, fn=fn, cfg=cfg)
    optimized = [text for text, _ok in results]
    fallbacks = sum(1 for _text, ok in results if not ok)
    publish_keyed_or_plain(s, "optimized_query_text", optimized, ids)
    if fallbacks:
        # Surfaced in report.provenance.optimization_fallbacks — a run whose LLM endpoint was
        # unreachable must not read as "optimization ran and changed nothing".
        # _SHARED field: lock the read-modify-write like engine.py does for stage_times.
        from ..executor.engine import _STAGE_TIMES_LOCK

        with _STAGE_TIMES_LOCK:
            s.optimization_fallbacks += fallbacks
        logger.warning(
            "Query optimization fell back to the ORIGINAL query for %d/%d items "
            "(the optimizer failed — check the LLM endpoint); the report's "
            "optimization arm is unoptimized for those items",
            fallbacks, len(results),
        )
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
    # Resolved per-node (global ⊕ this node's own params — llm backend/model/combine_strategy/…
    # can now diverge per multi_query_retrieval node, not just method/k); method always wins as
    # this node kind's own default/override, applied after resolution.
    base = _node_query_opt_config(s) or QueryOptimizationConfig(enabled=True, method=method)
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
    k = report_depth(s.node_params.get("k", s.k))
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
    """The optimization config for the current node — resolved before execution:
    the global config with this node's params overlaid (per-branch divergence — branch A
    rewrite vs branch B HyDE). Was a hand-rolled overlay with its own key list + casts."""
    return s.resolved_config(default=s.query_opt_config)


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
    from ...llm_client.parallel import map_completions

    params = s.node_params
    method = params.get("method", "rewrite_with_context")
    top_k = int(params.get("context_top_k", 3))
    cfg = s.query_opt_config
    if cfg is None:
        # The run only carries query_opt_config when `query_optimization.enabled` is true. The
        # fallback below defaults to OpenAI with an empty key, so every refine call fails, is
        # swallowed (optimization.py returns the original query) and the run LOOKS fine while
        # refining nothing. Say so instead.
        logger.warning(
            "query_refine: no query_optimization config for this run — falling back to "
            "library defaults (OpenAI endpoint). Set `query_optimization: {enabled: true}` "
            "so the node uses this run's LLM backend."
        )
        cfg = QueryOptimizationConfig(enabled=True)

    items = s.input_items("query_text")
    queries = list(items.values) if items is not None else list(s.input("query_text"))
    ids = (
        [str(i) for i in items.ids]
        if items is not None
        else [str(i) for i in range(len(queries))]
    )
    retrieved = list(s.get_artifact("retrieved", default=[]))

    def _one(idx: int) -> str:
        results = retrieved[idx] if idx < len(retrieved) else []
        doc_texts = _retrieved_doc_texts(results, top_k)
        return refine_query(q_list[idx], doc_texts, cfg, method=method, context_top_k=top_k)

    # One LLM call per query, so it gets the same concurrency treatment as generation and
    # judging — measured at 78s for 8 queries serially, the most expensive node in the run.
    q_list = queries
    refined = map_completions(
        range(len(queries)), _one,
        workers=getattr(cfg, "concurrency", 1),
        desc=f"Refining queries ({method})", unit="query",
    )
    publish_keyed_or_plain(s, "refined_query_text", refined, ids)
    # `changed` is the honest signal: a refine that silently no-ops (dead endpoint, unknown
    # method, empty context) still "reformulates" every query — into itself.
    changed = sum(1 for before, after in zip(queries, refined) if before != after)
    logger.info(
        "query_refine (%s): %d queries, %d rewritten, %d unchanged",
        method, len(refined), changed, len(refined) - changed,
    )
    if refined and changed == 0:
        logger.warning(
            "query_refine (%s): NO query was rewritten — check the LLM endpoint and that "
            "the retrieved context is non-empty (context_top_k=%d).", method, top_k,
        )
