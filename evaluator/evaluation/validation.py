"""Pre-flight validation: every check that must fail BEFORE any model loads.

One home for the chain that used to live inline in ``_run_core`` (with per-call
local imports working around layering) — determinism seeding, LLM cost budget,
embedding-space typing (config-level and per-node graph-level), and optional
store-backend availability. The embedding-space validators moved here from
``models/embedding_space.py``: they are *validation*, not model construction
(``resolve_embedding_space`` stays in models/ — a registry concern).

(This module previously held a dead ``validate_pubmed_dataset`` helper with zero
callers; replaced wholesale.)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..errors import ConfigurationError
from ..logging_config import get_logger
from ..models.embedding_space import resolve_embedding_space

logger = get_logger(__name__)

# Retrieval modes that compare vectors via inner product (so spaces must match).
_VECTOR_MODES = {"dense", "hybrid"}


def run_pre_flight(config: Any) -> Dict[str, Any]:
    """Run the full pre-flight chain; returns the determinism flags for provenance.

    Order matters: determinism first (everything downstream may consume RNGs),
    then the cheap config-level checks, then the graph-level ones.
    """
    from ..llm.cost import COST
    from ..pipeline.factory import check_graph_backend_dependencies
    from ..pipeline.stage_graph import build_graph_for_config
    from .provenance import set_global_determinism

    run_seed = getattr(getattr(config, "audio_synthesis", None), "seed", None)
    flags = set_global_determinism(run_seed)
    logger.info("determinism: %s", flags)

    budget = int(getattr(getattr(config, "llm", None), "max_tokens_budget", 0) or 0)
    COST.reset(budget_tokens=budget or None)

    validate_embedding_spaces(config)
    check_graph_backend_dependencies(config)

    try:
        graph = build_graph_for_config(config)
    except Exception:
        graph = None  # graph errors surface in their own validation path
    if graph is not None:
        validate_graph_embedding_spaces(graph, config)
    return flags


# ── Embedding-space typing (architecture A2 / §4.1 P1) ────────────────


def _resolve_space(
    registry: Any, model_type: Any, model_name: Optional[str], override: Optional[str]
) -> str:
    """Embedding-space id, honoring a per-instance ``embedding_space`` override (config field
    or node param) over the type/name-derived default — e.g. an APM trained to project audio
    into a text embedder's space declares that text space instead of its encoder name."""
    if override:
        return str(override)
    return resolve_embedding_space(registry, str(model_type), model_name)


def validate_embedding_spaces(config: Any) -> None:
    """Raise ``ConfigurationError`` if a dense/hybrid retrieval would dot vectors from
    different embedding spaces (query-side vs corpus-side). No-op for non-vector retrieval,
    asr_text (query embedder == corpus embedder), or fusion (combined space, not checked).

    Today the corpus is text-embedded for cross-modal retrieval, so the check that matters is
    ``audio_emb_retrieval``: the audio query embedder must share a space with the text corpus
    embedder.
    """
    from ..models.registry import audio_embedding_registry, text_embedding_registry

    model = getattr(config, "model", None)
    if model is None:
        return
    mode = str(getattr(model, "pipeline_mode", ""))
    vdb = getattr(config, "vector_db", None)
    retrieval_mode = str(getattr(vdb, "retrieval_mode", "dense")) if vdb else "dense"
    if retrieval_mode not in _VECTOR_MODES:
        return

    text_type = getattr(model, "text_emb_model_type", None)
    if mode == "audio_emb_retrieval":
        audio_type = getattr(model, "audio_emb_model_type", None)
        if not audio_type or not text_type:
            return  # not enough info to compare
        query_space = _resolve_space(
            audio_embedding_registry,
            audio_type,
            getattr(model, "audio_emb_model_name", None),
            getattr(model, "audio_emb_embedding_space", None),
        )
        corpus_space = _resolve_space(
            text_embedding_registry,
            text_type,
            getattr(model, "text_emb_model_name", None),
            getattr(model, "text_emb_embedding_space", None),
        )
        if query_space != corpus_space:
            raise ConfigurationError(
                f"Embedding-space mismatch for dense '{mode}': audio query embedder "
                f"'{audio_type}' is in space '{query_space}' but the text corpus embedder "
                f"'{text_type}' is in space '{corpus_space}'. Dense retrieval dots these "
                f"vectors, so the scores would be meaningless. Use embedders that share a "
                f"space (e.g. sonar_speech + sonar); for an APM trained to align to this text "
                f"embedder, set the SAME `embedding_space:` on both nodes (audio + text) to "
                f"declare the shared space; or switch to sparse retrieval."
            )
    # asr_text_retrieval: query and corpus both use text_emb → same space, always OK.
    # audio_text_retrieval (fusion): query is a fused vector — space combination is not
    # validated here (revisit with n-ary fusion).


def _node_space(node: Any, config: Any) -> Optional[str]:
    """The embedding space a graph node instance produces vectors in, or None.

    Per-node ``params.model``/``name`` win over the global model fields; a
    dataset_source vector column carries ``params.embedding_space`` directly;
    a fusion node's combined space is unknowable here (None = unchecked).
    """
    from ..models.registry import audio_embedding_registry, text_embedding_registry
    from ..pipeline.graph.operators import node_kind

    params = getattr(node, "params", None) or {}
    model = getattr(config, "model", None)
    stage = node_kind(node.stage, params)
    if stage == "dataset_source":
        return params.get("embedding_space")
    if stage in ("text_embedding", "corpus_embedding"):
        mtype = params.get("model") or getattr(model, "text_emb_model_type", None)
        if not mtype:
            return None
        return _resolve_space(
            text_embedding_registry,
            mtype,
            params.get("name") or getattr(model, "text_emb_model_name", None),
            params.get("embedding_space") or getattr(model, "text_emb_embedding_space", None),
        )
    if stage == "audio_embedding":
        mtype = params.get("model") or getattr(model, "audio_emb_model_type", None)
        if not mtype:
            return None
        return _resolve_space(
            audio_embedding_registry,
            mtype,
            params.get("name") or getattr(model, "audio_emb_model_name", None),
            params.get("embedding_space") or getattr(model, "audio_emb_embedding_space", None),
        )
    return None


def _producer_space(
    node_id: str, artifact: str, by_id: dict, config: Any, depth: int = 0
) -> Optional[str]:
    """Space of the vectors flowing out of ``node_id`` for ``artifact`` (follows
    pass-through nodes: vector_db ← corpus_vectors, corpus_merge ← its producers)."""
    if depth > 16:  # defensive: graphs are small, cycles already rejected
        return None
    node = by_id.get(node_id)
    if node is None:
        return None
    direct = _node_space(node, config)
    if direct is not None:
        return direct
    from ..pipeline.graph.operators import node_kind

    if node_kind(node.stage, node.params) in ("vector_db", "corpus_merge"):
        upstream = [p for a, p in node.bindings if a == "corpus_vectors"]
        spaces = {
            s
            for s in (
                _producer_space(p, "corpus_vectors", by_id, config, depth + 1)
                for p in upstream
            )
            if s is not None
        }
        return next(iter(spaces)) if len(spaces) == 1 else None
    return None


def validate_graph_embedding_spaces(graph: Any, config: Any) -> None:
    """Per-node V[s] check: reject a dense/hybrid retrieval whose bound query-vector
    producer and index chain live in different embedding spaces (§4.1 P1).

    Catches what the config-level check cannot: explicit graphs whose per-node
    ``params.model`` overrides diverge (e.g. ``corpus_embedding {model: labse}`` +
    ``text_embedding {model: jina_v4}``) and dataset vector columns whose declared
    ``embedding_space`` differs from the query embedder. Unresolvable spaces (fusion,
    no model info) stay unchecked — this is a trap-closer, not a prover.
    """
    vdb = getattr(config, "vector_db", None)
    retrieval_mode = str(getattr(vdb, "retrieval_mode", "dense")) if vdb else "dense"
    if retrieval_mode not in _VECTOR_MODES:
        return
    by_id = {n.id: n for n in graph.nodes}
    # Query-vector streams in one_of priority order (mirrors the retrieval input). The
    # effective stream is the highest-priority bound producer; fused has no resolvable
    # model space, so a fusion retrieval stays unchecked (trap-closer, not a prover).
    query_vector_artifacts = (
        "fused_query_vectors",
        "audio_query_vectors",
        "text_query_vectors",
        "query_vectors",
    )
    from ..pipeline.graph.operators import node_kind

    for node in graph.nodes:
        if node_kind(node.stage, node.params) != "retrieval":
            continue
        q_producer = q_artifact = None
        for art in query_vector_artifacts:
            producers = [p for a, p in node.bindings if a == art]
            if producers:
                q_producer, q_artifact = producers[-1], art
                break
        index_producers = [p for a, p in node.bindings if a == "vector_index"]
        if q_producer is None or not index_producers:
            continue
        q_space = _producer_space(q_producer, q_artifact, by_id, config)
        i_space = _producer_space(index_producers[-1], "vector_index", by_id, config)
        if q_space and i_space and q_space != i_space:
            raise ConfigurationError(
                f"Embedding-space mismatch at retrieval node '{node.id}': query "
                f"vectors from '{q_producer}' are in space '{q_space}' but "
                f"the index from '{index_producers[-1]}' holds space '{i_space}'. "
                f"Dense retrieval dots these vectors — the scores would be garbage. "
                f"Align the embedder models (or their registry `embedding_space` "
                f"metadata), or switch to sparse retrieval."
            )
