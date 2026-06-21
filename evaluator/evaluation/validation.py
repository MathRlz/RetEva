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

from typing import Any, Dict, List, Optional

from ..errors import ConfigurationError
from ..logging_config import get_logger
from ..models.embedding_space import resolve_embedding_space, spaces_compatible

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
    from ..plugins import discover_all_plugins
    from .provenance import set_global_determinism

    # Load any third-party plugins (models/nodes/handlers/metrics/datasets) before validation
    # so they are visible to the graph build + registries (§5). Idempotent, best-effort.
    discover_all_plugins()

    logger.info("Validating configuration, please wait…")

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
        validate_models(config, graph)
        validate_graph_embedding_spaces(graph, config)
    return flags


def validate_models(config: Any, graph: Any) -> None:
    """Fail fast — before any model loads — if the graph references a model type that isn't
    registered, or an adapter path that doesn't exist.

    Validates exactly the models the graph will build: ``config.model.*`` plus any per-node
    overrides (a graph node's ``model``/``adapter`` params), keyed to the right family registry
    via :func:`node_model_field`. An unused default family (e.g. the default ASR type on a pure
    audio_emb run) is never checked, since no node references it. Collects every problem and raises
    a single :class:`ConfigurationError` listing them all (with the available types)."""
    import os

    from ..models.registry import (
        asr_registry,
        text_embedding_registry,
        audio_embedding_registry,
    )
    from ..pipeline.graph.registry import node_model_field

    model = getattr(config, "model", None)
    if model is None:
        return

    # node_model_field(...) -> (type attr on config.model, adapter attr, the family registry)
    field_specs = {
        "model.asr_model_type": ("asr_model_type", "asr_adapter_path", asr_registry),
        "model.text_emb_model_type": (
            "text_emb_model_type", "text_emb_adapter_path", text_embedding_registry,
        ),
        "model.audio_emb_model_type": (
            "audio_emb_model_type", "audio_emb_adapter_path", audio_embedding_registry,
        ),
    }
    problems: list = []
    seen: set = set()
    for node in graph.nodes:
        spec = field_specs.get(node_model_field(node.stage, node.params))
        if spec is None:
            continue
        type_attr, adapter_attr, registry = spec
        params = node.params or {}
        # per-node override (graph-first) wins over config.model.*
        mtype = params.get("model") or getattr(model, type_attr, None)
        if mtype is not None and ("type", type_attr, str(mtype)) not in seen:
            seen.add(("type", type_attr, str(mtype)))
            if not registry.is_registered(str(mtype)):
                problems.append(
                    f"{type_attr}={mtype!r} is not a registered {registry.name} model "
                    f"(available: {', '.join(registry.list_types())})"
                )
        adapter = params.get("adapter") or getattr(model, adapter_attr, None)
        if adapter and ("adapter", str(adapter)) not in seen:
            seen.add(("adapter", str(adapter)))
            if not os.path.exists(adapter):
                problems.append(f"{adapter_attr} not found: {adapter}")

    if problems:
        raise ConfigurationError(
            "Configuration references model(s) that can't be resolved:\n  - "
            + "\n  - ".join(problems)
        )


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
    mode = str(getattr(config, "graph_template", None) or "")
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
        if not spaces_compatible(query_space, corpus_space):
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


def resolve_query_space(config: Any, stream_name: Optional[str]) -> Optional[str]:
    """The embedding space of a bound query-vector stream at runtime, or None when unknowable
    (fused vectors, missing model info). Used by the retrieval runtime guard (2b) to assert
    the query is comparable to the index it dots against. Mirrors the pre-flight resolution but
    keyed by the actually-bound stream name."""
    from ..models.registry import audio_embedding_registry, text_embedding_registry

    model = getattr(config, "model", None)
    if model is None:
        return None
    name = stream_name or "query_vectors"
    if name == "fused_query_vectors":
        return None  # a combined space is not a single model's space

    audio = (
        audio_embedding_registry,
        getattr(model, "audio_emb_model_type", None),
        getattr(model, "audio_emb_model_name", None),
        getattr(model, "audio_emb_embedding_space", None),
    )
    text = (
        text_embedding_registry,
        getattr(model, "text_emb_model_type", None),
        getattr(model, "text_emb_model_name", None),
        getattr(model, "text_emb_embedding_space", None),
    )
    if name == "audio_query_vectors":
        reg, mtype, mname, override = audio
    elif name == "text_query_vectors":
        reg, mtype, mname, override = text
    else:  # query_vectors / default: the template's primary query embedder
        is_audio = (getattr(config, "graph_template", None) or "") == "audio_emb_retrieval"
        reg, mtype, mname, override = audio if is_audio else text
    if not mtype:
        return None
    return _resolve_space(reg, mtype, mname, override)


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


def check_embedding_spaces(graph: Any, config: Any) -> List[str]:
    """Non-raising form of :func:`validate_graph_embedding_spaces`: collect *all*
    embedding-space mismatch messages (empty list = clean) instead of raising on the first.

    Feeds the builder UI, which warns on a mismatch at edit time rather than only failing at
    run time. Same per-node V[s] check; unresolvable spaces (fusion, no model info) stay
    unchecked — a trap-closer, not a prover.
    """
    vdb = getattr(config, "vector_db", None)
    retrieval_mode = str(getattr(vdb, "retrieval_mode", "dense")) if vdb else "dense"
    if retrieval_mode not in _VECTOR_MODES:
        return []
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

    issues: List[str] = []
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
        if not spaces_compatible(q_space, i_space):
            issues.append(
                f"Embedding-space mismatch at retrieval node '{node.id}': query "
                f"vectors from '{q_producer}' are in space '{q_space}' but "
                f"the index from '{index_producers[-1]}' holds space '{i_space}'. "
                f"Dense retrieval dots these vectors — the scores would be garbage. "
                f"Align the embedder models (or their registry `embedding_space` "
                f"metadata), or switch to sparse retrieval."
            )
    return issues


def validate_graph_embedding_spaces(graph: Any, config: Any) -> None:
    """Per-node V[s] check: reject a dense/hybrid retrieval whose bound query-vector
    producer and index chain live in different embedding spaces (§4.1 P1).

    Catches what the config-level check cannot: explicit graphs whose per-node
    ``params.model`` overrides diverge (e.g. ``corpus_embedding {model: labse}`` +
    ``text_embedding {model: jina_v4}``) and dataset vector columns whose declared
    ``embedding_space`` differs from the query embedder. Unresolvable spaces (fusion,
    no model info) stay unchecked — this is a trap-closer, not a prover.
    """
    issues = check_embedding_spaces(graph, config)
    if issues:
        raise ConfigurationError(issues[0])
