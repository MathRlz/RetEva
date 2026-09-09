"""Pre-flight validation: every check that must fail BEFORE any model loads — determinism
seeding, LLM cost budget, embedding-space typing (config-level and per-node graph-level),
and vector-store backend availability."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config.validation import EXTRA_VALIDATORS, configured_feature_configs
from ..errors import ConfigurationError
from ..logging_config import get_logger
from ..models.embedding_space import resolve_embedding_space, spaces_compatible
from ..llm_client.cost import COST
from ..pipeline.factory import check_graph_backend_dependencies, effective_vector_db_config
from ..pipeline.graph import build_graph_for_config
from .provenance import set_global_determinism
from .metric_registry import list_metrics
from ..config.graph_config import resolved_model_config
from ..pipeline.graph.operators import node_kind
from ..models.registry import (
    asr_registry,
    text_embedding_registry,
    audio_embedding_registry,
)
from ..models.retrieval.query.correction import list_correctors
from ..models.retrieval.query.optimization import list_combine_strategies
from ..pipeline.graph.registry import node_model_field

logger = get_logger(__name__)

# Retrieval modes that compare vectors via inner product (so spaces must match).
_VECTOR_MODES = {"dense", "hybrid"}


def run_pre_flight(config: Any) -> Dict[str, Any]:
    """Run the full pre-flight chain; returns the determinism flags for provenance.

    Order matters: determinism first (everything downstream may consume RNGs),
    then the cheap config-level checks, then the graph-level ones.
    """

    logger.info("Validating configuration, please wait…")

    validate_registries(config)

    run_seed = getattr(getattr(config, "audio_synthesis", None), "seed", None)
    flags = set_global_determinism(run_seed)
    logger.info("determinism: %s", flags)

    budget = int(getattr(getattr(config, "llm", None), "max_tokens_budget", 0) or 0)
    COST.reset(budget_tokens=budget or None)

    check_graph_backend_dependencies(config)

    try:
        graph = build_graph_for_config(config)
    except Exception:
        graph = None  # graph errors surface in their own validation path
    if graph is not None:
        validate_models(config, graph)
        validate_graph_embedding_spaces(graph, config)
        validate_graph_vector_db_configs(graph, config)
    return flags


_KIND_FAMILY = {"asr": "asr", "text_embedding": "text_emb", "audio_embedding": "audio_emb"}


def collect_registry_problems(config: Any) -> List[str]:
    """Every config check whose authority is a registry: metric names, model types, corrector
    and multi-query combine-strategy names. ``config/`` sits below all four registries and
    cannot import them, so it runs these through
    :data:`evaluator.config.validation.EXTRA_VALIDATORS` (appended at the bottom of this module).
    """
    errors: List[str] = []
    _check_metric_allowlist(config, errors)
    _check_model_types(config, errors)
    _check_query_registries(config, errors)
    return errors


def validate_registries(config: Any) -> None:
    """Raising form of :func:`collect_registry_problems`, for the run path: a config built with
    ``validate=False`` still meets these checks before pre-flight goes any further."""
    problems = collect_registry_problems(config)
    if problems:
        raise ConfigurationError(
            "Configuration validation failed:\n  - " + "\n  - ".join(problems)
        )


def _check_metric_allowlist(config: Any, errors: List[str]) -> None:
    """The `metrics:` allowlist must name registered metrics — caught at validate time
    with the registered list named, not as a silent no-op at report time."""
    names = getattr(config, "metrics", None)
    if not names:
        return

    registered = {spec.name for spec in list_metrics()}
    unknown = [n for n in names if n not in registered]
    if unknown:
        errors.append(
            f"unknown metric(s) in `metrics:` allowlist: {', '.join(sorted(unknown))}. "
            f"Registered: {', '.join(sorted(registered))}"
        )


def _configured_model_types(config: Any):
    """Yield (family, model_type) for the flat default (always, all three families) plus every
    graph node's resolved per-node override — so a branch-only model type fails here at
    validate time instead of only surfacing as a factory.py build crash. Graph-less configs
    (no ``graph.nodes``) contribute only the flat triple; ``config.graph`` raising is caught
    here rather than propagated."""
    yield "asr", config.model.asr_model_type
    yield "text_emb", config.model.text_emb_model_type
    yield "audio_emb", config.model.audio_emb_model_type


    try:
        nodes = config.graph.nodes
    except Exception:
        return
    for node in nodes:
        family = _KIND_FAMILY.get(node_kind(node.stage, node.params))
        if family is None:
            continue
        mcfg = resolved_model_config(config, node)
        yield family, getattr(mcfg, f"{family}_model_type")


def _check_model_types(config: Any, errors: List[str]) -> None:
    """Check every configured model type (the flat default AND any per-node branch override)
    is registered, deduped by (family, model_type)."""

    registries = {
        "asr": (asr_registry, "ASR"),
        "text_emb": (text_embedding_registry, "text embedding"),
        "audio_emb": (audio_embedding_registry, "audio embedding"),
    }
    seen = set()
    for family, model_type in _configured_model_types(config):
        if model_type is None or (family, model_type) in seen:
            continue
        seen.add((family, model_type))
        registry, label = registries[family]
        if not registry.is_registered(model_type):
            available = ", ".join(registry.list_types())
            errors.append(
                f"Unknown {label} model type: '{model_type}'. "
                f"Available types: {available}. "
                f"Tip: use one listed type or call EvaluationConfig.from_preset(...)."
            )


def _check_query_registries(config: Any, errors: List[str]) -> None:
    """``query_correction.method`` and ``query_optimization.combine_strategy`` name plug-in
    registries — a custom ``@register_corrector`` / ``@register_combine_strategy`` is valid
    config the moment it registers, so the registry is the authority. Checked on the flat
    default AND on every node's resolved override, since a node may set its own method."""

    correctors = sorted(set(list_correctors()))
    for cfg in configured_feature_configs(config, "query_correction"):
        if cfg.method not in correctors:
            errors.append(
                f"query_correction.method must be one of {correctors}, got {cfg.method!r}"
            )
            break

    strategies = list_combine_strategies()
    for cfg in configured_feature_configs(config, "query_optimization"):
        if cfg.combine_strategy not in strategies:
            errors.append(
                f"combine_strategy must be one of {strategies}, "
                f"got {cfg.combine_strategy!r}"
            )
            break


def validate_models(config: Any, graph: Any) -> None:
    """Fail fast — before any model loads — if the graph references a model type that isn't
    registered, or an adapter path that doesn't exist. Validates exactly the models the graph
    will build; an unused default family (e.g. the default ASR type on a pure audio_emb run)
    is never checked, since no node references it. Collects every problem and raises a single
    :class:`ConfigurationError` listing them all."""
    import os


    model = getattr(config, "model", None)
    if model is None:
        return

    # node_model_field(...) -> (type attr on the resolved ModelConfig, adapter attr, registry)
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
        # per-node override (graph-first) wins over config.model.* — same resolver factory.py
        # and node_pipeline.py use, instead of hand-rolling `params.get(x) or config.model.y`.
        mcfg = resolved_model_config(config, node)
        mtype = getattr(mcfg, type_attr, None)
        if mtype is not None and ("type", type_attr, str(mtype)) not in seen:
            seen.add(("type", type_attr, str(mtype)))
            if not registry.is_registered(str(mtype)):
                problems.append(
                    f"{type_attr}={mtype!r} is not a registered {registry.name} model "
                    f"(available: {', '.join(registry.list_types())})"
                )
        adapter = getattr(mcfg, adapter_attr, None)
        if adapter and ("adapter", str(adapter)) not in seen:
            seen.add(("adapter", str(adapter)))
            if not os.path.exists(adapter):
                problems.append(f"{adapter_attr} not found: {adapter}")

    if problems:
        raise ConfigurationError(
            "Configuration references model(s) that can't be resolved:\n  - "
            + "\n  - ".join(problems)
        )


def _resolve_space(
    registry: Any, model_type: Any, model_name: Optional[str], override: Optional[str]
) -> str:
    """Embedding-space id, honoring a per-instance ``embedding_space`` override (config field
    or node param) over the type/name-derived default — e.g. an APM trained to project audio
    into a text embedder's space declares that text space instead of its encoder name."""
    if override:
        return str(override)
    return resolve_embedding_space(registry, str(model_type), model_name)


def _node_space(node: Any, config: Any) -> Optional[str]:
    """The embedding space a graph node instance produces vectors in, or None when
    unknowable (e.g. a fusion node's combined space)."""
    from types import SimpleNamespace


    params = getattr(node, "params", None) or {}
    stage = node_kind(node.stage, params)
    if stage == "dataset_source":
        return params.get("embedding_space")
    if stage in ("text_embedding", "corpus_embedding"):
        # corpus_embedding shares the text-embedding param namespace (it builds its pipeline
        # via stage="text_embedding" too — see embedding.py's _stage_corpus_embedding), so
        # resolve it under that kind explicitly rather than its own unmapped "corpus_embedding".
        mcfg = resolved_model_config(config, SimpleNamespace(stage="text_embedding", params=params))
        if not mcfg.text_emb_model_type:
            return None
        return _resolve_space(
            text_embedding_registry,
            mcfg.text_emb_model_type,
            mcfg.text_emb_model_name,
            mcfg.text_emb_embedding_space,
        )
    if stage == "audio_embedding":
        mcfg = resolved_model_config(config, node)
        if not mcfg.audio_emb_model_type:
            return None
        return _resolve_space(
            audio_embedding_registry,
            mcfg.audio_emb_model_type,
            mcfg.audio_emb_model_name,
            mcfg.audio_emb_embedding_space,
        )
    return None


def resolve_query_space(
    config: Any, stream_name: Optional[str], *, producer_id: Optional[str] = None
) -> Optional[str]:
    """The embedding space of a bound query-vector stream at runtime, or None when unknowable
    (fused vectors, missing model info). Used by the retrieval runtime guard to assert the
    query is comparable to the index it dots against.

    ``producer_id`` is resolved via ``_node_space`` FIRST: a multi-variant graph's branches
    each carry their own per-node model override, so the flat ``config.model`` default is
    wrong for any node that overrides it (confirmed bug: every branch asserted the SAME
    flat-default space, false-positively raising ``EmbeddingSpaceMismatch`` for branches that
    differed from it). The flat-default path remains as a fallback when there's no live
    producer_id, or when ``_node_space`` can't resolve one (e.g. a pass-through)."""

    name = stream_name or "query_vectors"
    if name == "fused_query_vectors":
        return None  # a combined space is not a single model's space

    if producer_id is not None:
        graph = getattr(config, "graph", None)
        node = next(
            (n for n in getattr(graph, "nodes", ()) if n.id == producer_id), None
        )
        if node is not None:
            space = _node_space(node, config)
            if space is not None:
                return space

    model = getattr(config, "model", None)
    if model is None:
        return None

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
    else:  # query_vectors / default: the graph's primary query embedder
        is_audio = (getattr(config, "graph_label", None) or "") == "audio_emb_retrieval"
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
    run time. Unresolvable spaces (fusion, no model info) stay unchecked — a trap-closer,
    not a prover."""
    vdb = getattr(config, "vector_db", None)
    retrieval_mode = str(getattr(vdb, "retrieval_mode", "dense")) if vdb else "dense"
    if retrieval_mode not in _VECTOR_MODES:
        return []
    by_id = {n.id: n for n in graph.nodes}
    # Query-vector streams in one_of priority order (mirrors the retrieval input); the
    # effective stream is the highest-priority bound producer. Fused vectors have no
    # resolvable model space.
    query_vector_artifacts = (
        "fused_query_vectors",
        "audio_query_vectors",
        "text_query_vectors",
        "query_vectors",
    )

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
    """Per-node V[s] check: reject a dense/hybrid retrieval whose bound query-vector producer
    and index chain live in different embedding spaces. Catches what the config-level check
    cannot — explicit graphs whose per-node ``params.model`` overrides diverge, and dataset
    vector columns whose declared ``embedding_space`` differs from the query embedder."""
    issues = check_embedding_spaces(graph, config)
    if issues:
        raise ConfigurationError(issues[0])


def _vector_db_backend_key(vector_db_config: Any) -> Optional[tuple]:
    """A persistent-backend identity for *vector_db_config*, or None for a backend with
    no on-disk/remote collision surface (inmemory/faiss — each node gets its own
    process-local index)."""
    db_type = str(getattr(vector_db_config, "type", "")).lower()
    if db_type in ("chromadb",):
        return (db_type, vector_db_config.chromadb_path, vector_db_config.chromadb_collection_name)
    if db_type in ("qdrant",):
        location = vector_db_config.qdrant_url or vector_db_config.qdrant_path
        return (db_type, location, vector_db_config.qdrant_collection_name)
    return None


def validate_graph_vector_db_configs(graph: Any, config: Any) -> None:
    """Reject two ``vector_db`` nodes that resolve to the same persistent backend + collection
    — a silent index collision, since ``chromadb_collection_name`` and
    ``qdrant_collection_name`` both default to ``"documents"``. Each node's effective config
    is computed the same way the executor builds its index, so this only fires when
    per-node overrides genuinely collide — not on every multi-vector_db graph."""

    seen: Dict[tuple, str] = {}
    for node in graph.nodes:
        if node_kind(node.stage, node.params) != "vector_db":
            continue
        effective = effective_vector_db_config(config.vector_db, node.params or {})
        key = _vector_db_backend_key(effective)
        if key is None:
            continue
        if key in seen:
            raise ConfigurationError(
                f"vector_db nodes '{seen[key]}' and '{node.id}' both resolve to the same "
                f"backend/collection {key} — they will collide (silently share/overwrite "
                f"the same index). Give each a distinct `collection` and/or `path`/`url`."
            )
        seen[key] = node.id


#: config/ cannot import the registries these checks consult, so it discovers them
#: here instead — importing this module is what wires them into config.validate().
#: Guarded by tests/test_config_registry_hook.py.
EXTRA_VALIDATORS.append(collect_registry_problems)
