"""Run state + execution-context dataclasses for the DAG executor.

Holds the mutable ``RunState`` threaded through stage handlers plus the two input bundles
(``EvaluationContext`` runtime params, ``RunFeatures`` default-off knobs).
"""

from __future__ import annotations

from typing import Callable, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from ..run_context import RunContext
from ..item_isolation import DropSink
from ..result_schema import RunResults
from ...storage.cache import CacheManager


@dataclass
class EvaluationContext:
    """Bundles the pipeline + execution parameters of ``run_graph`` — its required
    ``context`` argument. ``features`` carries the optional, default-off feature
    configs (see ``RunFeatures``)."""

    # Pipelines (duck-typed: ASR / text-embedding / audio-embedding / retrieval pipelines)
    retrieval_pipeline: Optional[Any] = None
    asr_pipeline: Optional[Any] = None
    text_embedding_pipeline: Optional[Any] = None
    audio_embedding_pipeline: Optional[Any] = None

    # Execution parameters
    k: int = 10
    batch_size: int = 32
    trace_limit: int = 0
    num_workers: int = 0
    checkpoint_interval: int = 500
    experiment_id: Optional[str] = None
    resume_from_checkpoint: bool = True
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None
    oracle_mode: bool = False

    # Cache and features
    cache_manager: Optional[CacheManager] = None
    features: Optional["RunFeatures"] = None


@dataclass
class RunFeatures:
    """Optional feature configs + analysis flags for ``run_graph``.

    Groups the rarely-set, default-off knobs so the engine signature stays focused
    on the runtime inputs (pipelines, dataset, batch params). All default to
    disabled; ``run_from_bundle`` builds one from an ``EvaluationConfig``.
    """

    judge_config: Any = None
    answer_gen_config: Any = None
    query_opt_config: Any = None
    query_correction_config: Any = None
    embedding_fusion_config: Any = None
    term_weights: Optional[Dict[str, float]] = None
    compute_confidence_intervals: bool = False
    # B1: the config's `metrics:` allowlist — compute exactly these (None = collect-all).
    metric_allowlist: Any = None
    # A1: lineage-variant rollup reducer ("mean" | "min" | "max").
    variant_rollup: str = "mean"


# Field-scope markers (M1b): every RunState field declares whether a parallel branch
# worker keeps a *private* copy of it ("node") or shares the base instance ("shared").
# `_NodeView` derives its isolation set from these markers, and a test asserts every
# field is classified — adding a field without a scope is a test failure, not a
# silent cross-branch race.
_NODE = {"scope": "node"}  # per-branch private (swap-sensitive / per-query scratch)
_SHARED = {"scope": "shared"}  # shared across branches (thread-safe or serial-only)


def per_branch_field_names() -> frozenset:
    """Names of RunState fields marked ``scope: node`` (the `_NodeView` isolation set)."""
    from dataclasses import fields

    return frozenset(
        f.name for f in fields(RunState) if f.metadata.get("scope") == "node"
    )


@dataclass
class RunState:
    """Mutable execution context threaded through DAG stage handlers.

    Holds the pipelines/config inputs plus the accumulators each stage reads and
    writes. One instance per run_graph call; handlers mutate it in place.
    Every field carries a ``scope`` marker (M1b) — see ``per_branch_field_names``.
    """

    # inputs
    dataset: Any = field(metadata=_SHARED)
    mode: str = field(metadata=_SHARED)
    # Pipelines are node-scoped: `_node_pipeline` / corpus_index rebind them transiently
    # per branch, so concurrent branches must not see each other's swap.
    retrieval_pipeline: Any = field(metadata=_NODE)
    asr_pipeline: Any = field(metadata=_NODE)
    text_embedding_pipeline: Any = field(metadata=_NODE)
    audio_embedding_pipeline: Any = field(metadata=_NODE)
    cache_manager: Any = field(metadata=_SHARED)
    # config + load_info are needed only by the corpus_index node (it (re)builds the
    # vector index inside the graph). None for direct callers that pre-built the index —
    # the corpus_index handler then no-ops.
    config: Any = field(metadata=_SHARED)
    load_info: Any = field(metadata=_SHARED)
    k: int = field(metadata=_SHARED)
    batch_size: int = field(metadata=_SHARED)
    num_workers: int = field(metadata=_SHARED)
    checkpoint_interval: int = field(metadata=_SHARED)
    experiment_id: Any = field(metadata=_SHARED)
    resume_from_checkpoint: bool = field(metadata=_SHARED)
    oracle_mode: bool = field(metadata=_SHARED)
    embedding_fusion_config: Any = field(metadata=_SHARED)
    query_opt_config: Any = field(metadata=_SHARED)
    query_correction_config: Any = field(metadata=_SHARED)
    answer_gen_config: Any = field(metadata=_SHARED)
    judge_config: Any = field(metadata=_SHARED)
    trace_limit: int = field(metadata=_SHARED)
    term_weights: Any = field(metadata=_SHARED)
    compute_confidence_intervals: bool = field(metadata=_SHARED)
    total: int = field(metadata=_SHARED)
    cb: Callable = field(metadata=_SHARED)
    t_total: float = field(default=0.0, metadata=_SHARED)
    # Model lifecycle: when set, a stage's model is released after the last stage that
    # uses it (frees the device mid-run). Off unless a provider + on_finish policy apply.
    service_provider: Any = field(default=None, metadata=_SHARED)
    # GPUPool (evaluator.devices.pool), when `device_pool:` is configured — memory-aware
    # allocation + LRU eviction across ALL model builds, including per-node overrides
    # (_build_node_pipeline). None (the default) means no pool: models use their plain
    # configured device string for the run's lifetime, same as before this field existed.
    device_pool: Any = field(default=None, metadata=_SHARED)
    offload_after_stage: bool = field(default=False, metadata=_SHARED)
    # Soft-CPU offload (2c): release-after-last-use parks the model warm on host RAM instead
    # of freeing it, so a later stage/run reuses it with a CPU↔device move (no full reload).
    soft_cpu_offload: bool = field(default=False, metadata=_SHARED)
    # Multi-dataset runtime (B1): {dataset_id → loaded QueryDataset} for graphs with several
    # dataset_source nodes; empty in single-source mode (the dataset_source handler then uses
    # `dataset`). A node selects its source via `current_node.params.dataset`.
    dataset_sources: Dict[str, Any] = field(default_factory=dict, metadata=_SHARED)
    # B5: set when the questions↔corpus doc_id namespaces are disjoint — IR metrics are then
    # meaningless (every relevant doc absent from the corpus), so the report omits them (QA/judge
    # still run). The report records the join warning.
    disable_ir_metrics: bool = field(default=False, metadata=_SHARED)
    join_warning: str = field(default="", metadata=_SHARED)
    # Per-item query-optimization failures that fell back to the original query (surfaced in
    # provenance so a dead LLM endpoint can't masquerade as "optimization did nothing").
    optimization_fallbacks: int = field(default=0, metadata=_SHARED)
    correction_fallbacks: int = field(default=0, metadata=_SHARED)
    stage_times: Dict[str, float] = field(default_factory=dict, metadata=_SHARED)
    # Wall time per NODE id (every node, no exceptions — `self_timed` handlers used to
    # record nothing, so answer generation and the judge were invisible in the report).
    node_times: Dict[str, float] = field(default_factory=dict, metadata=_SHARED)
    results: "RunResults" = field(
        default_factory=dict, metadata=_SHARED
    )
    # RunContext: per-node artifact blackboard (Phase R). The executor sets current_node
    # before each handler; handlers exchange inter-node artifacts via put/get_artifact,
    # which key the context by the producing node's id (see run_context.RunContext).
    ctx: RunContext = field(default_factory=RunContext, metadata=_SHARED)
    current_node: Any = field(default=None, metadata=_NODE)
    # True when the graph contains a refine node (rerank / mmr / threshold) → retrieval
    # emits the fetch_k candidate pool for it instead of finalizing top-k itself.
    refine_in_graph: bool = field(default=False, metadata=_SHARED)
    # True when an mmr node is present → the rerank node keeps the fetch_k pool (MMR
    # re-selects k diverse from it) instead of truncating to k.
    mmr_in_graph: bool = field(default=False, metadata=_SHARED)
    # True when a hybrid result_fusion node is present → its dense + sparse arm retrievals
    # emit candidate pools (not finalized top-k) so the fusion sees the full depth.
    fuse_in_graph: bool = field(default=False, metadata=_SHARED)
    # Lazy per-run cache of the dataset corpus as a doc_id → doc lookup (get_corpus rebuilds
    # N dicts on every call). Corpus is global/immutable, so it is _SHARED and built once;
    # the retrieved-payload overlay stays per-call (branch-specific) in _answer_corpus_lookup.
    corpus_lookup_base: Optional[dict] = field(default=None, metadata=_SHARED)
    # The answer_gen node's resolved config (global ⊕ its own node params), stashed at
    # generation time so the answer_metrics node — a DIFFERENT node, so `resolved_config()`
    # can't see the answer_gen node's override — scores with the same effective config that
    # generated the answers, instead of silently falling back to the flat default.
    answer_gen_resolved_config: Any = field(default=None, metadata=_SHARED)
    # Same handoff for query_correction: the metrics node's `_corrected_metrics_enabled` check
    # runs under a different node, so it can't see a query_correction node's own override
    # without this stash (set by _stage_query_correction, read back in handlers/metrics.py).
    query_correction_resolved_config: Any = field(default=None, metadata=_SHARED)
    # Per-item failures dropped during the run (node_id → [query_id…]); surfaced in
    # report.provenance and excluded from the keyed report (drop-and-log, §3 / T1).
    drop_sink: "DropSink" = field(
        default_factory=lambda: DropSink(), metadata=_SHARED
    )
    # Resolved data flow (A3): consumer node id → {input key → the (artifact, producer)
    # actually read, plus "fallback": True when it was not the newest producer of the
    # highest-priority bound candidate}. The OneOf-priority + newest-published walk is
    # otherwise invisible in a saved run; surfaced as report.provenance.data_flow.
    # Shared: branch-namespaced node ids never collide across parallel branches.
    data_flow: Dict[str, Dict[str, Any]] = field(default_factory=dict, metadata=_SHARED)
    # B1: the config's `metrics:` allowlist (None = collect-all); read by _branch_scores.
    metric_allowlist: Any = field(default=None, metadata=_SHARED)
    # A1: lineage-variant rollup reducer ("mean" | "min" | "max"); read by _branch_scores.
    variant_rollup: str = field(default="mean", metadata=_SHARED)
    # {node_id → its resolved feature config} (global ⊕ node params, computed once
    # in _setup_execution_context). Read via `resolved_config()`; replaces node_overlay.
    node_configs: Dict[str, Any] = field(default_factory=dict, metadata=_SHARED)
    # Branch fail-fast: a node whose handler raised is recorded here (node_id → error
    # summary) instead of aborting the run; every transitive dependent is then skipped
    # (node_id → the root failed ancestor) so a broken branch stops at its error while
    # sibling branches keep running. Both surface in the final result.
    failed_nodes: Dict[str, str] = field(default_factory=dict, metadata=_SHARED)
    skipped_nodes: Dict[str, str] = field(default_factory=dict, metadata=_SHARED)

    def put_artifact(self, name: str, value: Any) -> None:
        """Publish ``name`` as an output of the currently-running node."""
        self.ctx.put(self.current_node.id, name, value)

    def put_items(self, name: str, items: Any) -> None:
        """Publish a per-item artifact as a keyed ``ItemSet`` (architecture W2/A1).

        Legacy consumers reading via :meth:`get_artifact` transparently get the plain
        ``values`` list; keyed consumers (metric nodes) read the ``ItemSet`` via
        :meth:`keyed_items`."""
        self.ctx.put(self.current_node.id, name, items)

    @property
    def node_params(self) -> dict:
        """The current node's params ({} when absent) — the per-instance config
        every handler overlays on its global config."""
        return getattr(self.current_node, "params", None) or {}

    def resolved_config(self, default: Any = None) -> Any:
        """This node's effective feature config, resolved before execution.

        The global sub-config with the node's params overlaid — allowlist ≡ the dataclass's
        fields, casts from the field types (``evaluation/node_config.py``). ``default`` is
        returned for a node with no feature config (or a direct-call path that skipped
        resolution)."""
        node_id = getattr(self.current_node, "id", None)
        if node_id is None or node_id not in self.node_configs:
            return default
        return self.node_configs[node_id]

    def _producers(self, name: str) -> list:
        """Producer node ids bound to input ``name`` for the current node (in order)."""
        bindings = getattr(self.current_node, "bindings", ())
        return [pid for art, pid in bindings if art == name]

    def sibling_artifact(self, bound_name: str, extra_key: str, default: Any = None) -> Any:
        """Read ``extra_key`` published by the producer bound to this node's ``bound_name``
        input (R4/multi-variant).

        For data a producer publishes ALONGSIDE its declared artifact, under a key that isn't
        itself a declared port (e.g. `answer_gen` also publishes the raw `answer_generation`
        dict next to its declared `generated_answers`) — there is no edge to resolve it through
        `get_artifact`. This finds the producer via the artifact that IS bound (`bound_name`,
        an existing edge), then reads `extra_key` straight off that same producer on the bus —
        so the lookup still lands on the right variant's producer, not a global scan."""
        producers = self._producers(bound_name)
        for pid in reversed(producers):
            if self.ctx.has(pid, extra_key):
                return self.ctx.get(pid, extra_key)
        return default

    def _record_flow(self, key: str, artifact: str, producer: str, expected) -> None:
        """A3: remember which (artifact, producer) actually served input ``key`` for the
        current node; a winner other than ``expected`` (the newest producer of the
        highest-priority bound candidate) is flagged as a fired fallback."""
        node_id = getattr(self.current_node, "id", None)
        if node_id is None:
            return
        entry: Dict[str, Any] = {"artifact": artifact, "producer": producer}
        if (artifact, producer) != expected:
            entry["fallback"] = True
        self.data_flow.setdefault(node_id, {})[key] = entry

    def resolved_producer(self, key: str) -> Tuple[Optional[str], Optional[str]]:
        """The ``(artifact_name, producer_id)`` that actually served canonical input
        ``key`` for the current node — must be called AFTER ``input()``/``get_artifact()``
        resolved it (reads the A3 ``data_flow`` record they leave behind).
        ``(None, None)`` if nothing has resolved ``key`` yet for this node."""
        node_id = getattr(self.current_node, "id", None)
        entry = self.data_flow.get(node_id, {}).get(key)
        if entry is None:
            return None, None
        return entry.get("artifact"), entry.get("producer")

    def _input_candidates(self, key: str) -> tuple:
        """The ordered candidate artifact names for a handler's canonical input key.

        For an ``OneOf`` input the wiring records ``(key → (cand1, cand2, …))`` (priority
        order) in the node's ``input_aliases``; a plain input resolves to ``(key,)``."""
        aliases = getattr(self.current_node, "input_aliases", ())
        for canonical, cands in aliases:
            if canonical == key:
                return cands
        return (key,)

    def input(self, key: str, default: Any = None) -> Any:
        """Read an input by its canonical key, resolving ``OneOf`` alternatives.

        Reads the highest-priority candidate that a bound producer actually published at
        run time (so a bailing producer — e.g. fusion with no text vectors — falls back to
        the next alternative). Use this for chained streams (query text / query vectors);
        use ``get_artifact`` directly for single-name artifacts. The resolved winner is
        recorded in ``data_flow`` (A3)."""
        from ..item_set import ItemSet

        expected = None
        for name in self._input_candidates(key):
            producers = self._producers(name)
            if expected is None and producers:
                expected = (name, producers[-1])
            for producer in reversed(producers):
                if self.ctx.has(producer, name):
                    self._record_flow(key, name, producer, expected)
                    value = self.ctx.get(producer, name)
                    return value.values if isinstance(value, ItemSet) else value
        return default

    def input_items(self, key: str, default: Any = None) -> Any:
        """Keyed (:class:`ItemSet`) sibling of :meth:`input` for per-item consumers."""
        from ..item_set import ItemSet

        expected = None
        for name in self._input_candidates(key):
            producers = self._producers(name)
            if expected is None and producers:
                expected = (name, producers[-1])
            for producer in reversed(producers):
                if self.ctx.has(producer, name):
                    value = self.ctx.get(producer, name)
                    if isinstance(value, ItemSet):
                        self._record_flow(key, name, producer, expected)
                        return value
        return default

    _MISSING = object()

    def get_artifact(self, name: str, default: Any = _MISSING) -> Any:
        """Read input ``name`` from the latest bound producer that actually published it.

        Resolves ``name``'s alias candidates the same way :meth:`input` does (B2: a
        same-modality artifact explicitly routed into this port reads through here too,
        under whatever real artifact name its producer published — see
        ``wiring.py:bind_explicit_edges``), then within each candidate resolves
        newest→oldest so a skipped producer (e.g. fusion bailing to audio-only) falls back
        to an earlier producer of the same artifact. An ``ItemSet`` is unwrapped to its
        ``values`` list so legacy (positional) consumers are unchanged (W2 shim). The
        resolved producer is recorded in ``data_flow`` (A3).
        """
        from ..item_set import ItemSet

        expected = None
        for cand in self._input_candidates(name):
            producers = self._producers(cand)
            if expected is None and producers:
                expected = (cand, producers[-1])
            for producer in reversed(producers):
                if self.ctx.has(producer, cand):
                    self._record_flow(name, cand, producer, expected)
                    value = self.ctx.get(producer, cand)
                    return value.values if isinstance(value, ItemSet) else value
        if default is RunState._MISSING:
            raise KeyError(f"no published producer for input '{name}'")
        return default

    def keyed_items(self, name: str, default: Any = None) -> Any:
        """Read input ``name`` only if a bound producer published a true keyed ``ItemSet``
        (M1d-2): the per-item identity source. Never wraps a plain publish positionally —
        index ids join nothing, so a keyed consumer would silently get an empty join;
        ``default`` is returned instead. Resolves alias candidates like :meth:`get_artifact`
        (B2 — a routed extra reads through here under its real artifact name)."""
        from ..item_set import ItemSet

        for cand in self._input_candidates(name):
            for producer in reversed(self._producers(cand)):
                if self.ctx.has(producer, cand):
                    value = self.ctx.get(producer, cand)
                    if isinstance(value, ItemSet):
                        return value
        return default
