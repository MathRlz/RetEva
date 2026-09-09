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
from ..item_set import ItemSet


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
    """Optional feature configs + analysis flags for ``run_graph``, grouped separately so
    the engine signature stays focused on the runtime inputs. All default to disabled;
    ``run_from_bundle`` builds one from an ``EvaluationConfig``."""

    judge_config: Any = None
    answer_gen_config: Any = None
    query_opt_config: Any = None
    query_correction_config: Any = None
    embedding_fusion_config: Any = None
    term_weights: Optional[Dict[str, float]] = None
    compute_confidence_intervals: bool = False
    # The config's `metrics:` allowlist — compute exactly these (None = collect-all).
    metric_allowlist: Any = None
    # Lineage-variant rollup reducer ("mean" | "min" | "max").
    variant_rollup: str = "mean"


# Field-scope markers: every RunState field declares whether a parallel branch worker
# keeps a private copy ("node") or shares the base instance ("shared") — a test asserts
# every field is classified, so a missing scope fails loudly instead of racing silently.
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
    """Mutable execution context threaded through DAG stage handlers: the pipelines/config
    inputs plus the accumulators each stage reads and writes. One instance per run_graph
    call, mutated in place; every field carries a ``scope`` marker (see
    ``per_branch_field_names``)."""

    # inputs
    dataset: Any = field(metadata=_SHARED)
    mode: str = field(metadata=_SHARED)
    # Pipelines are node-scoped: `_node_pipeline`/corpus_index rebind them per branch; concurrent
    # branches must not see each other's swap.
    retrieval_pipeline: Any = field(metadata=_NODE)
    asr_pipeline: Any = field(metadata=_NODE)
    text_embedding_pipeline: Any = field(metadata=_NODE)
    audio_embedding_pipeline: Any = field(metadata=_NODE)
    cache_manager: Any = field(metadata=_SHARED)
    # config + load_info: only the corpus_index node needs them (rebuilds the vector index); None
    # means a pre-built index, and the handler no-ops.
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
    # Model lifecycle: when set, releases a stage's model after its last use (mid-run); off unless
    # a provider + on_finish policy apply.
    service_provider: Any = field(default=None, metadata=_SHARED)
    # GPUPool (memory-aware allocation + LRU eviction) across all model builds incl. per-node
    # overrides, when `device_pool` is configured; None means no pool (plain configured device
    # string).
    device_pool: Any = field(default=None, metadata=_SHARED)
    offload_after_stage: bool = field(default=False, metadata=_SHARED)
    # Soft-CPU offload: release-after-last-use parks the model warm on host RAM instead of
    # freeing it, so a later stage reuses it via a CPU↔device move (no full reload).
    soft_cpu_offload: bool = field(default=False, metadata=_SHARED)
    # id → node kind of the EXECUTED graph (graph truth): handlers answer per-node questions
    # from bindings + this map, not run-global mode labels/pipeline presence, which misfire in
    # mixed graphs.
    node_kinds: Dict[str, str] = field(default_factory=dict, metadata=_SHARED)
    # Aggressive lifecycle (`on_use_soft_cpu`): a per-node model parks on CPU the moment its
    # node finishes so the GPU holds only the active node's models; reactivated via CPU→GPU
    # move when needed again.
    aggressive_offload: bool = field(default=False, metadata=_SHARED)
    # Multi-dataset runtime: {dataset_id → loaded QueryDataset}; empty in single-source mode
    # (handler then uses `dataset`). A node selects its source via `current_node.params.dataset`.
    dataset_sources: Dict[str, Any] = field(default_factory=dict, metadata=_SHARED)
    # Set when questions↔corpus doc_id namespaces are disjoint — IR metrics would then be
    # meaningless, so the report omits them (QA/judge still run); join_warning records why.
    disable_ir_metrics: bool = field(default=False, metadata=_SHARED)
    join_warning: str = field(default="", metadata=_SHARED)
    # Per-item query-optimization failures that fell back to the original query — surfaced in
    # provenance so a dead LLM endpoint can't masquerade as "optimization did nothing".
    optimization_fallbacks: int = field(default=0, metadata=_SHARED)
    correction_fallbacks: int = field(default=0, metadata=_SHARED)
    stage_times: Dict[str, float] = field(default_factory=dict, metadata=_SHARED)
    # Wall time per NODE id (every node, no exceptions — includes answer generation and the
    # judge, not just the bucketed `latency` stages).
    node_times: Dict[str, float] = field(default_factory=dict, metadata=_SHARED)
    results: "RunResults" = field(
        default_factory=dict, metadata=_SHARED
    )
    # RunContext: per-node artifact blackboard, keyed by producing node id; the executor sets
    # `current_node` before each handler (see run_context.RunContext).
    ctx: RunContext = field(default_factory=RunContext, metadata=_SHARED)
    current_node: Any = field(default=None, metadata=_NODE)
    # True when the graph contains a refine node (rerank/mmr/threshold) → retrieval emits the
    # fetch_k candidate pool instead of finalizing top-k itself.
    refine_in_graph: bool = field(default=False, metadata=_SHARED)
    # True when an mmr node is present → the rerank node keeps the fetch_k pool (MMR re-selects
    # k diverse from it) instead of truncating to k.
    mmr_in_graph: bool = field(default=False, metadata=_SHARED)
    # True when a hybrid result_fusion node is present → its dense + sparse arm retrievals emit
    # candidate pools (not finalized top-k) so the fusion sees the full depth.
    fuse_in_graph: bool = field(default=False, metadata=_SHARED)
    # Lazy per-run cache of the dataset corpus (doc_id → doc; get_corpus rebuilds N dicts per
    # call) — global/immutable so _SHARED and built once; the retrieved-payload overlay stays
    # per-call in _answer_corpus_lookup.
    corpus_lookup_base: Optional[dict] = field(default=None, metadata=_SHARED)
    # The answer_gen node's resolved config, stashed at generation time so answer_metrics — a
    # different node, so `resolved_config()` can't see this override — scores with the same
    # effective config instead of silently falling back to the flat default.
    answer_gen_resolved_config: Any = field(default=None, metadata=_SHARED)
    # Same handoff for query_correction: `_corrected_metrics_enabled` runs under a different
    # node and can't see the override without this stash (set by _stage_query_correction, read
    # in handlers/metrics.py).
    query_correction_resolved_config: Any = field(default=None, metadata=_SHARED)
    # Per-item failures dropped during the run (node_id → [query_id…]); surfaced in
    # report.provenance and excluded from the keyed report (drop-and-log).
    drop_sink: "DropSink" = field(
        default_factory=lambda: DropSink(), metadata=_SHARED
    )
    # Resolved data flow: consumer id → {input key → (artifact, producer, fallback?)} — the
    # OneOf-priority walk, otherwise invisible in a saved run; surfaced as
    # report.provenance.data_flow. Shared: branch-namespaced node ids never collide.
    data_flow: Dict[str, Dict[str, Any]] = field(default_factory=dict, metadata=_SHARED)
    # The config's `metrics:` allowlist (None = collect-all); read by _branch_scores.
    metric_allowlist: Any = field(default=None, metadata=_SHARED)
    # Lineage-variant rollup reducer ("mean" | "min" | "max"); read by _branch_scores.
    variant_rollup: str = field(default="mean", metadata=_SHARED)
    # {node_id → its resolved feature config} (global ⊕ node params, computed once). Read via
    # `resolved_config()`.
    node_configs: Dict[str, Any] = field(default_factory=dict, metadata=_SHARED)
    # Branch fail-fast: a node whose handler raised is recorded here (node_id → error summary)
    # instead of aborting the run; every transitive dependent is skipped (node_id → root failed
    # ancestor), so a broken branch stops while siblings keep running. Both surface in the final
    # result.
    failed_nodes: Dict[str, str] = field(default_factory=dict, metadata=_SHARED)
    skipped_nodes: Dict[str, str] = field(default_factory=dict, metadata=_SHARED)

    def put_artifact(self, name: str, value: Any) -> None:
        """Publish ``name`` as an output of the currently-running node."""
        self.ctx.put(self.current_node.id, name, value)

    def put_items(self, name: str, items: Any) -> None:
        """Publish a per-item artifact as a keyed ``ItemSet``.

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
        """This node's effective feature config, resolved before execution. ``default`` is
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
        input (multi-variant). For data a producer publishes ALONGSIDE its declared
        artifact, under a key that isn't itself a declared port (e.g. `answer_gen` also
        publishes the raw `answer_generation` dict next to its declared `generated_answers`)
        — this finds that producer via the bound artifact, then reads `extra_key` off the
        same producer, so the lookup lands on the right variant's producer, not a global
        scan."""
        producers = self._producers(bound_name)
        for pid in reversed(producers):
            if self.ctx.has(pid, extra_key):
                return self.ctx.get(pid, extra_key)
        return default

    def _record_flow(self, key: str, artifact: str, producer: str, expected) -> None:
        """Remember which (artifact, producer) actually served input ``key`` for the
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
        resolved it (reads the ``data_flow`` record they leave behind).
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

    def _published(self, key: str):
        """Yield ``(candidate, producer, expected)`` for every bound producer that published
        one of ``key``'s alias candidates, priority-ordered, newest→oldest within each;
        ``expected`` is what the wiring would have picked had nothing bailed."""
        expected = None
        for cand in self._input_candidates(key):
            producers = self._producers(cand)
            if expected is None and producers:
                expected = (cand, producers[-1])
            for producer in reversed(producers):
                if self.ctx.has(producer, cand):
                    yield cand, producer, expected

    def input(self, key: str, default: Any = None) -> Any:
        """Read an input by its canonical key, resolving ``OneOf`` alternatives: the
        highest-priority candidate a bound producer actually published (a bailing producer,
        e.g. fusion with no text vectors, falls back to the next alternative). Use this for
        chained streams (query text / query vectors); use ``get_artifact`` for single-name
        artifacts. The resolved winner is recorded in ``data_flow``."""

        for name, producer, expected in self._published(key):
            self._record_flow(key, name, producer, expected)
            value = self.ctx.get(producer, name)
            return value.values if isinstance(value, ItemSet) else value
        return default

    def input_items(self, key: str, default: Any = None) -> Any:
        """Keyed (:class:`ItemSet`) sibling of :meth:`input` for per-item consumers."""

        for name, producer, expected in self._published(key):
            value = self.ctx.get(producer, name)
            if isinstance(value, ItemSet):
                self._record_flow(key, name, producer, expected)
                return value
        return default

    _MISSING = object()

    def get_artifact(self, name: str, default: Any = _MISSING) -> Any:
        """Read input ``name`` from the latest bound producer that actually published it.
        Resolves alias candidates the same way :meth:`input` does (an explicitly-routed
        same-modality artifact reads through here under its producer's real name — see
        ``wiring.py:bind_explicit_edges``), then resolves newest→oldest within each candidate
        so a skipped producer (e.g. fusion bailing to audio-only) falls back to an earlier
        one. An ``ItemSet`` is unwrapped to its ``values`` list so legacy consumers are
        unchanged. The resolved producer is recorded in ``data_flow``."""

        for cand, producer, expected in self._published(name):
            self._record_flow(name, cand, producer, expected)
            value = self.ctx.get(producer, cand)
            return value.values if isinstance(value, ItemSet) else value
        if default is RunState._MISSING:
            raise KeyError(f"no published producer for input '{name}'")
        return default

    def keyed_items(self, name: str, default: Any = None) -> Any:
        """Read input ``name`` only if a bound producer published a true keyed ``ItemSet``:
        the per-item identity source. Never wraps a plain publish positionally —
        index ids join nothing, so a keyed consumer would silently get an empty join;
        ``default`` is returned instead. Resolves alias candidates like :meth:`get_artifact`
        (a routed extra reads through here under its real artifact name)."""

        for cand, producer, _expected in self._published(name):
            value = self.ctx.get(producer, cand)
            if isinstance(value, ItemSet):
                return value
        return default
