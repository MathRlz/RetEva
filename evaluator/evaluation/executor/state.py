"""Run state + execution-context dataclasses for the DAG executor.

Holds the mutable ``RunState`` threaded through stage handlers plus the two input bundles
(``EvaluationContext`` runtime params, ``RunFeatures`` default-off knobs).
"""

from __future__ import annotations

from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass, field

from ..run_context import RunContext
from ..item_isolation import DropSink
from ..result_schema import RunResults
from ...pipeline import (
    ASRPipelineProtocol,
    TextEmbeddingPipelineProtocol,
    AudioEmbeddingPipelineProtocol,
    RetrievalPipelineProtocol,
)
from ...storage.cache import CacheManager


@dataclass
class EvaluationContext:
    """Bundles the pipeline + execution parameters of ``run_graph``.

    Pass an instance as ``run_graph(dataset, context=ctx)`` (or
    ``run_graph(dataset, context=ctx, ...)``) instead of threading the dozen
    individual keyword arguments. When a context is given it supersedes the
    individual kwargs; omit it to keep using the explicit kwargs. ``features``
    carries the optional, default-off feature configs (see ``RunFeatures``).
    """

    # Pipelines
    retrieval_pipeline: Optional[RetrievalPipelineProtocol] = None
    asr_pipeline: Optional[ASRPipelineProtocol] = None
    text_embedding_pipeline: Optional[TextEmbeddingPipelineProtocol] = None
    audio_embedding_pipeline: Optional[AudioEmbeddingPipelineProtocol] = None

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
    stage_times: Dict[str, float] = field(default_factory=dict, metadata=_SHARED)
    results: "RunResults" = field(
        default_factory=lambda: RunResults(), metadata=_SHARED
    )
    # Intermediates the metrics stage hands to the answer-gen / finalize stages
    # (terminal, serial nodes — shared by construction).
    metrics_all_relevant: list = field(default_factory=list, metadata=_SHARED)
    wer_scores: list = field(default_factory=list, metadata=_SHARED)
    cer_scores: list = field(default_factory=list, metadata=_SHARED)
    per_query_recall5: list = field(default_factory=list, metadata=_SHARED)
    ans_detail_by_qid: dict = field(default_factory=dict, metadata=_SHARED)
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
    # Per-item failures dropped during the run (node_id → [query_id…]); surfaced in
    # report.provenance and excluded from the keyed report (drop-and-log, §3 / T1).
    drop_sink: "DropSink" = field(
        default_factory=lambda: DropSink(), metadata=_SHARED
    )

    def put_artifact(self, name: str, value: Any) -> None:
        """Publish ``name`` as an output of the currently-running node."""
        self.ctx.put(self.current_node.id, name, value)

    def put_items(self, name: str, items: Any) -> None:
        """Publish a per-item artifact as a keyed ``ItemSet`` (architecture W2/A1).

        Legacy consumers reading via :meth:`get_artifact` transparently get the plain
        ``values`` list; keyed consumers (metric nodes) read the ``ItemSet`` via
        :meth:`get_items`."""
        self.ctx.put(self.current_node.id, name, items)

    @property
    def node_params(self) -> dict:
        """The current node's params ({} when absent) — the per-instance config
        every handler overlays on its global config."""
        return getattr(self.current_node, "params", None) or {}

    def _producers(self, name: str) -> list:
        """Producer node ids bound to input ``name`` for the current node (in order)."""
        bindings = getattr(self.current_node, "bindings", ())
        return [pid for art, pid in bindings if art == name]

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
        use ``get_artifact`` directly for single-name artifacts."""
        from ..item_set import ItemSet

        for name in self._input_candidates(key):
            for producer in reversed(self._producers(name)):
                if self.ctx.has(producer, name):
                    value = self.ctx.get(producer, name)
                    return value.values if isinstance(value, ItemSet) else value
        return default

    def input_items(self, key: str, default: Any = None) -> Any:
        """Keyed (:class:`ItemSet`) sibling of :meth:`input` for per-item consumers."""
        from ..item_set import ItemSet

        for name in self._input_candidates(key):
            for producer in reversed(self._producers(name)):
                if self.ctx.has(producer, name):
                    value = self.ctx.get(producer, name)
                    if isinstance(value, ItemSet):
                        return value
        return default

    _MISSING = object()

    def get_artifact(self, name: str, default: Any = _MISSING) -> Any:
        """Read input ``name`` from the latest bound producer that actually published it.

        Resolves newest→oldest so a skipped producer (e.g. fusion bailing to audio-only)
        falls back to an earlier producer of the same artifact. An ``ItemSet`` is unwrapped
        to its ``values`` list so legacy (positional) consumers are unchanged (W2 shim).
        """
        from ..item_set import ItemSet

        for producer in reversed(self._producers(name)):
            if self.ctx.has(producer, name):
                value = self.ctx.get(producer, name)
                return value.values if isinstance(value, ItemSet) else value
        if default is RunState._MISSING:
            raise KeyError(f"no published producer for input '{name}'")
        return default

    def input_role(self, role: str, default: Any = None) -> Any:
        """Read the input bound to an edge ``role`` (operator-abstraction comparison-by-edge).

        Returns the value the producer tagged ``role`` (e.g. ``expected`` / ``actual``)
        published — so a generic ``measure`` node compares its pair by role rather than by
        hardcoded artifact names. Empty (→ ``default``) for every node that hasn't opted into
        role-tagged edges, so existing handlers are unaffected."""
        from ..item_set import ItemSet

        roles = getattr(self.current_node, "binding_roles", ())
        for art, producer, r in roles:
            if r == role and self.ctx.has(producer, art):
                value = self.ctx.get(producer, art)
                return value.values if isinstance(value, ItemSet) else value
        return default

    def get_items(self, name: str, default: Any = _MISSING) -> Any:
        """Read input ``name`` as a keyed ``ItemSet``. A producer that published a plain
        list is wrapped with positional index ids (best-effort) so keyed consumers work."""
        from ..item_set import ItemSet

        for producer in reversed(self._producers(name)):
            if self.ctx.has(producer, name):
                value = self.ctx.get(producer, name)
                if isinstance(value, ItemSet):
                    return value
                return ItemSet([str(i) for i in range(len(value))], list(value))
        if default is RunState._MISSING:
            raise KeyError(f"no published producer for input '{name}'")
        return default

    def keyed_items(self, name: str, default: Any = None) -> Any:
        """Read input ``name`` only if a bound producer published a true keyed ``ItemSet``
        (M1d-2): the per-item identity source. Returns ``default`` (no positional wrap)
        when only a plain list — or nothing — was published."""
        from ..item_set import ItemSet

        for producer in reversed(self._producers(name)):
            if self.ctx.has(producer, name):
                value = self.ctx.get(producer, name)
                if isinstance(value, ItemSet):
                    return value
        return default

    def get_artifacts(self, name: str) -> list:
        """Read input ``name`` from every bound producer (e.g. fusion's two embedders)."""
        return [self.ctx.get(pid, name) for pid in self._producers(name)]
