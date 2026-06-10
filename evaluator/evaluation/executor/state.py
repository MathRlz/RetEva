"""Run state + execution-context dataclasses for the DAG executor.

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X1). Holds the
mutable ``RunState`` threaded through stage handlers plus the two input bundles
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


@dataclass
class RunState:
    """Mutable execution context threaded through DAG stage handlers.

    Holds the pipelines/config inputs plus the accumulators each stage reads and
    writes. One instance per run_graph call; handlers mutate it in place.
    """

    # inputs
    dataset: Any
    mode: str
    retrieval_pipeline: Any
    asr_pipeline: Any
    text_embedding_pipeline: Any
    audio_embedding_pipeline: Any
    cache_manager: Any
    # config + load_info are needed only by the corpus_index node (it (re)builds the
    # vector index inside the graph). None for direct callers that pre-built the index —
    # the corpus_index handler then no-ops.
    config: Any
    load_info: Any
    k: int
    batch_size: int
    num_workers: int
    checkpoint_interval: int
    experiment_id: Any
    resume_from_checkpoint: bool
    oracle_mode: bool
    embedding_fusion_config: Any
    query_opt_config: Any
    query_correction_config: Any
    answer_gen_config: Any
    judge_config: Any
    trace_limit: int
    term_weights: Any
    compute_confidence_intervals: bool
    total: int
    cb: Callable
    t_total: float = 0.0
    # Model lifecycle: when set, a stage's model is released after the last stage that
    # uses it (frees the device mid-run). Off unless a provider + on_finish policy apply.
    service_provider: Any = None
    offload_after_stage: bool = False
    # Multi-dataset runtime (B1): {dataset_id → loaded QueryDataset} for graphs with several
    # dataset_source nodes; empty in single-source mode (the dataset_source handler then uses
    # `dataset`). A node selects its source via `current_node.params.dataset`.
    dataset_sources: Dict[str, Any] = field(default_factory=dict)
    # B5: set when the questions↔corpus doc_id namespaces are disjoint — IR metrics are then
    # meaningless (every relevant doc absent from the corpus), so the report omits them (QA/judge
    # still run). The report records the join warning.
    disable_ir_metrics: bool = False
    join_warning: str = ""
    # accumulators (parallel per-query lists unless noted)
    all_hypotheses: list = field(default_factory=list)
    all_ground_truth: list = field(default_factory=list)
    all_embeddings: Any = field(default_factory=list)  # retrieval input
    audio_embeddings: Any = None  # raw audio emb (for fusion)
    text_embeddings_for_fusion: Any = None
    all_relevance: list = field(default_factory=list)
    all_query_ids: list = field(default_factory=list)
    all_results_with_scores: list = field(default_factory=list)
    all_retrieved: list = field(default_factory=list)
    # Candidate results handed from the retrieval node to the rerank node (when present).
    retrieval_candidates: list = field(default_factory=list)
    retrieval_query_texts: Any = None
    audio_emb_for_alignment: Any = None
    text_emb_for_alignment: Any = None
    query_opt_bypassed: bool = False
    stage_times: Dict[str, float] = field(default_factory=dict)
    results: "RunResults" = field(default_factory=lambda: RunResults())
    # Intermediates the metrics stage hands to the answer-gen / finalize stages.
    metrics_all_relevant: list = field(default_factory=list)
    wer_scores: list = field(default_factory=list)
    cer_scores: list = field(default_factory=list)
    per_query_recall5: list = field(default_factory=list)
    ans_detail_by_qid: dict = field(default_factory=dict)
    # RunContext: per-node artifact blackboard (Phase R). The executor sets current_node
    # before each handler; handlers exchange inter-node artifacts via put/get_artifact,
    # which key the context by the producing node's id (see run_context.RunContext).
    ctx: RunContext = field(default_factory=RunContext)
    current_node: Any = None
    # True when the graph contains a rerank node → retrieval emits candidates for it
    # (even if the global pipeline has no reranker, e.g. explicit per-node instances).
    rerank_in_graph: bool = False
    # Per-item failures dropped during the run (node_id → [query_id…]); surfaced in
    # report.provenance and excluded from the keyed report (drop-and-log, §3 / T1).
    drop_sink: "DropSink" = field(default_factory=lambda: DropSink())

    def put_artifact(self, name: str, value: Any) -> None:
        """Publish ``name`` as an output of the currently-running node."""
        self.ctx.put(self.current_node.id, name, value)

    def put_items(self, name: str, items: Any) -> None:
        """Publish a per-item artifact as a keyed ``ItemSet`` (architecture W2/A1).

        Legacy consumers reading via :meth:`get_artifact` transparently get the plain
        ``values`` list; keyed consumers (metric nodes) read the ``ItemSet`` via
        :meth:`get_items`."""
        self.ctx.put(self.current_node.id, name, items)

    def _producers(self, name: str) -> list:
        """Producer node ids bound to input ``name`` for the current node (in order)."""
        bindings = getattr(self.current_node, "bindings", ())
        return [pid for art, pid in bindings if art == name]

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

    def get_items(self, name: str, default: Any = _MISSING) -> Any:
        """Read input ``name`` as a keyed ``ItemSet``. A producer that published a plain
        list is wrapped with the current ``all_query_ids`` (best-effort) so keyed consumers
        work during the transition."""
        from ..item_set import ItemSet

        for producer in reversed(self._producers(name)):
            if self.ctx.has(producer, name):
                value = self.ctx.get(producer, name)
                if isinstance(value, ItemSet):
                    return value
                ids = [str(i) for i in (self.all_query_ids or range(len(value)))]
                if len(ids) == len(value) and len(set(ids)) == len(ids):
                    return ItemSet(ids, list(value))
                return ItemSet([str(i) for i in range(len(value))], list(value))
        if default is RunState._MISSING:
            raise KeyError(f"no published producer for input '{name}'")
        return default

    def get_artifacts(self, name: str) -> list:
        """Read input ``name`` from every bound producer (e.g. fusion's two embedders)."""
        return [self.ctx.get(pid, name) for pid in self._producers(name)]
