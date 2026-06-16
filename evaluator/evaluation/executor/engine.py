"""DAG execution engine: topological dispatch + per-node run + run journal.

Moved verbatim from the former ``evaluation/phased.py`` (Phase 1, X2). Drives what
runs and in what order from the stage graph, dispatching each node to its registered
handler (serial, or intra-level concurrent when ``parallel_enabled``).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict

from ..stage_registry import get_stage_spec
from ...logging_config import get_logger
from .state import RunState

logger = get_logger(__name__)

_STAGE_TIMES_LOCK = threading.Lock()


def _run_one_node(
    state: "RunState", node: Any, sink: Any = None, level: int = 0
) -> None:
    """Run a single node's handler (with timing) against ``state`` (a base or a view).

    Emits ``node_start`` / ``node_complete`` events to ``sink`` (T10) so an observer sees the
    DAG advance node by node mid-run, with each node's wall time."""
    spec = get_stage_spec(node.stage)
    state.current_node = node  # so handlers put/get artifacts by node id
    # Node-granular console marker (replaces the old per-phase banners, N2): every DAG
    # node announces itself by id + stage, so the log tracks the graph, not fixed phases.
    logger.info("▶ node %s (stage=%s, level=%d)", node.id, node.stage, level)
    if sink is not None:
        sink.emit("node_start", node=node.id, level=level, stage=node.stage)
    _t = time.perf_counter()
    try:
        spec.fn(state)
    finally:
        dur = time.perf_counter() - _t
        if not spec.self_timed:
            # stage_times is shared; a lock keeps the += from losing updates under parallelism.
            with _STAGE_TIMES_LOCK:
                state.stage_times[spec.time_key] = (
                    state.stage_times.get(spec.time_key, 0.0) + dur
                )
        if sink is not None:
            sink.emit(
                "node_complete",
                node=node.id,
                level=level,
                stage=node.stage,
                duration_s=round(dur, 3),
            )


def _execute_stage_graph(state: "RunState", stage_graph, query_opt_config) -> None:
    """Dispatch each DAG node to its handler in topological order.

    The stage graph is the single source of truth for what runs and in what order. Independent
    nodes in the same topological level (e.g. the ref/asr/corr branches of the thesis apparatus)
    run **concurrently** when ``config.parallel_enabled`` — each on a private ``_NodeView`` so they
    don't clobber shared mutable state, and serialized per device so two nodes never contend for
    one GPU (T5). Single-node levels (the whole single-branch path) run exactly as before, serial.
    """
    from .offload import _plan_stage_offloads
    from .views import _NodeView
    from .parallel import _run_level_parallel
    from ..handlers.metrics import _branch_of

    query_opt_enabled = query_opt_config is not None and getattr(
        query_opt_config, "enabled", False
    )
    levels = list(stage_graph.topological_levels())
    flat_nodes = [node for level in levels for node in level]
    offloads = _plan_stage_offloads(state, flat_nodes, query_opt_enabled)
    branched = any("@" in n.id for n in flat_nodes)
    parallel = (
        bool(getattr(state.config, "parallel_enabled", False))
        if state.config is not None
        else False
    ) and branched
    # Persistent per-branch state: the "main" (shared-prefix) branch runs on the base; each
    # divergent branch carries its own _NodeView across all levels, so a branch's positional
    # node-scoped control attrs survive level→level for its own metrics node (T5).
    views: Dict[str, Any] = {}

    def _ctx_for(node: Any) -> Any:
        if not parallel:
            return state
        branch = _branch_of(node.id)
        if branch == "main":
            return state
        if branch not in views:
            views[branch] = _NodeView(state, node)
        return views[branch]

    # Full-run checkpoint/resume (T6): snapshot resumable state at each level boundary; a rerun
    # with the same config + graph resumes at the first incomplete level instead of recomputing.
    journal, start_level = _setup_run_journal(state, flat_nodes)

    # Live, node-granular progress to the callback + optional JSONL sink (T10).
    from ..progress import ProgressSink

    sink = ProgressSink.from_env(
        callback=getattr(state, "cb", None), total_levels=len(levels)
    )

    pos = 0
    for i, level in enumerate(levels):
        if i < start_level:
            pos += len(
                level
            )  # already completed in a prior run; keep offload index aligned
            continue
        if parallel and len(level) > 1:
            _run_level_parallel(state, level, _ctx_for, sink, i)
        else:
            for node in level:
                _run_one_node(_ctx_for(node), node, sink, i)
        # Offload models whose last use falls in this level (after the join, so a model a
        # parallel sibling still needs isn't freed early).
        for node in level:
            for model in offloads.get(pos, ()):
                try:
                    state.service_provider.release_model_instance(model)
                    logger.info("offloaded model after stage '%s'", node.stage)
                except Exception as exc:  # never let offload break the run
                    logger.warning(
                        "offload after stage '%s' failed: %s", node.stage, exc
                    )
            pos += 1
        if journal is not None:
            from ..run_journal import snapshot_state

            journal.save(i, snapshot_state(state))
    if journal is not None:
        journal.clear()  # ran to completion → no resume needed


def _setup_run_journal(state: "RunState", flat_nodes) -> tuple:
    """Build the run journal + resume point (T6). Returns ``(journal, start_level)``; the
    journal is ``None`` (no checkpointing) unless a cache dir + experiment id + a positive
    ``checkpoint_interval`` are configured. On a matching journal, restores the snapshot and
    resumes at the level after the last completed one."""
    if not (
        getattr(state, "cache_manager", None) is not None
        and getattr(state, "experiment_id", None)
        and getattr(state, "checkpoint_interval", 0) > 0
    ):
        return None, 0
    from ..run_journal import RunJournal, restore_state, run_key

    key = run_key(state.config, tuple(n.id for n in flat_nodes))
    journal = RunJournal(state.cache_manager.checkpoints_dir, key)
    start_level = 0
    if getattr(state, "resume_from_checkpoint", False):
        loaded = journal.load()
        if loaded is not None:
            last_level, blob = loaded
            try:
                restore_state(state, blob)
                start_level = last_level + 1
                logger.info("resuming run from level %d (journal %s)", start_level, key)
            except Exception as exc:
                logger.warning("journal restore failed (%s); running fresh", exc)
                start_level = 0
    return journal, start_level
