"""Windowed query-side execution (Roadmap 3a).

The corpus is embedded + indexed once; the *query* side runs one window at a time, so a
corpus-scale run holds only one window's audio / query vectors at a time. The per-item bus
artifacts each window produces (query_text, retrieved, …) accumulate across windows, and the
finalize nodes (per-item metrics + the report assembler + its bootstrap CIs / BH-FDR) run once
over the full accumulated set — so a windowed run reproduces the whole-run report exactly.

Three phases, by node taxonomy:

- **prelude** (once, full dataset): the ``source`` node (publishes the whole corpus + per-item
  GT) and the corpus embed/index nodes — window-independent, shared by every window.
- **windowed** (per window): the per-item query producers (asr, embed, query rewrites,
  retrieval, refine, fusion) — their output for item *i* depends only on item *i* + the shared
  index, so they slice cleanly.
- **finalize** (once, after windows): the metric / report / generation / sink nodes, which need
  the full per-item set (means, bootstrap CIs, BH-FDR, traces).

Opt-in via ``StreamingConfig.window_size``; default whole-dataset execution is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from ...pipeline.graph.operators import node_kind
from ...pipeline.graph.registry import node_category, node_domain

# Source kinds publish the whole corpus + per-item GT once (window-independent).
SOURCE_KINDS = frozenset({"dataset_source", "dataset_union"})
# Corpus-phase kinds embed/index the whole corpus once.
CORPUS_KINDS = frozenset({"corpus_embedding", "corpus_merge", "vector_db", "corpus_index"})
# Run-once prelude = source + corpus.
PRELUDE_KINDS = SOURCE_KINDS | CORPUS_KINDS


@dataclass(frozen=True)
class PhasePartition:
    """A graph's node ids split into the three windowed-execution phases."""

    prelude: Tuple[str, ...]      # run once, before windows (corpus index + whole GT)
    windowed: Tuple[str, ...]     # run per window (asr / embed / retrieve / refine / fusion)
    finalize: Tuple[str, ...]     # run once, after windows (metrics / report / CIs / traces)

    @property
    def all_ids(self) -> Tuple[str, ...]:
        return self.prelude + self.windowed + self.finalize


def _phase_of(node: Any) -> str:
    """Which windowed-execution phase a node belongs to (taxonomy-driven)."""
    params = getattr(node, "params", None)
    if node_kind(node.stage, params) in PRELUDE_KINDS:
        return "prelude"
    # Aggregators (metric/sink categories) and the reporting/generation areas need the full
    # per-item set → run once after all windows fold in.
    if node_category(node.stage, params) in ("metric", "sink") or node_domain(
        node.stage, params
    ) in ("reporting", "generation"):
        return "finalize"
    return "windowed"


def partition_phases(graph: Any) -> PhasePartition:
    """Split ``graph``'s nodes into prelude / windowed / finalize phases (each node in exactly
    one), preserving topological order within each phase (``graph.nodes`` order)."""
    buckets = {"prelude": [], "windowed": [], "finalize": []}
    for node in graph.nodes:
        buckets[_phase_of(node)].append(node.id)
    return PhasePartition(
        tuple(buckets["prelude"]), tuple(buckets["windowed"]), tuple(buckets["finalize"])
    )


def window_bounds(n: int, window_size: int) -> List[Tuple[int, int]]:
    """The ``[start, stop)`` query-window ranges over ``n`` items.

    A non-positive or ``>= n`` window collapses to a single whole-set window (so an over-large
    window degrades to today's whole-dataset pass). ``n == 0`` yields no windows."""
    if n <= 0:
        return []
    if window_size <= 0 or window_size >= n:
        return [(0, n)]
    return [(start, min(start + window_size, n)) for start in range(0, n, window_size)]


def merge_window_values(parts: List[Any]) -> Any:
    """Fold one bus slot's per-window values into the whole-run value (order-preserving).

    ItemSets concat by id; ndarrays stack rows; lists extend; a single part passes through. The
    order is window-then-row, i.e. dataset order — so a windowed run's accumulated per-item
    arrays match the whole run byte-for-byte (the bootstrap RNG resamples the same order)."""
    from ..item_set import ItemSet

    if len(parts) == 1:
        return parts[0]
    if all(isinstance(p, ItemSet) for p in parts):
        return ItemSet.concat(parts)
    if all(isinstance(p, np.ndarray) for p in parts):
        return np.concatenate(parts, axis=0)
    if all(isinstance(p, list) for p in parts):
        out: List[Any] = []
        for p in parts:
            out.extend(p)
        return out
    return parts[-1]  # non-per-item artifact (re-published each window) — last wins


def execute_windowed(state: Any, stage_graph: Any, window_size: int) -> None:
    """Run ``stage_graph`` windowing the query side (3a). Drop-in for ``_execute_stage_graph``
    when ``streaming.window_size`` is set; produces the same RunState the whole run would."""
    from .engine import _run_one_node
    from ...datasets.runtime import _QueryIdSubset
    from ...logging_config import get_logger

    logger = get_logger(__name__)
    flat = [node for level in stage_graph.topological_levels() for node in level]
    part = partition_phases(stage_graph)
    prelude, windowed, finalize = set(part.prelude), set(part.windowed), set(part.finalize)
    by_id = {n.id: n for n in flat}

    # Only the bus slots the finalize phase consumes need to survive across windows; every
    # other windowed slot (e.g. the heavy query vectors) is overwritten each window, so just
    # one window's worth is resident — that is the memory bound. A finalize slot is one a
    # finalize node binds as input: (producer_id, artifact_name).
    finalize_inputs = {
        (producer, artifact)
        for node in flat if node.id in finalize
        for artifact, producer in getattr(node, "bindings", ())
    }
    accumulate = {s for s in finalize_inputs if s[0] in windowed}

    full_dataset = state.dataset
    n = len(full_dataset) if hasattr(full_dataset, "__len__") else 0
    bounds = window_bounds(n, window_size)
    logger.info(
        "streaming: %d item(s) in %d window(s) of %d; prelude=%d windowed=%d finalize=%d "
        "(accumulating %d slot(s))",
        n, len(bounds), window_size, len(prelude), len(windowed), len(finalize),
        len(accumulate),
    )

    # Window-granular checkpoint/resume: a crashed corpus-scale run resumes at the first
    # incomplete window instead of redoing them all (the prelude re-runs — cache-fast — to
    # rebuild the shared index; only the light accumulator is persisted, not the index).
    journal, start_window, accum = _setup_window_journal(state, flat)

    # 1. Prelude — corpus index + whole per-item GT, over the full dataset, once (also on resume).
    for node in flat:
        if node.id in prelude:
            _run_one_node(state, node)

    # 2. Windowed query compute — accumulate only the finalize-bound per-item slots.
    for w, (start, stop) in enumerate(bounds):
        if w < start_window:
            continue  # already folded into the restored accumulator
        state.dataset = _QueryIdSubset(full_dataset, list(range(start, stop)))
        for node in flat:
            if node.id in windowed:
                _run_one_node(state, node)
        for slot in accumulate:
            if state.ctx.has(*slot):
                accum.setdefault(slot, []).append(state.ctx.get(*slot))
        if journal is not None:
            journal.save(w + 1, {"accum": accum})
        logger.debug("streaming: window %d/%d [%d:%d] done", w + 1, len(bounds), start, stop)
    for slot, parts in accum.items():
        state.ctx.put(slot[0], slot[1], merge_window_values(parts))

    # 3. Finalize — per-item metrics + report over the full accumulated bus, once.
    state.dataset = full_dataset
    for node in flat:
        if node.id in finalize:
            _run_one_node(state, node)
    if journal is not None:
        journal.clear()
    _ = by_id  # (kept for symmetry / future per-node hooks)


def _setup_window_journal(state: Any, flat: List[Any]):
    """``(journal, start_window, accum)`` for windowed resume — gated on the same
    cache-dir + experiment-id + checkpoint-interval as the level journal. A matching journal
    restores the accumulator and resumes at the first incomplete window."""
    from ...logging_config import get_logger
    from ..run_journal import RunJournal, run_key

    logger = get_logger(__name__)
    cm = getattr(state, "cache_manager", None)
    if not (
        cm is not None
        and getattr(state, "experiment_id", None)
        and getattr(state, "checkpoint_interval", 0) > 0
    ):
        return None, 0, {}
    # Distinct key from the level journal (different execution shape) via a sentinel node id.
    key = run_key(state.config, tuple(n.id for n in flat) + ("__windowed__",))
    journal = RunJournal(cm.checkpoints_dir, key)
    if getattr(state, "resume_from_checkpoint", False):
        loaded = journal.load()
        if loaded is not None:
            completed, blob = loaded
            try:
                accum = dict(blob.get("accum", {}))
                logger.info("resuming windowed run from window %d", int(completed))
                return journal, int(completed), accum
            except Exception as exc:  # noqa: BLE001 - a bad journal must never block a run
                logger.warning("windowed journal restore failed (%s); running fresh", exc)
    return journal, 0, {}
