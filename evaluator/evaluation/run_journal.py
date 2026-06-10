"""Full-run checkpoint/resume across the whole DAG, not just ASR (task T6).

Only ASR checkpointed before, so a crash at node 8/12 recomputed everything. The ``RunJournal``
snapshots the executor's resumable state at each *level* boundary (level-granular so it composes
with T5's parallel levels) and records the last completed level. A rerun with the same config +
graph restores that snapshot and resumes at the first incomplete level; the journal is cleared on
success. A different config/graph yields a different ``run_key`` so a stale journal is ignored
(never a wrong resume).

State is pickled (it holds ItemSets / numpy embeddings / retrieved payloads that JSON can't carry,
unlike the ASR JSON checkpoint). The ``RunContext`` and ``DropSink`` locks aren't picklable, so we
snapshot their plain data (``ctx._store`` / ``drop_sink.by_node``) and rebuild around them.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..logging_config import get_logger

logger = get_logger(__name__)

# Plain-data fields snapshotted from RunState (everything downstream nodes read). ctx and
# drop_sink are handled specially (their locks aren't picklable).
_SNAPSHOT_FIELDS = (
    "all_hypotheses",
    "all_ground_truth",
    "all_embeddings",
    "audio_embeddings",
    "text_embeddings_for_fusion",
    "all_relevance",
    "all_query_ids",
    "all_results_with_scores",
    "all_retrieved",
    "retrieval_candidates",
    "retrieval_query_texts",
    "audio_emb_for_alignment",
    "text_emb_for_alignment",
    "query_opt_bypassed",
    "metrics_all_relevant",
    "wer_scores",
    "cer_scores",
    "per_query_recall5",
    "ans_detail_by_qid",
    "stage_times",
    "results",
)


def run_key(config: Any, node_ids: Tuple[str, ...]) -> str:
    """Stable id for a run: same config identity + same graph ⇒ same key (resumable)."""
    from .provenance import config_hash

    base = config_hash(config) if config is not None else "noconfig"
    return hashlib.sha256(
        (base + "|" + ",".join(node_ids)).encode("utf-8")
    ).hexdigest()[:16]


def snapshot_state(state: Any) -> Dict[str, Any]:
    """Capture the resumable, picklable subset of ``state`` (no locks)."""
    blob: Dict[str, Any] = {f: getattr(state, f, None) for f in _SNAPSHOT_FIELDS}
    blob["__ctx_store__"] = dict(getattr(state.ctx, "_store", {}))
    blob["__dropped__"] = dict(getattr(state.drop_sink, "by_node", {}))
    return blob


def restore_state(state: Any, blob: Dict[str, Any]) -> None:
    """Restore a snapshot onto ``state`` in place (rebuilding the lock-bearing stores)."""
    for f in _SNAPSHOT_FIELDS:
        if f in blob:
            setattr(state, f, blob[f])
    state.ctx._store = dict(blob.get("__ctx_store__", {}))
    state.drop_sink.by_node = dict(blob.get("__dropped__", {}))


class RunJournal:
    """Persists the latest resumable snapshot + last completed level for one ``run_key``."""

    def __init__(self, checkpoints_dir: Path, key: str) -> None:
        self._path = Path(checkpoints_dir) / f"run_{key}.pkl"

    def save(self, level_idx: int, blob: Dict[str, Any]) -> None:
        """Overwrite the single journal file with the latest state (best-effort)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as fh:
                pickle.dump({"level": level_idx, "state": blob}, fh)
            tmp.replace(self._path)  # atomic swap so a crash mid-write can't corrupt
        except Exception as exc:  # checkpoint failure must never break the run
            logger.warning("run journal save failed: %s", exc)

    def load(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Return ``(last_completed_level, state_blob)`` or ``None`` when absent/unreadable."""
        if not self._path.exists():
            return None
        try:
            with open(self._path, "rb") as fh:
                data = pickle.load(fh)
            return int(data["level"]), data["state"]
        except Exception as exc:
            logger.warning("run journal load failed (%s); ignoring", exc)
            return None

    def clear(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass
