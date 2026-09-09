"""Full-run checkpoint/resume across the whole DAG, not just ASR.

Snapshots the executor's resumable state at each level boundary and resumes at the first
incomplete level on a rerun with the same config+graph. State is pickled (ItemSets/numpy
embeddings JSON can't carry); the unpicklable ``RunContext``/``DropSink`` locks are
snapshotted as plain data and rebuilt around instead.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..logging_config import get_logger
from .provenance import config_hash

logger = get_logger(__name__)

# Plain-data fields snapshotted from RunState (everything downstream nodes read). ctx and
# drop_sink are handled specially (their locks aren't picklable).
_SNAPSHOT_FIELDS = (
    "stage_times",
    "results",
)


_CODE_FINGERPRINT: Optional[str] = None


def _code_fingerprint() -> str:
    """Fingerprint of the installed evaluator source (path+size+mtime of every .py, cached per
    process): folded into ``run_key`` so a journal from OLD code is ignored after a code update
    (else a resumed run skips the very levels a fix changed, and the fix looks broken)."""
    global _CODE_FINGERPRINT
    if _CODE_FINGERPRINT is None:
        from pathlib import Path

        pkg = Path(__file__).resolve().parent.parent  # the evaluator package root
        h = hashlib.sha256()
        for p in sorted(pkg.rglob("*.py")):
            try:
                st = p.stat()
                # mtime on purpose: the journal is machine-local and a plain file copy (no git
                # metadata) must invalidate it too
                h.update(f"{p.relative_to(pkg)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
            except OSError:
                continue
        _CODE_FINGERPRINT = h.hexdigest()[:16]
    return _CODE_FINGERPRINT


def run_key(config: Any, node_ids: Tuple[str, ...]) -> str:
    """Stable id for a run: same config identity + same graph + same CODE ⇒ same key
    (resumable). Any source change yields a new key, so stale journals never resume."""

    base = config_hash(config) if config is not None else "noconfig"
    return hashlib.sha256(
        (base + "|" + ",".join(node_ids) + "|" + _code_fingerprint()).encode("utf-8")
    ).hexdigest()[:16]


def snapshot_state(state: Any) -> Dict[str, Any]:
    """Capture the resumable, picklable subset of ``state`` (no locks)."""
    blob: Dict[str, Any] = {f: getattr(state, f, None) for f in _SNAPSHOT_FIELDS}
    blob["__ctx_store__"] = dict(getattr(state.ctx, "_store", {}))
    blob["__dropped__"] = dict(getattr(state.drop_sink, "by_node", {}))
    return blob


def try_restore(state: Any, blob: Dict[str, Any]) -> bool:
    """Restore a snapshot, or leave ``state`` untouched and return False.

    Stages every value FIRST (the reads/copies are where a malformed blob fails), then
    applies — so "restore failed, running fresh" never proceeds on a half-restored state.
    """
    try:
        staged = [(f, blob[f]) for f in _SNAPSHOT_FIELDS if f in blob]
        ctx_store = dict(blob.get("__ctx_store__", {}))
        dropped = dict(blob.get("__dropped__", {}))
    except Exception as exc:  # noqa: BLE001 - a bad journal must never block a run
        logger.warning("journal restore failed (%s); running fresh", exc)
        return False
    for f, v in staged:
        setattr(state, f, v)
    state.ctx._store = ctx_store
    state.drop_sink.by_node = dropped
    return True


class RunJournal:
    """Persists the latest resumable snapshot + last completed level for one ``run_key``."""

    #: journals older than this are orphans (their code fingerprint / config no longer
    #: exists) — pruned on journal setup so key rotation doesn't accumulate stale .pkl.
    STALE_AFTER_S = 7 * 24 * 3600

    def __init__(self, checkpoints_dir: Path, key: str) -> None:
        self._path = Path(checkpoints_dir) / f"run_{key}.pkl"
        self._prune_stale(Path(checkpoints_dir))

    def _prune_stale(self, checkpoints_dir: Path) -> None:
        """Best-effort removal of old sibling journals (code-fingerprinted keys rotate on
        every source change, so orphans are expected, not exceptional)."""
        import time

        cutoff = time.time() - self.STALE_AFTER_S
        try:
            for p in checkpoints_dir.glob("run_*.pkl"):
                if p != self._path and p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
        except OSError:
            pass

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
