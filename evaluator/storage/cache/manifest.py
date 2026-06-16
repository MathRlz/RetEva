"""SQLite manifest read/write for the cache."""

import logging
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Optional

from .io import _file_checksum

logger = logging.getLogger(__name__)

# Manifest *index* schema version, stored in the SQLite file's PRAGMA user_version. Distinct
# from cache_keys.CACHE_SCHEMA_VERSION, which versions the *content* keys (a key mismatch is a
# clean miss). This versions the table layout: bump only on a NON-additive change (a new
# column migrates in place below). A manifest stamped NEWER than this is rejected — the code
# can't assume it understands that layout, so reading it could silently mis-resolve paths.
_MANIFEST_SCHEMA_VERSION = 1


class CacheManifestError(RuntimeError):
    """The on-disk cache manifest is a newer, incompatible schema than this evaluator."""


class ManifestMixin:
    """SQLite manifest index operations shared by the cache manager."""

    def _connect(self):
        """Open the manifest with a write timeout so concurrent runs wait for the lock
        instead of failing immediately with ``database is locked``."""
        return sqlite3.connect(self.manifest_db_path, timeout=30)

    def _initialize_manifest_db(self) -> None:
        """Initialize the SQLite cache manifest index, gating on its schema version.

        A manifest stamped with a version GREATER than ``_MANIFEST_SCHEMA_VERSION`` (written
        by a newer evaluator) is rejected with an actionable error instead of being read
        under assumptions that may not hold. A fresh DB (``user_version == 0``) — or a
        legacy one created before versioning, which carries the current additive layout —
        is stamped to the current version.
        """
        with closing(self._connect()) as conn, conn:
            # WAL persists in the file → readers don't block the writer (concurrent runs).
            conn.execute("PRAGMA journal_mode=WAL")
            stored = conn.execute("PRAGMA user_version").fetchone()[0]
            if stored > _MANIFEST_SCHEMA_VERSION:
                raise CacheManifestError(
                    f"Cache manifest at {self.manifest_db_path} is schema v{stored}, but "
                    f"this evaluator understands up to v{_MANIFEST_SCHEMA_VERSION} (the "
                    f"manifest was written by a newer version). Clear the cache "
                    f"(`evaluator cache clear`) or point EVALUATOR_CACHE_DIR at a fresh "
                    f"directory."
                )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_type TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    stage TEXT,
                    model_name TEXT,
                    model_version TEXT,
                    input_hash TEXT,
                    config_hash TEXT,
                    payload_hash TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (cache_type, cache_key)
                )
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_entries_type_updated
                ON cache_entries(cache_type, updated_at)
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_entries_type_model_key
                ON cache_entries(cache_type, model_name, cache_key)
                """)
            # Add columns if upgrading from older schema (additive — no version bump)
            for col, col_type in [("file_size_bytes", "INTEGER"), ("checksum", "TEXT")]:
                try:
                    conn.execute(
                        f"ALTER TABLE cache_entries ADD COLUMN {col} {col_type}"
                    )
                except sqlite3.OperationalError:
                    pass  # column already exists
            if stored != _MANIFEST_SCHEMA_VERSION:
                # Stamp the current version (PRAGMA forbids parameter binding; the value
                # is our own trusted int constant).
                conn.execute(f"PRAGMA user_version = {_MANIFEST_SCHEMA_VERSION}")
            conn.commit()

    def _upsert_manifest_entry(
        self,
        *,
        cache_type: str,
        cache_key: str,
        artifact_path: Path,
        stage: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        input_hash: Optional[str] = None,
        config_hash: Optional[str] = None,
        payload_hash: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        timestamp = datetime.now().isoformat()
        file_size = artifact_path.stat().st_size if artifact_path.exists() else 0
        checksum = _file_checksum(artifact_path) if artifact_path.exists() else None
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO cache_entries(
                    cache_type, cache_key, artifact_path, stage, model_name, model_version,
                    input_hash, config_hash, payload_hash, created_at, updated_at,
                    file_size_bytes, checksum
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_type, cache_key) DO UPDATE SET
                    artifact_path=excluded.artifact_path,
                    stage=excluded.stage,
                    model_name=excluded.model_name,
                    model_version=excluded.model_version,
                    input_hash=excluded.input_hash,
                    config_hash=excluded.config_hash,
                    payload_hash=excluded.payload_hash,
                    updated_at=excluded.updated_at,
                    file_size_bytes=excluded.file_size_bytes,
                    checksum=excluded.checksum
                """,
                (
                    cache_type,
                    cache_key,
                    self._rel_artifact_path(artifact_path),
                    stage,
                    model_name,
                    model_version,
                    input_hash,
                    config_hash,
                    payload_hash,
                    timestamp,
                    timestamp,
                    file_size,
                    checksum,
                ),
            )
            conn.commit()
        if self.max_size_gb and self.max_size_gb > 0:
            self._enforce_size_limit()

    def _rel_artifact_path(self, artifact_path: Path) -> str:
        """Path stored in the manifest, relative to ``cache_dir`` when possible (T9).

        A relative path keeps the cache portable: move the cache dir or mount it at a
        different absolute location (another machine/container) and the entry still resolves.
        A path outside ``cache_dir`` (shouldn't happen for our artifacts) is kept absolute.
        """
        try:
            return str(
                Path(artifact_path).resolve().relative_to(self.cache_dir.resolve())
            )
        except ValueError:
            return str(artifact_path)

    def _abs_artifact_path(self, stored: str) -> Path:
        """Resolve a manifest-stored path back to an absolute path against the *current*
        ``cache_dir`` (T9). Legacy absolute entries are returned as-is (back-compat)."""
        p = Path(stored)
        return p if p.is_absolute() else (self.cache_dir / p)

    def _resolve_manifest_path(
        self, cache_type: str, cache_key: str, fallback_path: Path
    ) -> Optional[Path]:
        """Resolve artifact path from manifest index, with deterministic fallback.

        Verifies file checksum when available and removes corrupted entries.
        """
        if not self.enabled:
            return None

        path: Optional[Path] = None
        stored_checksum: Optional[str] = None
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                """
                SELECT artifact_path, checksum FROM cache_entries
                WHERE cache_type = ? AND cache_key = ?
                """,
                (cache_type, cache_key),
            ).fetchone()
            if row:
                path = self._abs_artifact_path(row[0])
                stored_checksum = row[1]

        if path is not None:
            if path.exists():
                if stored_checksum and _file_checksum(path) != stored_checksum:
                    logger.warning(
                        "Cache corruption detected for %s/%s, removing",
                        cache_type,
                        cache_key,
                    )
                    path.unlink(missing_ok=True)
                else:
                    # Touch updated_at for LRU ordering
                    with closing(self._connect()) as conn, conn:
                        conn.execute(
                            "UPDATE cache_entries SET updated_at = ? "
                            "WHERE cache_type = ? AND cache_key = ?",
                            (datetime.now().isoformat(), cache_type, cache_key),
                        )
                        conn.commit()
                    return path
            # stale or corrupted manifest entry: remove
            with closing(self._connect()) as conn, conn:
                conn.execute(
                    "DELETE FROM cache_entries WHERE cache_type = ? AND cache_key = ?",
                    (cache_type, cache_key),
                )
                conn.commit()

        if fallback_path.exists():
            return fallback_path
        return None
