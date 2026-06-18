"""Eviction, size-budget, and cache-clearing logic."""

import logging
from contextlib import closing
from typing import Any, Dict

logger = logging.getLogger(__name__)


class EvictionMixin:
    """Cache statistics, size enforcement, and invalidation helpers."""

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics (consolidates size and file count)."""
        stats: Dict[str, Any] = {
            "total_size_bytes": 0,
            "total_size_human": "",
            "by_category": {},
        }

        for cache_type, cache_dir in self._cache_dirs.items():
            if not cache_dir.exists():
                stats["by_category"][cache_type] = {
                    "size_bytes": 0,
                    "size_human": "0 B",
                    "file_count": 0,
                }
                continue

            # Calculate size and count in single pass
            size_bytes = 0
            file_count = 0
            for file_path in cache_dir.rglob("*"):
                if file_path.is_file():
                    size_bytes += file_path.stat().st_size
                    file_count += 1

            stats["by_category"][cache_type] = {
                "size_bytes": size_bytes,
                "size_human": self._human_readable_size(size_bytes),
                "file_count": file_count,
            }
            stats["total_size_bytes"] += size_bytes

        stats["total_size_human"] = self._human_readable_size(stats["total_size_bytes"])
        stats["sizes_bytes"] = {
            cache_type: info["size_bytes"]
            for cache_type, info in stats["by_category"].items()
        }
        stats["sizes_human"] = {
            cache_type: info["size_human"]
            for cache_type, info in stats["by_category"].items()
        }
        stats["file_counts"] = {
            cache_type: info["file_count"]
            for cache_type, info in stats["by_category"].items()
        }
        stats["sizes_bytes"]["total"] = stats["total_size_bytes"]
        stats["sizes_human"]["total"] = stats["total_size_human"]
        stats["file_counts"]["total"] = sum(
            info["file_count"] for info in stats["by_category"].values()
        )
        return stats

    def get_cache_size(self) -> Dict[str, int]:
        """Get size of each cache type in bytes (legacy method, uses get_cache_stats)."""
        full_stats = self.get_cache_stats()
        sizes = {
            cache_type: info["size_bytes"]
            for cache_type, info in full_stats["by_category"].items()
        }
        sizes["total"] = full_stats["total_size_bytes"]
        return sizes

    def _human_readable_size(self, size_bytes: int | float) -> str:
        """Convert bytes to human readable format."""
        size: float = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"

    # Cache invalidation
    def clear_all(self):
        """Clear all caches."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.__init__(str(self.cache_dir), self.enabled)

    def clear_type(self, cache_type: str):
        """Clear specific cache type."""
        import shutil

        if cache_type not in self._cache_dirs:
            raise ValueError(
                f"Unknown cache type: {cache_type}. "
                f"Valid types: {list(self._cache_dirs.keys())}"
            )

        cache_dir = self._cache_dirs[cache_type]

        if cache_dir.exists():
            file_count = len(list(cache_dir.rglob("*")))
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            if self.enabled and self.manifest_db_path.exists():
                with closing(self._connect()) as conn, conn:
                    conn.execute(
                        "DELETE FROM cache_entries WHERE cache_type = ?", (cache_type,)
                    )
                    conn.commit()
            logger.info(f"Cleared {file_count} files from {cache_type} cache")

    def get_cache_size_bytes(self) -> int:
        """Return total size of cached artifacts tracked by manifest."""
        if not self.enabled or not self.manifest_db_path.exists():
            return 0
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(file_size_bytes), 0) FROM cache_entries"
            ).fetchone()
        return int(row[0]) if row else 0

    def compact_manifest(self) -> int:
        """Drop manifest rows whose artifact file no longer exists (audit M4).

        The size-limit eviction only runs when ``max_size_gb`` triggers (default 0 =
        disabled), so rows for externally-deleted / stale artifacts otherwise accumulate
        forever on long-lived servers. Runs at manager init; returns the row count removed.
        """
        if not self.enabled or not self.manifest_db_path.exists():
            return 0
        removed = 0
        with closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT cache_type, cache_key, artifact_path FROM cache_entries"
            ).fetchall()
            for cache_type, cache_key, artifact_path in rows:
                try:
                    if self._abs_artifact_path(artifact_path).exists():
                        continue
                except OSError:
                    pass  # unresolvable path == stale
                conn.execute(
                    "DELETE FROM cache_entries WHERE cache_type = ? AND cache_key = ?",
                    (cache_type, cache_key),
                )
                removed += 1
            conn.commit()
        if removed:
            logger.info(
                "Cache manifest compaction: removed %d orphaned entries", removed
            )
        return removed

    def _enforce_size_limit(self) -> None:
        """Evict least-recently-used entries until cache is within max_size_gb."""
        if not self.max_size_gb or self.max_size_gb <= 0 or not self.enabled:
            return
        limit_bytes = int(self.max_size_gb * (1 << 30))
        current = self.get_cache_size_bytes()
        if current <= limit_bytes:
            return

        evicted = 0
        with closing(self._connect()) as conn, conn:
            rows = conn.execute("""
                SELECT cache_type, cache_key, artifact_path,
                       COALESCE(file_size_bytes, 0)
                FROM cache_entries
                ORDER BY updated_at ASC
                """).fetchall()
            for cache_type, cache_key, artifact_path, size in rows:
                if current <= limit_bytes:
                    break
                try:
                    p = self._abs_artifact_path(artifact_path)
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass
                conn.execute(
                    "DELETE FROM cache_entries WHERE cache_type = ? AND cache_key = ?",
                    (cache_type, cache_key),
                )
                current -= size
                evicted += 1
            conn.commit()
        if evicted:
            logger.info(
                "Cache eviction: removed %d entries to stay within %.1f GB limit",
                evicted,
                self.max_size_gb,
            )

    def clear_model(self, model_name: str) -> int:
        """
        Clear all caches related to a specific model.

        This searches for cached files that contain the model name in their key
        and removes them. This is a heuristic approach based on how cache keys
        are generated (they include model names).

        Args:
            model_name: Name of the model to clear caches for.

        Returns:
            Number of cache entries cleared.
        """
        cleared_count = 0

        if self.enabled and self.manifest_db_path.exists():
            with closing(self._connect()) as conn, conn:
                rows = conn.execute(
                    """
                    SELECT cache_type, cache_key, artifact_path
                    FROM cache_entries
                    WHERE model_name = ?
                    """,
                    (model_name,),
                ).fetchall()
                for cache_type, cache_key, artifact_path in rows:
                    try:
                        path = self._abs_artifact_path(artifact_path)
                        if path.exists():
                            path.unlink()
                            cleared_count += 1
                        conn.execute(
                            "DELETE FROM cache_entries WHERE cache_type = ? AND cache_key = ?",
                            (cache_type, cache_key),
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete manifest entry {artifact_path}: {e}"
                        )
                conn.commit()

        logger.info(f"Cleared {cleared_count} cache entries for model: {model_name}")
        return cleared_count
