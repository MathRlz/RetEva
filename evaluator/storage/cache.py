"""Caching infrastructure for expensive computations."""
import hashlib
import json
import logging
import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

from .cache_keys import (
    model_key,
    embedding_key,
    transcription_key,
    audio_embedding_key,
    unique_texts_key,
    unique_texts_manifest_key,
)

logger = logging.getLogger(__name__)


def _file_checksum(path: Path, block_size: int = 1 << 16) -> str:
    """Compute SHA-256 checksum of a file (first 64KB for speed)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            h.update(f.read(block_size))
    except OSError:
        return ""
    return h.hexdigest()


class CacheManager:
    """Manages disk-based caching for expensive computations."""

    def __init__(self, cache_dir: str = ".cache", enabled: bool = True,
                 max_size_gb: float = 0):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.max_size_gb = max_size_gb
        self.manifest_db_path = self.cache_dir / "cache_manifest.sqlite"
        
        # Subdirectories for different cache types
        self.asr_features_dir = self.cache_dir / "asr_features"
        self.transcriptions_dir = self.cache_dir / "transcriptions"
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.audio_embeddings_dir = self.cache_dir / "audio_embeddings"
        self.vector_db_dir = self.cache_dir / "vector_dbs"
        self.checkpoints_dir = self.cache_dir / "checkpoints"
        
        # Centralized mapping for cache directories (replaces if-elif chains)
        self._cache_dirs: Dict[str, Path] = {
            "asr_features": self.asr_features_dir,
            "transcriptions": self.transcriptions_dir,
            "embeddings": self.embeddings_dir,
            "audio_embeddings": self.audio_embeddings_dir,
            "vector_db": self.vector_db_dir,
            "checkpoints": self.checkpoints_dir,
        }
        
        if self.enabled:
            for dir_path in self._cache_dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)
            self._initialize_manifest_db()

    def _initialize_manifest_db(self) -> None:
        """Initialize SQLite cache manifest index."""
        with sqlite3.connect(self.manifest_db_path) as conn:
            conn.execute(
                """
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
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_entries_type_updated
                ON cache_entries(cache_type, updated_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_entries_type_model_key
                ON cache_entries(cache_type, model_name, cache_key)
                """
            )
            # Add columns if upgrading from older schema
            for col, col_type in [("file_size_bytes", "INTEGER"), ("checksum", "TEXT")]:
                try:
                    conn.execute(f"ALTER TABLE cache_entries ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass  # column already exists
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
        with sqlite3.connect(self.manifest_db_path) as conn:
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
                    str(artifact_path),
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
        with sqlite3.connect(self.manifest_db_path) as conn:
            row = conn.execute(
                """
                SELECT artifact_path, checksum FROM cache_entries
                WHERE cache_type = ? AND cache_key = ?
                """,
                (cache_type, cache_key),
            ).fetchone()
            if row:
                path = Path(row[0])
                stored_checksum = row[1]

        if path is not None:
            if path.exists():
                if stored_checksum and _file_checksum(path) != stored_checksum:
                    logger.warning(
                        "Cache corruption detected for %s/%s, removing",
                        cache_type, cache_key,
                    )
                    path.unlink(missing_ok=True)
                else:
                    # Touch updated_at for LRU ordering
                    with sqlite3.connect(self.manifest_db_path) as conn:
                        conn.execute(
                            "UPDATE cache_entries SET updated_at = ? "
                            "WHERE cache_type = ? AND cache_key = ?",
                            (datetime.now().isoformat(), cache_type, cache_key),
                        )
                        conn.commit()
                    return path
            # stale or corrupted manifest entry: remove
            with sqlite3.connect(self.manifest_db_path) as conn:
                conn.execute(
                    "DELETE FROM cache_entries WHERE cache_type = ? AND cache_key = ?",
                    (cache_type, cache_key),
                )
                conn.commit()

        if fallback_path.exists():
            return fallback_path
        return None
    
    def _get_cache_path(self, cache_type: str, cache_key: str, extension: str = "") -> Path:
        """Get cache file path for a given type and key."""
        if cache_type not in self._cache_dirs:
            raise ValueError(
                f"Unknown cache type: {cache_type}. "
                f"Valid types: {list(self._cache_dirs.keys())}"
            )
        
        base_dir = self._cache_dirs[cache_type]
        return base_dir / f"{cache_key}{extension}"

    def get_asr_features(self, audio_hash: str, model_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve cached ASR features (features, attention_mask)."""
        if not self.enabled:
            return None
        
        cache_key = model_key(audio_hash, model_name)
        fallback_path = self._get_cache_path("asr_features", cache_key, ".npz")
        features_path = self._resolve_manifest_path("asr_features", cache_key, fallback_path)
        
        if features_path is not None and features_path.exists():
            data = np.load(features_path)
            return data['features'], data['attention_mask']
        return None
    
    def set_asr_features(self, audio_hash: str, model_name: str, 
                        features: np.ndarray, attention_mask: np.ndarray):
        """Cache ASR features."""
        if not self.enabled:
            return
        
        cache_key = model_key(audio_hash, model_name)
        features_path = self._get_cache_path("asr_features", cache_key, ".npz")
        
        np.savez_compressed(features_path, features=features, attention_mask=attention_mask)
        self._upsert_manifest_entry(
            cache_type="asr_features",
            cache_key=cache_key,
            artifact_path=features_path,
            stage="asr_features",
            model_name=model_name,
            input_hash=audio_hash,
        )
    
    def get_transcription(self, audio_hash: str, model_name: str, language: Optional[str] = None) -> Optional[str]:
        """Retrieve cached transcription."""
        if not self.enabled:
            return None
        
        cache_key = transcription_key(audio_hash, model_name, language)
        fallback_path = self._get_cache_path("transcriptions", cache_key, ".json")
        trans_path = self._resolve_manifest_path("transcriptions", cache_key, fallback_path)
        
        if trans_path is not None and trans_path.exists():
            with open(trans_path, 'r') as f:
                data = json.load(f)
                return data['transcription']
        return None
    
    def set_transcription(self, audio_hash: str, model_name: str, 
                         transcription: str, language: Optional[str] = None):
        """Cache transcription."""
        if not self.enabled:
            return
        
        cache_key = transcription_key(audio_hash, model_name, language)
        trans_path = self._get_cache_path("transcriptions", cache_key, ".json")
        
        with open(trans_path, 'w') as f:
            json.dump({
                'transcription': transcription,
                'model_name': model_name,
                'language': language,
                'timestamp': datetime.now().isoformat()
            }, f)
        self._upsert_manifest_entry(
            cache_type="transcriptions",
            cache_key=cache_key,
            artifact_path=trans_path,
            stage="transcription",
            model_name=model_name,
            input_hash=audio_hash,
            config_hash=str(language) if language is not None else None,
        )
    
    def get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Retrieve cached text embedding."""
        if not self.enabled:
            return None
        
        cache_key = embedding_key(text, model_name)
        fallback_path = self._get_cache_path("embeddings", cache_key, ".npy")
        emb_path = self._resolve_manifest_path("embeddings", cache_key, fallback_path)
        
        if emb_path is not None and emb_path.exists():
            return np.load(emb_path)
        return None
    
    def set_embedding(self, text: str, model_name: str, embedding: np.ndarray):
        """Cache text embedding."""
        if not self.enabled:
            return
        
        cache_key = embedding_key(text, model_name)
        emb_path = self._get_cache_path("embeddings", cache_key, ".npy")
        
        np.save(emb_path, embedding)
        self._upsert_manifest_entry(
            cache_type="embeddings",
            cache_key=cache_key,
            artifact_path=emb_path,
            stage="text_embedding",
            model_name=model_name,
            input_hash=text,
        )
    
    def get_embeddings_batch(self, texts: list[str], model_name: str) -> Optional[np.ndarray]:
        """Retrieve cached embeddings for a batch of texts."""
        if not self.enabled:
            return None
        
        embeddings = []
        for text in texts:
            emb = self.get_embedding(text, model_name)
            if emb is None:
                return None  # If any is missing, return None
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def set_embeddings_batch(self, texts: list[str], model_name: str, embeddings: np.ndarray):
        """Cache embeddings for a batch of texts."""
        if not self.enabled:
            return
        
        for text, embedding in zip(texts, embeddings):
            self.set_embedding(text, model_name, embedding)
    
    def get_audio_embedding(self, audio_hash: str, model_name: str) -> Optional[np.ndarray]:
        """Retrieve cached audio embedding."""
        if not self.enabled:
            return None
        
        cache_key = audio_embedding_key(audio_hash, model_name)
        fallback_path = self._get_cache_path("audio_embeddings", cache_key, ".npy")
        emb_path = self._resolve_manifest_path("audio_embeddings", cache_key, fallback_path)
        
        if emb_path is not None and emb_path.exists():
            return np.load(emb_path)
        return None
    
    def set_audio_embedding(self, audio_hash: str, model_name: str, embedding: np.ndarray):
        """Cache audio embedding."""
        if not self.enabled:
            return
        
        cache_key = audio_embedding_key(audio_hash, model_name)
        emb_path = self._get_cache_path("audio_embeddings", cache_key, ".npy")
        
        np.save(emb_path, embedding)
        self._upsert_manifest_entry(
            cache_type="audio_embeddings",
            cache_key=cache_key,
            artifact_path=emb_path,
            stage="audio_embedding",
            model_name=model_name,
            input_hash=audio_hash,
        )
    
    def get_vector_db(self, db_key: str) -> Optional[Tuple[np.ndarray, list[str]]]:
        """Retrieve cached vector database (vectors, texts)."""
        if not self.enabled:
            return None
        
        fallback_vectors_path = self._get_cache_path("vector_db", db_key, "_vectors.npy")
        vectors_path = self._resolve_manifest_path("vector_db", db_key, fallback_vectors_path)
        metadata_path = self._get_cache_path("vector_db", db_key, "_metadata.json")
        
        if vectors_path is not None and vectors_path.exists() and metadata_path.exists():
            vectors = np.load(vectors_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return vectors, metadata['texts']
        return None
    
    def set_vector_db(self, db_key: str, vectors: np.ndarray, texts: list[str]):
        """Cache vector database."""
        if not self.enabled:
            return
        
        vectors_path = self._get_cache_path("vector_db", db_key, "_vectors.npy")
        metadata_path = self._get_cache_path("vector_db", db_key, "_metadata.json")
        
        np.save(vectors_path, vectors)
        with open(metadata_path, 'w') as f:
            json.dump({
                'texts': texts,
                'num_vectors': len(vectors),
                'vector_dim': vectors.shape[1] if len(vectors) > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }, f)
        self._upsert_manifest_entry(
            cache_type="vector_db",
            cache_key=db_key,
            artifact_path=vectors_path,
            stage="vector_db",
            input_hash=db_key,
        )
    
    def get_checkpoint(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load evaluation checkpoint."""
        if not self.enabled:
            return None
        
        fallback_path = self._get_cache_path("checkpoints", experiment_id, ".json")
        checkpoint_path = self._resolve_manifest_path("checkpoints", experiment_id, fallback_path)
        
        if checkpoint_path is not None and checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_unique_texts(
        self,
        dataset_name: Optional[str] = None,
        dataset_size: Optional[int] = None,
        *,
        dataset_fingerprint: Optional[str] = None,
        preprocessing_fingerprint: Optional[str] = None,
    ) -> Optional[List[str]]:
        """Retrieve cached unique texts for a dataset."""
        if not self.enabled:
            return None

        if dataset_fingerprint is not None:
            cache_key = unique_texts_manifest_key(
                dataset_fp=dataset_fingerprint,
                preprocessing_fp=preprocessing_fingerprint,
            )
        elif dataset_name is not None and dataset_size is not None:
            cache_key = unique_texts_key(dataset_name, dataset_size)
        else:
            raise ValueError(
                "Provide either dataset_fingerprint or both dataset_name and dataset_size"
            )
        cache_path = self._get_cache_path("vector_db", f"unique_texts_{cache_key}", ".json")
        
        manifest_key = f"unique_texts_{cache_key}"
        resolved = self._resolve_manifest_path("vector_db", manifest_key, cache_path)
        if resolved is not None and resolved.exists():
            cache_path = resolved
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return data['unique_texts']
        return None
    
    def set_unique_texts(
        self,
        dataset_name: Optional[str] = None,
        dataset_size: Optional[int] = None,
        unique_texts: Optional[List[str]] = None,
        *,
        dataset_fingerprint: Optional[str] = None,
        preprocessing_fingerprint: Optional[str] = None,
    ):
        """Cache unique texts for a dataset."""
        if not self.enabled:
            return

        if unique_texts is None:
            raise ValueError("unique_texts must be provided")

        if dataset_fingerprint is not None:
            cache_key = unique_texts_manifest_key(
                dataset_fp=dataset_fingerprint,
                preprocessing_fp=preprocessing_fingerprint,
            )
            metadata: Dict[str, Any] = {
                'dataset_fingerprint': dataset_fingerprint,
                'preprocessing_fingerprint': preprocessing_fingerprint,
            }
        elif dataset_name is not None and dataset_size is not None:
            cache_key = unique_texts_key(dataset_name, dataset_size)
            metadata = {
                'dataset_name': dataset_name,
                'dataset_size': dataset_size,
            }
        else:
            raise ValueError(
                "Provide either dataset_fingerprint or both dataset_name and dataset_size"
            )

        cache_path = self._get_cache_path("vector_db", f"unique_texts_{cache_key}", ".json")
        
        with open(cache_path, 'w') as f:
            json.dump({
                **metadata,
                'unique_texts': unique_texts,
                'num_unique': len(unique_texts),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        self._upsert_manifest_entry(
            cache_type="vector_db",
            cache_key=f"unique_texts_{cache_key}",
            artifact_path=cache_path,
            stage="unique_texts",
            input_hash=dataset_fingerprint or dataset_name,
            config_hash=preprocessing_fingerprint,
        )
    
    def set_checkpoint(self, experiment_id: str, checkpoint_data: Dict[str, Any]):
        """Save evaluation checkpoint."""
        if not self.enabled:
            return
        
        checkpoint_path = self._get_cache_path("checkpoints", experiment_id, ".json")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        self._upsert_manifest_entry(
            cache_type="checkpoints",
            cache_key=experiment_id,
            artifact_path=checkpoint_path,
            stage="checkpoint",
            input_hash=experiment_id,
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics (consolidates size and file count)."""
        stats: Dict[str, Any] = {
            "total_size_bytes": 0,
            "total_size_human": "",
            "by_category": {}
        }
        
        for cache_type, cache_dir in self._cache_dirs.items():
            if not cache_dir.exists():
                stats["by_category"][cache_type] = {
                    "size_bytes": 0,
                    "size_human": "0 B",
                    "file_count": 0
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
                "file_count": file_count
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
        for unit in ['B', 'KB', 'MB', 'GB']:
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
                with sqlite3.connect(self.manifest_db_path) as conn:
                    conn.execute("DELETE FROM cache_entries WHERE cache_type = ?", (cache_type,))
                    conn.commit()
            logger.info(f"Cleared {file_count} files from {cache_type} cache")
    
    def get_cache_size_bytes(self) -> int:
        """Return total size of cached artifacts tracked by manifest."""
        if not self.enabled or not self.manifest_db_path.exists():
            return 0
        with sqlite3.connect(self.manifest_db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(file_size_bytes), 0) FROM cache_entries"
            ).fetchone()
        return int(row[0]) if row else 0

    def _enforce_size_limit(self) -> None:
        """Evict least-recently-used entries until cache is within max_size_gb."""
        if not self.max_size_gb or self.max_size_gb <= 0 or not self.enabled:
            return
        limit_bytes = int(self.max_size_gb * (1 << 30))
        current = self.get_cache_size_bytes()
        if current <= limit_bytes:
            return

        evicted = 0
        with sqlite3.connect(self.manifest_db_path) as conn:
            rows = conn.execute(
                """
                SELECT cache_type, cache_key, artifact_path,
                       COALESCE(file_size_bytes, 0)
                FROM cache_entries
                ORDER BY updated_at ASC
                """
            ).fetchall()
            for cache_type, cache_key, artifact_path, size in rows:
                if current <= limit_bytes:
                    break
                try:
                    p = Path(artifact_path)
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
                evicted, self.max_size_gb,
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
            with sqlite3.connect(self.manifest_db_path) as conn:
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
                        path = Path(artifact_path)
                        if path.exists():
                            path.unlink()
                            cleared_count += 1
                        conn.execute(
                            "DELETE FROM cache_entries WHERE cache_type = ? AND cache_key = ?",
                            (cache_type, cache_key),
                        )
                    except Exception as e:
                        logger.warning(f"Failed to delete manifest entry {artifact_path}: {e}")
                conn.commit()

        logger.info(f"Cleared {cleared_count} cache entries for model: {model_name}")
        return cleared_count
