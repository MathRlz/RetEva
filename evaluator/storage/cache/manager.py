"""Caching infrastructure for expensive computations."""

import logging
import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

from ..cache_keys import (
    model_key,
    embedding_key,
    transcription_key,
    audio_embedding_key,
    unique_texts_key,
    unique_texts_manifest_key,
)
from .eviction import EvictionMixin
from .io import IoMixin
from .manifest import ManifestMixin

logger = logging.getLogger(__name__)


class CacheManager(ManifestMixin, IoMixin, EvictionMixin):
    """Manages disk-based caching for expensive computations."""

    def __init__(
        self, cache_dir: str = ".cache", enabled: bool = True, max_size_gb: float = 0
    ):
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
        self.synthesized_audio_dir = self.cache_dir / "synthesized_audio"

        # Centralized mapping for cache directories (replaces if-elif chains)
        self._cache_dirs: Dict[str, Path] = {
            "asr_features": self.asr_features_dir,
            "transcriptions": self.transcriptions_dir,
            "embeddings": self.embeddings_dir,
            "audio_embeddings": self.audio_embeddings_dir,
            "vector_db": self.vector_db_dir,
            "checkpoints": self.checkpoints_dir,
            "synthesized_audio": self.synthesized_audio_dir,
        }

        if self.enabled:
            for dir_path in self._cache_dirs.values():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    # A single unwritable category shouldn't crash the whole cache; that
                    # category just degrades to no-op (get returns None, writes are skipped).
                    logger.warning("Could not create cache dir %s: %s", dir_path, exc)
            try:
                self._initialize_manifest_db()
                # Drop rows whose artifact file is gone (M4) — keeps the manifest bounded
                # even with the size limit disabled (max_size_gb=0, the default).
                self.compact_manifest()
            except (sqlite3.OperationalError, OSError) as exc:
                # An unwritable / read-only cache_dir (e.g. owned by another user) must not
                # crash the run — disable the cache and continue uncached. Every cache op
                # gates on `self.enabled`, so this degrades the whole manager to a no-op.
                logger.warning(
                    "Cache manifest at %s is not writable (%s) — disabling cache for "
                    "this run.", self.manifest_db_path, exc,
                )
                self.enabled = False

    def _get_cache_path(
        self, cache_type: str, cache_key: str, extension: str = ""
    ) -> Path:
        """Get cache file path for a given type and key."""
        if cache_type not in self._cache_dirs:
            raise ValueError(
                f"Unknown cache type: {cache_type}. "
                f"Valid types: {list(self._cache_dirs.keys())}"
            )

        base_dir = self._cache_dirs[cache_type]
        return base_dir / f"{cache_key}{extension}"

    def get_asr_features(
        self, audio_hash: str, model_name: str, model_version: Optional[str] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Retrieve cached ASR features (features, attention_mask)."""
        cache_key = model_key(audio_hash, model_name, model_version)
        data = self._load_npz_cache("asr_features", cache_key, ".npz")
        if data is None:
            return None
        return data["features"], data["attention_mask"]

    def set_asr_features(
        self,
        audio_hash: str,
        model_name: str,
        features: np.ndarray,
        attention_mask: np.ndarray,
        model_version: Optional[str] = None,
    ):
        """Cache ASR features."""
        cache_key = model_key(audio_hash, model_name, model_version)
        self._save_npz_cache(
            cache_type="asr_features",
            cache_key=cache_key,
            extension=".npz",
            stage="asr_features",
            model_name=model_name,
            model_version=model_version,
            input_hash=audio_hash,
            features=features,
            attention_mask=attention_mask,
        )

    def get_transcription(
        self,
        audio_hash: str,
        model_name: str,
        language: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> Optional[str]:
        """Retrieve cached transcription."""
        cache_key = transcription_key(audio_hash, model_name, language, model_version)
        payload = self._load_json_cache("transcriptions", cache_key, ".json")
        if payload is None:
            return None
        return payload.get("transcription")

    def set_transcription(
        self,
        audio_hash: str,
        model_name: str,
        transcription: str,
        language: Optional[str] = None,
        model_version: Optional[str] = None,
    ):
        """Cache transcription."""
        cache_key = transcription_key(audio_hash, model_name, language, model_version)
        payload = {
            "transcription": transcription,
            "model_name": model_name,
            "language": language,
            "timestamp": datetime.now().isoformat(),
        }
        self._save_json_cache(
            cache_type="transcriptions",
            cache_key=cache_key,
            extension=".json",
            payload=payload,
            stage="transcription",
            model_name=model_name,
            model_version=model_version,
            input_hash=audio_hash,
            config_hash=str(language) if language is not None else None,
        )

    def get_embedding(
        self, text: str, model_name: str, model_version: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Retrieve cached text embedding."""
        cache_key = embedding_key(text, model_name, model_version)
        return self._load_numpy_cache("embeddings", cache_key, ".npy")

    def set_embedding(
        self,
        text: str,
        model_name: str,
        embedding: np.ndarray,
        model_version: Optional[str] = None,
    ):
        """Cache text embedding."""
        cache_key = embedding_key(text, model_name, model_version)
        self._save_numpy_cache(
            cache_type="embeddings",
            cache_key=cache_key,
            extension=".npy",
            array=embedding,
            stage="text_embedding",
            model_name=model_name,
            model_version=model_version,
            input_hash=text,
        )

    def get_embeddings_batch(
        self, texts: List[str], model_name: str, model_version: Optional[str] = None
    ) -> List[Optional[np.ndarray]]:
        """Retrieve cached embeddings for a batch of texts.

        Returns a list parallel to ``texts`` where each element is the cached
        embedding or ``None`` if not cached.  Callers should only embed the
        ``None`` slots rather than re-embedding the entire batch.
        """
        if not self.enabled:
            return [None] * len(texts)
        return [self.get_embedding(text, model_name, model_version) for text in texts]

    def set_embeddings_batch(
        self,
        texts: List[str],
        model_name: str,
        embeddings: np.ndarray,
        model_version: Optional[str] = None,
    ):
        """Cache embeddings for a batch of texts."""
        for text, embedding in zip(texts, embeddings):
            self.set_embedding(text, model_name, embedding, model_version)

    def get_audio_embedding(
        self, audio_hash: str, model_name: str, model_version: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Retrieve cached audio embedding."""
        cache_key = audio_embedding_key(audio_hash, model_name, model_version)
        return self._load_numpy_cache("audio_embeddings", cache_key, ".npy")

    def set_audio_embedding(
        self,
        audio_hash: str,
        model_name: str,
        embedding: np.ndarray,
        model_version: Optional[str] = None,
    ):
        """Cache audio embedding."""
        cache_key = audio_embedding_key(audio_hash, model_name, model_version)
        self._save_numpy_cache(
            cache_type="audio_embeddings",
            cache_key=cache_key,
            extension=".npy",
            array=embedding,
            stage="audio_embedding",
            model_name=model_name,
            model_version=model_version,
            input_hash=audio_hash,
        )

    def get_vector_db(self, db_key: str) -> Optional[Tuple[np.ndarray, list[str]]]:
        """Retrieve cached vector database (vectors, texts)."""
        vectors = self._load_numpy_cache("vector_db", db_key, "_vectors.npy")
        metadata_path = self._get_cache_path("vector_db", db_key, "_metadata.json")
        if vectors is None or not metadata_path.exists():
            return None
        metadata = self._read_json_file(metadata_path)
        return vectors, metadata["texts"]

    def set_vector_db(self, db_key: str, vectors: np.ndarray, texts: list[str]):
        """Cache vector database."""
        if not self.enabled:
            return
        metadata_path = self._get_cache_path("vector_db", db_key, "_metadata.json")
        self._save_numpy_cache(
            cache_type="vector_db",
            cache_key=db_key,
            extension="_vectors.npy",
            array=vectors,
            stage="vector_db",
            input_hash=db_key,
        )
        payload = {
            "texts": texts,
            "num_vectors": len(vectors),
            "vector_dim": vectors.shape[1] if len(vectors) > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }
        self._write_json_file(metadata_path, payload)

    def delete_vector_db(self, db_key: str) -> bool:
        """Delete a cached vector DB (artifacts + manifest entry).

        Returns True if anything was removed. Used to drop the index cache tied to a
        specific run (its ``db_key`` is stored in the run's metadata).
        """
        from contextlib import closing

        removed = False
        # Deterministic artifact paths (_vectors.npy + _metadata.json).
        for extension in ("_vectors.npy", "_metadata.json"):
            path = self._get_cache_path("vector_db", db_key, extension)
            if path.exists():
                path.unlink(missing_ok=True)
                removed = True
        # Manifest row (the .npy is tracked there; remove it and its artifact too).
        if self.manifest_db_path.exists():
            with closing(self._connect()) as conn, conn:
                row = conn.execute(
                    "SELECT artifact_path FROM cache_entries "
                    "WHERE cache_type='vector_db' AND cache_key=?",
                    (db_key,),
                ).fetchone()
                if row and row[0]:
                    artifact = self._abs_artifact_path(row[0])
                    if artifact.exists():
                        artifact.unlink(missing_ok=True)
                        removed = True
                cursor = conn.execute(
                    "DELETE FROM cache_entries WHERE cache_type='vector_db' AND cache_key=?",
                    (db_key,),
                )
                conn.commit()
                removed = removed or cursor.rowcount > 0
        return removed

    def get_checkpoint(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load evaluation checkpoint."""
        return self._load_json_cache("checkpoints", experiment_id, ".json")

    # ── Synthesized (TTS) audio ───────────────────────────────────────
    def synthesized_audio_path(self, cache_key: str) -> Path:
        """Path where a synthesized-audio WAV for *cache_key* lives in the cache."""
        return self._get_cache_path("synthesized_audio", cache_key, ".wav")

    def get_synthesized_audio(self, cache_key: str) -> Optional[Path]:
        """Return the cached WAV path for *cache_key* if present (and caching enabled)."""
        if not self.enabled:
            return None
        path = self.synthesized_audio_path(cache_key)
        return path if path.exists() else None

    def register_synthesized_audio(self, cache_key: str, path: Path) -> None:
        """Record a synthesized-audio WAV in the manifest (for stats + clear_all/clear_type)."""
        if not self.enabled:
            return
        self._upsert_manifest_entry(
            cache_type="synthesized_audio",
            cache_key=cache_key,
            artifact_path=path,
            stage="synthesis",
        )

    def get_unique_texts(
        self,
        dataset_name: Optional[str] = None,
        dataset_size: Optional[int] = None,
        *,
        dataset_fingerprint: Optional[str] = None,
        preprocessing_fingerprint: Optional[str] = None,
    ) -> Optional[List[str]]:
        """Retrieve cached unique texts for a dataset."""
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
        manifest_key = f"unique_texts_{cache_key}"
        payload = self._load_json_cache("vector_db", manifest_key, ".json")
        if payload is None:
            return None
        return payload.get("unique_texts")

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
                "dataset_fingerprint": dataset_fingerprint,
                "preprocessing_fingerprint": preprocessing_fingerprint,
            }
        elif dataset_name is not None and dataset_size is not None:
            cache_key = unique_texts_key(dataset_name, dataset_size)
            metadata = {
                "dataset_name": dataset_name,
                "dataset_size": dataset_size,
            }
        else:
            raise ValueError(
                "Provide either dataset_fingerprint or both dataset_name and dataset_size"
            )

        payload = {
            **metadata,
            "unique_texts": unique_texts,
            "num_unique": len(unique_texts),
            "timestamp": datetime.now().isoformat(),
        }
        self._save_json_cache(
            cache_type="vector_db",
            cache_key=f"unique_texts_{cache_key}",
            extension=".json",
            payload=payload,
            stage="unique_texts",
            input_hash=dataset_fingerprint or dataset_name,
            config_hash=preprocessing_fingerprint,
            indent=2,
        )

    def set_checkpoint(self, experiment_id: str, checkpoint_data: Dict[str, Any]):
        """Save evaluation checkpoint."""
        self._save_json_cache(
            cache_type="checkpoints",
            cache_key=experiment_id,
            extension=".json",
            payload=checkpoint_data,
            stage="checkpoint",
            input_hash=experiment_id,
            indent=2,
        )
