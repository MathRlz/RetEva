"""Artifact serialization helpers (json / npy / npz load + store)."""

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Any, Callable, Optional, Dict


def _file_checksum(path: Path, block_size: int = 1 << 16) -> str:
    """Compute SHA-256 checksum of a file (first 64KB for speed)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            h.update(f.read(block_size))
    except OSError:
        return ""
    return h.hexdigest()


class IoMixin:
    """File (de)serialization helpers shared by the cache manager."""

    def _read_json_file(self, path: Path) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    def _write_json_file(
        self, path: Path, payload: Dict[str, Any], *, indent: Optional[int] = None
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=indent)

    def _resolve_cached_artifact(
        self, cache_type: str, cache_key: str, extension: str
    ) -> Optional[Path]:
        """The on-disk artifact for a cache entry, or None if caching is off or it is absent.

        The manifest's recorded path wins; the conventional path is the fallback."""
        if not self.enabled:
            return None
        fallback_path = self._get_cache_path(cache_type, cache_key, extension)
        resolved = self._resolve_manifest_path(cache_type, cache_key, fallback_path)
        if resolved is None or not resolved.exists():
            return None
        return resolved

    def _store_artifact(
        self,
        cache_type: str,
        cache_key: str,
        extension: str,
        write: Callable[[Path], None],
        *,
        stage: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        input_hash: Optional[str] = None,
        config_hash: Optional[str] = None,
        payload_hash: Optional[str] = None,
    ) -> None:
        """Write one artifact through ``write(path)`` and record it in the manifest."""
        if not self.enabled:
            return
        cache_path = self._get_cache_path(cache_type, cache_key, extension)
        write(cache_path)
        self._upsert_manifest_entry(
            cache_type=cache_type,
            cache_key=cache_key,
            artifact_path=cache_path,
            stage=stage,
            model_name=model_name,
            model_version=model_version,
            input_hash=input_hash,
            config_hash=config_hash,
            payload_hash=payload_hash,
        )

    def _load_json_cache(
        self, cache_type: str, cache_key: str, extension: str
    ) -> Optional[Dict[str, Any]]:
        resolved = self._resolve_cached_artifact(cache_type, cache_key, extension)
        return self._read_json_file(resolved) if resolved is not None else None

    def _save_json_cache(
        self,
        cache_type: str,
        cache_key: str,
        extension: str,
        payload: Dict[str, Any],
        *,
        indent: Optional[int] = None,
        **manifest_fields: Any,
    ) -> None:
        self._store_artifact(
            cache_type,
            cache_key,
            extension,
            lambda path: self._write_json_file(path, payload, indent=indent),
            **manifest_fields,
        )

    def _load_numpy_cache(
        self, cache_type: str, cache_key: str, extension: str
    ) -> Optional[np.ndarray]:
        resolved = self._resolve_cached_artifact(cache_type, cache_key, extension)
        return np.load(resolved) if resolved is not None else None

    def _save_numpy_cache(
        self,
        cache_type: str,
        cache_key: str,
        extension: str,
        array: np.ndarray,
        **manifest_fields: Any,
    ) -> None:
        self._store_artifact(
            cache_type,
            cache_key,
            extension,
            lambda path: np.save(path, array),
            **manifest_fields,
        )

    def _load_npz_cache(
        self, cache_type: str, cache_key: str, extension: str
    ) -> Optional[Dict[str, np.ndarray]]:
        resolved = self._resolve_cached_artifact(cache_type, cache_key, extension)
        if resolved is None:
            return None
        with np.load(resolved) as data:
            return {
                "features": data["features"],
                "attention_mask": data["attention_mask"],
            }

    def _save_npz_cache(
        self,
        cache_type: str,
        cache_key: str,
        extension: str,
        arrays: Dict[str, np.ndarray],
        **manifest_fields: Any,
    ) -> None:
        self._store_artifact(
            cache_type,
            cache_key,
            extension,
            lambda path: np.savez_compressed(path, **arrays),
            **manifest_fields,
        )
