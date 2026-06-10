"""Artifact serialization helpers (json / npy / npz load + store)."""

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any


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

    def _load_json_cache(
        self, cache_type: str, cache_key: str, extension: str
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        fallback_path = self._get_cache_path(cache_type, cache_key, extension)
        resolved = self._resolve_manifest_path(cache_type, cache_key, fallback_path)
        if resolved is not None and resolved.exists():
            return self._read_json_file(resolved)
        return None

    def _save_json_cache(
        self,
        cache_type: str,
        cache_key: str,
        extension: str,
        payload: Dict[str, Any],
        *,
        stage: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        input_hash: Optional[str] = None,
        config_hash: Optional[str] = None,
        payload_hash: Optional[str] = None,
        indent: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        cache_path = self._get_cache_path(cache_type, cache_key, extension)
        self._write_json_file(cache_path, payload, indent=indent)
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

    def _load_numpy_cache(
        self, cache_type: str, cache_key: str, extension: str
    ) -> Optional[np.ndarray]:
        if not self.enabled:
            return None
        fallback_path = self._get_cache_path(cache_type, cache_key, extension)
        resolved = self._resolve_manifest_path(cache_type, cache_key, fallback_path)
        if resolved is not None and resolved.exists():
            return np.load(resolved)
        return None

    def _save_numpy_cache(
        self,
        cache_type: str,
        cache_key: str,
        extension: str,
        array: np.ndarray,
        *,
        stage: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        input_hash: Optional[str] = None,
        config_hash: Optional[str] = None,
        payload_hash: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        cache_path = self._get_cache_path(cache_type, cache_key, extension)
        np.save(cache_path, array)
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

    def _load_npz_cache(
        self, cache_type: str, cache_key: str, extension: str
    ) -> Optional[Dict[str, np.ndarray]]:
        if not self.enabled:
            return None
        fallback_path = self._get_cache_path(cache_type, cache_key, extension)
        resolved = self._resolve_manifest_path(cache_type, cache_key, fallback_path)
        if resolved is None or not resolved.exists():
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
        *,
        stage: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        input_hash: Optional[str] = None,
        config_hash: Optional[str] = None,
        payload_hash: Optional[str] = None,
        **arrays: np.ndarray,
    ) -> None:
        if not self.enabled:
            return
        cache_path = self._get_cache_path(cache_type, cache_key, extension)
        np.savez_compressed(cache_path, **arrays)
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
