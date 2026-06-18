from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Union
from pathlib import Path
import logging
import numpy as np
import json

from ..constants import MIN_NORM_THRESHOLD

# faiss is imported lazily: loading it eagerly (on any `import evaluator.storage`)
# pulls faiss + its OpenMP runtime into every process, which clashes with torch's
# OpenMP and segfaults torch ops — even for runs that use the in-memory store and
# never touch faiss. Constructed only when a Faiss store is actually built.
faiss = None  # set by _ensure_faiss()


def _ensure_faiss():
    """Import faiss on first use and bind it to the module global."""
    global faiss
    if faiss is None:
        import faiss as _faiss
        faiss = _faiss
    return faiss


class VectorStore(ABC):
    @abstractmethod
    def build(self, vectors: np.ndarray, payloads: List[Any]) -> None:
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> List[Tuple[Any, float]]:
        pass

    def search_batch(self, queries: np.ndarray, k: int) -> List[List[Tuple[Any, float]]]:
        """Batch search — default loops over search(). Override for vectorized impl."""
        return [self.search(q, k) for q in queries]

    def _payload_at(self, idx: int) -> Any:
        """Bounds-checked payload lookup (A4/F4; H3). A backend can hand back a stale/oversized
        index (or a ``load()`` with mismatched payloads); index it blindly and the search either
        returns the wrong document or raises ``IndexError`` mid-run. Return None for an
        out-of-range index (logged, not raised) so callers skip the bad hit. Handles both the
        ``payloads`` (in-memory/FAISS stores) and ``_payloads`` (chromadb/qdrant) attribute names.
        """
        payloads = getattr(self, "payloads", None)
        if payloads is None:
            payloads = getattr(self, "_payloads", None)
        if payloads is None:
            payloads = []
        if 0 <= idx < len(payloads):
            return payloads[idx]
        logging.getLogger("evaluator").warning(
            "%s: payload index %d out of range (n=%d) — skipping hit",
            type(self).__name__, idx, len(payloads),
        )
        return None

    def _verify_payload_count(self, n_vectors: int, payloads: list, where: str) -> None:
        """Fail loudly when a reloaded index's vector count and payload count disagree (R5).

        Save/load writes the index and payload sidecar separately, so a stale/truncated
        sidecar otherwise surfaces as wrong documents or an ``IndexError`` deep in a search
        (the H3 runtime guard catches it per-hit; this catches the whole-file mismatch at load,
        loudly, before any query runs)."""
        if len(payloads) != n_vectors:
            raise ValueError(
                f"{type(self).__name__}.load: {where} has {n_vectors} vectors but "
                f"{len(payloads)} payloads — index/payload mismatch (corrupt or stale sidecar)"
            )

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save vector store to disk."""
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Load vector store from disk."""
        pass


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None
        self.payloads: List[Any] = []

    def build(self, vectors: np.ndarray, payloads: List[Any]) -> None:
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, MIN_NORM_THRESHOLD)  # Avoid division by zero
        self.vectors = vectors / norms
        self.payloads = payloads

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[Any, float]]:
        if self.vectors is None:
            raise ValueError("Cannot search: vectors not built yet. Call build() first.")
        # Normalize query for cosine similarity
        query_norm = np.linalg.norm(query)
        if query_norm > MIN_NORM_THRESHOLD:
            query = query / query_norm

        sims = np.dot(self.vectors, query.T).squeeze()
        indices = np.argsort(-sims)[:k]
        return [
            (p, sims[i]) for i in indices
            if (p := self._payload_at(int(i))) is not None
        ]

    def search_batch(self, queries: np.ndarray, k: int) -> List[List[Tuple[Any, float]]]:
        if self.vectors is None:
            raise ValueError("Cannot search: vectors not built yet.")
        queries = np.asarray(queries, dtype=np.float32)
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms = np.maximum(norms, MIN_NORM_THRESHOLD)
        queries = queries / norms
        # Vectorized dot product: (N, D) @ (D, M) → (N, M)
        sims = queries @ self.vectors.T
        # Top-k per query via argpartition
        results: List[List[Tuple[Any, float]]] = []
        for row in sims:
            if k >= len(row):
                top_idx = np.argsort(-row)
            else:
                part_idx = np.argpartition(-row, k)[:k]
                top_idx = part_idx[np.argsort(-row[part_idx])]
            results.append([
                (p, float(row[i])) for i in top_idx
                if (p := self._payload_at(int(i))) is not None
            ])
        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save vector store to disk (numpy + JSON format)."""
        if self.vectors is None:
            raise ValueError("Cannot save: vectors not built yet. Call build() first.")
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save vectors
        np.save(save_path / "vectors.npy", self.vectors)

        # Save payloads
        with open(save_path / "payloads.json", 'w') as f:
            json.dump(self.payloads, f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """Load vector store from disk."""
        load_path = Path(path)

        # Load vectors
        self.vectors = np.load(load_path / "vectors.npy")

        # Load payloads
        with open(load_path / "payloads.json", 'r') as f:
            self.payloads = json.load(f)
        self._verify_payload_count(self.vectors.shape[0], self.payloads, str(load_path))


class FaissVectorStore(VectorStore):
    def __init__(self, dim: int) -> None:
        _ensure_faiss()
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.payloads: List[Any] = []

    def build(self, vectors: np.ndarray, payloads: List[Any]) -> None:
        faiss.normalize_L2(vectors)
        self.index.add(vectors.astype(np.float32))
        self.payloads = payloads

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[Any, float]]:
        q = query.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)

        scores, indices = self.index.search(q, k)
        return [(p, scores[0][j])
                for j, i in enumerate(indices[0])
                if i >= 0 and (p := self._payload_at(int(i))) is not None]

    def search_batch(self, queries: np.ndarray, k: int) -> List[List[Tuple[Any, float]]]:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        results: List[List[Tuple[Any, float]]] = []
        for row_scores, row_indices in zip(scores, indices):
            results.append([(p, float(s))
                            for s, i in zip(row_scores, row_indices)
                            if i >= 0 and (p := self._payload_at(int(i))) is not None])
        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save FAISS index to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_path / "faiss.index"))
        with open(save_path / "payloads.json", 'w') as f:
            json.dump(self.payloads, f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """Load FAISS index from disk."""
        load_path = Path(path)
        self.index = faiss.read_index(str(load_path / "faiss.index"))
        with open(load_path / "payloads.json", 'r') as f:
            self.payloads = json.load(f)
        self._verify_payload_count(self.index.ntotal, self.payloads, str(load_path))


def _read_index_mmap(index_path: str):
    """Read a FAISS index memory-mapped (pages from disk on demand → bounded RAM). Flat indexes
    that the build can't mmap fall back to a normal read (logged)."""
    try:
        return faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
    except Exception as exc:  # noqa: BLE001 - mmap is an optimization, never a hard requirement
        logging.getLogger("evaluator").info(
            "faiss mmap unavailable for this index (%s); reading in-RAM", exc
        )
        return faiss.read_index(index_path)


class FaissMmapVectorStore(VectorStore):
    """Off-RAM FAISS store (Roadmap 3b): the index is memory-mapped from disk and payloads are
    fetched on demand from a Parquet store, so neither the full index nor the full corpus is
    bounded by one box's RAM. The search is the same normalized ``IndexFlatIP`` dot product, so
    results are identical to :class:`FaissVectorStore` — off-RAM is a memory trade, not a
    different ranking."""

    def __init__(self, dim: int, *, work_dir: Union[str, Path, None] = None,
                 row_group_size: int = 1024) -> None:
        _ensure_faiss()
        import tempfile

        self.dim = dim
        self._row_group_size = int(row_group_size)
        self._work_dir = Path(work_dir) if work_dir else Path(
            tempfile.mkdtemp(prefix="faiss_mmap_")
        )
        self.index = None
        self._store = None  # ParquetPayloadStore (off-RAM payloads)
        self._n = 0

    def build(self, vectors: np.ndarray, payloads: List[Any]) -> None:
        from .payload_store import ParquetPayloadStore

        v = np.asarray(vectors, dtype=np.float32).copy()
        faiss.normalize_L2(v)
        idx = faiss.IndexFlatIP(self.dim)
        idx.add(v)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._work_dir / "faiss.index"
        faiss.write_index(idx, str(index_path))
        del idx  # drop the in-RAM index; reopen it memory-mapped
        self.index = _read_index_mmap(str(index_path))
        self._store = ParquetPayloadStore.write(
            payloads, self._work_dir / "payloads.parquet",
            row_group_size=self._row_group_size,
        )
        self._n = len(payloads)

    def _payload_at(self, idx: int) -> Any:
        """Fetch the payload off-disk (one row group resident) instead of from a RAM list."""
        if self._store is None or not 0 <= idx < self._n:
            if self._store is not None:
                logging.getLogger("evaluator").warning(
                    "FaissMmapVectorStore: payload index %d out of range (n=%d)", idx, self._n
                )
            return None
        return self._store.get(idx)

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[Any, float]]:
        q = query.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        return [(p, scores[0][j])
                for j, i in enumerate(indices[0])
                if i >= 0 and (p := self._payload_at(int(i))) is not None]

    def search_batch(self, queries: np.ndarray, k: int) -> List[List[Tuple[Any, float]]]:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        results: List[List[Tuple[Any, float]]] = []
        for row_scores, row_indices in zip(scores, indices):
            results.append([(p, float(s))
                            for s, i in zip(row_scores, row_indices)
                            if i >= 0 and (p := self._payload_at(int(i))) is not None])
        return results

    def save(self, path: Union[str, Path]) -> None:
        """Copy the on-disk index + payload file to ``path`` (both already materialized)."""
        import shutil

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(self._work_dir / "faiss.index", save_path / "faiss.index")
        shutil.copy(self._work_dir / "payloads.parquet", save_path / "payloads.parquet")

    def load(self, path: Union[str, Path]) -> None:
        """Memory-map the index + open the Parquet payload store from ``path``."""
        from .payload_store import ParquetPayloadStore

        load_path = Path(path)
        self.index = _read_index_mmap(str(load_path / "faiss.index"))
        self._store = ParquetPayloadStore(
            load_path / "payloads.parquet", row_group_size=self._row_group_size
        )
        self._n = len(self._store)
        self._verify_payload_count(self.index.ntotal, list(range(self._n)), str(load_path))


class FaissGpuVectorStore(VectorStore):
    def __init__(self, dim: int, gpu_id: int = 0) -> None:
        _ensure_faiss()
        self.dim = dim
        self.gpu_id = gpu_id
        self.res = faiss.StandardGpuResources()

        cpu_index = faiss.IndexFlatIP(dim)
        self.index = faiss.index_cpu_to_gpu(
            self.res, gpu_id, cpu_index
        )
        self.payloads: List[Any] = []

    def build(self, vectors: np.ndarray, payloads: List[Any]) -> None:
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.payloads = payloads

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[Any, float]]:
        q = query.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)

        scores, indices = self.index.search(q, k)
        return [(p, scores[0][j])
                for j, i in enumerate(indices[0])
                if i >= 0 and (p := self._payload_at(int(i))) is not None]

    def search_batch(self, queries: np.ndarray, k: int) -> List[List[Tuple[Any, float]]]:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        results: List[List[Tuple[Any, float]]] = []
        for row_scores, row_indices in zip(scores, indices):
            results.append([(p, float(s))
                            for s, i in zip(row_scores, row_indices)
                            if i >= 0 and (p := self._payload_at(int(i))) is not None])
        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save GPU FAISS index to disk (converts to CPU first)."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Convert GPU index to CPU for saving
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(cpu_index, str(save_path / "faiss.index"))

        # Save payloads
        with open(save_path / "payloads.json", 'w') as f:
            json.dump(self.payloads, f, indent=2)

        # Save metadata
        with open(save_path / "metadata.json", 'w') as f:
            json.dump({'dim': self.dim, 'gpu_id': self.gpu_id}, f)

    def load(self, path: Union[str, Path]) -> None:
        """Load FAISS index from disk and move to GPU."""
        load_path = Path(path)

        # Load CPU index
        cpu_index = faiss.read_index(str(load_path / "faiss.index"))

        # Move to GPU
        self.index = faiss.index_cpu_to_gpu(self.res, self.gpu_id, cpu_index)

        # Load payloads
        with open(load_path / "payloads.json", 'r') as f:
            self.payloads = json.load(f)

        # Load metadata
        with open(load_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
            self.dim = metadata['dim']
            self.gpu_id = metadata.get('gpu_id', 0)


def build_gpu_ivf(vectors: np.ndarray, dim: int, nlist: int = 1024, gpu_id: int = 0):
    _ensure_faiss()
    res = faiss.StandardGpuResources()

    quantizer = faiss.IndexFlatIP(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist)

    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)

    vectors = vectors.astype(np.float32)
    faiss.normalize_L2(vectors)

    gpu_index.train(vectors)
    gpu_index.add(vectors)

    gpu_index.nprobe = 10
    return gpu_index
