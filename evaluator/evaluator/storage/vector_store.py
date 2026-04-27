from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Union
from pathlib import Path
import numpy as np
import faiss
import json

from ..constants import MIN_NORM_THRESHOLD

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
        return [(self.payloads[i], sims[i]) for i in indices]

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
            results.append([(self.payloads[i], float(row[i])) for i in top_idx])
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

class FaissVectorStore(VectorStore):
    def __init__(self, dim: int) -> None:
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
        return [(self.payloads[i], scores[0][j])
                for j, i in enumerate(indices[0]) if i >= 0]

    def search_batch(self, queries: np.ndarray, k: int) -> List[List[Tuple[Any, float]]]:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        results: List[List[Tuple[Any, float]]] = []
        for row_scores, row_indices in zip(scores, indices):
            results.append([(self.payloads[i], float(s))
                            for s, i in zip(row_scores, row_indices) if i >= 0])
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
    
class FaissGpuVectorStore(VectorStore):
    def __init__(self, dim: int, gpu_id: int = 0) -> None:
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
        return [(self.payloads[i], scores[0][j])
                for j, i in enumerate(indices[0]) if i >= 0]

    def search_batch(self, queries: np.ndarray, k: int) -> List[List[Tuple[Any, float]]]:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        results: List[List[Tuple[Any, float]]] = []
        for row_scores, row_indices in zip(scores, indices):
            results.append([(self.payloads[i], float(s))
                            for s, i in zip(row_scores, row_indices) if i >= 0])
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
