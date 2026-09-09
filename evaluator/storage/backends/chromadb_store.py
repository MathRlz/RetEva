"""ChromaDB vector store implementation."""

from typing import List, Any, Tuple, Union, Optional, Dict
from pathlib import Path
import json
import numpy as np

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from ..vector_store import VectorStore
from ...utils.numeric import l2_normalize
from ...logging_config import get_logger
from ._common import (
    check_vector_payload_lengths,
    load_payloads_and_metadata,
    primitive_payload,
    restore_common_metadata,
)

logger = get_logger(__name__)


class ChromaDBVectorStore(VectorStore):
    """Vector store implementation using ChromaDB.

    Supports both persistent and in-memory storage, with metadata filtering.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_path: Path for persistent storage. If None, uses in-memory storage.
        distance_fn: Distance function to use. One of "cosine", "l2", "ip" (inner product).
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_path: Optional[str] = None,
        distance_fn: str = "cosine",
    ) -> None:
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Install it with: pip install chromadb"
            )

        self.collection_name = collection_name
        self.persist_path = persist_path
        self.distance_fn = distance_fn
        self._payloads: List[Any] = []

        self._init_client()

    def _init_client(self) -> None:
        if self.persist_path:
            self.client = chromadb.PersistentClient(path=self.persist_path)
        else:
            self.client = chromadb.Client()

        # Get or create collection with specified distance function
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_fn}
        )

    def _recreate_collection(self) -> None:
        """Delete then recreate the collection so build/load start from empty state."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as exc:
            logger.debug("delete_collection %s skipped: %s", self.collection_name, exc)

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_fn}
        )

    def build(self, vectors: np.ndarray, payloads: List[Any]) -> None:
        """Build the vector store from vectors and payloads."""
        check_vector_payload_lengths(vectors, payloads)

        self._payloads = payloads

        normalized_vectors = l2_normalize(vectors, axis=1)  # cosine similarity

        self._recreate_collection()

        ids = [str(i) for i in range(len(vectors))]

        # Convert payloads to metadata (ChromaDB requires string/int/float/bool values)
        metadatas = [
            primitive_payload(payload, i, allow_lists=False)
            for i, payload in enumerate(payloads)
        ]

        # Add in batches to avoid memory issues
        batch_size = 5000
        for i in range(0, len(vectors), batch_size):
            end_idx = min(i + batch_size, len(vectors))
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=normalized_vectors[i:end_idx].tolist(),
                metadatas=metadatas[i:end_idx],
            )

    def _hits_to_results(self, doc_ids, distances) -> List[Tuple[Any, float]]:
        """Convert one query's ChromaDB ids/distances into (payload, score) tuples.

        ChromaDB returns distances, not similarities, so cosine/l2 need converting;
        inner product distances are already the similarity score.
        """
        output = []
        for doc_id, distance in zip(doc_ids, distances):
            payload_idx = int(doc_id)

            if self.distance_fn == "cosine":
                score = 1.0 - distance
            elif self.distance_fn == "ip":
                score = distance
            else:  # l2
                score = 1.0 / (1.0 + distance)

            payload = self._payload_at(payload_idx)
            if payload is not None:
                output.append((payload, float(score)))

        return output

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> List[Tuple[Any, float]]:
        """Search for similar vectors; ``where``/``where_document`` are optional
        ChromaDB filters. Returns (payload, score) tuples sorted by relevance."""
        count = self.collection.count()
        if count == 0:
            return []

        query = l2_normalize(query)

        query_kwargs = {
            "query_embeddings": [query.tolist()],
            "n_results": min(k, count),
        }

        if where:
            query_kwargs["where"] = where
        if where_document:
            query_kwargs["where_document"] = where_document

        results = self.collection.query(**query_kwargs)

        output = []
        if results["ids"] and results["ids"][0]:
            distances = (results["distances"][0] if results["distances"]
                         else [0.0] * len(results["ids"][0]))
            output = self._hits_to_results(results["ids"][0], distances)

        return output

    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[List[Tuple[Any, float]]]:
        """Search for similar vectors in batch; ``where`` is an optional ChromaDB
        metadata filter. Returns one result list per query."""
        normalized_queries = l2_normalize(queries, axis=1)

        query_kwargs = {
            "query_embeddings": normalized_queries.tolist(),
            "n_results": min(k, self.collection.count()) if self.collection.count() > 0 else k,
        }

        if where:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)

        all_outputs = []
        for query_idx in range(len(queries)):
            output = []
            if results["ids"] and query_idx < len(results["ids"]) and results["ids"][query_idx]:
                distances = (
                    results["distances"][query_idx]
                    if results["distances"] else [0.0] * len(results["ids"][query_idx])
                )
                output = self._hits_to_results(results["ids"][query_idx], distances)

            all_outputs.append(output)

        return all_outputs

    def save(self, path: Union[str, Path]) -> None:
        """Save vector store to disk: payloads + metadata, plus the full collection
        data when running in-memory (persistent mode is already on disk)."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "payloads.json", "w") as f:
            json.dump(self._payloads, f, indent=2)

        metadata = {
            "collection_name": self.collection_name,
            "persist_path": self.persist_path,
            "distance_fn": self.distance_fn,
            "count": self.collection.count(),
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if not self.persist_path:
            all_data = self.collection.get(include=["embeddings", "metadatas"])

            # Convert numpy arrays to lists for JSON serialization
            embeddings = all_data["embeddings"]
            if embeddings is not None and hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            elif embeddings is not None:
                embeddings = [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]

            export_data = {
                "ids": all_data["ids"],
                "embeddings": embeddings,
                "metadatas": all_data["metadatas"],
            }
            with open(save_path / "collection_data.json", "w") as f:
                json.dump(export_data, f)

    def load(self, path: Union[str, Path]) -> None:
        """Load vector store from disk."""
        load_path = Path(path)

        self._payloads, metadata = load_payloads_and_metadata(load_path)
        restore_common_metadata(self, metadata)

        self._init_client()

        # If collection data was exported (in-memory mode), restore it
        collection_data_path = load_path / "collection_data.json"
        if collection_data_path.exists():
            with open(collection_data_path, "r") as f:
                export_data = json.load(f)

            self._recreate_collection()

            if export_data["ids"]:
                self.collection.add(
                    ids=export_data["ids"],
                    embeddings=export_data["embeddings"],
                    metadatas=export_data["metadatas"],
                )
