"""ChromaDB vector store implementation."""

from typing import List, Any, Tuple, Union, Optional, Dict
from pathlib import Path
import json
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from ..vector_store import VectorStore
from ...constants import MIN_NORM_THRESHOLD


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
        
        # Initialize client
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize ChromaDB client and collection."""
        if self.persist_path:
            self.client = chromadb.PersistentClient(path=self.persist_path)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection with specified distance function
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_fn}
        )
    
    def build(self, vectors: np.ndarray, payloads: List[Any]) -> None:
        """Build the vector store from vectors and payloads.
        
        Args:
            vectors: Array of vectors with shape (n_samples, dim).
            payloads: List of payload objects corresponding to each vector.
        """
        if len(vectors) != len(payloads):
            raise ValueError(
                f"Number of vectors ({len(vectors)}) must match number of payloads ({len(payloads)})"
            )
        
        # Store payloads locally for retrieval
        self._payloads = payloads
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, MIN_NORM_THRESHOLD)
        normalized_vectors = vectors / norms
        
        # Clear existing collection data
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_fn}
        )
        
        # Add vectors to collection
        ids = [str(i) for i in range(len(vectors))]
        
        # Convert payloads to metadata (ChromaDB requires string/int/float/bool values)
        metadatas = []
        for i, payload in enumerate(payloads):
            if isinstance(payload, dict):
                # Filter out non-primitive values for metadata
                metadata = {
                    k: v for k, v in payload.items() 
                    if isinstance(v, (str, int, float, bool))
                }
                metadata["_payload_idx"] = i
            else:
                metadata = {"_payload_idx": i, "_payload_str": str(payload)}
            metadatas.append(metadata)
        
        # Add in batches to avoid memory issues
        batch_size = 5000
        for i in range(0, len(vectors), batch_size):
            end_idx = min(i + batch_size, len(vectors))
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=normalized_vectors[i:end_idx].tolist(),
                metadatas=metadatas[i:end_idx],
            )
    
    def search(
        self, 
        query: np.ndarray, 
        k: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> List[Tuple[Any, float]]:
        """Search for similar vectors.
        
        Args:
            query: Query vector.
            k: Number of results to return.
            where: Optional metadata filter dict for ChromaDB.
            where_document: Optional document content filter for ChromaDB.
            
        Returns:
            List of (payload, score) tuples, sorted by relevance.
        """
        # Handle empty collection
        count = self.collection.count()
        if count == 0:
            return []
        
        # Normalize query
        query_norm = np.linalg.norm(query)
        if query_norm > MIN_NORM_THRESHOLD:
            query = query / query_norm
        
        # Prepare query kwargs
        query_kwargs = {
            "query_embeddings": [query.tolist()],
            "n_results": min(k, count),
        }
        
        if where:
            query_kwargs["where"] = where
        if where_document:
            query_kwargs["where_document"] = where_document
        
        # Query collection
        results = self.collection.query(**query_kwargs)
        
        # Extract results
        output = []
        if results["ids"] and results["ids"][0]:
            distances = results["distances"][0] if results["distances"] else [0.0] * len(results["ids"][0])
            
            for idx, (doc_id, distance) in enumerate(zip(results["ids"][0], distances)):
                payload_idx = int(doc_id)
                
                # Convert distance to similarity score
                # ChromaDB returns distances, not similarities for cosine
                if self.distance_fn == "cosine":
                    score = 1.0 - distance
                elif self.distance_fn == "ip":
                    score = distance
                else:  # l2
                    score = 1.0 / (1.0 + distance)
                
                output.append((self._payloads[payload_idx], float(score)))
        
        return output
    
    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[List[Tuple[Any, float]]]:
        """Search for similar vectors in batch.
        
        Args:
            queries: Array of query vectors with shape (n_queries, dim).
            k: Number of results to return per query.
            where: Optional metadata filter dict for ChromaDB.
            
        Returns:
            List of result lists, one per query.
        """
        # Normalize queries
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms = np.maximum(norms, MIN_NORM_THRESHOLD)
        normalized_queries = queries / norms
        
        # Prepare query kwargs
        query_kwargs = {
            "query_embeddings": normalized_queries.tolist(),
            "n_results": min(k, self.collection.count()) if self.collection.count() > 0 else k,
        }
        
        if where:
            query_kwargs["where"] = where
        
        # Query collection
        results = self.collection.query(**query_kwargs)
        
        # Extract results for each query
        all_outputs = []
        for query_idx in range(len(queries)):
            output = []
            if results["ids"] and query_idx < len(results["ids"]) and results["ids"][query_idx]:
                distances = (
                    results["distances"][query_idx] 
                    if results["distances"] else [0.0] * len(results["ids"][query_idx])
                )
                
                for doc_id, distance in zip(results["ids"][query_idx], distances):
                    payload_idx = int(doc_id)
                    
                    # Convert distance to similarity score
                    if self.distance_fn == "cosine":
                        score = 1.0 - distance
                    elif self.distance_fn == "ip":
                        score = distance
                    else:  # l2
                        score = 1.0 / (1.0 + distance)
                    
                    output.append((self._payloads[payload_idx], float(score)))
            
            all_outputs.append(output)
        
        return all_outputs
    
    def save(self, path: Union[str, Path]) -> None:
        """Save vector store to disk.
        
        For persistent storage, this persists the ChromaDB data.
        Also saves payloads separately for full reconstruction.
        
        Args:
            path: Directory path to save to.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save payloads
        with open(save_path / "payloads.json", "w") as f:
            json.dump(self._payloads, f, indent=2)
        
        # Save metadata
        metadata = {
            "collection_name": self.collection_name,
            "persist_path": self.persist_path,
            "distance_fn": self.distance_fn,
            "count": self.collection.count(),
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # If using in-memory storage, export collection data
        if not self.persist_path:
            # Get all data from collection
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
        """Load vector store from disk.
        
        Args:
            path: Directory path to load from.
        """
        load_path = Path(path)
        
        # Load payloads
        with open(load_path / "payloads.json", "r") as f:
            self._payloads = json.load(f)
        
        # Load metadata
        with open(load_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        self.collection_name = metadata["collection_name"]
        self.distance_fn = metadata["distance_fn"]
        
        # Re-initialize client
        self._init_client()
        
        # If collection data was exported (in-memory mode), restore it
        collection_data_path = load_path / "collection_data.json"
        if collection_data_path.exists():
            with open(collection_data_path, "r") as f:
                export_data = json.load(f)
            
            # Clear and recreate collection
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_fn}
            )
            
            # Add data back
            if export_data["ids"]:
                self.collection.add(
                    ids=export_data["ids"],
                    embeddings=export_data["embeddings"],
                    metadatas=export_data["metadatas"],
                )
