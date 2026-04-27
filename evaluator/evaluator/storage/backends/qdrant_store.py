"""Qdrant vector store implementation."""

from typing import List, Any, Tuple, Union, Optional, Dict
from pathlib import Path
import json
import numpy as np
import uuid

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams, 
        Distance, 
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        QueryRequest,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from ..vector_store import VectorStore
from ...constants import MIN_NORM_THRESHOLD


class QdrantVectorStore(VectorStore):
    """Vector store implementation using Qdrant.
    
    Supports three modes:
    - In-memory: No url or path specified (default)
    - Local persistent: path specified (stores on disk)
    - Remote server: url specified (connects to Qdrant server)
    
    Args:
        collection_name: Name of the Qdrant collection.
        url: URL of Qdrant server (for remote mode). Example: "http://localhost:6333"
        path: Path for local persistent storage. If None and url is None, uses in-memory.
        distance_fn: Distance function to use. One of "cosine", "euclidean", "dot_product".
        api_key: Optional API key for Qdrant Cloud authentication.
    """
    
    DISTANCE_MAP = {
        "cosine": Distance.COSINE if QDRANT_AVAILABLE else None,
        "euclidean": Distance.EUCLID if QDRANT_AVAILABLE else None,
        "dot_product": Distance.DOT if QDRANT_AVAILABLE else None,
        # Aliases
        "l2": Distance.EUCLID if QDRANT_AVAILABLE else None,
        "ip": Distance.DOT if QDRANT_AVAILABLE else None,
    }
    
    def __init__(
        self, 
        collection_name: str = "documents",
        url: Optional[str] = None,
        path: Optional[str] = None,
        distance_fn: str = "cosine",
        api_key: Optional[str] = None,
    ) -> None:
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client is not installed. Install it with: pip install qdrant-client"
            )
        
        self.collection_name = collection_name
        self.url = url
        self.path = path
        self.distance_fn = distance_fn
        self.api_key = api_key
        self._payloads: List[Any] = []
        self._dim: Optional[int] = None
        
        # Initialize client
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize Qdrant client based on configuration."""
        if self.url:
            # Remote server mode
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
            )
        elif self.path:
            # Local persistent mode
            self.client = QdrantClient(path=self.path)
        else:
            # In-memory mode
            self.client = QdrantClient(":memory:")
    
    def _get_distance(self) -> "Distance":
        """Get Qdrant Distance enum from string."""
        distance = self.DISTANCE_MAP.get(self.distance_fn)
        if distance is None:
            valid = list(self.DISTANCE_MAP.keys())
            raise ValueError(
                f"Unknown distance function: '{self.distance_fn}'. "
                f"Valid options: {', '.join(valid)}"
            )
        return distance
    
    def _ensure_collection(self, dim: int) -> None:
        """Ensure collection exists with correct configuration."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name in collection_names:
            # Delete existing collection to rebuild
            self.client.delete_collection(self.collection_name)
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=self._get_distance(),
            ),
        )
        self._dim = dim
    
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
        normalized_vectors = (vectors / norms).astype(np.float32)
        
        # Ensure collection exists
        dim = vectors.shape[1]
        self._ensure_collection(dim)
        
        # Prepare points for upsert
        points = []
        for i, (vector, payload) in enumerate(zip(normalized_vectors, payloads)):
            # Convert payload to Qdrant-compatible format
            if isinstance(payload, dict):
                qdrant_payload = {
                    k: v for k, v in payload.items() 
                    if isinstance(v, (str, int, float, bool, list))
                }
                qdrant_payload["_payload_idx"] = i
            else:
                qdrant_payload = {"_payload_idx": i, "_payload_str": str(payload)}
            
            points.append(PointStruct(
                id=i,
                vector=vector.tolist(),
                payload=qdrant_payload,
            ))
        
        # Upsert in batches
        batch_size = 1000
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
    
    def search(
        self, 
        query: np.ndarray, 
        k: int = 5,
        filter_conditions: Optional[Dict] = None,
    ) -> List[Tuple[Any, float]]:
        """Search for similar vectors.
        
        Args:
            query: Query vector.
            k: Number of results to return.
            filter_conditions: Optional dict for filtering. Keys are field names,
                values are the values to match.
            
        Returns:
            List of (payload, score) tuples, sorted by relevance.
        """
        # Check if collection exists and has points
        try:
            collection_info = self.client.get_collection(self.collection_name)
            if collection_info.points_count == 0:
                return []
        except Exception:
            return []
        
        # Normalize query
        query_norm = np.linalg.norm(query)
        if query_norm > MIN_NORM_THRESHOLD:
            query = query / query_norm
        
        # Build filter if provided
        query_filter = None
        if filter_conditions:
            conditions = [
                FieldCondition(key=field_key, match=MatchValue(value=field_val))
                for field_key, field_val in filter_conditions.items()
            ]
            query_filter = Filter(must=conditions)
        
        # Search using query_points API
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query.astype(np.float32).tolist(),
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )
        
        # Extract results
        output = []
        for hit in results.points:
            payload_idx = hit.payload.get("_payload_idx", 0)
            score = hit.score
            
            # For cosine distance, score is already similarity
            # For euclidean, we need to convert
            if self.distance_fn in ("euclidean", "l2"):
                score = 1.0 / (1.0 + score)
            
            output.append((self._payloads[payload_idx], float(score)))
        
        return output
    
    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 5,
        filter_conditions: Optional[Dict] = None,
    ) -> List[List[Tuple[Any, float]]]:
        """Search for similar vectors in batch.
        
        Args:
            queries: Array of query vectors with shape (n_queries, dim).
            k: Number of results to return per query.
            filter_conditions: Optional dict for filtering.
            
        Returns:
            List of result lists, one per query.
        """
        # Normalize queries
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms = np.maximum(norms, MIN_NORM_THRESHOLD)
        normalized_queries = (queries / norms).astype(np.float32)
        
        # Build filter if provided
        query_filter = None
        if filter_conditions:
            conditions = [
                FieldCondition(key=field_key, match=MatchValue(value=field_val))
                for field_key, field_val in filter_conditions.items()
            ]
            query_filter = Filter(must=conditions)
        
        # Batch search using query_batch_points
        requests = [
            QueryRequest(
                query=q.tolist(),
                limit=k,
                filter=query_filter,
                with_payload=True,
            )
            for q in normalized_queries
        ]
        
        batch_results = self.client.query_batch_points(
            collection_name=self.collection_name,
            requests=requests,
        )
        
        # Extract results
        all_outputs = []
        for response in batch_results:
            output = []
            for hit in response.points:
                payload_idx = hit.payload.get("_payload_idx", 0)
                score = hit.score
                
                if self.distance_fn in ("euclidean", "l2"):
                    score = 1.0 / (1.0 + score)
                
                output.append((self._payloads[payload_idx], float(score)))
            all_outputs.append(output)
        
        return all_outputs
    
    def save(self, path: Union[str, Path]) -> None:
        """Save vector store to disk.
        
        For remote/persistent mode, this saves payloads and metadata.
        For in-memory mode, also exports vectors.
        
        Args:
            path: Directory path to save to.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save payloads
        with open(save_path / "payloads.json", "w") as f:
            json.dump(self._payloads, f, indent=2)
        
        # Get collection info
        try:
            collection_info = self.client.get_collection(self.collection_name)
            count = collection_info.points_count
        except Exception:
            count = 0
        
        # Save metadata
        metadata = {
            "collection_name": self.collection_name,
            "url": self.url,
            "path": self.path,
            "distance_fn": self.distance_fn,
            "dim": self._dim,
            "count": count,
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # If using in-memory storage, export vectors
        if not self.url and not self.path:
            # Scroll through all points
            all_points = []
            offset = None
            
            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_vectors=True,
                    with_payload=True,
                )
                points, next_offset = result
                
                for point in points:
                    all_points.append({
                        "id": point.id,
                        "vector": point.vector,
                        "payload": point.payload,
                    })
                
                if next_offset is None:
                    break
                offset = next_offset
            
            with open(save_path / "vectors.json", "w") as f:
                json.dump(all_points, f)
    
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
        self._dim = metadata.get("dim")
        
        # Re-initialize client (in-memory for loaded data)
        self.url = None
        self.path = None
        self._init_client()
        
        # If vectors were exported, restore them
        vectors_path = load_path / "vectors.json"
        if vectors_path.exists():
            with open(vectors_path, "r") as f:
                all_points = json.load(f)
            
            if all_points and self._dim:
                # Recreate collection
                self._ensure_collection(self._dim)
                
                # Restore points
                points = [
                    PointStruct(
                        id=p["id"],
                        vector=p["vector"],
                        payload=p["payload"],
                    )
                    for p in all_points
                ]
                
                # Upsert in batches
                batch_size = 1000
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                    )
