"""Vector store implementations for the evaluator.

This module provides different vector store backends for similarity search.
"""

from ..vector_store import VectorStore, InMemoryVectorStore, FaissVectorStore, FaissGpuVectorStore

__all__ = [
    "VectorStore",
    "InMemoryVectorStore", 
    "FaissVectorStore",
    "FaissGpuVectorStore",
]

# Try to import ChromaDB store (optional dependency)
try:
    from .chromadb_store import ChromaDBVectorStore
    __all__.append("ChromaDBVectorStore")
except ImportError:
    # ChromaDB not installed, skip
    pass

# Try to import Qdrant store (optional dependency)
try:
    from .qdrant_store import QdrantVectorStore
    __all__.append("QdrantVectorStore")
except ImportError:
    # Qdrant client not installed, skip
    pass
