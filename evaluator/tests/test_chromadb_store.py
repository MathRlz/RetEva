"""Unit tests for ChromaDB vector store implementation."""
import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if chromadb is available
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


@unittest.skipUnless(CHROMADB_AVAILABLE, "ChromaDB not installed")
class TestChromaDBVectorStore(unittest.TestCase):
    """Test ChromaDB vector store implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        from evaluator.storage.backends.chromadb_store import ChromaDBVectorStore
        self.store = ChromaDBVectorStore(collection_name="test_collection")
        self.temp_dir = None
    
    def tearDown(self):
        """Clean up test resources."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_build_and_search(self):
        """Test building index and searching."""
        # Create test vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)
        payloads = ["doc1", "doc2", "doc3", "doc4"]
        
        # Build index
        self.store.build(vectors, payloads)
        
        # Search for similar to first vector
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = self.store.search(query, k=2)
        
        self.assertEqual(len(results), 2)
        # First result should be most similar
        self.assertEqual(results[0][0], "doc1")
        self.assertGreater(results[0][1], 0.9)  # High similarity
    
    def test_search_batch(self):
        """Test batch search functionality."""
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        payloads = ["doc1", "doc2", "doc3"]
        
        self.store.build(vectors, payloads)
        
        # Batch query
        queries = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        results = self.store.search_batch(queries, k=2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0][0], "doc1")  # First query matches doc1
        self.assertEqual(results[1][0][0], "doc2")  # Second query matches doc2
    
    def test_save_and_load_inmemory(self):
        """Test saving and loading in-memory store."""
        self.temp_dir = tempfile.mkdtemp()
        
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        payloads = ["doc1", "doc2"]
        
        # Build and save
        self.store.build(vectors, payloads)
        self.store.save(self.temp_dir)
        
        # Create new store and load
        from evaluator.storage.backends.chromadb_store import ChromaDBVectorStore
        new_store = ChromaDBVectorStore(collection_name="test_load")
        new_store.load(self.temp_dir)
        
        # Verify search works
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = new_store.search(query, k=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "doc1")
    
    def test_persistent_storage(self):
        """Test persistent storage mode."""
        self.temp_dir = tempfile.mkdtemp()
        persist_path = str(Path(self.temp_dir) / "chroma_db")
        
        from evaluator.storage.backends.chromadb_store import ChromaDBVectorStore
        
        # Create store with persistence
        store1 = ChromaDBVectorStore(
            collection_name="persistent_test",
            persist_path=persist_path
        )
        
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        payloads = ["doc1", "doc2"]
        
        store1.build(vectors, payloads)
        
        # Create new store pointing to same path
        store2 = ChromaDBVectorStore(
            collection_name="persistent_test",
            persist_path=persist_path
        )
        # Reload payloads for store2
        store2._payloads = payloads
        
        # Should be able to search
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store2.search(query, k=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "doc1")
    
    def test_dict_payloads(self):
        """Test with dict payloads containing metadata."""
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        payloads = [
            {"id": 1, "text": "first document", "category": "A"},
            {"id": 2, "text": "second document", "category": "B"},
        ]
        
        self.store.build(vectors, payloads)
        
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = self.store.search(query, k=1)
        
        self.assertEqual(results[0][0]["id"], 1)
        self.assertEqual(results[0][0]["text"], "first document")
    
    def test_metadata_filtering(self):
        """Test metadata filtering in search."""
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],  # Similar to first
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        payloads = [
            {"id": 1, "category": "A"},
            {"id": 2, "category": "B"},
            {"id": 3, "category": "A"},
        ]
        
        self.store.build(vectors, payloads)
        
        # Search with metadata filter
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = self.store.search(query, k=2, where={"category": "A"})
        
        # Should only return category A documents
        for result in results:
            self.assertEqual(result[0]["category"], "A")
    
    def test_empty_collection_search(self):
        """Test search on empty collection."""
        # Create a fresh store with unique collection name to ensure it's empty
        from evaluator.storage.backends.chromadb_store import ChromaDBVectorStore
        empty_store = ChromaDBVectorStore(collection_name="test_empty_collection")
        
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = empty_store.search(query, k=5)
        
        self.assertEqual(results, [])
    
    def test_build_vector_payload_mismatch(self):
        """Test error when vectors and payloads count mismatch."""
        vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        payloads = ["doc1", "doc2"]  # More payloads than vectors
        
        with self.assertRaises(ValueError):
            self.store.build(vectors, payloads)
    
    def test_different_distance_functions(self):
        """Test different distance functions."""
        from evaluator.storage.backends.chromadb_store import ChromaDBVectorStore
        
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)
        payloads = ["doc1", "doc2"]
        
        # Test L2 distance
        store_l2 = ChromaDBVectorStore(
            collection_name="test_l2",
            distance_fn="l2"
        )
        store_l2.build(vectors, payloads)
        
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store_l2.search(query, k=1)
        
        self.assertEqual(results[0][0], "doc1")


@unittest.skipUnless(CHROMADB_AVAILABLE, "ChromaDB not installed")
class TestChromaDBFactoryIntegration(unittest.TestCase):
    """Test ChromaDB integration with factory."""
    
    def test_factory_creates_chromadb_store(self):
        """Test that factory can create ChromaDB store."""
        from evaluator.config import VectorDBConfig
        from evaluator.pipeline.factory import create_vector_store_from_config
        
        config = VectorDBConfig(
            type="chromadb",
            chromadb_collection_name="factory_test",
            chromadb_path=None,  # In-memory
        )
        
        store = create_vector_store_from_config(config)
        
        from evaluator.storage.backends.chromadb_store import ChromaDBVectorStore
        self.assertIsInstance(store, ChromaDBVectorStore)
    
    def test_factory_passes_config_options(self):
        """Test that factory passes config options correctly."""
        from evaluator.config import VectorDBConfig
        from evaluator.pipeline.factory import create_vector_store_from_config
        
        config = VectorDBConfig(
            type="chromadb",
            chromadb_collection_name="custom_collection",
            distance_metric="l2",
        )
        
        store = create_vector_store_from_config(config)
        
        self.assertEqual(store.collection_name, "custom_collection")
        self.assertEqual(store.distance_fn, "l2")


class TestChromaDBImportError(unittest.TestCase):
    """Test ChromaDB import error handling."""
    
    def test_import_error_message(self):
        """Test that helpful error is raised when chromadb not installed."""
        # Mock chromadb as unavailable
        with patch.dict('sys.modules', {'chromadb': None}):
            # Re-import to trigger error
            import importlib
            from evaluator.storage.backends import chromadb_store
            
            # Reload to pick up mocked import
            original_available = chromadb_store.CHROMADB_AVAILABLE
            chromadb_store.CHROMADB_AVAILABLE = False
            
            try:
                with self.assertRaises(ImportError) as context:
                    chromadb_store.ChromaDBVectorStore()
                
                self.assertIn("chromadb", str(context.exception).lower())
                self.assertIn("pip install", str(context.exception))
            finally:
                chromadb_store.CHROMADB_AVAILABLE = original_available
    
    def test_factory_unknown_type_includes_chromadb(self):
        """Test that unknown type error lists chromadb as available."""
        from evaluator.config import VectorDBConfig
        
        # With enum validation, error is raised during config creation
        with self.assertRaises(ValueError) as context:
            config = VectorDBConfig(type="nonexistent")
        
        self.assertIn("chromadb", str(context.exception).lower())


class TestVectorDBConfigChromaDB(unittest.TestCase):
    """Test VectorDBConfig ChromaDB fields."""
    
    def test_chromadb_config_defaults(self):
        """Test default values for ChromaDB config."""
        from evaluator.config import VectorDBConfig
        
        config = VectorDBConfig()
        
        self.assertIsNone(config.chromadb_path)
        self.assertEqual(config.chromadb_collection_name, "documents")
    
    def test_chromadb_config_custom_values(self):
        """Test custom values for ChromaDB config."""
        from evaluator.config import VectorDBConfig
        
        config = VectorDBConfig(
            type="chromadb",
            chromadb_path="/path/to/db",
            chromadb_collection_name="my_collection",
        )
        
        self.assertEqual(config.type, "chromadb")
        self.assertEqual(config.chromadb_path, "/path/to/db")
        self.assertEqual(config.chromadb_collection_name, "my_collection")


if __name__ == "__main__":
    unittest.main()
