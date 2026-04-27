"""Unit tests for Qdrant vector store implementation."""
import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if qdrant-client is available
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


@unittest.skipUnless(QDRANT_AVAILABLE, "Qdrant client not installed")
class TestQdrantVectorStore(unittest.TestCase):
    """Test Qdrant vector store implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        from evaluator.storage.backends.qdrant_store import QdrantVectorStore
        # Use unique collection name to avoid conflicts
        import uuid
        self.collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        self.store = QdrantVectorStore(collection_name=self.collection_name)
        self.temp_dir = None
    
    def tearDown(self):
        """Clean up test resources."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        # Clean up collection
        try:
            self.store.client.delete_collection(self.collection_name)
        except Exception:
            pass
    
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
        from evaluator.storage.backends.qdrant_store import QdrantVectorStore
        new_store = QdrantVectorStore(collection_name="test_load")
        new_store.load(self.temp_dir)
        
        # Verify search works
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = new_store.search(query, k=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "doc1")
    
    def test_persistent_storage(self):
        """Test persistent storage mode."""
        self.temp_dir = tempfile.mkdtemp()
        persist_path = str(Path(self.temp_dir) / "qdrant_db")
        
        from evaluator.storage.backends.qdrant_store import QdrantVectorStore
        
        # Create store with persistence
        store1 = QdrantVectorStore(
            collection_name="persistent_test",
            path=persist_path
        )
        
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        payloads = ["doc1", "doc2"]
        
        store1.build(vectors, payloads)
        
        # Close the first client before creating a new one
        store1.client.close()
        del store1
        
        # Create new store pointing to same path
        store2 = QdrantVectorStore(
            collection_name="persistent_test",
            path=persist_path
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
        results = self.store.search(query, k=2, filter_conditions={"category": "A"})
        
        # Should only return category A documents
        for result in results:
            self.assertEqual(result[0]["category"], "A")
    
    def test_empty_collection_search(self):
        """Test search on empty collection."""
        from evaluator.storage.backends.qdrant_store import QdrantVectorStore
        import uuid
        empty_store = QdrantVectorStore(collection_name=f"test_empty_{uuid.uuid4().hex[:8]}")
        
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
        from evaluator.storage.backends.qdrant_store import QdrantVectorStore
        import uuid
        
        vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)
        payloads = ["doc1", "doc2"]
        
        # Test euclidean distance
        store_l2 = QdrantVectorStore(
            collection_name=f"test_l2_{uuid.uuid4().hex[:8]}",
            distance_fn="euclidean"
        )
        store_l2.build(vectors, payloads)
        
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store_l2.search(query, k=1)
        
        self.assertEqual(results[0][0], "doc1")
        
        # Test dot product
        store_dot = QdrantVectorStore(
            collection_name=f"test_dot_{uuid.uuid4().hex[:8]}",
            distance_fn="dot_product"
        )
        store_dot.build(vectors, payloads)
        
        results = store_dot.search(query, k=1)
        self.assertEqual(results[0][0], "doc1")
    
    def test_invalid_distance_function(self):
        """Test error on invalid distance function."""
        from evaluator.storage.backends.qdrant_store import QdrantVectorStore
        
        store = QdrantVectorStore(
            collection_name="test_invalid_dist",
            distance_fn="invalid"
        )
        
        vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        payloads = ["doc1"]
        
        with self.assertRaises(ValueError) as context:
            store.build(vectors, payloads)
        
        self.assertIn("invalid", str(context.exception).lower())


@unittest.skipUnless(QDRANT_AVAILABLE, "Qdrant client not installed")
class TestQdrantFactoryIntegration(unittest.TestCase):
    """Test Qdrant integration with factory."""
    
    def test_factory_creates_qdrant_store(self):
        """Test that factory can create Qdrant store."""
        from evaluator.config import VectorDBConfig
        from evaluator.pipeline.factory import create_vector_store_from_config
        
        config = VectorDBConfig(
            type="qdrant",
            qdrant_collection_name="factory_test",
            qdrant_path=None,  # In-memory
            qdrant_url=None,
        )
        
        store = create_vector_store_from_config(config)
        
        from evaluator.storage.backends.qdrant_store import QdrantVectorStore
        self.assertIsInstance(store, QdrantVectorStore)
    
    def test_factory_passes_config_options(self):
        """Test that factory passes config options correctly."""
        from evaluator.config import VectorDBConfig
        from evaluator.pipeline.factory import create_vector_store_from_config
        
        config = VectorDBConfig(
            type="qdrant",
            qdrant_collection_name="custom_collection",
            distance_metric="euclidean",
        )
        
        store = create_vector_store_from_config(config)
        
        self.assertEqual(store.collection_name, "custom_collection")
        self.assertEqual(store.distance_fn, "euclidean")


class TestQdrantImportError(unittest.TestCase):
    """Test Qdrant import error handling."""
    
    def test_import_error_message(self):
        """Test that helpful error is raised when qdrant-client not installed."""
        # Mock qdrant_client as unavailable
        with patch.dict('sys.modules', {'qdrant_client': None}):
            from evaluator.storage.backends import qdrant_store
            
            # Save original and set to unavailable
            original_available = qdrant_store.QDRANT_AVAILABLE
            qdrant_store.QDRANT_AVAILABLE = False
            
            try:
                with self.assertRaises(ImportError) as context:
                    qdrant_store.QdrantVectorStore()
                
                self.assertIn("qdrant", str(context.exception).lower())
                self.assertIn("pip install", str(context.exception))
            finally:
                qdrant_store.QDRANT_AVAILABLE = original_available
    
    def test_factory_unknown_type_includes_qdrant(self):
        """Test that unknown type error lists qdrant as available."""
        from evaluator.config import VectorDBConfig
        
        # With enum validation, error is raised during config creation
        with self.assertRaises(ValueError) as context:
            config = VectorDBConfig(type="nonexistent")
        
        self.assertIn("qdrant", str(context.exception).lower())


class TestVectorDBConfigQdrant(unittest.TestCase):
    """Test VectorDBConfig Qdrant fields."""
    
    def test_qdrant_config_defaults(self):
        """Test default values for Qdrant config."""
        from evaluator.config import VectorDBConfig
        
        config = VectorDBConfig()
        
        self.assertIsNone(config.qdrant_url)
        self.assertIsNone(config.qdrant_path)
        self.assertEqual(config.qdrant_collection_name, "documents")
        self.assertIsNone(config.qdrant_api_key)
    
    def test_qdrant_config_custom_values(self):
        """Test custom values for Qdrant config."""
        from evaluator.config import VectorDBConfig
        
        config = VectorDBConfig(
            type="qdrant",
            qdrant_url="http://localhost:6333",
            qdrant_path=None,
            qdrant_collection_name="my_collection",
            qdrant_api_key="secret_key",
        )
        
        self.assertEqual(config.type, "qdrant")
        self.assertEqual(config.qdrant_url, "http://localhost:6333")
        self.assertIsNone(config.qdrant_path)
        self.assertEqual(config.qdrant_collection_name, "my_collection")
        self.assertEqual(config.qdrant_api_key, "secret_key")
    
    def test_qdrant_config_persistent_path(self):
        """Test Qdrant config with persistent path."""
        from evaluator.config import VectorDBConfig
        
        config = VectorDBConfig(
            type="qdrant",
            qdrant_path="/path/to/db",
            qdrant_collection_name="persistent_collection",
        )
        
        self.assertEqual(config.qdrant_path, "/path/to/db")
        self.assertIsNone(config.qdrant_url)


if __name__ == "__main__":
    unittest.main()
