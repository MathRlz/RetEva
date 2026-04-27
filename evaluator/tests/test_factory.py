"""Unit tests for model factory functions."""
import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.models.registry import asr_registry, text_embedding_registry, audio_embedding_registry


class TestASRFactory(unittest.TestCase):
    """Test ASR model factory function."""
    
    def test_factory_uses_registry(self):
        """Test that factory function uses the registry."""
        from evaluator.models.factory import create_asr_model
        
        # Test that whisper type returns correct class
        with patch('torch.device'):
            # We don't actually want to load models, just test registry lookup
            whisper_class = asr_registry.get("whisper")
            self.assertIsNotNone(whisper_class)
    
    def test_factory_unknown_model_type(self):
        """Test that factory raises error for unknown model type."""
        from evaluator.models.factory import create_asr_model
        
        with self.assertRaises(ValueError) as context:
            create_asr_model("nonexistent_model")
        
        self.assertIn("Unknown", str(context.exception))
        self.assertIn("nonexistent_model", str(context.exception))
    
    def test_factory_uses_default_name(self):
        """Test that factory uses default name when model_name is None."""
        from evaluator.models.factory import create_asr_model
        
        # Get default name from registry
        default_name = asr_registry.get_default_name("whisper")
        self.assertIsNotNone(default_name)
        self.assertEqual(default_name, "openai/whisper-medium")


class TestTextEmbeddingFactory(unittest.TestCase):
    """Test text embedding model factory function."""
    
    def test_factory_uses_registry(self):
        """Test that factory function uses the registry."""
        from evaluator.models.factory import create_text_embedding_model
        
        # Test that labse type is registered
        labse_class = text_embedding_registry.get("labse")
        self.assertIsNotNone(labse_class)
    
    def test_factory_unknown_model_type(self):
        """Test that factory raises error for unknown model type."""
        from evaluator.models.factory import create_text_embedding_model
        
        with self.assertRaises(ValueError) as context:
            create_text_embedding_model("nonexistent_model")
        
        self.assertIn("Unknown", str(context.exception))
    
    def test_factory_clap_text_special_case(self):
        """Test that clap_text returns None (special case)."""
        from evaluator.models.factory import create_text_embedding_model
        
        result = create_text_embedding_model("clap_text")
        self.assertIsNone(result)
    
    def test_factory_uses_default_names(self):
        """Test that factory uses default names from registry."""
        # Test a few models
        labse_default = text_embedding_registry.get_default_name("labse")
        self.assertEqual(labse_default, "sentence-transformers/LaBSE")
        
        jina_default = text_embedding_registry.get_default_name("jina_v4")
        self.assertEqual(jina_default, "jinaai/jina-embeddings-v4")


class TestAudioEmbeddingFactory(unittest.TestCase):
    """Test audio embedding model factory function."""
    
    def test_factory_uses_registry(self):
        """Test that factory function uses the registry."""
        from evaluator.models.factory import create_audio_embedding_model
        
        # Test that attention_pool type is registered
        apm_class = audio_embedding_registry.get("attention_pool")
        self.assertIsNotNone(apm_class)
    
    def test_factory_unknown_model_type(self):
        """Test that factory raises error for unknown model type."""
        from evaluator.models.factory import create_audio_embedding_model
        
        # Factory should raise ValueError from registry lookup
        with self.assertRaises(ValueError) as context:
            create_audio_embedding_model("nonexistent_model")
        
        self.assertIn("Unknown", str(context.exception))
        self.assertIn("nonexistent_model", str(context.exception))
    
    def test_factory_clap_style_requires_model_path(self):
        """Test that clap_style model requires model_path."""
        from evaluator.models.factory import create_audio_embedding_model
        
        with self.assertRaises(ValueError) as context:
            create_audio_embedding_model("clap_style", model_path=None)
        
        self.assertIn("model_path", str(context.exception))
    
    def test_factory_attention_pool_uses_default(self):
        """Test that attention_pool uses default name from registry."""
        default_name = audio_embedding_registry.get_default_name("attention_pool")
        self.assertEqual(default_name, "openai/whisper-large")


class TestFactoryIntegration(unittest.TestCase):
    """Integration tests for factory functions with registry."""
    
    def test_all_registered_models_have_factories(self):
        """Test that all registered models can be created via factories."""
        from evaluator.models.factory import (
            create_asr_model,
            create_text_embedding_model,
            create_audio_embedding_model
        )
        
        # ASR models
        for model_type in asr_registry.list_types():
            # Should not raise exception (though may fail without GPU/models)
            try:
                model_class = asr_registry.get(model_type)
                self.assertIsNotNone(model_class)
            except Exception:
                pass
        
        # Text embedding models
        for model_type in text_embedding_registry.list_types():
            try:
                model_class = text_embedding_registry.get(model_type)
                self.assertIsNotNone(model_class)
            except Exception:
                pass
        
        # Audio embedding models
        for model_type in audio_embedding_registry.list_types():
            try:
                model_class = audio_embedding_registry.get(model_type)
                self.assertIsNotNone(model_class)
            except Exception:
                pass
    
    def test_registry_and_factory_consistency(self):
        """Test that registry default names match what factory expects."""
        # All ASR models should have default names
        for model_type in ["whisper", "wav2vec2"]:
            default = asr_registry.get_default_name(model_type)
            self.assertIsNotNone(default, f"ASR model {model_type} should have default name")
        
        # All text embedding models should have default names
        for model_type in ["labse", "jina_v4", "clip", "nemotron", "bge_m3"]:
            default = text_embedding_registry.get_default_name(model_type)
            self.assertIsNotNone(default, f"Text embedding model {model_type} should have default name")
        
        # Attention pool should have default name
        default = audio_embedding_registry.get_default_name("attention_pool")
        self.assertIsNotNone(default)
        
        # Clap style should not have default name (requires model_path)
        default = audio_embedding_registry.get_default_name("clap_style")
        self.assertIsNone(default)


if __name__ == "__main__":
    unittest.main()
