"""Unit tests for model registry system."""
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.models.registry import (
    ModelRegistry,
    asr_registry,
    text_embedding_registry,
    audio_embedding_registry,
    register_asr_model,
    register_text_embedding_model,
    register_audio_embedding_model,
    get_all_registered_models
)
from evaluator.models.base import ASRModel, TextEmbeddingModel, AudioEmbeddingModel


class TestModelRegistry(unittest.TestCase):
    """Test the ModelRegistry class."""
    
    def setUp(self):
        """Create a fresh registry for each test."""
        self.registry = ModelRegistry("Test")
    
    def test_registry_initialization(self):
        """Test that registry initializes correctly."""
        self.assertEqual(self.registry.name, "Test")
        self.assertEqual(len(self.registry.list_types()), 0)
    
    def test_register_model(self):
        """Test registering a model class."""
        class DummyModel:
            pass
        
        self.registry.register("dummy", DummyModel, default_name="dummy-model")
        
        self.assertTrue(self.registry.is_registered("dummy"))
        self.assertEqual(self.registry.get("dummy"), DummyModel)
        self.assertEqual(self.registry.get_default_name("dummy"), "dummy-model")
    
    def test_register_with_metadata(self):
        """Test registering a model with metadata."""
        class DummyModel:
            pass
        
        self.registry.register(
            "dummy", 
            DummyModel,
            description="Test model",
            tags=["test", "dummy"]
        )
        
        metadata = self.registry.get_metadata("dummy")
        self.assertEqual(metadata["description"], "Test model")
        self.assertEqual(metadata["tags"], ["test", "dummy"])
    
    def test_get_nonexistent_model(self):
        """Test that getting a nonexistent model raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.registry.get("nonexistent")
        
        self.assertIn("Unknown", str(context.exception))
        self.assertIn("nonexistent", str(context.exception))
    
    def test_list_types(self):
        """Test listing registered model types."""
        class Model1:
            pass
        class Model2:
            pass
        
        self.registry.register("model1", Model1)
        self.registry.register("model2", Model2)
        
        types = self.registry.list_types()
        self.assertEqual(len(types), 2)
        self.assertIn("model1", types)
        self.assertIn("model2", types)
    
    def test_decorator_registration(self):
        """Test registering models using the decorator."""
        @self.registry.decorator("decorated", default_name="test-model")
        class DecoratedModel:
            pass
        
        self.assertTrue(self.registry.is_registered("decorated"))
        self.assertEqual(self.registry.get("decorated"), DecoratedModel)
        self.assertEqual(self.registry.get_default_name("decorated"), "test-model")
    
    def test_overwrite_warning(self):
        """Test that overwriting a model type logs a warning."""
        class Model1:
            pass
        class Model2:
            pass
        
        self.registry.register("model", Model1)
        # This should log a warning but not fail
        self.registry.register("model", Model2)
        
        # The second registration should overwrite
        self.assertEqual(self.registry.get("model"), Model2)


class TestGlobalRegistries(unittest.TestCase):
    """Test the global registry instances."""
    
    def test_asr_registry_exists(self):
        """Test that ASR registry exists and has correct name."""
        self.assertEqual(asr_registry.name, "ASR")
    
    def test_text_embedding_registry_exists(self):
        """Test that text embedding registry exists and has correct name."""
        self.assertEqual(text_embedding_registry.name, "TextEmbedding")
    
    def test_audio_embedding_registry_exists(self):
        """Test that audio embedding registry exists and has correct name."""
        self.assertEqual(audio_embedding_registry.name, "AudioEmbedding")
    
    def test_asr_models_registered(self):
        """Test that ASR models are registered."""
        types = asr_registry.list_types()
        self.assertIn("whisper", types)
        self.assertIn("wav2vec2", types)
    
    def test_text_embedding_models_registered(self):
        """Test that text embedding models are registered."""
        types = text_embedding_registry.list_types()
        self.assertIn("labse", types)
        self.assertIn("jina_v4", types)
        self.assertIn("clip", types)
        self.assertIn("nemotron", types)
        self.assertIn("bge_m3", types)
    
    def test_audio_embedding_models_registered(self):
        """Test that audio embedding models are registered."""
        types = audio_embedding_registry.list_types()
        self.assertIn("attention_pool", types)
        self.assertIn("clap_style", types)
    
    def test_default_names_exist(self):
        """Test that default names are set for models."""
        # ASR models
        self.assertIsNotNone(asr_registry.get_default_name("whisper"))
        self.assertIsNotNone(asr_registry.get_default_name("wav2vec2"))
        
        # Text embedding models
        self.assertIsNotNone(text_embedding_registry.get_default_name("labse"))
        self.assertIsNotNone(text_embedding_registry.get_default_name("jina_v4"))
        
        # Audio embedding models
        self.assertIsNotNone(audio_embedding_registry.get_default_name("attention_pool"))
    
    def test_metadata_exists(self):
        """Test that metadata (descriptions) are set for models."""
        # Check a few models have descriptions
        whisper_meta = asr_registry.get_metadata("whisper")
        self.assertIn("description", whisper_meta)
        self.assertTrue(len(whisper_meta["description"]) > 0)
        
        labse_meta = text_embedding_registry.get_metadata("labse")
        self.assertIn("description", labse_meta)
        self.assertTrue(len(labse_meta["description"]) > 0)


class TestRegistryDecorators(unittest.TestCase):
    """Test the decorator functions."""
    
    def test_register_asr_model_decorator(self):
        """Test the ASR model registration decorator."""
        # Create a temporary registry for testing
        test_registry = ModelRegistry("TestASR")
        
        @test_registry.decorator("test_asr", default_name="test/asr-model")
        class TestASRModel:
            pass
        
        self.assertTrue(test_registry.is_registered("test_asr"))
        self.assertEqual(test_registry.get("test_asr"), TestASRModel)
    
    def test_get_all_registered_models(self):
        """Test getting all registered models."""
        all_models = get_all_registered_models()
        
        self.assertIn("asr", all_models)
        self.assertIn("text_embedding", all_models)
        self.assertIn("audio_embedding", all_models)
        
        self.assertIsInstance(all_models["asr"], list)
        self.assertIsInstance(all_models["text_embedding"], list)
        self.assertIsInstance(all_models["audio_embedding"], list)
        
        # Check that models are present
        self.assertGreater(len(all_models["asr"]), 0)
        self.assertGreater(len(all_models["text_embedding"]), 0)
        self.assertGreater(len(all_models["audio_embedding"]), 0)


class TestModelLookup(unittest.TestCase):
    """Test looking up actual model classes."""
    
    def test_whisper_model_lookup(self):
        """Test looking up WhisperModel."""
        from evaluator.models import WhisperModel
        
        model_class = asr_registry.get("whisper")
        self.assertEqual(model_class, WhisperModel)
        self.assertTrue(issubclass(model_class, ASRModel))
    
    def test_wav2vec2_model_lookup(self):
        """Test looking up Wav2Vec2Model."""
        from evaluator.models import Wav2Vec2Model
        
        model_class = asr_registry.get("wav2vec2")
        self.assertEqual(model_class, Wav2Vec2Model)
        self.assertTrue(issubclass(model_class, ASRModel))
    
    def test_labse_model_lookup(self):
        """Test looking up LabseModel."""
        from evaluator.models import LabseModel
        
        model_class = text_embedding_registry.get("labse")
        self.assertEqual(model_class, LabseModel)
        self.assertTrue(issubclass(model_class, TextEmbeddingModel))
    
    def test_jina_model_lookup(self):
        """Test looking up JinaV4Model."""
        from evaluator.models import JinaV4Model
        
        model_class = text_embedding_registry.get("jina_v4")
        self.assertEqual(model_class, JinaV4Model)
        self.assertTrue(issubclass(model_class, TextEmbeddingModel))
    
    def test_attention_pool_model_lookup(self):
        """Test looking up AttentionPoolAudioModel."""
        from evaluator.models import AttentionPoolAudioModel
        
        model_class = audio_embedding_registry.get("attention_pool")
        self.assertEqual(model_class, AttentionPoolAudioModel)
        self.assertTrue(issubclass(model_class, AudioEmbeddingModel))


class TestParamsAndSizeResolution(unittest.TestCase):
    """Test Params dataclass auto-detection and size resolution."""

    def setUp(self):
        self.registry = ModelRegistry("ParamsTest")

    def _register_with_params(self):
        from dataclasses import dataclass
        from typing import ClassVar, Dict

        class FakeModel:
            @dataclass
            class Params:
                size: str = "base"
                pooling: str = "mean"
                SIZES: ClassVar[Dict[str, str]] = {
                    "base": "org/model-base",
                    "large": "org/model-large",
                }
        self.registry.register("fake", FakeModel, default_name="org/model-base")
        return FakeModel

    def test_params_auto_detected(self):
        cls = self._register_with_params()
        self.assertIs(self.registry.get_params_class("fake"), cls.Params)

    def test_get_sizes(self):
        self._register_with_params()
        sizes = self.registry.get_sizes("fake")
        self.assertEqual(sizes, {"base": "org/model-base", "large": "org/model-large"})

    def test_get_sizes_no_params(self):
        class NoParams:
            pass
        self.registry.register("plain", NoParams)
        self.assertEqual(self.registry.get_sizes("plain"), {})

    def test_get_default_size(self):
        self._register_with_params()
        self.assertEqual(self.registry.get_default_size("fake"), "base")

    def test_get_params_schema(self):
        self._register_with_params()
        schema = self.registry.get_params_schema("fake")
        self.assertIn("size", schema)
        self.assertEqual(schema["size"]["default"], "base")
        self.assertEqual(sorted(schema["size"]["choices"]), ["base", "large"])
        self.assertIn("pooling", schema)
        self.assertEqual(schema["pooling"]["default"], "mean")

    def test_resolve_model_name_explicit(self):
        self._register_with_params()
        self.assertEqual(
            self.registry.resolve_model_name("fake", model_name="custom/x"),
            "custom/x",
        )

    def test_resolve_model_name_by_size(self):
        self._register_with_params()
        self.assertEqual(
            self.registry.resolve_model_name("fake", size="large"),
            "org/model-large",
        )

    def test_resolve_model_name_default(self):
        self._register_with_params()
        self.assertEqual(
            self.registry.resolve_model_name("fake"),
            "org/model-base",
        )

    def test_resolve_model_name_bad_size(self):
        self._register_with_params()
        with self.assertRaises(ValueError) as ctx:
            self.registry.resolve_model_name("fake", size="xl")
        self.assertIn("xl", str(ctx.exception))

    def test_resolve_model_name_wins_over_size(self):
        """model_name takes priority when both given."""
        self._register_with_params()
        self.assertEqual(
            self.registry.resolve_model_name("fake", size="large", model_name="override/x"),
            "override/x",
        )

    def test_real_whisper_sizes(self):
        """Verify whisper Params registered in global registry."""
        sizes = asr_registry.get_sizes("whisper")
        self.assertIn("large-v3", sizes)
        self.assertEqual(sizes["large-v3"], "openai/whisper-large-v3")

    def test_real_clip_sizes(self):
        sizes = text_embedding_registry.get_sizes("clip")
        self.assertIn("large", sizes)

    def test_real_hubert_schema(self):
        schema = audio_embedding_registry.get_params_schema("hubert")
        self.assertIn("pooling", schema)
        self.assertEqual(schema["pooling"]["default"], "mean")


class TestRegistryEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Create a fresh registry for each test."""
        self.registry = ModelRegistry("EdgeCase")
    
    def test_register_none_as_model(self):
        """Test that registering None works (though not recommended)."""
        # This should not crash, registry is flexible
        self.registry.register("none_model", None)
        self.assertEqual(self.registry.get("none_model"), None)
    
    def test_empty_model_type(self):
        """Test registering with empty string as model type."""
        class DummyModel:
            pass
        
        # Empty string is a valid key
        self.registry.register("", DummyModel)
        self.assertTrue(self.registry.is_registered(""))
    
    def test_get_default_name_nonexistent(self):
        """Test getting default name for nonexistent model."""
        result = self.registry.get_default_name("nonexistent")
        self.assertIsNone(result)
    
    def test_get_metadata_nonexistent(self):
        """Test getting metadata for nonexistent model."""
        result = self.registry.get_metadata("nonexistent")
        self.assertEqual(result, {})
    
    def test_list_types_empty_registry(self):
        """Test listing types from empty registry."""
        types = self.registry.list_types()
        self.assertEqual(types, [])
        self.assertIsInstance(types, list)


if __name__ == "__main__":
    unittest.main()
