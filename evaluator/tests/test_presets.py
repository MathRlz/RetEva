"""Unit tests for evaluation presets."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.config.model_presets import (
    get_preset, list_presets,
    WHISPER_LABSE_PRESET, WAV2VEC_JINA_PRESET, AUDIO_ONLY_PRESET, FAST_DEV_PRESET
)
from evaluator.config import EvaluationConfig


class TestPresets(unittest.TestCase):
    """Test preset definitions and retrieval."""
    
    def test_list_presets(self):
        """Test that list_presets returns expected presets."""
        presets = list_presets()
        self.assertIn("whisper_labse", presets)
        self.assertIn("wav2vec_jina", presets)
        self.assertIn("audio_only", presets)
        self.assertIn("fast_dev", presets)
    
    def test_get_preset_returns_copy(self):
        """Test that get_preset returns a copy, not the original."""
        preset1 = get_preset("fast_dev")
        preset2 = get_preset("fast_dev")
        
        preset1["experiment_name"] = "modified"
        self.assertNotEqual(preset1["experiment_name"], preset2["experiment_name"])
    
    def test_get_preset_unknown_raises(self):
        """Test that unknown preset name raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_preset("nonexistent_preset")
        
        self.assertIn("nonexistent_preset", str(context.exception))
        self.assertIn("Available presets", str(context.exception))
    
    def test_whisper_labse_preset_structure(self):
        """Test WHISPER_LABSE_PRESET has expected structure."""
        preset = WHISPER_LABSE_PRESET
        
        self.assertEqual(preset["model"]["pipeline_mode"], "asr_text_retrieval")
        self.assertEqual(preset["model"]["asr_model_type"], "whisper")
        self.assertEqual(preset["model"]["text_emb_model_type"], "labse")
    
    def test_wav2vec_jina_preset_structure(self):
        """Test WAV2VEC_JINA_PRESET has expected structure."""
        preset = WAV2VEC_JINA_PRESET
        
        self.assertEqual(preset["model"]["pipeline_mode"], "asr_text_retrieval")
        self.assertEqual(preset["model"]["asr_model_type"], "wav2vec2")
        self.assertEqual(preset["model"]["text_emb_model_type"], "jina_v4")
    
    def test_audio_only_preset_structure(self):
        """Test AUDIO_ONLY_PRESET has expected structure."""
        preset = AUDIO_ONLY_PRESET
        
        self.assertEqual(preset["model"]["pipeline_mode"], "audio_emb_retrieval")
        self.assertIsNone(preset["model"]["asr_model_type"])
        self.assertIsNotNone(preset["model"]["audio_emb_model_type"])
    
    def test_fast_dev_preset_structure(self):
        """Test FAST_DEV_PRESET has expected structure."""
        preset = FAST_DEV_PRESET
        
        self.assertEqual(preset["model"]["asr_size"], "tiny")
        # Device is no longer hardcoded - it uses auto-detection
        self.assertNotIn("asr_device", preset["model"])  # Devices configured by get_preset
        self.assertEqual(preset["data"]["trace_limit"], 50)
        self.assertFalse(preset["checkpoint_enabled"])


class TestEvaluationConfigFromPreset(unittest.TestCase):
    """Test EvaluationConfig.from_preset method."""
    
    def test_from_preset_basic(self):
        """Test creating config from preset."""
        config = EvaluationConfig.from_preset("fast_dev")
        
        self.assertEqual(config.experiment_name, "fast_dev")
        self.assertEqual(config.model.asr_model_type, "whisper")
        self.assertEqual(config.model.asr_size, "tiny")
        # Device is auto-detected based on available hardware
        self.assertIn(config.model.asr_device, ["cpu", "cuda:0"])
        self.assertEqual(config.data.trace_limit, 50)

    def test_from_preset_with_overrides(self):
        """Test creating config from preset with overrides."""
        config = EvaluationConfig.from_preset(
            "fast_dev",
            experiment_name="my_experiment",
            model_asr_device="cuda:0",
            data_batch_size=64
        )

        self.assertEqual(config.experiment_name, "my_experiment")
        self.assertEqual(config.model.asr_device, "cuda:0")
        self.assertEqual(config.data.batch_size, 64)
        # Unchanged from preset
        self.assertEqual(config.model.asr_size, "tiny")
    
    def test_from_preset_whisper_labse(self):
        """Test whisper_labse preset creates valid config."""
        config = EvaluationConfig.from_preset("whisper_labse")
        
        self.assertEqual(config.model.pipeline_mode, "asr_text_retrieval")
        self.assertEqual(config.model.asr_model_type, "whisper")
        self.assertEqual(config.model.text_emb_model_type, "labse")
    
    def test_from_preset_audio_only(self):
        """Test audio_only preset creates valid config."""
        config = EvaluationConfig.from_preset("audio_only")
        
        self.assertEqual(config.model.pipeline_mode, "audio_emb_retrieval")
        self.assertIsNone(config.model.asr_model_type)
        self.assertEqual(config.model.audio_emb_model_type, "attention_pool")
    
    def test_from_preset_unknown_raises(self):
        """Test that unknown preset name raises ValueError."""
        with self.assertRaises(ValueError):
            EvaluationConfig.from_preset("nonexistent")
    
    def test_from_preset_vector_db_override(self):
        """Test vector_db section overrides work."""
        config = EvaluationConfig.from_preset(
            "fast_dev",
            vector_db_k=10,
            vector_db_retrieval_mode="hybrid"
        )
        
        self.assertEqual(config.vector_db.k, 10)
        self.assertEqual(config.vector_db.retrieval_mode, "hybrid")


class TestEvaluationConfigFromDict(unittest.TestCase):
    """Test EvaluationConfig.from_dict method."""
    
    def test_from_dict_minimal(self):
        """Test creating config from minimal dict."""
        config = EvaluationConfig.from_dict({})
        
        # Should use all defaults
        self.assertEqual(config.experiment_name, "evaluation")
        self.assertEqual(config.model.asr_model_type, "wav2vec2")
    
    def test_from_dict_with_nested(self):
        """Test creating config with nested sections."""
        config = EvaluationConfig.from_dict({
            "experiment_name": "test_exp",
            "model": {
                "asr_model_type": "whisper",
                "pipeline_mode": "asr_text_retrieval"
            },
            "data": {
                "batch_size": 64
            }
        })
        
        self.assertEqual(config.experiment_name, "test_exp")
        self.assertEqual(config.model.asr_model_type, "whisper")
        self.assertEqual(config.data.batch_size, 64)


if __name__ == "__main__":
    unittest.main()
