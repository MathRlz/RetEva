"""Unit tests for device detection and auto-configuration."""
import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.config import (
    detect_device,
    get_available_gpu_count,
    ModelConfig,
    EvaluationConfig,
)


class TestDetectDevice(unittest.TestCase):
    """Test detect_device function."""
    
    def test_detect_device_returns_cpu_when_preferred(self):
        """Test that cpu is returned when explicitly preferred."""
        result = detect_device(preferred="cpu")
        self.assertEqual(result, "cpu")
    
    def test_detect_device_returns_cuda_when_available_and_preferred(self):
        """Test that preferred cuda device is returned when valid."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            # Need to re-import to pick up the mock
            import importlib
            import evaluator.config
            importlib.reload(evaluator.config)
            
            result = evaluator.config.detect_device(preferred="cuda:0")
            self.assertEqual(result, "cuda:0")
            
            result = evaluator.config.detect_device(preferred="cuda:1")
            self.assertEqual(result, "cuda:1")
    
    def test_detect_device_falls_back_when_preferred_cuda_unavailable(self):
        """Test fallback to auto-detect when preferred CUDA device doesn't exist."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1  # Only cuda:0 exists
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            import importlib
            import evaluator.config
            importlib.reload(evaluator.config)
            
            # cuda:5 doesn't exist, should fall back to cuda:0
            result = evaluator.config.detect_device(preferred="cuda:5")
            self.assertEqual(result, "cuda:0")
    
    def test_detect_device_falls_back_to_cpu_when_no_cuda(self):
        """Test fallback to cpu when preferred is CUDA but no CUDA available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            import importlib
            import evaluator.config
            importlib.reload(evaluator.config)
            
            result = evaluator.config.detect_device(preferred="cuda:0")
            self.assertEqual(result, "cpu")
    
    def test_detect_device_auto_selects_cuda(self):
        """Test auto-detection selects cuda:0 when available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            import importlib
            import evaluator.config
            importlib.reload(evaluator.config)
            
            result = evaluator.config.detect_device()
            self.assertEqual(result, "cuda:0")
    
    def test_detect_device_auto_selects_cpu(self):
        """Test auto-detection selects cpu when no CUDA available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            import importlib
            import evaluator.config
            importlib.reload(evaluator.config)
            
            result = evaluator.config.detect_device()
            self.assertEqual(result, "cpu")


class TestGetAvailableGpuCount(unittest.TestCase):
    """Test get_available_gpu_count function."""
    
    def test_returns_zero_when_no_cuda(self):
        """Test returns 0 when CUDA is not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            import importlib
            import evaluator.config
            importlib.reload(evaluator.config)
            
            result = evaluator.config.get_available_gpu_count()
            self.assertEqual(result, 0)
    
    def test_returns_device_count_when_cuda_available(self):
        """Test returns device count when CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            import importlib
            import evaluator.config
            importlib.reload(evaluator.config)
            
            result = evaluator.config.get_available_gpu_count()
            self.assertEqual(result, 4)


class TestModelConfigAutoConfigureDevices(unittest.TestCase):
    """Test ModelConfig.auto_configure_devices method."""
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_auto_configure_no_gpus(self, mock_gpu_count):
        """Test all devices set to cpu when no GPUs available."""
        mock_gpu_count.return_value = 0
        
        config = ModelConfig()
        config.auto_configure_devices()
        
        self.assertEqual(config.asr_device, "cpu")
        self.assertEqual(config.text_emb_device, "cpu")
        self.assertEqual(config.audio_emb_device, "cpu")
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_auto_configure_single_gpu(self, mock_gpu_count):
        """Test all devices set to cuda:0 when single GPU available."""
        mock_gpu_count.return_value = 1
        
        config = ModelConfig()
        config.auto_configure_devices()
        
        self.assertEqual(config.asr_device, "cuda:0")
        self.assertEqual(config.text_emb_device, "cuda:0")
        self.assertEqual(config.audio_emb_device, "cuda:0")
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_auto_configure_multiple_gpus(self, mock_gpu_count):
        """Test devices distributed across GPUs when multiple available."""
        mock_gpu_count.return_value = 2
        
        config = ModelConfig()
        config.auto_configure_devices()
        
        self.assertEqual(config.asr_device, "cuda:0")
        self.assertEqual(config.text_emb_device, "cuda:1")
        self.assertEqual(config.audio_emb_device, "cuda:0")


class TestEvaluationConfigWithAutoDevices(unittest.TestCase):
    """Test EvaluationConfig.with_auto_devices method."""
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_with_auto_devices_returns_copy(self, mock_gpu_count):
        """Test that with_auto_devices returns a new config, not mutating original."""
        mock_gpu_count.return_value = 0
        
        original = EvaluationConfig()
        original_asr_device = original.model.asr_device
        
        new_config = original.with_auto_devices()
        
        # Original should be unchanged
        self.assertEqual(original.model.asr_device, original_asr_device)
        # New config should have cpu
        self.assertEqual(new_config.model.asr_device, "cpu")
        # Should be different objects
        self.assertIsNot(original, new_config)
        self.assertIsNot(original.model, new_config.model)
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_with_auto_devices_preserves_other_settings(self, mock_gpu_count):
        """Test that with_auto_devices preserves non-device settings."""
        mock_gpu_count.return_value = 0
        
        original = EvaluationConfig(
            experiment_name="test_experiment",
            output_dir="test_output",
        )
        original.model.asr_model_type = "whisper"
        original.model.pipeline_mode = "asr_text_retrieval"
        
        new_config = original.with_auto_devices()
        
        self.assertEqual(new_config.experiment_name, "test_experiment")
        self.assertEqual(new_config.output_dir, "test_output")
        self.assertEqual(new_config.model.asr_model_type, "whisper")
        self.assertEqual(new_config.model.pipeline_mode, "asr_text_retrieval")
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_with_auto_devices_single_gpu(self, mock_gpu_count):
        """Test with_auto_devices with single GPU."""
        mock_gpu_count.return_value = 1
        
        config = EvaluationConfig().with_auto_devices()
        
        self.assertEqual(config.model.asr_device, "cuda:0")
        self.assertEqual(config.model.text_emb_device, "cuda:0")
        self.assertEqual(config.model.audio_emb_device, "cuda:0")


class TestPresetAutoDevices(unittest.TestCase):
    """Test that presets use auto-device detection."""
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_preset_auto_devices_no_gpu(self, mock_gpu_count):
        """Test presets auto-configure to cpu when no GPU."""
        mock_gpu_count.return_value = 0
        
        from evaluator.config.model_presets import get_preset
        preset = get_preset("whisper_labse")
        
        self.assertEqual(preset["model"]["asr_device"], "cpu")
        self.assertEqual(preset["model"]["text_emb_device"], "cpu")
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_preset_auto_devices_single_gpu(self, mock_gpu_count):
        """Test presets auto-configure to cuda:0 when single GPU."""
        mock_gpu_count.return_value = 1
        
        from evaluator.config.model_presets import get_preset
        preset = get_preset("audio_only")
        
        self.assertEqual(preset["model"]["asr_device"], "cuda:0")
        self.assertEqual(preset["model"]["audio_emb_device"], "cuda:0")
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_preset_auto_devices_multiple_gpus(self, mock_gpu_count):
        """Test presets distribute across GPUs when multiple available."""
        mock_gpu_count.return_value = 2
        
        from evaluator.config.model_presets import get_preset
        preset = get_preset("wav2vec_jina")
        
        self.assertEqual(preset["model"]["asr_device"], "cuda:0")
        self.assertEqual(preset["model"]["text_emb_device"], "cuda:1")
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_preset_auto_devices_disabled(self, mock_gpu_count):
        """Test presets don't add device keys when auto_devices=False."""
        mock_gpu_count.return_value = 0
        
        from evaluator.config.model_presets import get_preset
        preset = get_preset("whisper_labse", auto_devices=False)
        
        # Should not have device keys since we removed them from presets
        self.assertNotIn("asr_device", preset["model"])
        self.assertNotIn("text_emb_device", preset["model"])
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_from_preset_with_auto_devices(self, mock_gpu_count):
        """Test EvaluationConfig.from_preset uses auto-detection."""
        mock_gpu_count.return_value = 0
        
        config = EvaluationConfig.from_preset("whisper_labse")
        
        self.assertEqual(config.model.asr_device, "cpu")
        self.assertEqual(config.model.text_emb_device, "cpu")


class TestFromYamlEdgeCases(unittest.TestCase):
    """Test EvaluationConfig.from_yaml edge cases."""
    
    def test_from_yaml_missing_file_raises_error(self):
        """Test that missing YAML file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            EvaluationConfig.from_yaml("/nonexistent/path/config.yaml")
    
    def test_from_yaml_invalid_yaml_raises_error(self):
        """Test that invalid YAML content raises error."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            temp_path = f.name
        
        try:
            with self.assertRaises(Exception):  # yaml.scanner.ScannerError
                EvaluationConfig.from_yaml(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_from_yaml_empty_file_raises_error(self):
        """Test that empty YAML file raises error (None cannot be processed)."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            # Empty YAML returns None, which can't be processed
            with self.assertRaises(AttributeError):
                EvaluationConfig.from_yaml(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_from_yaml_with_unknown_keys_ignored(self):
        """Test that unknown keys in YAML are passed through."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("experiment_name: test\nunknown_key: value\n")
            temp_path = f.name
        
        try:
            # This should raise TypeError due to unexpected keyword argument
            with self.assertRaises(TypeError):
                EvaluationConfig.from_yaml(temp_path)
        finally:
            os.unlink(temp_path)


class TestFromPresetEdgeCases(unittest.TestCase):
    """Test EvaluationConfig.from_preset edge cases."""
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_from_preset_invalid_name_raises_error(self, mock_gpu_count):
        """Test that invalid preset name raises ValueError."""
        mock_gpu_count.return_value = 0
        
        with self.assertRaises(ValueError) as context:
            EvaluationConfig.from_preset("nonexistent_preset")
        
        self.assertIn("Unknown preset", str(context.exception))
        self.assertIn("nonexistent_preset", str(context.exception))
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_from_preset_empty_name_raises_error(self, mock_gpu_count):
        """Test that empty preset name raises ValueError."""
        mock_gpu_count.return_value = 0
        
        with self.assertRaises(ValueError):
            EvaluationConfig.from_preset("")
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_from_preset_with_nested_overrides(self, mock_gpu_count):
        """Test from_preset with nested key overrides."""
        mock_gpu_count.return_value = 0
        
        config = EvaluationConfig.from_preset(
            "fast_dev",
            model_asr_model_name="custom/model",
            data_batch_size=64,
            cache_enabled=False
        )
        
        self.assertEqual(config.model.asr_model_name, "custom/model")
        self.assertEqual(config.data.batch_size, 64)
        self.assertFalse(config.cache.enabled)
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_from_preset_vector_db_override(self, mock_gpu_count):
        """Test from_preset with vector_db nested override."""
        mock_gpu_count.return_value = 0
        
        config = EvaluationConfig.from_preset(
            "fast_dev",
            vector_db_k=10,
            vector_db_type="faiss"
        )
        
        self.assertEqual(config.vector_db.k, 10)
        self.assertEqual(config.vector_db.type, "faiss")


class TestConfigValidationEdgeCases(unittest.TestCase):
    """Test configuration validation edge cases."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig has sensible defaults."""
        config = ModelConfig()
        
        self.assertEqual(config.asr_model_type, "wav2vec2")
        self.assertEqual(config.pipeline_mode, "asr_text_retrieval")
        self.assertIsNone(config.asr_model_name)


class TestConfigValidation(unittest.TestCase):
    """Test EvaluationConfig.validate method."""
    
    def test_validate_valid_config_returns_no_warnings(self):
        """Test that a valid config with defaults returns no warnings."""
        # Use validate=False initially to construct, then call validate
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        
        warnings = config.validate()
        self.assertEqual(warnings, [])
    
    def test_validate_invalid_device_format(self):
        """Test that invalid device format raises ConfigurationError."""
        from evaluator.config import ConfigurationError
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "invalid_device"
        
        with self.assertRaises(ConfigurationError) as context:
            config.validate()
        
        self.assertIn("Invalid device format", str(context.exception))
        self.assertIn("asr_device", str(context.exception))
    
    def test_validate_cuda_without_number(self):
        """Test that 'cuda' without number is valid."""
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cuda"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        
        # Should not raise, may return warnings about CUDA availability
        config.validate()
    
    def test_validate_mps_device(self):
        """Test that 'mps' device is valid."""
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "mps"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        
        # Should not raise
        config.validate()
    
    def test_validate_invalid_asr_model_type(self):
        """Test that unknown ASR model type raises ConfigurationError."""
        from evaluator.config import ConfigurationError
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.model.asr_model_type = "nonexistent_asr"
        
        with self.assertRaises(ConfigurationError) as context:
            config.validate()
        
        self.assertIn("Unknown ASR model type", str(context.exception))
        self.assertIn("nonexistent_asr", str(context.exception))
    
    def test_validate_invalid_text_emb_model_type(self):
        """Test that unknown text embedding model type raises ConfigurationError."""
        from evaluator.config import ConfigurationError
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.model.text_emb_model_type = "nonexistent_emb"
        
        with self.assertRaises(ConfigurationError) as context:
            config.validate()
        
        self.assertIn("Unknown text embedding model type", str(context.exception))
    
    def test_validate_invalid_audio_emb_model_type(self):
        """Test that unknown audio embedding model type raises ConfigurationError."""
        from evaluator.config import ConfigurationError
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.model.audio_emb_model_type = "nonexistent_audio"
        
        with self.assertRaises(ConfigurationError) as context:
            config.validate()
        
        self.assertIn("Unknown audio embedding model type", str(context.exception))
    
    def test_validate_none_model_types_allowed(self):
        """Test that None model types are allowed (for optional components)."""
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.model.asr_model_type = None
        config.model.text_emb_model_type = None
        config.model.audio_emb_model_type = None
        
        # Should not raise
        warnings = config.validate()
        self.assertIsInstance(warnings, list)
    
    def test_validate_nonexistent_questions_path(self):
        """Test that non-existent questions path raises ConfigurationError."""
        from evaluator.config import ConfigurationError
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.data.questions_path = "/nonexistent/path/questions.json"
        
        with self.assertRaises(ConfigurationError) as context:
            config.validate()
        
        self.assertIn("Questions path does not exist", str(context.exception))
    
    def test_validate_nonexistent_corpus_path(self):
        """Test that non-existent corpus path raises ConfigurationError."""
        from evaluator.config import ConfigurationError
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.data.corpus_path = "/nonexistent/path/corpus.json"
        
        with self.assertRaises(ConfigurationError) as context:
            config.validate()
        
        self.assertIn("Corpus path does not exist", str(context.exception))
    
    def test_validate_zero_batch_size(self):
        """Test that zero batch size raises ConfigurationError."""
        from evaluator.config import ConfigurationError
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.data.batch_size = 0
        
        with self.assertRaises(ConfigurationError) as context:
            config.validate()
        
        self.assertIn("Batch size must be positive", str(context.exception))
    
    def test_validate_negative_batch_size(self):
        """Test that negative batch size raises ConfigurationError."""
        from evaluator.config import ConfigurationError
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.data.batch_size = -5
        
        with self.assertRaises(ConfigurationError) as context:
            config.validate()
        
        self.assertIn("Batch size must be positive", str(context.exception))
    
    def test_validate_large_batch_size_warning(self):
        """Test that very large batch size returns warning."""
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.data.batch_size = 512
        
        warnings = config.validate()
        
        self.assertTrue(any("large batch size" in w.lower() for w in warnings))
    
    def test_validate_zero_vector_db_k(self):
        """Test that zero k value raises ConfigurationError."""
        from evaluator.config import ConfigurationError
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.vector_db.k = 0
        
        with self.assertRaises(ConfigurationError) as context:
            config.validate()
        
        self.assertIn("vector_db.k must be positive", str(context.exception))
    
    def test_validate_negative_reranker_top_k(self):
        """Test that negative reranker_top_k raises ConfigurationError."""
        from evaluator.config import ConfigurationError
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.vector_db.reranker_top_k = -1
        
        with self.assertRaises(ConfigurationError) as context:
            config.validate()
        
        self.assertIn("reranker_top_k must be positive", str(context.exception))
    
    def test_validate_multiple_errors(self):
        """Test that multiple errors are collected and reported."""
        from evaluator.config import ConfigurationError
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "invalid"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.model.asr_model_type = "nonexistent"
        config.data.batch_size = 0
        
        with self.assertRaises(ConfigurationError) as context:
            config.validate()
        
        error_msg = str(context.exception)
        self.assertIn("Invalid device format", error_msg)
        self.assertIn("Unknown ASR model type", error_msg)
        self.assertIn("Batch size must be positive", error_msg)
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_validate_cuda_availability_warning(self, mock_gpu_count):
        """Test that using CUDA when unavailable produces warning."""
        mock_gpu_count.return_value = 0
        
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.device_count.return_value = 0
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            import importlib
            import evaluator.config
            importlib.reload(evaluator.config)
            
            config = evaluator.config.EvaluationConfig.from_dict({}, validate=False)
            config.model.asr_device = "cuda:0"
            config.model.text_emb_device = "cpu"
            config.model.audio_emb_device = "cpu"
            
            warnings = config.validate()
            
            self.assertTrue(any("CUDA is not available" in w for w in warnings))
    
    def test_from_dict_validates_by_default(self):
        """Test that from_dict validates by default."""
        from evaluator.config import ConfigurationError
        
        with self.assertRaises(ConfigurationError):
            EvaluationConfig.from_dict({
                "model": {"asr_model_type": "nonexistent"},
            })
    
    def test_from_dict_skip_validation(self):
        """Test that from_dict can skip validation."""
        # Should not raise even with invalid config
        config = EvaluationConfig.from_dict(
            {"model": {"asr_model_type": "nonexistent"}},
            validate=False
        )
        self.assertEqual(config.model.asr_model_type, "nonexistent")
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_from_preset_validates_by_default(self, mock_gpu_count):
        """Test that from_preset validates by default."""
        mock_gpu_count.return_value = 0
        
        # Presets should be valid so this should not raise
        config = EvaluationConfig.from_preset("fast_dev")
        self.assertIsNotNone(config)
    
    @patch('evaluator.config.get_available_gpu_count')
    def test_from_preset_skip_validation(self, mock_gpu_count):
        """Test that from_preset can skip validation."""
        mock_gpu_count.return_value = 0
        
        config = EvaluationConfig.from_preset(
            "fast_dev",
            validate=False,
            model_asr_model_type="nonexistent"
        )
        self.assertEqual(config.model.asr_model_type, "nonexistent")
    
    def test_from_yaml_validates_by_default(self):
        """Test that from_yaml validates by default."""
        from evaluator.config import ConfigurationError
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("model:\n  asr_model_type: nonexistent\n")
            temp_path = f.name
        
        try:
            with self.assertRaises(ConfigurationError):
                EvaluationConfig.from_yaml(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_from_yaml_skip_validation(self):
        """Test that from_yaml can skip validation."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("model:\n  asr_model_type: nonexistent\n")
            temp_path = f.name
        
        try:
            config = EvaluationConfig.from_yaml(temp_path, validate=False)
            self.assertEqual(config.model.asr_model_type, "nonexistent")
        finally:
            os.unlink(temp_path)
    
    def test_cache_config_defaults(self):
        """Test CacheConfig has sensible defaults."""
        from evaluator.config import CacheConfig
        
        config = CacheConfig()
        
        self.assertTrue(config.enabled)
        self.assertEqual(config.cache_dir, ".cache")
        self.assertTrue(config.cache_asr_features)
    
    def test_data_config_defaults(self):
        """Test DataConfig has sensible defaults."""
        from evaluator.config import DataConfig
        
        config = DataConfig()
        
        self.assertEqual(config.dataset_name, "admed_voice")
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.strict_validation)
    
    def test_evaluation_config_to_runtime_and_experiment_dict(self):
        """Test grouped runtime/experiment config surfaces."""
        config = EvaluationConfig(
            experiment_name="test_exp",
            output_dir="results/"
        )
        
        runtime = config.to_runtime_dict()
        experiment = config.to_experiment_dict()
        
        self.assertIn("model", runtime)
        self.assertIn("pipeline_mode", runtime["model"])
        self.assertEqual(experiment["experiment_name"], "test_exp")
        self.assertEqual(experiment["output_dir"], "results/")
    
    def test_evaluation_config_to_yaml_and_back(self):
        """Test round-trip YAML serialization."""
        import tempfile
        import os
        
        original = EvaluationConfig(
            experiment_name="roundtrip_test",
            output_dir="test_output"
        )
        original.model.asr_model_type = "whisper"
        original.data.batch_size = 16
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            original.to_yaml(temp_path)
            loaded = EvaluationConfig.from_yaml(temp_path)
            
            self.assertEqual(loaded.experiment_name, "roundtrip_test")
            self.assertEqual(loaded.output_dir, "test_output")
            self.assertEqual(loaded.model.asr_model_type, "whisper")
            self.assertEqual(loaded.data.batch_size, 16)
        finally:
            os.unlink(temp_path)
    
    def test_logging_config_get_levels(self):
        """Test LoggingConfig level conversion methods."""
        from evaluator.config import LoggingConfig
        import logging
        
        config = LoggingConfig(console_level="WARNING", file_level="DEBUG")
        
        self.assertEqual(config.get_console_level(), logging.WARNING)
        self.assertEqual(config.get_file_level(), logging.DEBUG)


class TestFromDictEdgeCases(unittest.TestCase):
    """Test EvaluationConfig.from_dict edge cases."""
    
    def test_from_dict_empty(self):
        """Test from_dict with empty dictionary uses defaults."""
        config = EvaluationConfig.from_dict({})
        
        self.assertEqual(config.experiment_name, "evaluation")
        self.assertEqual(config.output_dir, "evaluation_results")
    
    def test_from_dict_partial_nested(self):
        """Test from_dict with partial nested configuration."""
        config = EvaluationConfig.from_dict({
            "experiment_name": "partial_test",
            "model": {
                "asr_model_type": "whisper"
                # Other model fields use defaults
            }
        })
        
        self.assertEqual(config.experiment_name, "partial_test")
        self.assertEqual(config.model.asr_model_type, "whisper")
        self.assertEqual(config.model.pipeline_mode, "asr_text_retrieval")  # default
    
    def test_from_dict_all_sections(self):
        """Test from_dict with all configuration sections."""
        config_dict = {
            "experiment_name": "full_test",
            "cache": {"enabled": False},
            "logging": {"console_level": "DEBUG"},
            "model": {"pipeline_mode": "audio_emb_retrieval"},
            "data": {"batch_size": 64},
            "audio_synthesis": {"enabled": True},
            "judge": {"enabled": True, "model": "gpt-4"},
            "vector_db": {"k": 10, "type": "faiss"}
        }
        
        config = EvaluationConfig.from_dict(config_dict)
        
        self.assertEqual(config.experiment_name, "full_test")
        self.assertFalse(config.cache.enabled)
        self.assertEqual(config.logging.console_level, "DEBUG")
        self.assertEqual(config.model.pipeline_mode, "audio_emb_retrieval")
        self.assertEqual(config.data.batch_size, 64)
        self.assertTrue(config.audio_synthesis.enabled)
        self.assertTrue(config.judge.enabled)
        self.assertEqual(config.judge.model, "gpt-4")
        self.assertEqual(config.vector_db.k, 10)


class TestGpuMemoryFunctions(unittest.TestCase):
    """Test GPU memory helper functions."""
    
    def test_get_gpu_memory_gb_returns_none_when_no_cuda(self):
        """Test get_gpu_memory_gb returns None when CUDA unavailable."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            import importlib
            import evaluator.config
            importlib.reload(evaluator.config)
            
            result = evaluator.config.get_gpu_memory_gb(0)
            self.assertIsNone(result)
    
    def test_get_gpu_memory_gb_returns_memory_when_available(self):
        """Test get_gpu_memory_gb returns memory size when available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        
        mock_props = MagicMock()
        mock_props.total_memory = 8 * (1024 ** 3)  # 8 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            import importlib
            import evaluator.config
            importlib.reload(evaluator.config)
            
            result = evaluator.config.get_gpu_memory_gb(0)
            self.assertAlmostEqual(result, 8.0, places=1)
    
    def test_get_gpu_free_memory_gb_returns_none_when_no_cuda(self):
        """Test get_gpu_free_memory_gb returns None when CUDA unavailable."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        with patch.dict('sys.modules', {'torch': mock_torch}):
            import importlib
            import evaluator.config
            importlib.reload(evaluator.config)
            
            result = evaluator.config.get_gpu_free_memory_gb(0)
            self.assertIsNone(result)
    
    def test_estimate_model_memory_gb_returns_estimate(self):
        """Test estimate_model_memory_gb returns expected estimates."""
        from evaluator.config import estimate_model_memory_gb
        
        whisper_mem = estimate_model_memory_gb("asr", "whisper")
        self.assertGreater(whisper_mem, 0)
        
        labse_mem = estimate_model_memory_gb("text_embedding", "labse")
        self.assertGreater(labse_mem, 0)
        
        # None model type returns 0
        none_mem = estimate_model_memory_gb("asr", None)
        self.assertEqual(none_mem, 0.0)
    
    def test_estimate_model_memory_gb_uses_default(self):
        """Test estimate_model_memory_gb uses default for unknown models."""
        from evaluator.config import estimate_model_memory_gb
        
        mem = estimate_model_memory_gb("asr", "unknown_model")
        self.assertGreater(mem, 0)


class TestEmbeddingDimensionValidation(unittest.TestCase):
    """Test embedding dimension compatibility checks."""
    
    def test_validate_warns_on_dimension_mismatch(self):
        """Test that dimension mismatch produces warning in audio_emb_retrieval mode."""
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.model.pipeline_mode = "audio_emb_retrieval"
        config.model.text_emb_model_type = "labse"  # 768 dim
        config.model.audio_emb_model_type = "attention_pool"
        config.model.audio_emb_dim = 1024  # Mismatch!
        
        warnings = config.validate()
        
        self.assertTrue(
            any("dimension mismatch" in w.lower() for w in warnings),
            f"Expected dimension mismatch warning, got: {warnings}"
        )
    
    def test_validate_no_warning_on_matching_dimensions(self):
        """Test that matching dimensions produce no warning."""
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.model.pipeline_mode = "audio_emb_retrieval"
        config.model.text_emb_model_type = "labse"  # 768 dim
        config.model.audio_emb_model_type = "attention_pool"
        config.model.audio_emb_dim = 768  # Matches!
        
        warnings = config.validate()
        
        self.assertFalse(
            any("dimension mismatch" in w.lower() for w in warnings),
            f"Unexpected dimension mismatch warning: {warnings}"
        )
    
    def test_validate_no_dimension_check_for_asr_pipeline(self):
        """Test that dimension check is skipped for asr_text_retrieval mode."""
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.model.pipeline_mode = "asr_text_retrieval"
        config.model.text_emb_model_type = "labse"
        config.model.audio_emb_dim = 2048  # Different but shouldn't matter
        
        warnings = config.validate()
        
        self.assertFalse(
            any("dimension mismatch" in w.lower() for w in warnings),
            f"Should not check dimensions for asr_text_retrieval: {warnings}"
        )


class TestPreflightCheck(unittest.TestCase):
    """Test preflight_check function."""
    
    def test_preflight_check_returns_warnings_list(self):
        """Test that preflight_check returns a list."""
        from evaluator.config import preflight_check
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        
        result = preflight_check(config)
        self.assertIsInstance(result, list)
    
    def test_preflight_check_raises_on_fatal_errors(self):
        """Test that preflight_check raises ConfigurationError for fatal issues."""
        from evaluator.config import preflight_check, ConfigurationError
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "invalid_device"  # Fatal error
        
        with self.assertRaises(ConfigurationError):
            preflight_check(config)
    
    def test_preflight_check_warns_on_missing_asr_for_asr_pipeline(self):
        """Test that preflight_check warns when ASR model missing for ASR pipeline."""
        from evaluator.config import preflight_check
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.model.pipeline_mode = "asr_text_retrieval"
        config.model.asr_model_type = None  # Missing!
        
        warnings = preflight_check(config)
        
        self.assertTrue(
            any("asr_model_type is None" in w for w in warnings),
            f"Expected warning about missing ASR model, got: {warnings}"
        )
    
    def test_preflight_check_warns_on_missing_audio_emb_for_audio_pipeline(self):
        """Test that preflight_check warns when audio_emb missing for audio pipeline."""
        from evaluator.config import preflight_check
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.model.pipeline_mode = "audio_emb_retrieval"
        config.model.audio_emb_model_type = None  # Missing!
        
        warnings = preflight_check(config)
        
        self.assertTrue(
            any("audio_emb_model_type is None" in w for w in warnings),
            f"Expected warning about missing audio emb model, got: {warnings}"
        )
    
    def test_preflight_check_warns_on_missing_judge_api_key(self):
        """Test that preflight_check warns when judge API key is missing."""
        from evaluator.config import preflight_check
        import os
        
        config = EvaluationConfig.from_dict({}, validate=False)
        config.model.asr_device = "cpu"
        config.model.text_emb_device = "cpu"
        config.model.audio_emb_device = "cpu"
        config.judge.enabled = True
        config.judge.api_key_env = "TEST_NONEXISTENT_API_KEY_12345"
        
        # Ensure the env var is not set
        original = os.environ.pop("TEST_NONEXISTENT_API_KEY_12345", None)
        try:
            warnings = preflight_check(config)
            
            self.assertTrue(
                any("environment variable" in w and "not set" in w for w in warnings),
                f"Expected warning about missing API key, got: {warnings}"
            )
        finally:
            if original is not None:
                os.environ["TEST_NONEXISTENT_API_KEY_12345"] = original
    
    def test_preflight_check_warns_on_insufficient_gpu_memory(self):
        """Test that preflight_check warns when GPU memory is insufficient."""
        from evaluator.config import preflight_check
        
        # Use patch at the module level where it's actually called
        with patch('evaluator.config.get_gpu_free_memory_gb') as mock_free_mem, \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1):
            
            mock_free_mem.return_value = 0.5  # Only 0.5 GB free
            
            config = EvaluationConfig.from_dict({}, validate=False)
            config.model.asr_device = "cuda:0"
            config.model.text_emb_device = "cpu"
            config.model.audio_emb_device = "cpu"
            config.model.asr_model_type = "whisper"  # Needs ~2.5 GB
            
            warnings = preflight_check(config)
            
            self.assertTrue(
                any("free memory" in w.lower() or "exceeds" in w.lower() for w in warnings),
                f"Expected warning about GPU memory, got: {warnings}"
            )


class TestGetTextEmbeddingDim(unittest.TestCase):
    """Test get_text_embedding_dim function."""
    
    def test_returns_known_dimensions(self):
        """Test that known model dimensions are returned."""
        from evaluator.config import get_text_embedding_dim
        
        self.assertEqual(get_text_embedding_dim("labse"), 768)
        self.assertEqual(get_text_embedding_dim("jina_v4"), 1024)
        self.assertEqual(get_text_embedding_dim("bge_m3"), 1024)
    
    def test_returns_none_for_unknown(self):
        """Test that None is returned for unknown models."""
        from evaluator.config import get_text_embedding_dim
        
        self.assertIsNone(get_text_embedding_dim("unknown_model"))
        self.assertIsNone(get_text_embedding_dim(None))


if __name__ == "__main__":
    unittest.main()
