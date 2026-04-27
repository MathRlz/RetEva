"""Unit tests for the high-level convenience API."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.api import (
    evaluate_from_config,
    evaluate_from_preset,
    quick_evaluate,
    run_evaluation,
    run_evaluation_matrix,
    EvaluationError,
    ConfigurationError,
)
from evaluator.config import EvaluationConfig


class TestEvaluateFromConfig(unittest.TestCase):
    """Test evaluate_from_config function."""
    
    def test_raises_error_for_missing_file(self):
        """Test that ConfigurationError is raised for missing config file."""
        with self.assertRaises(ConfigurationError) as cm:
            evaluate_from_config("/nonexistent/path/config.yaml")
        self.assertIn("not found", str(cm.exception))
    
    def test_raises_error_for_non_yaml_file(self):
        """Test that ConfigurationError is raised for non-YAML files."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"key": "value"}')
            temp_path = f.name
        
        try:
            with self.assertRaises(ConfigurationError) as cm:
                evaluate_from_config(temp_path)
            self.assertIn("YAML", str(cm.exception))
        finally:
            Path(temp_path).unlink()
    
    @patch('evaluator.api.run_evaluation')
    @patch('evaluator.config.EvaluationConfig.from_yaml')
    def test_loads_config_and_runs_evaluation(self, mock_from_yaml, mock_run):
        """Test that config is loaded and evaluation runs."""
        mock_config = MagicMock(spec=EvaluationConfig)
        mock_config.with_auto_devices.return_value = mock_config
        mock_from_yaml.return_value = mock_config
        mock_run.return_value = {"MRR": 0.75}
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"experiment_name: test\n")
            temp_path = f.name
        
        try:
            result = evaluate_from_config(temp_path)
            
            mock_from_yaml.assert_called_once_with(temp_path)
            mock_config.with_auto_devices.assert_called_once()
            mock_run.assert_called_once_with(mock_config)
            self.assertEqual(result, {"MRR": 0.75})
        finally:
            Path(temp_path).unlink()
    
    @patch('evaluator.api.run_evaluation')
    @patch('evaluator.config.EvaluationConfig.from_yaml')
    def test_auto_devices_can_be_disabled(self, mock_from_yaml, mock_run):
        """Test that auto_devices=False skips device auto-configuration."""
        mock_config = MagicMock(spec=EvaluationConfig)
        mock_from_yaml.return_value = mock_config
        mock_run.return_value = {}
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"experiment_name: test\n")
            temp_path = f.name
        
        try:
            evaluate_from_config(temp_path, auto_devices=False)
            mock_config.with_auto_devices.assert_not_called()
        finally:
            Path(temp_path).unlink()


class TestEvaluateFromPreset(unittest.TestCase):
    """Test evaluate_from_preset function."""
    
    def test_raises_error_for_unknown_preset(self):
        """Test that ConfigurationError is raised for unknown presets."""
        with self.assertRaises(ConfigurationError) as cm:
            evaluate_from_preset("nonexistent_preset")
        self.assertIn("Unknown preset", str(cm.exception))
        self.assertIn("whisper_labse", str(cm.exception))  # Should list available
    
    @patch('evaluator.api.run_evaluation')
    def test_loads_preset_and_runs_evaluation(self, mock_run):
        """Test that preset is loaded and evaluation runs."""
        mock_run.return_value = {"WER": 0.12, "MRR": 0.85}
        
        # This will fail at _run_evaluation but we mock it
        result = evaluate_from_preset("fast_dev")
        
        mock_run.assert_called_once()
        config = mock_run.call_args[0][0]
        self.assertIsInstance(config, EvaluationConfig)
        self.assertEqual(result, {"WER": 0.12, "MRR": 0.85})
    
    @patch('evaluator.api.run_evaluation')
    def test_data_path_override(self, mock_run):
        """Test that data_path overrides preset's questions_path."""
        mock_run.return_value = {}
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{}')
            temp_path = f.name
        
        try:
            evaluate_from_preset("fast_dev", data_path=temp_path)
            
            config = mock_run.call_args[0][0]
            self.assertEqual(config.data.questions_path, temp_path)
        finally:
            Path(temp_path).unlink()
    
    @patch('evaluator.api.run_evaluation')
    def test_corpus_path_override(self, mock_run):
        """Test that corpus_path overrides preset's corpus_path."""
        mock_run.return_value = {}
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{}')
            temp_path = f.name
        
        try:
            evaluate_from_preset("fast_dev", corpus_path=temp_path)
            
            config = mock_run.call_args[0][0]
            self.assertEqual(config.data.corpus_path, temp_path)
        finally:
            Path(temp_path).unlink()
    
    @patch('evaluator.api.run_evaluation')
    def test_additional_overrides(self, mock_run):
        """Test that **overrides are applied to config."""
        mock_run.return_value = {}
        
        evaluate_from_preset(
            "fast_dev",
            data_batch_size=64,
            model_asr_device="cpu"
        )
        
        config = mock_run.call_args[0][0]
        self.assertEqual(config.data.batch_size, 64)
        self.assertEqual(config.model.asr_device, "cpu")


class TestQuickEvaluate(unittest.TestCase):
    """Test quick_evaluate function."""
    
    def test_raises_error_for_missing_audio_dir(self):
        """Test that ConfigurationError is raised for missing audio directory."""
        with self.assertRaises(ConfigurationError) as cm:
            quick_evaluate("/nonexistent/audio/dir")
        self.assertIn("not found", str(cm.exception))
    
    def test_raises_error_for_unknown_asr_model(self):
        """Test that ConfigurationError is raised for unknown ASR model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ConfigurationError) as cm:
                quick_evaluate(tmpdir, model="unknown_model")
            self.assertIn("Unknown ASR model", str(cm.exception))
    
    def test_raises_error_for_unknown_embedding_model(self):
        """Test that ConfigurationError is raised for unknown embedding model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ConfigurationError) as cm:
                quick_evaluate(tmpdir, embedding="unknown_embedding")
            self.assertIn("Unknown embedding model", str(cm.exception))
    
    @patch('evaluator.api.run_evaluation')
    def test_creates_config_with_defaults(self, mock_run):
        """Test that default config is created correctly."""
        mock_run.return_value = {"MRR": 0.5}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = quick_evaluate(tmpdir)
            
            config = mock_run.call_args[0][0]
            self.assertEqual(config.model.asr_model_type, "whisper")
            self.assertEqual(config.model.text_emb_model_type, "labse")
            self.assertEqual(config.vector_db.k, 5)
            self.assertEqual(config.data.batch_size, 32)
    
    @patch('evaluator.api.run_evaluation')
    def test_accepts_model_aliases(self, mock_run):
        """Test that model aliases work correctly."""
        mock_run.return_value = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test case insensitivity
            quick_evaluate(tmpdir, model="WHISPER", embedding="LABSE")
            config = mock_run.call_args[0][0]
            self.assertEqual(config.model.asr_model_type, "whisper")
            
            # Test wav2vec alias
            quick_evaluate(tmpdir, model="wav2vec")
            config = mock_run.call_args[0][0]
            self.assertEqual(config.model.asr_model_type, "wav2vec2")
            
            # Test jina alias
            quick_evaluate(tmpdir, embedding="jina")
            config = mock_run.call_args[0][0]
            self.assertEqual(config.model.text_emb_model_type, "jina_v4")
    
    @patch('evaluator.api.run_evaluation')
    def test_custom_parameters(self, mock_run):
        """Test that custom parameters are applied."""
        mock_run.return_value = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temp corpus file for validation
            corpus_file = Path(tmpdir) / "corpus.json"
            corpus_file.write_text("{}")
            
            quick_evaluate(
                tmpdir,
                k=10,
                batch_size=16,
                trace_limit=100,
                corpus_path=str(corpus_file)
            )
            
            config = mock_run.call_args[0][0]
            self.assertEqual(config.vector_db.k, 10)
            self.assertEqual(config.data.batch_size, 16)
            self.assertEqual(config.data.trace_limit, 100)
            self.assertEqual(config.data.corpus_path, str(corpus_file))


class TestRunEvaluation(unittest.TestCase):
    """Test run_evaluation error handling."""
    
    @patch('evaluator.api._service_run_evaluation')
    def test_wraps_runtime_errors(self, mock_run):
        """Test that runtime errors are wrapped in EvaluationError."""
        mock_run.side_effect = RuntimeError("Logging failed")
        
        config = EvaluationConfig()
        with self.assertRaises(EvaluationError) as cm:
            run_evaluation(config)
        self.assertIn("logging", str(cm.exception).lower())

    @patch("evaluator.services.evaluation_service._evaluate_metrics")
    @patch("evaluator.services.evaluation_service.load_dataset")
    @patch("evaluator.services.evaluation_service.create_pipeline_from_config")
    @patch("evaluator.services.evaluation_service.CacheManager")
    @patch("evaluator.services.evaluation_service.setup_logging")
    def test_includes_cache_metadata_in_results(
        self,
        _mock_setup_logging,
        mock_cache_manager_cls,
        mock_create_pipeline,
        mock_load_dataset,
        mock_eval_metrics,
    ):
        from evaluator.services.evaluation_service import run_evaluation as run_eval_service

        cfg = EvaluationConfig(experiment_name="cache_meta")
        mock_cache = MagicMock()
        mock_cache.enabled = True
        mock_cache.get_cache_stats.side_effect = [
            {"sizes_bytes": {"total": 10}, "file_counts": {"total": 1}},
            {"sizes_bytes": {"total": 20}, "file_counts": {"total": 3}},
        ]
        mock_cache_manager_cls.return_value = mock_cache

        mock_bundle = MagicMock()
        mock_bundle.retrieval_pipeline = MagicMock()
        mock_bundle.text_embedding_pipeline = MagicMock()
        mock_bundle.asr_pipeline = None
        mock_bundle.audio_embedding_pipeline = None
        mock_create_pipeline.return_value = mock_bundle

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 5

        def _load_side_effect(*args, **kwargs):
            kwargs["load_info"]["vector_cache_hit"] = True
            return mock_dataset

        mock_load_dataset.side_effect = _load_side_effect
        mock_eval_metrics.return_value = {"MRR": 0.5}

        results = run_eval_service(cfg)

        self.assertIn("cache", results.metadata)
        self.assertTrue(results.metadata["cache"]["enabled"])
        self.assertEqual(results.metadata["cache"]["load"]["vector_cache_hit"], True)
        self.assertEqual(results.metadata["cache"]["delta"]["size_bytes_delta"]["total"], 10)


class TestRunEvaluationMatrix(unittest.TestCase):
    """Test run_evaluation_matrix behavior."""

    @patch("evaluator.api._service_run_evaluation_matrix")
    def test_runs_matrix_and_returns_bundle(self, mock_run_matrix):
        config = EvaluationConfig()
        test_setups = [{"setup_id": "s1", "overrides": {"vector_db_k": 10}}]
        mock_run_matrix.return_value = {"num_setups": 1, "runs": []}

        result = run_evaluation_matrix(config, test_setups)

        mock_run_matrix.assert_called_once_with(config, test_setups)
        self.assertEqual(result["num_setups"], 1)

    @patch("evaluator.api._service_run_evaluation_matrix")
    def test_wraps_runtime_errors(self, mock_run_matrix):
        mock_run_matrix.side_effect = RuntimeError("matrix failed")
        with self.assertRaises(EvaluationError):
            run_evaluation_matrix(EvaluationConfig(), [])


class TestRunEvaluationMatrixService(unittest.TestCase):
    """Test comparison bundle generated by service layer."""

    @patch("evaluator.services.evaluation_service.run_evaluation")
    def test_comparison_bundle_contains_baseline_and_leaderboard(self, mock_run_eval):
        from evaluator.services.evaluation_service import run_evaluation_matrix as run_matrix_service

        result_a = MagicMock()
        result_a.to_dict.return_value = {"MRR": 0.5, "Recall@5": 0.7, "_metadata": {}}
        result_b = MagicMock()
        result_b.to_dict.return_value = {"MRR": 0.6, "Recall@5": 0.8, "_metadata": {}}
        mock_run_eval.side_effect = [result_a, result_b]

        cfg = EvaluationConfig(experiment_name="base")
        bundle = run_matrix_service(
            cfg,
            [
                {"setup_id": "a", "overrides": {}},
                {"setup_id": "b", "overrides": {"vector_db_k": 10}},
            ],
        )

        self.assertEqual(bundle["comparison"]["baseline_setup_id"], "a")
        self.assertEqual(bundle["comparison"]["ranking_metric"], "MRR")
        self.assertEqual(bundle["comparison"]["leaderboard"][0]["setup_id"], "b")
        deltas = {d["setup_id"]: d["deltas_vs_baseline"] for d in bundle["comparison"]["metric_deltas"]}
        self.assertAlmostEqual(deltas["b"]["MRR"], 0.1, places=6)


class TestExceptionTypes(unittest.TestCase):
    """Test custom exception types."""
    
    def test_evaluation_error_inherits_from_exception(self):
        """Test that EvaluationError is a proper exception."""
        self.assertTrue(issubclass(EvaluationError, Exception))
        
        with self.assertRaises(EvaluationError):
            raise EvaluationError("Test error")
    
    def test_configuration_error_inherits_from_exception(self):
        """Test that ConfigurationError is a proper exception."""
        self.assertTrue(issubclass(ConfigurationError, Exception))
        
        with self.assertRaises(ConfigurationError):
            raise ConfigurationError("Test error")


class TestAPIExports(unittest.TestCase):
    """Test that API functions are properly exported from the package."""
    
    def test_tier1_exports_available(self):
        """Test that Tier 1 API functions are exported from evaluator package."""
        import evaluator
        
        # All Tier 1 functions should be accessible
        self.assertTrue(hasattr(evaluator, 'evaluate_from_config'))
        self.assertTrue(hasattr(evaluator, 'evaluate_from_preset'))
        self.assertTrue(hasattr(evaluator, 'quick_evaluate'))
        self.assertTrue(hasattr(evaluator, 'EvaluationError'))
        self.assertTrue(hasattr(evaluator, 'ConfigurationError'))
        self.assertTrue(hasattr(evaluator, 'run_evaluation_matrix'))
    
    def test_tier1_in_all(self):
        """Test that Tier 1 API functions are in __all__."""
        import evaluator
        
        self.assertIn('evaluate_from_config', evaluator.__all__)
        self.assertIn('evaluate_from_preset', evaluator.__all__)
        self.assertIn('quick_evaluate', evaluator.__all__)
        self.assertIn('EvaluationError', evaluator.__all__)
        self.assertIn('ConfigurationError', evaluator.__all__)
        self.assertIn('run_evaluation_matrix', evaluator.__all__)


if __name__ == "__main__":
    unittest.main()
