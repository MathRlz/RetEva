"""Tests for experiment tracking module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import os
import sys

from evaluator.tracking import BaseTracker, NoOpTracker, MLflowTracker
from evaluator.config import TrackingConfig, EvaluationConfig


class TestNoOpTracker:
    """Tests for NoOpTracker."""
    
    def test_init(self):
        """Test NoOpTracker initialization."""
        tracker = NoOpTracker()
        assert tracker is not None
        
    def test_init_with_args(self):
        """Test NoOpTracker ignores initialization arguments."""
        tracker = NoOpTracker(
            experiment_name="test",
            tracking_uri="http://localhost:5000",
            extra_param="ignored"
        )
        assert tracker is not None
    
    def test_start_run(self):
        """Test start_run does nothing."""
        tracker = NoOpTracker()
        tracker.start_run()  # Should not raise
        tracker.start_run(run_name="test_run")  # Should not raise
    
    def test_log_params(self):
        """Test log_params does nothing."""
        tracker = NoOpTracker()
        tracker.log_params({"batch_size": 32, "model": "whisper"})  # Should not raise
    
    def test_log_metrics(self):
        """Test log_metrics does nothing."""
        tracker = NoOpTracker()
        tracker.log_metrics({"MRR": 0.75, "MAP": 0.68})  # Should not raise
        tracker.log_metrics({"loss": 0.5}, step=10)  # Should not raise
    
    def test_log_artifact(self):
        """Test log_artifact does nothing."""
        tracker = NoOpTracker()
        tracker.log_artifact("/path/to/artifact.json")  # Should not raise
    
    def test_end_run(self):
        """Test end_run does nothing."""
        tracker = NoOpTracker()
        tracker.end_run()  # Should not raise
    
    def test_context_manager(self):
        """Test NoOpTracker as context manager."""
        with NoOpTracker() as tracker:
            tracker.log_params({"param": "value"})
            tracker.log_metrics({"metric": 1.0})
        # Should not raise
    
    def test_implements_protocol(self):
        """Test NoOpTracker implements BaseTracker protocol."""
        tracker = NoOpTracker()
        assert isinstance(tracker, BaseTracker)


@pytest.fixture
def mock_mlflow():
    """Create a mock mlflow module."""
    mock = MagicMock()
    return mock


class TestMLflowTracker:
    """Tests for MLflowTracker with mocked mlflow."""
    
    def test_init(self):
        """Test MLflowTracker initialization."""
        tracker = MLflowTracker("test_experiment")
        assert tracker.experiment_name == "test_experiment"
        assert tracker.tracking_uri is None
        assert tracker._run_active is False
    
    def test_init_with_uri(self):
        """Test MLflowTracker with custom tracking URI."""
        tracker = MLflowTracker(
            experiment_name="test",
            tracking_uri="http://localhost:5000"
        )
        assert tracker.tracking_uri == "http://localhost:5000"
    
    def test_start_run(self, mock_mlflow):
        """Test starting a run."""
        tracker = MLflowTracker("test_experiment")
        tracker._mlflow = mock_mlflow
        
        tracker.start_run(run_name="test_run")
        
        mock_mlflow.start_run.assert_called_once_with(run_name="test_run")
        assert tracker._run_active is True
    
    def test_start_run_sets_experiment(self, mock_mlflow):
        """Test that start_run calls set_experiment when mlflow not yet initialized."""
        tracker = MLflowTracker("test_experiment")
        
        with patch.object(tracker, '_ensure_mlflow', return_value=mock_mlflow):
            tracker.start_run()
            mock_mlflow.start_run.assert_called_once()
    
    def test_log_params(self, mock_mlflow):
        """Test logging parameters."""
        tracker = MLflowTracker("test")
        tracker._mlflow = mock_mlflow
        tracker._run_active = True
        
        tracker.log_params({"batch_size": 32, "model": "whisper"})
        
        mock_mlflow.log_params.assert_called()
    
    def test_log_params_flattens_nested_dicts(self, mock_mlflow):
        """Test that nested dicts are flattened."""
        tracker = MLflowTracker("test")
        tracker._mlflow = mock_mlflow
        tracker._run_active = True
        
        tracker.log_params({
            "model": {
                "type": "whisper",
                "device": "cuda:0"
            }
        })
        
        # Check that log_params was called with flattened keys
        call_args = mock_mlflow.log_params.call_args
        params = call_args[0][0]
        assert "model.type" in params
        assert "model.device" in params
    
    def test_log_params_without_active_run(self, mock_mlflow):
        """Test log_params warns when no run is active."""
        tracker = MLflowTracker("test")
        tracker._mlflow = mock_mlflow
        tracker._run_active = False
        
        tracker.log_params({"key": "value"})
        
        # Should not call mlflow.log_params
        mock_mlflow.log_params.assert_not_called()
    
    def test_log_metrics(self, mock_mlflow):
        """Test logging metrics."""
        tracker = MLflowTracker("test")
        tracker._mlflow = mock_mlflow
        tracker._run_active = True
        
        tracker.log_metrics({"MRR": 0.75, "MAP": 0.68})
        
        mock_mlflow.log_metrics.assert_called_once_with({"MRR": 0.75, "MAP": 0.68})
    
    def test_log_metrics_with_step(self, mock_mlflow):
        """Test logging metrics with step."""
        tracker = MLflowTracker("test")
        tracker._mlflow = mock_mlflow
        tracker._run_active = True
        
        tracker.log_metrics({"loss": 0.5}, step=10)
        
        mock_mlflow.log_metrics.assert_called_once_with({"loss": 0.5}, step=10)
    
    def test_log_metrics_filters_non_numeric(self, mock_mlflow):
        """Test that non-numeric values are filtered."""
        tracker = MLflowTracker("test")
        tracker._mlflow = mock_mlflow
        tracker._run_active = True
        
        tracker.log_metrics({
            "MRR": 0.75,
            "pipeline_mode": "asr_text_retrieval",  # String, should be filtered
            "phased": True,  # Bool, should be filtered
            "total_samples": 100,  # Int, should be kept
        })
        
        call_args = mock_mlflow.log_metrics.call_args
        metrics = call_args[0][0]
        assert "MRR" in metrics
        assert "total_samples" in metrics
        assert "pipeline_mode" not in metrics
        assert "phased" not in metrics
    
    def test_log_artifact(self, mock_mlflow):
        """Test logging artifact."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"test": "data"}')
            temp_path = f.name
        
        try:
            tracker = MLflowTracker("test")
            tracker._mlflow = mock_mlflow
            tracker._run_active = True
            
            tracker.log_artifact(temp_path)
            
            mock_mlflow.log_artifact.assert_called_once_with(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_log_artifact_nonexistent_file(self, mock_mlflow):
        """Test logging nonexistent artifact."""
        tracker = MLflowTracker("test")
        tracker._mlflow = mock_mlflow
        tracker._run_active = True
        
        tracker.log_artifact("/nonexistent/path/file.json")
        
        # Should not call mlflow.log_artifact
        mock_mlflow.log_artifact.assert_not_called()
    
    def test_end_run(self, mock_mlflow):
        """Test ending a run."""
        tracker = MLflowTracker("test")
        tracker._mlflow = mock_mlflow
        tracker._run_active = True
        
        tracker.end_run()
        
        mock_mlflow.end_run.assert_called_once()
        assert tracker._run_active is False
    
    def test_end_run_when_not_active(self, mock_mlflow):
        """Test end_run when no run is active."""
        tracker = MLflowTracker("test")
        tracker._mlflow = mock_mlflow
        tracker._run_active = False
        
        tracker.end_run()
        
        # Should not call mlflow.end_run
        mock_mlflow.end_run.assert_not_called()
    
    def test_context_manager(self, mock_mlflow):
        """Test MLflowTracker as context manager."""
        tracker = MLflowTracker("test")
        
        with patch.object(tracker, '_ensure_mlflow', return_value=mock_mlflow):
            with tracker:
                mock_mlflow.start_run.assert_called_once()
                tracker.log_params({"param": "value"})
                tracker.log_metrics({"metric": 1.0})
            
            mock_mlflow.end_run.assert_called_once()
    
    def test_start_new_run_ends_previous(self, mock_mlflow):
        """Test starting a new run ends the previous one."""
        tracker = MLflowTracker("test")
        tracker._mlflow = mock_mlflow
        
        tracker.start_run(run_name="run1")
        tracker.start_run(run_name="run2")
        
        # end_run should be called once for run1
        assert mock_mlflow.end_run.call_count == 1
        # start_run should be called twice
        assert mock_mlflow.start_run.call_count == 2
    
    def test_mlflow_import_error(self):
        """Test graceful handling when mlflow is not installed."""
        tracker = MLflowTracker("test")
        tracker._mlflow = None  # Reset so _ensure_mlflow tries to import
        
        # Mock the import to fail
        with patch.dict('sys.modules', {'mlflow': None}):
            # Force reimport by clearing the cached mlflow
            tracker._mlflow = None
            with pytest.raises(ImportError) as exc_info:
                tracker._ensure_mlflow()
            assert "MLflow is required" in str(exc_info.value)


class TestTrackingConfig:
    """Tests for TrackingConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = TrackingConfig()
        assert config.enabled is False
        assert config.backend == "mlflow"
        assert config.mlflow_tracking_uri is None
        assert config.mlflow_experiment_name is None
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrackingConfig(
            enabled=True,
            backend="mlflow",
            mlflow_tracking_uri="http://localhost:5000",
            mlflow_experiment_name="my_experiment"
        )
        assert config.enabled is True
        assert config.mlflow_tracking_uri == "http://localhost:5000"
        assert config.mlflow_experiment_name == "my_experiment"
    
    def test_invalid_backend(self):
        """Test validation of invalid backend."""
        with pytest.raises(ValueError) as exc_info:
            TrackingConfig(backend="invalid_backend")
        assert "Invalid tracking backend" in str(exc_info.value)


class TestEvaluationConfigTracking:
    """Tests for tracking integration in EvaluationConfig."""
    
    def test_default_tracking_config(self):
        """Test default tracking configuration."""
        config = EvaluationConfig()
        assert config.tracking.enabled is False
        assert config.tracking.backend == "mlflow"
    
    def test_from_dict_with_tracking(self):
        """Test creating config from dict with tracking."""
        config_dict = {
            "experiment_name": "test_eval",
            "tracking": {
                "enabled": True,
                "backend": "mlflow",
                "mlflow_tracking_uri": "http://localhost:5000",
                "mlflow_experiment_name": "my_mlflow_exp"
            }
        }
        config = EvaluationConfig.from_dict(config_dict, validate=False)
        assert config.tracking.enabled is True
        assert config.tracking.mlflow_tracking_uri == "http://localhost:5000"
        assert config.tracking.mlflow_experiment_name == "my_mlflow_exp"
    
    def test_experiment_dict_includes_tracking(self):
        """Test that experiment config includes tracking fields."""
        config = EvaluationConfig()
        experiment_dict = config.to_experiment_dict()
        assert "tracking" in experiment_dict
        assert "enabled" in experiment_dict["tracking"]
        assert "backend" in experiment_dict["tracking"]


class TestTrackingIntegration:
    """Integration tests for tracking with evaluation."""
    
    def test_tracker_workflow(self, mock_mlflow):
        """Test complete tracking workflow."""
        tracker = MLflowTracker("integration_test")
        tracker._mlflow = mock_mlflow
        
        tracker.start_run()
        
        # Log configuration
        tracker.log_params({
            "experiment_name": "test",
            "model": {"asr": "whisper", "embedding": "labse"},
            "batch_size": 32,
        })
        
        # Simulate logging metrics at different steps
        tracker.log_metrics({"loss": 0.8}, step=1)
        tracker.log_metrics({"loss": 0.5}, step=2)
        tracker.log_metrics({"loss": 0.3}, step=3)
        
        # Log final metrics
        tracker.log_metrics({
            "MRR": 0.75,
            "MAP": 0.68,
            "Recall@5": 0.82,
            "NDCG@5": 0.71,
        })
        
        tracker.end_run()
        
        # Verify mlflow was called correctly
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.end_run.assert_called_once()
        assert mock_mlflow.log_params.called
        assert mock_mlflow.log_metrics.call_count == 4
