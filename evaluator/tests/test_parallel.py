"""Tests for parallel evaluation module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List


class TestBatchDistributor:
    """Tests for BatchDistributor class."""
    
    def test_distribute_even_split(self):
        """Test even distribution of items."""
        from evaluator.parallel.batch_distributor import BatchDistributor
        
        distributor = BatchDistributor()
        items = list(range(100))
        batches = distributor.distribute(items, num_workers=4)
        
        assert len(batches) == 4
        assert all(len(b) == 25 for b in batches)
        # Verify all items are present
        all_items = [item for batch in batches for item in batch]
        assert sorted(all_items) == items
    
    def test_distribute_uneven_split(self):
        """Test distribution when items don't divide evenly."""
        from evaluator.parallel.batch_distributor import BatchDistributor
        
        distributor = BatchDistributor()
        items = list(range(10))
        batches = distributor.distribute(items, num_workers=3)
        
        assert len(batches) == 3
        # First worker gets extra item
        assert len(batches[0]) == 4
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        # Verify all items present
        all_items = [item for batch in batches for item in batch]
        assert sorted(all_items) == items
    
    def test_distribute_empty_list(self):
        """Test distribution of empty list."""
        from evaluator.parallel.batch_distributor import BatchDistributor
        
        distributor = BatchDistributor()
        batches = distributor.distribute([], num_workers=4)
        
        assert len(batches) == 4
        assert all(len(b) == 0 for b in batches)
    
    def test_distribute_single_worker(self):
        """Test distribution with single worker."""
        from evaluator.parallel.batch_distributor import BatchDistributor
        
        distributor = BatchDistributor()
        items = list(range(50))
        batches = distributor.distribute(items, num_workers=1)
        
        assert len(batches) == 1
        assert batches[0] == items
    
    def test_distribute_more_workers_than_items(self):
        """Test when there are more workers than items."""
        from evaluator.parallel.batch_distributor import BatchDistributor
        
        distributor = BatchDistributor()
        items = [1, 2, 3]
        batches = distributor.distribute(items, num_workers=5)
        
        assert len(batches) == 5
        # First 3 workers get 1 item each, rest get 0
        assert len(batches[0]) == 1
        assert len(batches[1]) == 1
        assert len(batches[2]) == 1
        assert len(batches[3]) == 0
        assert len(batches[4]) == 0
    
    def test_distribute_invalid_workers(self):
        """Test that invalid num_workers raises error."""
        from evaluator.parallel.batch_distributor import BatchDistributor
        
        distributor = BatchDistributor()
        
        with pytest.raises(ValueError, match="num_workers must be positive"):
            distributor.distribute([1, 2, 3], num_workers=0)
        
        with pytest.raises(ValueError, match="num_workers must be positive"):
            distributor.distribute([1, 2, 3], num_workers=-1)
    
    def test_distribute_indices(self):
        """Test distributing indices instead of items."""
        from evaluator.parallel.batch_distributor import BatchDistributor
        
        distributor = BatchDistributor()
        batches = distributor.distribute_indices(total_items=15, num_workers=4)
        
        assert len(batches) == 4
        all_indices = [idx for batch in batches for idx in batch]
        assert sorted(all_indices) == list(range(15))


class TestWorkerResult:
    """Tests for WorkerResult dataclass."""
    
    def test_worker_result_creation(self):
        """Test creating a WorkerResult."""
        from evaluator.parallel.batch_distributor import WorkerResult
        
        result = WorkerResult(
            worker_id=0,
            device="cuda:0",
            results={"MRR": 0.75, "MAP": 0.65},
            num_samples=100,
            error=None,
        )
        
        assert result.worker_id == 0
        assert result.device == "cuda:0"
        assert result.results["MRR"] == 0.75
        assert result.num_samples == 100
        assert result.error is None
    
    def test_worker_result_with_error(self):
        """Test WorkerResult with an error."""
        from evaluator.parallel.batch_distributor import WorkerResult
        
        result = WorkerResult(
            worker_id=1,
            device="cuda:1",
            results=None,
            num_samples=0,
            error="CUDA out of memory",
        )
        
        assert result.error == "CUDA out of memory"
        assert result.results is None


class TestAggregateResults:
    """Tests for aggregate_results function."""
    
    def test_aggregate_single_result(self):
        """Test aggregating a single result."""
        from evaluator.parallel.batch_distributor import WorkerResult, aggregate_results
        
        results = [
            WorkerResult(
                worker_id=0,
                device="cuda:0",
                results={
                    "MRR": 0.75,
                    "MAP": 0.65,
                    "pipeline_mode": "asr_text_retrieval",
                },
                num_samples=100,
                error=None,
            )
        ]
        
        aggregated = aggregate_results(results)
        
        assert aggregated["MRR"] == 0.75
        assert aggregated["MAP"] == 0.65
        assert aggregated["total_samples"] == 100
        assert aggregated["parallel"] is True
    
    def test_aggregate_multiple_results_equal_samples(self):
        """Test aggregating multiple results with equal samples."""
        from evaluator.parallel.batch_distributor import WorkerResult, aggregate_results
        
        results = [
            WorkerResult(
                worker_id=0,
                device="cuda:0",
                results={"MRR": 0.8, "MAP": 0.7, "pipeline_mode": "asr_text"},
                num_samples=50,
                error=None,
            ),
            WorkerResult(
                worker_id=1,
                device="cuda:1",
                results={"MRR": 0.6, "MAP": 0.5, "pipeline_mode": "asr_text"},
                num_samples=50,
                error=None,
            ),
        ]
        
        aggregated = aggregate_results(results)
        
        # Weighted average: (0.8*50 + 0.6*50) / 100 = 0.7
        assert aggregated["MRR"] == pytest.approx(0.7)
        # (0.7*50 + 0.5*50) / 100 = 0.6
        assert aggregated["MAP"] == pytest.approx(0.6)
        assert aggregated["total_samples"] == 100
        assert aggregated["num_workers"] == 2
    
    def test_aggregate_weighted_by_samples(self):
        """Test that aggregation properly weights by sample count."""
        from evaluator.parallel.batch_distributor import WorkerResult, aggregate_results
        
        results = [
            WorkerResult(
                worker_id=0,
                device="cuda:0",
                results={"MRR": 1.0, "pipeline_mode": "test"},
                num_samples=10,
                error=None,
            ),
            WorkerResult(
                worker_id=1,
                device="cuda:1",
                results={"MRR": 0.5, "pipeline_mode": "test"},
                num_samples=90,
                error=None,
            ),
        ]
        
        aggregated = aggregate_results(results)
        
        # Weighted: (1.0*10 + 0.5*90) / 100 = (10 + 45) / 100 = 0.55
        assert aggregated["MRR"] == pytest.approx(0.55)
    
    def test_aggregate_with_partial_failure(self):
        """Test aggregation when some workers fail."""
        from evaluator.parallel.batch_distributor import WorkerResult, aggregate_results
        
        results = [
            WorkerResult(
                worker_id=0,
                device="cuda:0",
                results={"MRR": 0.8, "pipeline_mode": "test"},
                num_samples=50,
                error=None,
            ),
            WorkerResult(
                worker_id=1,
                device="cuda:1",
                results=None,
                num_samples=0,
                error="CUDA error",
            ),
        ]
        
        aggregated = aggregate_results(results)
        
        assert aggregated["MRR"] == 0.8
        assert aggregated["total_samples"] == 50
        assert aggregated["num_workers"] == 1
        assert "failed_workers" in aggregated
        assert len(aggregated["failed_workers"]) == 1
        assert aggregated["failed_workers"][0]["error"] == "CUDA error"
    
    def test_aggregate_all_failed(self):
        """Test that aggregation raises when all workers fail."""
        from evaluator.parallel.batch_distributor import WorkerResult, aggregate_results
        
        results = [
            WorkerResult(
                worker_id=0,
                device="cuda:0",
                results=None,
                num_samples=0,
                error="Error 1",
            ),
            WorkerResult(
                worker_id=1,
                device="cuda:1",
                results=None,
                num_samples=0,
                error="Error 2",
            ),
        ]
        
        with pytest.raises(RuntimeError, match="All workers failed"):
            aggregate_results(results)
    
    def test_aggregate_query_traces(self):
        """Test that query traces are concatenated."""
        from evaluator.parallel.batch_distributor import WorkerResult, aggregate_results
        
        results = [
            WorkerResult(
                worker_id=0,
                device="cuda:0",
                results={
                    "MRR": 0.8,
                    "pipeline_mode": "test",
                    "query_traces": [{"id": 1}, {"id": 2}],
                },
                num_samples=50,
                error=None,
            ),
            WorkerResult(
                worker_id=1,
                device="cuda:1",
                results={
                    "MRR": 0.6,
                    "pipeline_mode": "test",
                    "query_traces": [{"id": 3}],
                },
                num_samples=50,
                error=None,
            ),
        ]
        
        aggregated = aggregate_results(results)
        
        assert len(aggregated["query_traces"]) == 3
        assert aggregated["query_traces"][0]["id"] == 1
        assert aggregated["query_traces"][2]["id"] == 3


class TestWorker:
    """Tests for Worker class."""
    
    def test_worker_init(self):
        """Test Worker initialization."""
        from evaluator.parallel.batch_distributor import Worker
        
        worker = Worker(worker_id=2, device="cuda:2")
        
        assert worker.worker_id == 2
        assert worker.device == "cuda:2"
    
    def test_worker_run_success(self):
        """Test successful worker execution."""
        from evaluator.parallel.batch_distributor import Worker
        
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = MagicMock(side_effect=lambda i: {"data": i})
        
        # Create mock evaluate function
        def mock_eval_fn(dataset, **kwargs):
            return {"MRR": 0.8, "samples": len(dataset)}
        
        worker = Worker(worker_id=0, device="cuda:0")
        result = worker.run(
            indices=[0, 1, 2],
            dataset=mock_dataset,
            evaluate_fn=mock_eval_fn,
        )
        
        assert result.error is None
        assert result.num_samples == 3
        assert result.results is not None
    
    def test_worker_run_handles_exception(self):
        """Test that worker handles exceptions gracefully."""
        from evaluator.parallel.batch_distributor import Worker
        
        mock_dataset = MagicMock()
        
        def failing_eval_fn(dataset, **kwargs):
            raise ValueError("Test error")
        
        worker = Worker(worker_id=0, device="cuda:0")
        result = worker.run(
            indices=[0, 1, 2],
            dataset=mock_dataset,
            evaluate_fn=failing_eval_fn,
        )
        
        assert result.error is not None
        assert "Test error" in result.error
        assert result.results is None


class TestParallelEvaluator:
    """Tests for ParallelEvaluator class."""
    
    @patch('evaluator.parallel.data_parallel.get_available_gpu_count')
    def test_init_auto_detect_gpus(self, mock_gpu_count):
        """Test auto-detection of GPU count."""
        from evaluator.parallel.data_parallel import ParallelEvaluator
        
        mock_gpu_count.return_value = 4
        mock_config = MagicMock()
        
        evaluator = ParallelEvaluator(config=mock_config, num_workers=None)
        
        assert evaluator.num_workers == 4
        assert evaluator.devices == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    
    @patch('evaluator.parallel.data_parallel.get_available_gpu_count')
    def test_init_explicit_workers(self, mock_gpu_count):
        """Test explicit worker count."""
        from evaluator.parallel.data_parallel import ParallelEvaluator
        
        mock_gpu_count.return_value = 4
        mock_config = MagicMock()
        
        evaluator = ParallelEvaluator(config=mock_config, num_workers=2)
        
        assert evaluator.num_workers == 2
        assert evaluator.devices == ["cuda:0", "cuda:1"]
    
    @patch('evaluator.parallel.data_parallel.get_available_gpu_count')
    def test_init_no_gpus_fallback(self, mock_gpu_count):
        """Test fallback to CPU when no GPUs available."""
        from evaluator.parallel.data_parallel import ParallelEvaluator
        
        mock_gpu_count.return_value = 0
        mock_config = MagicMock()
        
        evaluator = ParallelEvaluator(config=mock_config, num_workers=None)
        
        assert evaluator.num_workers == 1
        assert evaluator.devices == ["cpu"]
    
    @patch('evaluator.parallel.data_parallel.get_available_gpu_count')
    def test_init_more_workers_than_gpus(self, mock_gpu_count):
        """Test worker distribution when requesting more workers than GPUs."""
        from evaluator.parallel.data_parallel import ParallelEvaluator
        
        mock_gpu_count.return_value = 2
        mock_config = MagicMock()
        
        evaluator = ParallelEvaluator(config=mock_config, num_workers=4)
        
        assert evaluator.num_workers == 4
        # Workers should round-robin across available GPUs
        assert evaluator.devices == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    
    @patch('evaluator.parallel.data_parallel.get_available_gpu_count')
    def test_evaluate_parallel_empty_dataset(self, mock_gpu_count):
        """Test parallel evaluation with empty dataset."""
        from evaluator.parallel.data_parallel import ParallelEvaluator
        
        mock_gpu_count.return_value = 2
        mock_config = MagicMock()
        
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)
        
        evaluator = ParallelEvaluator(config=mock_config, num_workers=2)
        results = evaluator.evaluate_parallel(dataset=mock_dataset)
        
        assert results["total_samples"] == 0
        assert results["parallel"] is True

    @patch("evaluator.storage.cache.CacheManager")
    @patch("evaluator.pipeline.create_pipeline_from_config")
    @patch("evaluator.evaluation.phased.evaluate_phased")
    @patch("evaluator.parallel.data_parallel.get_available_gpu_count")
    def test_single_worker_uses_current_cache_manager_signature(
        self,
        mock_gpu_count,
        mock_evaluate_phased,
        mock_create_pipeline,
        mock_cache_manager,
    ):
        from evaluator.parallel.data_parallel import ParallelEvaluator

        mock_gpu_count.return_value = 0
        mock_config = MagicMock()
        mock_config.cache.enabled = True
        mock_config.cache.cache_dir = ".cache-test"
        mock_config.model.asr_device = "cpu"
        mock_config.model.text_emb_device = "cpu"
        mock_config.model.audio_emb_device = "cpu"
        mock_bundle = MagicMock()
        mock_create_pipeline.return_value = mock_bundle
        mock_evaluate_phased.return_value = {"MRR": 0.5}

        mock_dataset = list(range(3))

        evaluator = ParallelEvaluator(config=mock_config, num_workers=1)
        result = evaluator._run_single_worker(
            worker_id=0,
            device="cpu",
            indices=[0, 1],
            dataset=mock_dataset,
            k=5,
            batch_size=2,
            trace_limit=0,
        )

        assert result.error is None
        mock_cache_manager.assert_called_once_with(
            cache_dir=".cache-test",
            enabled=True,
        )


class TestEvaluationConfigParallelFields:
    """Tests for parallel fields in EvaluationConfig."""
    
    def test_config_parallel_defaults(self):
        """Test that parallel fields have correct defaults."""
        from evaluator.config import EvaluationConfig
        
        config = EvaluationConfig()
        
        assert config.parallel_enabled is False
        assert config.num_parallel_workers == 0
    
    def test_config_parallel_from_dict(self):
        """Test loading parallel config from dict."""
        from evaluator.config import EvaluationConfig
        
        config_dict = {
            "experiment_name": "test",
            "parallel_enabled": True,
            "num_parallel_workers": 4,
        }
        
        config = EvaluationConfig.from_dict(config_dict, validate=False)
        
        assert config.parallel_enabled is True
        assert config.num_parallel_workers == 4
    
    def test_config_parallel_to_runtime_dict(self):
        """Test that parallel fields are included in runtime config."""
        from evaluator.config import EvaluationConfig
        
        config = EvaluationConfig(
            parallel_enabled=True,
            num_parallel_workers=2,
        )
        
        config_dict = config.to_runtime_dict()
        
        assert config_dict["parallel_enabled"] is True
        assert config_dict["num_parallel_workers"] == 2


class TestIntegration:
    """Integration tests for parallel evaluation (without actual GPUs)."""
    
    def test_distributor_with_worker(self):
        """Test BatchDistributor integrates with Worker."""
        from evaluator.parallel.batch_distributor import BatchDistributor, Worker
        
        distributor = BatchDistributor()
        indices_batches = distributor.distribute_indices(total_items=100, num_workers=4)
        
        # Verify each batch can be used to create a worker
        for i, indices in enumerate(indices_batches):
            worker = Worker(worker_id=i, device=f"cuda:{i}")
            assert worker.worker_id == i
            assert len(indices) in [24, 25, 26]  # Approximately even
    
    def test_full_distribution_and_aggregation_flow(self):
        """Test the full flow from distribution to aggregation."""
        from evaluator.parallel.batch_distributor import (
            BatchDistributor, 
            Worker, 
            WorkerResult,
            aggregate_results,
        )
        
        # Distribute items
        distributor = BatchDistributor()
        batches = distributor.distribute(list(range(100)), num_workers=4)
        
        # Simulate worker results
        results = []
        for i, batch in enumerate(batches):
            results.append(WorkerResult(
                worker_id=i,
                device=f"cuda:{i}",
                results={
                    "MRR": 0.7 + (i * 0.05),  # Vary slightly per worker
                    "MAP": 0.6 + (i * 0.05),
                    "pipeline_mode": "asr_text_retrieval",
                },
                num_samples=len(batch),
                error=None,
            ))
        
        # Aggregate
        aggregated = aggregate_results(results)
        
        assert aggregated["total_samples"] == 100
        assert aggregated["num_workers"] == 4
        assert "MRR" in aggregated
        assert "MAP" in aggregated
        # Weighted average should be between min and max worker values
        assert 0.7 <= aggregated["MRR"] <= 0.85
