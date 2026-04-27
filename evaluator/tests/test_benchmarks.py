"""Tests for benchmarking utilities."""

import tempfile
import json
import csv
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest
import numpy as np
import torch

from evaluator.benchmarks import (
    Timer,
    PerformanceStats,
    aggregate_timings,
    ModelBenchmark,
    BenchmarkResult,
    generate_benchmark_report,
    export_to_json,
    export_to_csv,
)
from evaluator.benchmarks.timer import timed


class TestPerformanceStats:
    """Tests for PerformanceStats dataclass."""

    def test_creation(self):
        """PerformanceStats can be created with all fields."""
        stats = PerformanceStats(
            mean=1.5,
            std=0.2,
            min=1.0,
            max=2.0,
            samples=10
        )
        assert stats.mean == 1.5
        assert stats.std == 0.2
        assert stats.min == 1.0
        assert stats.max == 2.0
        assert stats.samples == 10

    def test_str_representation(self):
        """String representation is formatted correctly."""
        stats = PerformanceStats(mean=1.5, std=0.2, min=1.0, max=2.0, samples=10)
        s = str(stats)
        assert "mean=1.5000s" in s
        assert "std=0.2000s" in s
        assert "n=10" in s


class TestAggregateTimings:
    """Tests for aggregate_timings function."""

    def test_single_value(self):
        """Single value returns correct stats."""
        stats = aggregate_timings([1.0])
        assert stats.mean == 1.0
        assert stats.std == 0.0
        assert stats.min == 1.0
        assert stats.max == 1.0
        assert stats.samples == 1

    def test_multiple_values(self):
        """Multiple values compute correct mean and std."""
        timings = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = aggregate_timings(timings)
        
        assert stats.mean == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.samples == 5
        # Sample std for [1,2,3,4,5] is sqrt(2.5) ≈ 1.58
        assert abs(stats.std - 1.5811) < 0.001

    def test_empty_list_raises_error(self):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot aggregate empty"):
            aggregate_timings([])


class TestTimer:
    """Tests for Timer context manager."""

    def test_context_manager_basic(self):
        """Timer records elapsed time as context manager."""
        timer = Timer("test")
        with timer:
            pass  # Instant operation
        
        assert timer.elapsed >= 0
        assert len(timer.timings) == 1

    def test_multiple_measurements(self):
        """Timer accumulates multiple measurements."""
        timer = Timer("test")
        
        for _ in range(5):
            with timer:
                pass
        
        assert len(timer.timings) == 5
        stats = timer.stats
        assert stats.samples == 5

    def test_start_stop_manual(self):
        """Timer can be used with manual start/stop."""
        timer = Timer("test")
        timer.start()
        elapsed = timer.stop()
        
        assert elapsed >= 0
        assert timer.elapsed == elapsed

    def test_stop_without_start_raises(self):
        """Stopping timer without starting raises error."""
        timer = Timer("test")
        with pytest.raises(RuntimeError, match="Timer was not started"):
            timer.stop()

    def test_reset_clears_timings(self):
        """Reset clears all recorded timings."""
        timer = Timer("test")
        with timer:
            pass
        
        timer.reset()
        
        assert len(timer.timings) == 0
        assert timer.elapsed == 0.0

    def test_total_time(self):
        """Total time sums all measurements."""
        timer = Timer("test")
        timer.timings = [1.0, 2.0, 3.0]
        
        assert timer.total_time == 6.0

    def test_stats_with_no_timings_raises(self):
        """Getting stats with no timings raises error."""
        timer = Timer("test")
        with pytest.raises(ValueError):
            _ = timer.stats


class TestTimedContextManager:
    """Tests for timed() helper."""

    def test_timed_yields_timer(self):
        """timed() yields a Timer object."""
        with timed("test") as t:
            assert isinstance(t, Timer)
            assert t.name == "test"
        
        assert t.elapsed >= 0


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_creation_minimal(self):
        """BenchmarkResult can be created with minimal fields."""
        result = BenchmarkResult(name="test")
        assert result.name == "test"
        assert result.throughput == 0.0

    def test_to_dict(self):
        """to_dict returns serializable dictionary."""
        stats = PerformanceStats(mean=0.1, std=0.01, min=0.05, max=0.15, samples=10)
        result = BenchmarkResult(
            name="test",
            timing_stats=stats,
            throughput=100.0,
            num_samples=100,
        )
        
        d = result.to_dict()
        
        assert d["name"] == "test"
        assert d["throughput"] == 100.0
        assert "timing" in d
        assert d["timing"]["mean_sec"] == 0.1

    def test_str_representation(self):
        """String representation includes key info."""
        stats = PerformanceStats(mean=0.1, std=0.01, min=0.05, max=0.15, samples=10)
        result = BenchmarkResult(
            name="test benchmark",
            timing_stats=stats,
            throughput=100.0,
            num_samples=50,
        )
        
        s = str(result)
        
        assert "test benchmark" in s
        assert "Samples: 50" in s
        assert "Throughput" in s


class TestModelBenchmark:
    """Tests for ModelBenchmark class."""

    def test_benchmark_embedding(self):
        """Benchmark embedding model produces valid result."""
        # Create mock model
        mock_model = Mock()
        mock_model.name.return_value = "mock-embedding"
        mock_model.encode.return_value = np.random.randn(10, 768)
        
        benchmark = ModelBenchmark(verbose=False)
        texts = ["text " + str(i) for i in range(10)]
        
        result = benchmark.benchmark_embedding(
            model=mock_model,
            texts=texts,
            warmup=1,
            batch_size=5,
        )
        
        assert result.name == "Embedding (mock-embedding)"
        assert result.num_samples == 10
        assert result.timing_stats is not None
        assert result.throughput > 0

    def test_benchmark_asr(self):
        """Benchmark ASR model produces valid result."""
        # Create mock model
        mock_model = Mock()
        mock_model.name.return_value = "mock-asr"
        mock_model.transcribe.return_value = ["transcription"]
        
        benchmark = ModelBenchmark(verbose=False)
        samples = [
            {"audio": torch.randn(16000), "sampling_rate": 16000}
            for _ in range(5)
        ]
        
        result = benchmark.benchmark_asr(
            model=mock_model,
            samples=samples,
            warmup=1,
        )
        
        assert result.name == "ASR (mock-asr)"
        assert result.num_samples == 5
        assert "real_time_factor" in result.extra_metrics

    def test_benchmark_retrieval(self):
        """Benchmark retrieval store produces valid result."""
        # Create mock store
        mock_store = Mock()
        mock_store.search.return_value = [("doc1", 0.9), ("doc2", 0.8)]
        
        benchmark = ModelBenchmark(verbose=False)
        queries = np.random.randn(10, 768)
        
        result = benchmark.benchmark_retrieval(
            store=mock_store,
            queries=queries,
            warmup=1,
            k=5,
        )
        
        assert result.name == "Retrieval"
        assert result.num_samples == 10
        assert "k" in result.extra_metrics
        assert result.extra_metrics["k"] == 5

    def test_benchmark_retrieval_single_query(self):
        """Benchmark retrieval handles single query vector."""
        mock_store = Mock()
        mock_store.search.return_value = [("doc1", 0.9)]
        
        benchmark = ModelBenchmark()
        query = np.random.randn(768)  # 1D array
        
        result = benchmark.benchmark_retrieval(
            store=mock_store,
            queries=query,
            warmup=1,
            k=1,
        )
        
        assert result.num_samples == 1

    def test_empty_samples_raises_error(self):
        """Empty samples list raises ValueError."""
        benchmark = ModelBenchmark()
        mock_model = Mock()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            benchmark.benchmark_embedding(mock_model, texts=[])
        
        with pytest.raises(ValueError, match="cannot be empty"):
            benchmark.benchmark_asr(mock_model, samples=[])
        
        with pytest.raises(ValueError, match="cannot be empty"):
            benchmark.benchmark_retrieval(mock_model, queries=[])

    def test_benchmark_audio_embedding(self):
        """Benchmark audio embedding model produces valid result."""
        mock_model = Mock()
        mock_model.name.return_value = "mock-audio-emb"
        mock_model.encode_audio.return_value = np.random.randn(1, 768)
        
        benchmark = ModelBenchmark(verbose=False)
        samples = [
            {"audio": torch.randn(16000), "sampling_rate": 16000}
            for _ in range(3)
        ]
        
        result = benchmark.benchmark_audio_embedding(
            model=mock_model,
            samples=samples,
            warmup=1,
        )
        
        assert result.name == "Audio Embedding (mock-audio-emb)"
        assert result.num_samples == 3
        assert "real_time_factor" in result.extra_metrics


class TestReportGeneration:
    """Tests for report generation and export."""

    def test_generate_benchmark_report(self):
        """Generate report from results."""
        stats = PerformanceStats(mean=0.1, std=0.01, min=0.05, max=0.15, samples=10)
        results = [
            BenchmarkResult(
                name="Model A",
                timing_stats=stats,
                throughput=100.0,
                num_samples=100,
            ),
            BenchmarkResult(
                name="Model B",
                timing_stats=stats,
                throughput=200.0,
                num_samples=100,
            ),
        ]
        
        report = generate_benchmark_report(results, title="Test Report")
        
        assert "Test Report" in report
        assert "Model A" in report
        assert "Model B" in report
        assert "SUMMARY" in report

    def test_generate_report_empty_results(self):
        """Generate report handles empty results."""
        report = generate_benchmark_report([])
        assert "No benchmark results" in report

    def test_export_to_json(self):
        """Export results to JSON file."""
        stats = PerformanceStats(mean=0.1, std=0.01, min=0.05, max=0.15, samples=10)
        results = [
            BenchmarkResult(name="test", timing_stats=stats, throughput=100.0)
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            export_to_json(results, path, metadata={"device": "cpu"})
            
            assert path.exists()
            
            with open(path) as f:
                data = json.load(f)
            
            assert "results" in data
            assert len(data["results"]) == 1
            assert data["results"][0]["name"] == "test"
            assert data["metadata"]["device"] == "cpu"

    def test_export_to_csv(self):
        """Export results to CSV file."""
        stats = PerformanceStats(mean=0.1, std=0.01, min=0.05, max=0.15, samples=10)
        results = [
            BenchmarkResult(name="test1", timing_stats=stats, throughput=100.0),
            BenchmarkResult(name="test2", timing_stats=stats, throughput=200.0),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            export_to_csv(results, path)
            
            assert path.exists()
            
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 2
            assert rows[0]["name"] == "test1"
            assert rows[1]["name"] == "test2"

    def test_export_creates_parent_dirs(self):
        """Export creates parent directories if needed."""
        results = [BenchmarkResult(name="test")]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "results.json"
            export_to_json(results, path)
            
            assert path.exists()


class TestBenchmarkCLI:
    """Tests for benchmark CLI module."""

    def test_parse_args_defaults(self):
        """Parse args with defaults."""
        from evaluator.cli.benchmark import parse_benchmark_args
        
        args = parse_benchmark_args([])
        
        assert args.model == "all"
        assert args.samples == 100
        assert args.warmup == 3
        assert args.batch_size == 32

    def test_parse_args_custom(self):
        """Parse args with custom values."""
        from evaluator.cli.benchmark import parse_benchmark_args
        
        args = parse_benchmark_args([
            "--model", "whisper",
            "--samples", "50",
            "--warmup", "5",
            "--output", "results.json",
        ])
        
        assert args.model == "whisper"
        assert args.samples == 50
        assert args.warmup == 5
        assert args.output == "results.json"

    def test_create_synthetic_audio(self):
        """Create synthetic audio for benchmarking."""
        from evaluator.cli.benchmark import create_synthetic_audio
        
        audio = create_synthetic_audio(duration_sec=2.0, sample_rate=16000)
        
        assert isinstance(audio, torch.Tensor)
        assert len(audio) == 32000  # 2 seconds at 16kHz

    def test_create_synthetic_texts(self):
        """Create synthetic texts for benchmarking."""
        from evaluator.cli.benchmark import create_synthetic_texts
        
        texts = create_synthetic_texts(num_texts=50)
        
        assert len(texts) == 50
        assert all(isinstance(t, str) for t in texts)
        assert "[Sample 1]" in texts[0]
