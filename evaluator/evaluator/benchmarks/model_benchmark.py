"""Benchmarking utilities for models and retrieval pipelines."""

import gc
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union
from datetime import datetime

import numpy as np
import torch

from .timer import Timer, PerformanceStats, aggregate_timings


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB.
    
    Returns:
        Dictionary with 'cpu_mb' and optionally 'gpu_mb' keys.
    """
    memory = {"cpu_mb": 0.0}
    
    # CPU memory via torch
    try:
        import psutil
        process = psutil.Process()
        memory["cpu_mb"] = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    
    # GPU memory if available
    if torch.cuda.is_available():
        memory["gpu_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        memory["gpu_max_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return memory


def clear_memory() -> None:
    """Clear Python garbage and GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    
    name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    timing_stats: Optional[PerformanceStats] = None
    throughput: float = 0.0  # items per second
    throughput_unit: str = "items/sec"
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_peak_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    num_samples: int = 0
    num_warmup: int = 0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "timestamp": self.timestamp,
            "throughput": self.throughput,
            "throughput_unit": self.throughput_unit,
            "memory_before_mb": self.memory_before_mb,
            "memory_after_mb": self.memory_after_mb,
            "memory_peak_mb": self.memory_peak_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_memory_peak_mb": self.gpu_memory_peak_mb,
            "num_samples": self.num_samples,
            "num_warmup": self.num_warmup,
        }
        
        if self.timing_stats:
            result["timing"] = {
                "mean_sec": self.timing_stats.mean,
                "std_sec": self.timing_stats.std,
                "min_sec": self.timing_stats.min,
                "max_sec": self.timing_stats.max,
                "samples": self.timing_stats.samples,
            }
        
        if self.extra_metrics:
            result["extra_metrics"] = self.extra_metrics
            
        return result
    
    def __str__(self) -> str:
        lines = [
            f"Benchmark: {self.name}",
            f"  Samples: {self.num_samples} (warmup: {self.num_warmup})",
        ]
        
        if self.timing_stats:
            lines.append(f"  Timing: {self.timing_stats}")
        
        lines.append(f"  Throughput: {self.throughput:.2f} {self.throughput_unit}")
        lines.append(f"  Memory: {self.memory_before_mb:.1f} -> {self.memory_after_mb:.1f} MB")
        
        if self.gpu_memory_mb > 0:
            lines.append(f"  GPU Memory: {self.gpu_memory_mb:.1f} MB (peak: {self.gpu_memory_peak_mb:.1f} MB)")
        
        return "\n".join(lines)


class ModelBenchmark:
    """Benchmarking utility for ASR, embedding, and retrieval models.
    
    Example:
        benchmark = ModelBenchmark()
        
        # Benchmark ASR model
        result = benchmark.benchmark_asr(whisper_model, audio_samples)
        print(result)
        
        # Benchmark embedding model
        result = benchmark.benchmark_embedding(labse_model, texts)
        print(result)
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize benchmark utility.
        
        Args:
            verbose: If True, print progress during benchmarking.
        """
        self.verbose = verbose
    
    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[Benchmark] {message}")
    
    def benchmark_asr(
        self,
        model: Any,
        samples: List[Dict[str, Any]],
        warmup: int = 3,
        language: Optional[str] = None,
    ) -> BenchmarkResult:
        """Benchmark ASR model transcription performance.
        
        Args:
            model: ASR model with transcribe() method.
            samples: List of audio samples. Each sample should have 'audio' (tensor)
                     and 'sampling_rate' (int) keys.
            warmup: Number of warmup iterations before timing.
            language: Optional language code for transcription.
            
        Returns:
            BenchmarkResult with timing and throughput statistics.
        """
        if not samples:
            raise ValueError("samples list cannot be empty")
        
        self._log(f"Starting ASR benchmark with {len(samples)} samples...")
        
        # Prepare audio data
        audio_tensors = [s["audio"] for s in samples]
        sampling_rates = [s.get("sampling_rate", 16000) for s in samples]
        
        clear_memory()
        mem_before = get_memory_usage()
        
        # Warmup
        self._log(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            model.transcribe(audio_tensors[:1], sampling_rates[:1], language=language)
        
        clear_memory()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        timer = Timer("asr_transcribe")
        self._log("Running benchmark iterations...")
        
        for i, (audio, sr) in enumerate(zip(audio_tensors, sampling_rates)):
            with timer:
                model.transcribe([audio], [sr], language=language)
            
            if self.verbose and (i + 1) % 10 == 0:
                self._log(f"  Processed {i + 1}/{len(samples)} samples")
        
        mem_after = get_memory_usage()
        
        stats = timer.stats
        total_audio_duration = sum(
            len(audio) / sr for audio, sr in zip(audio_tensors, sampling_rates)
        )
        
        return BenchmarkResult(
            name=f"ASR ({model.name()})",
            timing_stats=stats,
            throughput=len(samples) / timer.total_time if timer.total_time > 0 else 0,
            throughput_unit="samples/sec",
            memory_before_mb=mem_before.get("cpu_mb", 0),
            memory_after_mb=mem_after.get("cpu_mb", 0),
            memory_peak_mb=mem_after.get("cpu_mb", 0),
            gpu_memory_mb=mem_after.get("gpu_mb", 0),
            gpu_memory_peak_mb=mem_after.get("gpu_max_mb", 0),
            num_samples=len(samples),
            num_warmup=warmup,
            extra_metrics={
                "total_audio_duration_sec": total_audio_duration,
                "real_time_factor": timer.total_time / total_audio_duration if total_audio_duration > 0 else 0,
            }
        )
    
    def benchmark_embedding(
        self,
        model: Any,
        texts: List[str],
        warmup: int = 3,
        batch_size: int = 32,
    ) -> BenchmarkResult:
        """Benchmark text embedding model performance.
        
        Args:
            model: Text embedding model with encode() method.
            texts: List of texts to encode.
            warmup: Number of warmup iterations before timing.
            batch_size: Batch size for encoding.
            
        Returns:
            BenchmarkResult with timing and throughput statistics.
        """
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        self._log(f"Starting embedding benchmark with {len(texts)} texts...")
        
        clear_memory()
        mem_before = get_memory_usage()
        
        # Warmup
        self._log(f"Running {warmup} warmup iterations...")
        warmup_texts = texts[:min(batch_size, len(texts))]
        for _ in range(warmup):
            model.encode(warmup_texts)
        
        clear_memory()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark - process in batches
        timer = Timer("embedding_encode")
        self._log("Running benchmark iterations...")
        
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch = texts[start_idx:end_idx]
            
            with timer:
                model.encode(batch)
            
            if self.verbose and (i + 1) % 5 == 0:
                self._log(f"  Processed batch {i + 1}/{num_batches}")
        
        mem_after = get_memory_usage()
        stats = timer.stats
        
        return BenchmarkResult(
            name=f"Embedding ({model.name()})",
            timing_stats=stats,
            throughput=len(texts) / timer.total_time if timer.total_time > 0 else 0,
            throughput_unit="texts/sec",
            memory_before_mb=mem_before.get("cpu_mb", 0),
            memory_after_mb=mem_after.get("cpu_mb", 0),
            memory_peak_mb=mem_after.get("cpu_mb", 0),
            gpu_memory_mb=mem_after.get("gpu_mb", 0),
            gpu_memory_peak_mb=mem_after.get("gpu_max_mb", 0),
            num_samples=len(texts),
            num_warmup=warmup,
            extra_metrics={
                "batch_size": batch_size,
                "num_batches": num_batches,
                "avg_batch_time_sec": stats.mean,
            }
        )
    
    def benchmark_retrieval(
        self,
        store: Any,
        queries: Union[List[np.ndarray], np.ndarray],
        warmup: int = 3,
        k: int = 10,
    ) -> BenchmarkResult:
        """Benchmark vector store retrieval performance.
        
        Args:
            store: Vector store with search() method.
            queries: List of query vectors or 2D array of shape (N, D).
            warmup: Number of warmup iterations before timing.
            k: Number of results to retrieve per query.
            
        Returns:
            BenchmarkResult with timing and throughput statistics.
        """
        # Convert to list if numpy array
        if isinstance(queries, np.ndarray):
            if queries.ndim == 1:
                queries = [queries]
            else:
                queries = [queries[i] for i in range(len(queries))]
        
        if not queries:
            raise ValueError("queries list cannot be empty")
        
        self._log(f"Starting retrieval benchmark with {len(queries)} queries...")
        
        clear_memory()
        mem_before = get_memory_usage()
        
        # Warmup
        self._log(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            store.search(queries[0], k=k)
        
        clear_memory()
        
        # Benchmark
        timer = Timer("retrieval_search")
        self._log("Running benchmark iterations...")
        
        for i, query in enumerate(queries):
            with timer:
                store.search(query, k=k)
            
            if self.verbose and (i + 1) % 100 == 0:
                self._log(f"  Processed {i + 1}/{len(queries)} queries")
        
        mem_after = get_memory_usage()
        stats = timer.stats
        
        return BenchmarkResult(
            name="Retrieval",
            timing_stats=stats,
            throughput=len(queries) / timer.total_time if timer.total_time > 0 else 0,
            throughput_unit="queries/sec",
            memory_before_mb=mem_before.get("cpu_mb", 0),
            memory_after_mb=mem_after.get("cpu_mb", 0),
            memory_peak_mb=mem_after.get("cpu_mb", 0),
            gpu_memory_mb=mem_after.get("gpu_mb", 0),
            gpu_memory_peak_mb=mem_after.get("gpu_max_mb", 0),
            num_samples=len(queries),
            num_warmup=warmup,
            extra_metrics={
                "k": k,
                "avg_query_time_ms": stats.mean * 1000,
                "queries_per_second": 1 / stats.mean if stats.mean > 0 else 0,
            }
        )
    
    def benchmark_audio_embedding(
        self,
        model: Any,
        samples: List[Dict[str, Any]],
        warmup: int = 3,
    ) -> BenchmarkResult:
        """Benchmark audio embedding model performance.
        
        Args:
            model: Audio embedding model with encode_audio() method.
            samples: List of audio samples with 'audio' and 'sampling_rate' keys.
            warmup: Number of warmup iterations before timing.
            
        Returns:
            BenchmarkResult with timing and throughput statistics.
        """
        if not samples:
            raise ValueError("samples list cannot be empty")
        
        self._log(f"Starting audio embedding benchmark with {len(samples)} samples...")
        
        audio_tensors = [s["audio"] for s in samples]
        sampling_rates = [s.get("sampling_rate", 16000) for s in samples]
        
        clear_memory()
        mem_before = get_memory_usage()
        
        # Warmup
        self._log(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            model.encode_audio(audio_tensors[:1], sampling_rates[:1])
        
        clear_memory()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        timer = Timer("audio_embedding")
        self._log("Running benchmark iterations...")
        
        for i, (audio, sr) in enumerate(zip(audio_tensors, sampling_rates)):
            with timer:
                model.encode_audio([audio], [sr])
            
            if self.verbose and (i + 1) % 10 == 0:
                self._log(f"  Processed {i + 1}/{len(samples)} samples")
        
        mem_after = get_memory_usage()
        stats = timer.stats
        
        total_audio_duration = sum(
            len(audio) / sr for audio, sr in zip(audio_tensors, sampling_rates)
        )
        
        return BenchmarkResult(
            name=f"Audio Embedding ({model.name()})",
            timing_stats=stats,
            throughput=len(samples) / timer.total_time if timer.total_time > 0 else 0,
            throughput_unit="samples/sec",
            memory_before_mb=mem_before.get("cpu_mb", 0),
            memory_after_mb=mem_after.get("cpu_mb", 0),
            memory_peak_mb=mem_after.get("cpu_mb", 0),
            gpu_memory_mb=mem_after.get("gpu_mb", 0),
            gpu_memory_peak_mb=mem_after.get("gpu_max_mb", 0),
            num_samples=len(samples),
            num_warmup=warmup,
            extra_metrics={
                "total_audio_duration_sec": total_audio_duration,
                "real_time_factor": timer.total_time / total_audio_duration if total_audio_duration > 0 else 0,
            }
        )
