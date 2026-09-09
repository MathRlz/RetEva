"""Benchmarking utilities for models and retrieval pipelines."""

import gc
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Any, Dict
from datetime import datetime

import torch

from .timer import Timer, PerformanceStats


def get_memory_usage() -> Dict[str, float]:
    """Current memory usage in MB: 'cpu_mb', plus 'gpu_mb'/'gpu_max_mb' when CUDA
    is available."""
    memory = {"cpu_mb": 0.0}

    try:
        import psutil
        process = psutil.Process()
        memory["cpu_mb"] = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    if torch.cuda.is_available():
        memory["gpu_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        memory["gpu_max_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return memory


def clear_memory() -> None:
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
            lines.append(f"  GPU Memory: {self.gpu_memory_mb:.1f} MB "
                         f"(peak: {self.gpu_memory_peak_mb:.1f} MB)")

        return "\n".join(lines)


class ModelBenchmark:
    """Benchmarking utility for ASR, embedding, and retrieval models."""

    def __init__(self, verbose: bool = False):
        """If ``verbose``, print progress during benchmarking."""
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[Benchmark] {message}")

    def _warm_up(self, warmup: int, iteration: Callable[[], Any], timer_name: str):
        """Snapshot memory, run ``warmup`` untimed iterations, reset the CUDA peak counters and
        hand back ``(memory_before, timer)`` ready for the measured loop."""
        clear_memory()
        mem_before = get_memory_usage()

        self._log(f"Running {warmup} warmup iterations...")
        for _ in range(warmup):
            iteration()

        clear_memory()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        timer = Timer(timer_name)
        self._log("Running benchmark iterations...")
        return mem_before, timer

    def benchmark_asr(
        self,
        model: Any,
        samples: List[Dict[str, Any]],
        warmup: int = 3,
        language: Optional[str] = None,
    ) -> BenchmarkResult:
        """Benchmark ASR model transcribe() performance. Each sample needs 'audio'
        (tensor) and 'sampling_rate' (int) keys; ``warmup`` untimed iterations run
        first."""
        if not samples:
            raise ValueError("samples list cannot be empty")

        self._log(f"Starting ASR benchmark with {len(samples)} samples...")

        audio_tensors = [s["audio"] for s in samples]
        sampling_rates = [s.get("sampling_rate", 16000) for s in samples]

        mem_before, timer = self._warm_up(
            warmup,
            lambda: model.transcribe(
                audio_tensors[:1], sampling_rates[:1], language=language
            ),
            "asr_transcribe",
        )

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
                "real_time_factor": (timer.total_time / total_audio_duration
                                     if total_audio_duration > 0 else 0),
            }
        )

    def benchmark_embedding(
        self,
        model: Any,
        texts: List[str],
        warmup: int = 3,
        batch_size: int = 32,
    ) -> BenchmarkResult:
        """Benchmark text embedding model encode() performance in batches of
        ``batch_size``; ``warmup`` untimed iterations run first."""
        if not texts:
            raise ValueError("texts list cannot be empty")

        self._log(f"Starting embedding benchmark with {len(texts)} texts...")

        warmup_texts = texts[:min(batch_size, len(texts))]
        mem_before, timer = self._warm_up(
            warmup, lambda: model.encode(warmup_texts), "embedding_encode"
        )

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

