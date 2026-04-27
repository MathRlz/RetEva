"""Benchmark reporting and export utilities."""

import json
import csv
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
from datetime import datetime

from .model_benchmark import BenchmarkResult

CURRENT_RESULT_FORMAT_VERSION = "1.0"


def generate_benchmark_report(
    results: List[BenchmarkResult],
    title: str = "Benchmark Report",
    include_extra_metrics: bool = True,
) -> str:
    """Generate a human-readable benchmark report.
    
    Args:
        results: List of benchmark results to include.
        title: Report title.
        include_extra_metrics: Whether to include extra metrics in report.
        
    Returns:
        Formatted report string.
    """
    if not results:
        return "No benchmark results to report."
    
    lines = [
        "=" * 70,
        title.center(70),
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total benchmarks: {len(results)}",
        "",
    ]
    
    for i, result in enumerate(results, 1):
        lines.append("-" * 70)
        lines.append(f"[{i}] {result.name}")
        lines.append("-" * 70)
        lines.append(f"  Timestamp: {result.timestamp}")
        lines.append(f"  Samples: {result.num_samples} (warmup: {result.num_warmup})")
        lines.append("")
        
        # Timing stats
        if result.timing_stats:
            stats = result.timing_stats
            lines.append("  Timing Statistics:")
            lines.append(f"    Mean:  {stats.mean * 1000:.2f} ms")
            lines.append(f"    Std:   {stats.std * 1000:.2f} ms")
            lines.append(f"    Min:   {stats.min * 1000:.2f} ms")
            lines.append(f"    Max:   {stats.max * 1000:.2f} ms")
            lines.append("")
        
        # Throughput
        lines.append("  Throughput:")
        lines.append(f"    {result.throughput:.2f} {result.throughput_unit}")
        lines.append("")
        
        # Memory
        lines.append("  Memory Usage:")
        lines.append(f"    Before: {result.memory_before_mb:.1f} MB")
        lines.append(f"    After:  {result.memory_after_mb:.1f} MB")
        lines.append(f"    Delta:  {result.memory_after_mb - result.memory_before_mb:+.1f} MB")
        
        if result.gpu_memory_mb > 0:
            lines.append(f"    GPU:    {result.gpu_memory_mb:.1f} MB (peak: {result.gpu_memory_peak_mb:.1f} MB)")
        
        lines.append("")
        
        # Extra metrics
        if include_extra_metrics and result.extra_metrics:
            lines.append("  Additional Metrics:")
            for key, value in result.extra_metrics.items():
                if isinstance(value, float):
                    lines.append(f"    {key}: {value:.4f}")
                else:
                    lines.append(f"    {key}: {value}")
            lines.append("")
    
    lines.append("=" * 70)
    
    # Summary table
    lines.append("SUMMARY")
    lines.append("=" * 70)
    lines.append(f"{'Benchmark':<35} {'Throughput':>15} {'Mean Time':>12}")
    lines.append("-" * 70)
    
    for result in results:
        throughput_str = f"{result.throughput:.2f}"
        mean_str = f"{result.timing_stats.mean * 1000:.2f} ms" if result.timing_stats else "N/A"
        lines.append(f"{result.name:<35} {throughput_str:>15} {mean_str:>12}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def export_to_json(
    results: List[BenchmarkResult],
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Export benchmark results to JSON file.
    
    Args:
        results: List of benchmark results.
        path: Output file path.
        metadata: Optional metadata to include in the export.
    """
    output = {
        "result_format_version": CURRENT_RESULT_FORMAT_VERSION,
        "generated_at": datetime.now().isoformat(),
        "num_benchmarks": len(results),
        "results": [r.to_dict() for r in results],
    }
    
    if metadata:
        output["metadata"] = metadata
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def export_to_csv(
    results: List[BenchmarkResult],
    path: Union[str, Path],
) -> None:
    """Export benchmark results to CSV file.
    
    Args:
        results: List of benchmark results.
        path: Output file path.
    """
    if not results:
        return
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "name",
        "timestamp",
        "num_samples",
        "num_warmup",
        "throughput",
        "throughput_unit",
        "timing_mean_sec",
        "timing_std_sec",
        "timing_min_sec",
        "timing_max_sec",
        "memory_before_mb",
        "memory_after_mb",
        "memory_delta_mb",
        "gpu_memory_mb",
        "gpu_memory_peak_mb",
    ]
    
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                "name": result.name,
                "timestamp": result.timestamp,
                "num_samples": result.num_samples,
                "num_warmup": result.num_warmup,
                "throughput": result.throughput,
                "throughput_unit": result.throughput_unit,
                "memory_before_mb": result.memory_before_mb,
                "memory_after_mb": result.memory_after_mb,
                "memory_delta_mb": result.memory_after_mb - result.memory_before_mb,
                "gpu_memory_mb": result.gpu_memory_mb,
                "gpu_memory_peak_mb": result.gpu_memory_peak_mb,
            }
            
            if result.timing_stats:
                row["timing_mean_sec"] = result.timing_stats.mean
                row["timing_std_sec"] = result.timing_stats.std
                row["timing_min_sec"] = result.timing_stats.min
                row["timing_max_sec"] = result.timing_stats.max
            else:
                row["timing_mean_sec"] = ""
                row["timing_std_sec"] = ""
                row["timing_min_sec"] = ""
                row["timing_max_sec"] = ""
            
            writer.writerow(row)


def load_benchmark_results(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load benchmark results from JSON file.
    
    Args:
        path: Path to JSON file.
        
    Returns:
        List of benchmark result dictionaries.
    """
    path = Path(path)
    
    with open(path) as f:
        data = json.load(f)
    
    return data.get("results", [])


def compare_benchmark_results(
    baseline: List[BenchmarkResult],
    current: List[BenchmarkResult],
) -> str:
    """Compare two sets of benchmark results.
    
    Args:
        baseline: Baseline benchmark results.
        current: Current benchmark results to compare.
        
    Returns:
        Comparison report string.
    """
    lines = [
        "=" * 70,
        "BENCHMARK COMPARISON".center(70),
        "=" * 70,
        "",
    ]
    
    # Create lookup by name
    baseline_by_name = {r.name: r for r in baseline}
    current_by_name = {r.name: r for r in current}
    
    all_names = set(baseline_by_name.keys()) | set(current_by_name.keys())
    
    lines.append(f"{'Benchmark':<30} {'Baseline':>12} {'Current':>12} {'Change':>12}")
    lines.append("-" * 70)
    
    for name in sorted(all_names):
        base = baseline_by_name.get(name)
        curr = current_by_name.get(name)
        
        if base and curr and base.timing_stats and curr.timing_stats:
            base_mean = base.timing_stats.mean * 1000
            curr_mean = curr.timing_stats.mean * 1000
            
            if base_mean > 0:
                change_pct = ((curr_mean - base_mean) / base_mean) * 100
                change_str = f"{change_pct:+.1f}%"
                if change_pct < -5:
                    change_str += " ✓"  # Improvement
                elif change_pct > 5:
                    change_str += " ✗"  # Regression
            else:
                change_str = "N/A"
            
            lines.append(
                f"{name:<30} {base_mean:>10.2f}ms {curr_mean:>10.2f}ms {change_str:>12}"
            )
        elif base and base.timing_stats:
            base_mean = base.timing_stats.mean * 1000
            lines.append(f"{name:<30} {base_mean:>10.2f}ms {'N/A':>12} {'removed':>12}")
        elif curr and curr.timing_stats:
            curr_mean = curr.timing_stats.mean * 1000
            lines.append(f"{name:<30} {'N/A':>12} {curr_mean:>10.2f}ms {'new':>12}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)
