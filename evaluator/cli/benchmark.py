"""CLI command for running benchmarks."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from evaluator import models
from evaluator.benchmarks import (
    ModelBenchmark,
    BenchmarkResult,
    generate_benchmark_report,
    export_to_json,
    export_to_csv,
)
from evaluator.models import WhisperModel


def create_synthetic_audio(
    duration_sec: float = 3.0,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Create synthetic audio for benchmarking."""
    num_samples = int(duration_sec * sample_rate)
    t = torch.linspace(0, duration_sec, num_samples)
    audio = 0.5 * torch.sin(2 * np.pi * 440 * t) + 0.1 * torch.randn(num_samples)
    return audio


def create_synthetic_texts(num_texts: int = 100) -> List[str]:
    """Create synthetic texts for benchmarking."""
    base_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Medical research has shown significant advances in treatment options.",
        "Machine learning algorithms process large amounts of data efficiently.",
        "Clinical trials demonstrate the efficacy of new pharmaceutical compounds.",
        "Natural language processing enables computers to understand human speech.",
        "Diagnostic imaging techniques have revolutionized medical practice.",
        "Deep learning models achieve state-of-the-art results in various tasks.",
        "Patient outcomes improve with early detection and intervention.",
        "Speech recognition systems convert audio to text with high accuracy.",
        "Healthcare systems benefit from automated data analysis tools.",
    ]

    texts = []
    for i in range(num_texts):
        base = base_texts[i % len(base_texts)]
        texts.append(f"[Sample {i+1}] {base}")

    return texts


def parse_benchmark_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse benchmark command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run performance benchmarks on models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluator.cli.benchmark --model whisper --samples 100
  python -m evaluator.cli.benchmark --model labse --samples 500 --output results.json
  python -m evaluator.cli.benchmark --model all --warmup 5
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["whisper", "labse", "bge-m3", "jina", "all"],
        default="all",
        help="Model to benchmark (default: all)",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to benchmark (default: 100)",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding models (default: 32)",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (supports .json and .csv)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run benchmarks on (default: cuda if available)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--audio-duration",
        type=float,
        default=3.0,
        help="Duration of synthetic audio samples in seconds (default: 3.0)",
    )

    return parser.parse_args(args)


# (--model value, evaluator.models class, banner text, short name used in error output)
_EMBEDDING_BENCHMARKS = (
    ("labse", "LabseModel", "LaBSE embedding model", "LaBSE"),
    ("bge-m3", "BgeM3Model", "BGE-M3 embedding model", "BGE-M3"),
    ("jina", "JinaV4Model", "Jina v4 embedding model", "Jina v4"),
)


def _benchmark_embedding_model(
    benchmark: ModelBenchmark,
    args: argparse.Namespace,
    device: torch.device,
    texts: List[str],
    class_name: str,
    banner: str,
    label: str,
) -> Optional[BenchmarkResult]:
    """Benchmark one embedding model. A missing optional dependency or a runtime failure is
    reported and skipped so the remaining models still get benchmarked."""
    try:
        model_class = getattr(models, class_name)

        print("=" * 60)
        print(f"Benchmarking {banner}...")
        print("=" * 60)

        model = model_class()
        model.to(device)

        result = benchmark.benchmark_embedding(
            model=model,
            texts=texts,
            warmup=args.warmup,
            batch_size=args.batch_size,
        )
        print(result)
        print()
        return result

    except ImportError as e:
        print(f"Warning: Could not import {class_name}: {e}")
    except Exception as e:
        print(f"Error benchmarking {label}: {e}")
    return None


def run_benchmark_cli(args: argparse.Namespace) -> List[BenchmarkResult]:
    """Run benchmarks based on CLI arguments."""
    results = []
    benchmark = ModelBenchmark(verbose=args.verbose)
    device = torch.device(args.device)

    print(f"Running benchmarks on device: {device}")
    print(f"Samples: {args.samples}, Warmup: {args.warmup}")
    print()

    # Benchmark ASR models
    if args.model in ["whisper", "all"]:
        try:
            print("=" * 60)
            print("Benchmarking Whisper ASR model...")
            print("=" * 60)

            audio_samples = [
                {
                    "audio": create_synthetic_audio(args.audio_duration),
                    "sampling_rate": 16000,
                }
                for _ in range(args.samples)
            ]

            model = WhisperModel(model_size="base")
            model.to(device)

            result = benchmark.benchmark_asr(
                model=model,
                samples=audio_samples,
                warmup=args.warmup,
            )
            results.append(result)
            print(result)
            print()

        except ImportError as e:
            print(f"Warning: Could not import WhisperModel: {e}")
        except Exception as e:
            print(f"Error benchmarking Whisper: {e}")

    # Benchmark embedding models
    texts = create_synthetic_texts(args.samples)

    for key, class_name, banner, label in _EMBEDDING_BENCHMARKS:
        if args.model in [key, "all"]:
            result = _benchmark_embedding_model(
                benchmark, args, device, texts, class_name, banner, label
            )
            if result is not None:
                results.append(result)

    return results


def main(args: Optional[List[str]] = None) -> None:
    parsed_args = parse_benchmark_args(args)

    results = run_benchmark_cli(parsed_args)

    if not results:
        print("No benchmark results collected.")
        sys.exit(1)

    report = generate_benchmark_report(results)
    print()
    print(report)

    if parsed_args.output:
        output_path = Path(parsed_args.output)

        if output_path.suffix == ".json":
            export_to_json(
                results,
                output_path,
                metadata={
                    "device": parsed_args.device,
                    "samples": parsed_args.samples,
                    "warmup": parsed_args.warmup,
                },
            )
            print(f"\nResults exported to: {output_path}")

        elif output_path.suffix == ".csv":
            export_to_csv(results, output_path)
            print(f"\nResults exported to: {output_path}")

        else:
            # Default to JSON
            json_path = output_path.with_suffix(".json")
            export_to_json(
                results,
                json_path,
                metadata={
                    "device": parsed_args.device,
                    "samples": parsed_args.samples,
                    "warmup": parsed_args.warmup,
                },
            )
            print(f"\nResults exported to: {json_path}")


if __name__ == "__main__":
    main()
