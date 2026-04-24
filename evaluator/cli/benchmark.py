"""CLI command for running benchmarks."""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from evaluator.benchmarks import (
    ModelBenchmark,
    BenchmarkResult,
    generate_benchmark_report,
    export_to_json,
    export_to_csv,
)


def create_synthetic_audio(
    duration_sec: float = 3.0,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Create synthetic audio for benchmarking.
    
    Args:
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.
        
    Returns:
        Synthetic audio tensor.
    """
    num_samples = int(duration_sec * sample_rate)
    # Generate simple sine wave with some noise
    t = torch.linspace(0, duration_sec, num_samples)
    audio = 0.5 * torch.sin(2 * np.pi * 440 * t) + 0.1 * torch.randn(num_samples)
    return audio


def create_synthetic_texts(num_texts: int = 100) -> List[str]:
    """Create synthetic texts for benchmarking.
    
    Args:
        num_texts: Number of texts to generate.
        
    Returns:
        List of synthetic texts.
    """
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
        # Add variation
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


def run_benchmark_cli(args: argparse.Namespace) -> List[BenchmarkResult]:
    """Run benchmarks based on CLI arguments.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        List of benchmark results.
    """
    results = []
    benchmark = ModelBenchmark(verbose=args.verbose)
    device = torch.device(args.device)
    
    print(f"Running benchmarks on device: {device}")
    print(f"Samples: {args.samples}, Warmup: {args.warmup}")
    print()
    
    # Benchmark ASR models
    if args.model in ["whisper", "all"]:
        try:
            from evaluator.models import WhisperModel
            
            print("=" * 60)
            print("Benchmarking Whisper ASR model...")
            print("=" * 60)
            
            # Create synthetic audio samples
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
    
    if args.model in ["labse", "all"]:
        try:
            from evaluator.models import LabseModel
            
            print("=" * 60)
            print("Benchmarking LaBSE embedding model...")
            print("=" * 60)
            
            model = LabseModel()
            model.to(device)
            
            result = benchmark.benchmark_embedding(
                model=model,
                texts=texts,
                warmup=args.warmup,
                batch_size=args.batch_size,
            )
            results.append(result)
            print(result)
            print()
            
        except ImportError as e:
            print(f"Warning: Could not import LabseModel: {e}")
        except Exception as e:
            print(f"Error benchmarking LaBSE: {e}")
    
    if args.model in ["bge-m3", "all"]:
        try:
            from evaluator.models import BgeM3Model
            
            print("=" * 60)
            print("Benchmarking BGE-M3 embedding model...")
            print("=" * 60)
            
            model = BgeM3Model()
            model.to(device)
            
            result = benchmark.benchmark_embedding(
                model=model,
                texts=texts,
                warmup=args.warmup,
                batch_size=args.batch_size,
            )
            results.append(result)
            print(result)
            print()
            
        except ImportError as e:
            print(f"Warning: Could not import BgeM3Model: {e}")
        except Exception as e:
            print(f"Error benchmarking BGE-M3: {e}")
    
    if args.model in ["jina", "all"]:
        try:
            from evaluator.models import JinaV4Model
            
            print("=" * 60)
            print("Benchmarking Jina v4 embedding model...")
            print("=" * 60)
            
            model = JinaV4Model()
            model.to(device)
            
            result = benchmark.benchmark_embedding(
                model=model,
                texts=texts,
                warmup=args.warmup,
                batch_size=args.batch_size,
            )
            results.append(result)
            print(result)
            print()
            
        except ImportError as e:
            print(f"Warning: Could not import JinaV4Model: {e}")
        except Exception as e:
            print(f"Error benchmarking Jina v4: {e}")
    
    return results


def main(args: Optional[List[str]] = None) -> None:
    """Main entry point for benchmark CLI."""
    parsed_args = parse_benchmark_args(args)
    
    results = run_benchmark_cli(parsed_args)
    
    if not results:
        print("No benchmark results collected.")
        sys.exit(1)
    
    # Generate and print report
    report = generate_benchmark_report(results)
    print()
    print(report)
    
    # Export if output path specified
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
