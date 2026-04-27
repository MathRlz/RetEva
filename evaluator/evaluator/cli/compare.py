"""CLI command for comparing experiment results with statistical significance testing."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from evaluator.analysis import (
    compare_result_files,
    format_comparison_report,
)


def parse_compare_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for compare command."""
    parser = argparse.ArgumentParser(
        description="Compare two evaluation result files with statistical significance testing"
    )
    
    parser.add_argument(
        "file_a",
        type=str,
        help="Path to first results JSON file"
    )
    parser.add_argument(
        "file_b",
        type=str,
        help="Path to second results JSON file"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Specific metrics to compare (default: all common numeric metrics)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for JSON comparison results (default: print to stdout)"
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    return parser.parse_args(args)


def run_compare(args: argparse.Namespace) -> int:
    """Run experiment comparison.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    path_a = Path(args.file_a)
    path_b = Path(args.file_b)
    
    # Validate paths
    if not path_a.exists():
        print(f"Error: File not found: {path_a}", file=sys.stderr)
        return 1
    if not path_b.exists():
        print(f"Error: File not found: {path_b}", file=sys.stderr)
        return 1
    
    try:
        comparison = compare_result_files(path_a, path_b, args.metrics)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Format output
    if args.format == "json":
        output = json.dumps(comparison, indent=2)
    else:
        output = format_comparison_report(comparison)
    
    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        print(f"Comparison saved to: {output_path}")
    else:
        print(output)
    
    return 0


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for compare CLI."""
    parsed_args = parse_compare_args(args)
    return run_compare(parsed_args)


if __name__ == "__main__":
    sys.exit(main())
