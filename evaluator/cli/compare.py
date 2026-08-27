"""CLI command for comparing experiment results with statistical significance testing.

Input shapes, auto-detected:
- Two flat result JSON files (the original, unchanged: `evaluator compare a.json b.json`).
- Two or more variant/run directories (`variants/<id>/`, or a whole run `output_dir` that
  resolves to its single variant) — baseline-vs-each against the first path.
- ONE multi-variant run's `output_dir` (or its `variants/` dir) — a run with several compared
  paths in its graph (e.g. several ASR models or audio encoders) auto-expands to compare all
  of them, baseline = the first: `evaluator compare path/to/run`.
Reuses the same significance testing plus a resolved-config diff and per-query answer diffs
(`analysis/variant_compare.py`, the "unified comparison tool" roadmap item).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from evaluator.analysis import (
    compare_result_files,
    format_comparison_report,
)
from evaluator.analysis.variant_compare import (
    VariantCompareError,
    compare_paths,
    format_variant_comparison_report,
)


def parse_compare_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for compare command."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare evaluation results with statistical significance testing: two flat "
            "result JSON files, or two+ variant/run directories (baseline-vs-each, first = "
            "baseline)"
        )
    )

    parser.add_argument(
        "paths",
        type=str,
        nargs="+",
        help=(
            "Two result JSON files, or two+ variant/run directories "
            "(a variants/<id>/ dir, or a whole output_dir with exactly one variant)"
        ),
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
    paths = [Path(p) for p in args.paths]
    for p in paths:
        if not p.exists():
            print(f"Error: path not found: {p}", file=sys.stderr)
            return 1

    # Two flat JSON files → the original file-diff+stats path, unchanged.
    legacy_mode = len(paths) == 2 and all(p.is_file() for p in paths)

    try:
        if legacy_mode:
            comparison = compare_result_files(paths[0], paths[1], args.metrics)
        else:
            comparison = compare_paths(paths, args.metrics)
    except VariantCompareError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Format output
    if args.format == "json":
        output = json.dumps(comparison, indent=2)
    elif legacy_mode:
        output = format_comparison_report(comparison)
    else:
        output = format_variant_comparison_report(comparison)

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
