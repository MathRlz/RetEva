"""CLI command for exporting evaluation results to various formats."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from evaluator.analysis import (
    export_to_csv,
    export_to_excel,
    export_to_latex,
    compare_experiments_to_latex,
    export_sample_results,
)
from evaluator.analysis.export import load_results


def parse_export_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for export command."""
    parser = argparse.ArgumentParser(
        description="Export evaluation results to CSV, Excel, or LaTeX formats"
    )

    parser.add_argument(
        "input",
        type=str,
        nargs="+",
        help="Path to results JSON file(s). Multiple files for comparison table."
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["csv", "excel", "latex", "latex-compare", "samples",
                 "metrics-table", "traces", "traces-parquet", "mlflow", "wandb"],
        default="csv",
        help="Output format (default: csv). 'metrics-table' = tidy branch×metric CSV; "
             "'traces' = per-query JSONL; 'traces-parquet' = per-query Parquet; "
             "'mlflow'/'wandb' = log the report to that tracker ('--output' = run name)."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="Evaluation Results",
        help="Table caption for LaTeX output"
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="Experiment names for comparison table (required for latex-compare)"
    )

    return parser.parse_args(args)


def run_export(args: argparse.Namespace) -> int:
    """Run export command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    input_paths = [Path(p) for p in args.input]

    # Validate input files exist
    for path in input_paths:
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            return 1

    try:
        if args.format == "latex-compare":
            # Multi-file comparison
            if len(input_paths) < 2:
                print(
                    "Error: latex-compare requires at least 2 input files",
                    file=sys.stderr
                )
                return 1

            # Generate names if not provided
            names = args.names
            if names is None:
                names = [p.stem for p in input_paths]
            elif len(names) != len(input_paths):
                print(
                    f"Error: Number of names ({len(names)}) must match "
                    f"number of input files ({len(input_paths)})",
                    file=sys.stderr
                )
                return 1

            results_list = [load_results(p) for p in input_paths]
            compare_experiments_to_latex(
                results_list,
                names,
                output_path=args.output,
                caption=args.caption
            )
            print(f"LaTeX comparison table saved to: {args.output}")

        else:
            # Single file export
            if len(input_paths) > 1:
                print(
                    f"Warning: {args.format} format only uses first input file",
                    file=sys.stderr
                )

            results = load_results(input_paths[0])

            if args.format == "csv":
                export_to_csv(results, args.output)
                print(f"CSV exported to: {args.output}")

            elif args.format == "excel":
                export_to_excel(results, args.output)
                print(f"Excel file exported to: {args.output}")

            elif args.format == "latex":
                export_to_latex(results, args.output, caption=args.caption)
                print(f"LaTeX table saved to: {args.output}")

            elif args.format == "samples":
                export_sample_results(results, args.output)
                print(f"Per-sample results exported to: {args.output}")

            elif args.format == "metrics-table":
                from evaluator.analysis.report_export import write_metrics_table_csv

                n = write_metrics_table_csv(results, args.output)
                print(f"Metrics table ({n} rows) exported to: {args.output}")

            elif args.format == "traces":
                from evaluator.analysis.report_export import write_traces_jsonl

                n = write_traces_jsonl(results, args.output)
                print(f"Per-query traces ({n} rows) exported to: {args.output}")

            elif args.format == "traces-parquet":
                from evaluator.analysis.report_export import write_traces_parquet

                n = write_traces_parquet(results, args.output)
                print(f"Per-query traces ({n} rows) exported to: {args.output}")

            elif args.format == "mlflow":
                from evaluator.analysis.tracking_export import to_mlflow

                to_mlflow(results, run_name=args.output)
                print(f"Logged report to MLflow (run: {args.output})")

            elif args.format == "wandb":
                from evaluator.analysis.tracking_export import to_wandb

                to_wandb(results, run_name=args.output)
                print(f"Logged report to W&B (run: {args.output})")

        return 0

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        return 1
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for export CLI."""
    parsed_args = parse_export_args(args)
    return run_export(parsed_args)


if __name__ == "__main__":
    sys.exit(main())
