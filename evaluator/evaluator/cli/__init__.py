"""CLI package for the evaluation tool."""

from .parser import parse_args
from .commands import run_evaluation
from .utils import get_adapter_suffix, generate_output_filename
from .compare import main as compare_main, run_compare
from .export import main as export_main, run_export
from .gpu_status import show_gpu_status, print_gpu_summary
from .benchmark import main as benchmark_main, run_benchmark_cli

__all__ = [
    "parse_args",
    "run_evaluation",
    "get_adapter_suffix",
    "generate_output_filename",
    "compare_main",
    "run_compare",
    "export_main",
    "run_export",
    "show_gpu_status",
    "print_gpu_summary",
    "benchmark_main",
    "run_benchmark_cli",
]
