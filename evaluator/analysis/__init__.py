"""Analysis module for statistical significance testing and experiment comparison."""

from .significance import (
    paired_ttest,
    wilcoxon_test,
    bootstrap_confidence_interval,
    compare_experiments,
    load_results,
    compare_result_files,
    format_comparison_report,
)

from .export import (
    export_to_csv,
    export_to_excel,
    export_to_latex,
    compare_experiments_to_latex,
    export_sample_results,
)

from .branch_report import (
    latex_branch_table,
    plot_branch_deltas,
    export_per_query_failures,
)

from .errors import (
    analyze_asr_errors,
    analyze_retrieval_failures,
    categorize_errors,
    generate_error_report,
)

from .grid_search import GridSearch

__all__ = [
    # Significance testing
    "paired_ttest",
    "wilcoxon_test",
    "bootstrap_confidence_interval",
    "compare_experiments",
    "load_results",
    "compare_result_files",
    "format_comparison_report",
    # Export functions
    "export_to_csv",
    "export_to_excel",
    "export_to_latex",
    "compare_experiments_to_latex",
    "export_sample_results",
    # Branched-report thesis artifacts
    "latex_branch_table",
    "plot_branch_deltas",
    "export_per_query_failures",
    # Error analysis
    "analyze_asr_errors",
    "analyze_retrieval_failures",
    "categorize_errors",
    "generate_error_report",
    # Grid search
    "GridSearch",
]
