"""Visualization utilities for evaluation results.

This package provides plotting and interactive visualization functions
for evaluation results.
"""

from .plots import (
    plot_metric_comparison,
    plot_multi_metric_comparison,
    plot_metric_heatmap,
    plot_metric_distribution,
    plot_correlation_matrix,
)

from .interactive import (
    create_interactive_bar_chart,
    create_multi_metric_bar_chart,
    create_scatter_plot,
    create_radar_chart,
    create_heatmap,
    save_interactive_html,
)

__all__ = [
    # Static plots
    "plot_metric_comparison",
    "plot_multi_metric_comparison",
    "plot_metric_heatmap",
    "plot_metric_distribution",
    "plot_correlation_matrix",
    # Interactive plots
    "create_interactive_bar_chart",
    "create_multi_metric_bar_chart",
    "create_scatter_plot",
    "create_radar_chart",
    "create_heatmap",
    "save_interactive_html",
]
