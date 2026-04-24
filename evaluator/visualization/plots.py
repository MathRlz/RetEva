"""Plotting utilities for evaluation results.

This module provides reusable plotting functions for common visualizations
of evaluation results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_metric_comparison(
    results: List[Dict[str, Any]],
    metric: str = "MRR",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Create a bar chart comparing a metric across results.
    
    Args:
        results: List of result dictionaries
        metric: Metric to plot
        title: Plot title (auto-generated if None)
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    # Extract metric values
    labels = []
    values = []
    
    for result in results:
        if metric in result:
            label = result.get('_filename', result.get('asr', 'Unknown'))[:40]
            labels.append(label)
            values.append(result[metric])
    
    if not values:
        raise ValueError(f"No results contain metric '{metric}'")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(values)), values, color='steelblue', alpha=0.8)
    
    # Highlight best value
    best_idx = np.argmax(values) if metric not in ['WER', 'CER'] else np.argmin(values)
    bars[best_idx].set_color('forestgreen')
    
    ax.set_xlabel('Model Configuration', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title or f'Model Comparison by {metric}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multi_metric_comparison(
    results: List[Dict[str, Any]],
    metrics: List[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Create a grouped bar chart comparing multiple metrics.
    
    Args:
        results: List of result dictionaries
        metrics: List of metrics to plot (default: MRR, Recall@5, NDCG@5)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ['MRR', 'Recall@5', 'NDCG@5']
    
    # Build DataFrame
    rows = []
    for result in results:
        row = {'Model': result.get('_filename', result.get('asr', 'Unknown'))[:30]}
        for metric in metrics:
            if metric in result:
                row[metric] = result[metric]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            values = df[metric].fillna(0)
            offset = width * (i - len(metrics)/2 + 0.5)
            ax.bar(x + offset, values, width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title or 'Multi-Metric Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metric_heatmap(
    results: List[Dict[str, Any]],
    metrics: List[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Create a heatmap showing metric values across experiments.
    
    Args:
        results: List of result dictionaries
        metrics: List of metrics to include
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ['MRR', 'Recall@1', 'Recall@5', 'NDCG@5']
    
    # Build DataFrame
    rows = []
    for result in results:
        row = {'Model': result.get('_filename', result.get('asr', 'Unknown'))[:30]}
        for metric in metrics:
            if metric in result:
                row[metric] = result[metric]
        rows.append(row)
    
    df = pd.DataFrame(rows).set_index('Model')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df,
        annot=True,
        fmt='.4f',
        cmap='YlGnBu',
        cbar_kws={'label': 'Score'},
        vmin=0,
        vmax=1,
        ax=ax
    )
    
    ax.set_title(title or 'Metric Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metric_distribution(
    results: List[Dict[str, Any]],
    metric: str = "MRR",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Create a box plot showing metric distribution.
    
    Args:
        results: List of result dictionaries
        metric: Metric to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    values = [r[metric] for r in results if metric in r]
    
    if not values:
        raise ValueError(f"No results contain metric '{metric}'")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bp = ax.boxplot([values], labels=[metric], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    
    # Add individual points
    y = np.random.normal(1, 0.04, len(values))
    ax.scatter(y, values, alpha=0.5, color='steelblue', s=50)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title or f'{metric} Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(
    results: List[Dict[str, Any]],
    metrics: List[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Create a correlation matrix heatmap.
    
    Args:
        results: List of result dictionaries
        metrics: List of metrics to correlate
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ['WER', 'MRR', 'Recall@5', 'NDCG@5']
    
    # Build DataFrame
    rows = []
    for result in results:
        row = {}
        for metric in metrics:
            if metric in result:
                row[metric] = result[metric]
        if row:
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if df.empty or len(df) < 2:
        raise ValueError("Not enough data for correlation analysis")
    
    # Calculate correlation
    corr = df.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
        ax=ax
    )
    
    ax.set_title(title or 'Metric Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
