"""Interactive visualization utilities using Plotly.

This module provides functions for creating interactive plots of evaluation results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_interactive_bar_chart(
    results: List[Dict[str, Any]],
    metric: str = "MRR",
    title: Optional[str] = None,
    height: int = 600,
    width: int = 1000
) -> go.Figure:
    """Create interactive bar chart comparing metric across results.
    
    Args:
        results: List of result dictionaries
        metric: Metric to plot
        title: Plot title
        height: Figure height
        width: Figure width
        
    Returns:
        Plotly Figure object
    """
    labels = []
    values = []
    
    for result in results:
        if metric in result:
            label = result.get('_filename', result.get('asr', 'Unknown'))[:40]
            labels.append(label)
            values.append(result[metric])
    
    if not values:
        raise ValueError(f"No results contain metric '{metric}'")
    
    # Sort by value
    sorted_data = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(labels),
            y=list(values),
            text=[f'{v:.4f}' for v in values],
            textposition='outside',
            marker_color='steelblue',
            hovertemplate='<b>%{x}</b><br>' + f'{metric}: %{{y:.4f}}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=title or f'Model Comparison by {metric}',
        xaxis_title='Model',
        yaxis_title=metric,
        width=width,
        height=height,
        xaxis_tickangle=-45,
        yaxis_range=[0, max(values) * 1.1]
    )
    
    return fig


def create_multi_metric_bar_chart(
    results: List[Dict[str, Any]],
    metrics: List[str] = None,
    title: Optional[str] = None,
    height: int = 600,
    width: int = 1000
) -> go.Figure:
    """Create interactive grouped bar chart for multiple metrics.
    
    Args:
        results: List of result dictionaries
        metrics: Metrics to plot
        title: Plot title
        height: Figure height
        width: Figure width
        
    Returns:
        Plotly Figure object
    """
    if metrics is None:
        metrics = ['MRR', 'Recall@5', 'NDCG@5']
    
    # Build data
    rows = []
    for result in results:
        row = {'Model': result.get('_filename', result.get('asr', 'Unknown'))[:30]}
        for metric in metrics:
            if metric in result:
                row[metric] = result[metric]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    fig = go.Figure()
    
    for metric in metrics:
        if metric in df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=df['Model'],
                y=df[metric],
                text=df[metric].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A'),
                textposition='outside',
                hovertemplate=f'<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=title or 'Multi-Metric Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        width=width,
        height=height,
        xaxis_tickangle=-45,
        yaxis_range=[0, 1.0],
        barmode='group'
    )
    
    return fig


def create_scatter_plot(
    results: List[Dict[str, Any]],
    x_metric: str = "WER",
    y_metric: str = "MRR",
    color_by: Optional[str] = None,
    title: Optional[str] = None,
    height: int = 600,
    width: int = 900
) -> go.Figure:
    """Create interactive scatter plot comparing two metrics.
    
    Args:
        results: List of result dictionaries
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
        color_by: Field to color points by (e.g., 'embedder')
        title: Plot title
        height: Figure height
        width: Figure width
        
    Returns:
        Plotly Figure object
    """
    rows = []
    for result in results:
        if x_metric in result and y_metric in result:
            row = {
                'Model': result.get('_filename', 'Unknown')[:30],
                x_metric: result[x_metric],
                y_metric: result[y_metric]
            }
            if color_by and color_by in result:
                row[color_by] = result[color_by]
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        raise ValueError(f"No results contain both {x_metric} and {y_metric}")
    
    if color_by and color_by in df.columns:
        fig = px.scatter(
            df,
            x=x_metric,
            y=y_metric,
            color=color_by,
            hover_data=['Model'],
            title=title or f'{y_metric} vs {x_metric}',
            labels={x_metric: x_metric, y_metric: y_metric}
        )
    else:
        fig = px.scatter(
            df,
            x=x_metric,
            y=y_metric,
            hover_data=['Model'],
            title=title or f'{y_metric} vs {x_metric}',
            labels={x_metric: x_metric, y_metric: y_metric}
        )
    
    fig.update_layout(width=width, height=height)
    fig.update_traces(marker=dict(size=12))
    
    return fig


def create_radar_chart(
    results: List[Dict[str, Any]],
    metrics: List[str] = None,
    max_models: int = 5,
    title: Optional[str] = None,
    height: int = 600,
    width: int = 800
) -> go.Figure:
    """Create interactive radar chart for multi-metric profile.
    
    Args:
        results: List of result dictionaries
        metrics: Metrics to include
        max_models: Maximum number of models to show
        title: Plot title
        height: Figure height
        width: Figure width
        
    Returns:
        Plotly Figure object
    """
    if metrics is None:
        metrics = ['MRR', 'Recall@1', 'Recall@5', 'NDCG@5']
    
    # Filter results with all metrics
    valid_results = []
    for result in results:
        if all(m in result for m in metrics):
            valid_results.append(result)
    
    if not valid_results:
        raise ValueError("No results contain all specified metrics")
    
    # Sort by first metric and take top N
    if 'MRR' in metrics:
        valid_results.sort(key=lambda x: x.get('MRR', 0), reverse=True)
    valid_results = valid_results[:max_models]
    
    fig = go.Figure()
    
    for result in valid_results:
        values = [result[m] for m in metrics]
        values.append(values[0])  # Close the polygon
        
        model_name = result.get('_filename', result.get('asr', 'Unknown'))[:30]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=title or 'Multi-Metric Profile',
        width=width,
        height=height,
        showlegend=True
    )
    
    return fig


def create_heatmap(
    results: List[Dict[str, Any]],
    metrics: List[str] = None,
    title: Optional[str] = None,
    height: int = 600,
    width: int = 900
) -> go.Figure:
    """Create interactive heatmap of metrics across models.
    
    Args:
        results: List of result dictionaries
        metrics: Metrics to include
        title: Plot title
        height: Figure height
        width: Figure width
        
    Returns:
        Plotly Figure object
    """
    if metrics is None:
        metrics = ['MRR', 'Recall@1', 'Recall@5', 'NDCG@5']
    
    # Build data matrix
    rows = []
    labels = []
    
    for result in results:
        row = []
        for metric in metrics:
            if metric in result:
                row.append(result[metric])
            else:
                row.append(np.nan)
        rows.append(row)
        labels.append(result.get('_filename', result.get('asr', 'Unknown'))[:30])
    
    fig = go.Figure(data=go.Heatmap(
        z=rows,
        x=metrics,
        y=labels,
        colorscale='YlGnBu',
        text=[[f'{v:.4f}' if not np.isnan(v) else 'N/A' for v in row] for row in rows],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Score"),
        hovertemplate='Model: %{y}<br>Metric: %{x}<br>Score: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title or 'Metric Heatmap',
        xaxis_title='Metric',
        yaxis_title='Model',
        width=width,
        height=height
    )
    
    return fig


def save_interactive_html(
    fig: go.Figure,
    path: Path,
    include_plotlyjs: str = 'cdn'
) -> None:
    """Save interactive plot as HTML file.
    
    Args:
        fig: Plotly figure
        path: Output path
        include_plotlyjs: How to include plotly.js ('cdn', 'inline', or False)
    """
    fig.write_html(str(path), include_plotlyjs=include_plotlyjs)
