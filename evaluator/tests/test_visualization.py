"""Tests for visualization module."""

import json
import tempfile
from pathlib import Path

import pytest

# Check if visualization dependencies are available
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt
    from evaluator.visualization.plots import (
        plot_metric_comparison,
        plot_multi_metric_comparison,
        plot_metric_heatmap,
        plot_metric_distribution,
        plot_correlation_matrix,
    )
    from evaluator.visualization.interactive import (
        create_interactive_bar_chart,
        create_multi_metric_bar_chart,
        create_scatter_plot,
        create_radar_chart,
        create_heatmap,
        save_interactive_html,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="Visualization dependencies not installed")


if not VISUALIZATION_AVAILABLE:
    # Skip module if dependencies not available
    pytest.skip("Visualization dependencies not available", allow_module_level=True)


@pytest.fixture
def sample_results():
    """Create sample results for testing."""
    return [
        {
            '_filename': 'exp1',
            'asr': 'Whisper',
            'embedder': 'LaBSE',
            'WER': 0.15,
            'MRR': 0.85,
            'Recall@1': 0.80,
            'Recall@5': 0.90,
            'NDCG@5': 0.88,
        },
        {
            '_filename': 'exp2',
            'asr': 'Wav2Vec2',
            'embedder': 'Jina',
            'WER': 0.20,
            'MRR': 0.80,
            'Recall@1': 0.75,
            'Recall@5': 0.85,
            'NDCG@5': 0.83,
        },
        {
            '_filename': 'exp3',
            'asr': 'Whisper',
            'embedder': 'Jina',
            'WER': 0.12,
            'MRR': 0.90,
            'Recall@1': 0.85,
            'Recall@5': 0.95,
            'NDCG@5': 0.92,
        },
    ]


class TestStaticPlots:
    """Tests for static matplotlib/seaborn plots."""
    
    def test_plot_metric_comparison(self, sample_results):
        """Test basic metric comparison plot."""
        fig = plot_metric_comparison(sample_results, metric='MRR')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_metric_comparison_with_save(self, sample_results, tmp_path):
        """Test saving plot to file."""
        save_path = tmp_path / "test_plot.png"
        fig = plot_metric_comparison(sample_results, metric='MRR', save_path=save_path)
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_metric_comparison_wer(self, sample_results):
        """Test WER plot (lower is better)."""
        fig = plot_metric_comparison(sample_results, metric='WER')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_metric_comparison_missing_metric(self, sample_results):
        """Test error when metric not found."""
        with pytest.raises(ValueError, match="No results contain metric"):
            plot_metric_comparison(sample_results, metric='NonExistent')
    
    def test_plot_multi_metric_comparison(self, sample_results):
        """Test multi-metric comparison plot."""
        fig = plot_multi_metric_comparison(sample_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_multi_metric_comparison_custom(self, sample_results):
        """Test with custom metrics."""
        fig = plot_multi_metric_comparison(
            sample_results, 
            metrics=['MRR', 'Recall@5']
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_metric_heatmap(self, sample_results):
        """Test metric heatmap."""
        fig = plot_metric_heatmap(sample_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_metric_distribution(self, sample_results):
        """Test metric distribution plot."""
        fig = plot_metric_distribution(sample_results, metric='MRR')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_correlation_matrix(self, sample_results):
        """Test correlation matrix."""
        fig = plot_correlation_matrix(sample_results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_correlation_matrix_insufficient_data(self):
        """Test error with insufficient data."""
        results = [{'MRR': 0.85}]  # Only 1 result
        with pytest.raises(ValueError, match="Not enough data"):
            plot_correlation_matrix(results)


class TestInteractivePlots:
    """Tests for interactive Plotly plots."""
    
    def test_create_interactive_bar_chart(self, sample_results):
        """Test interactive bar chart."""
        fig = create_interactive_bar_chart(sample_results, metric='MRR')
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_create_interactive_bar_chart_custom(self, sample_results):
        """Test with custom parameters."""
        fig = create_interactive_bar_chart(
            sample_results,
            metric='MRR',
            title='Custom Title',
            height=500,
            width=800
        )
        assert fig is not None
        assert fig.layout.title.text == 'Custom Title'
    
    def test_create_multi_metric_bar_chart(self, sample_results):
        """Test multi-metric bar chart."""
        fig = create_multi_metric_bar_chart(sample_results)
        assert fig is not None
        assert len(fig.data) > 0
    
    def test_create_scatter_plot(self, sample_results):
        """Test scatter plot."""
        fig = create_scatter_plot(
            sample_results,
            x_metric='WER',
            y_metric='MRR'
        )
        assert fig is not None
    
    def test_create_scatter_plot_with_color(self, sample_results):
        """Test scatter plot with color grouping."""
        fig = create_scatter_plot(
            sample_results,
            x_metric='WER',
            y_metric='MRR',
            color_by='embedder'
        )
        assert fig is not None
    
    def test_create_radar_chart(self, sample_results):
        """Test radar chart."""
        fig = create_radar_chart(sample_results)
        assert fig is not None
    
    def test_create_radar_chart_custom_metrics(self, sample_results):
        """Test radar chart with custom metrics."""
        fig = create_radar_chart(
            sample_results,
            metrics=['MRR', 'Recall@5'],
            max_models=2
        )
        assert fig is not None
    
    def test_create_heatmap(self, sample_results):
        """Test interactive heatmap."""
        fig = create_heatmap(sample_results)
        assert fig is not None
    
    def test_save_interactive_html(self, sample_results, tmp_path):
        """Test saving interactive plot as HTML."""
        fig = create_interactive_bar_chart(sample_results, metric='MRR')
        html_path = tmp_path / "test_plot.html"
        save_interactive_html(fig, html_path)
        assert html_path.exists()
        
        # Check that HTML contains plotly
        content = html_path.read_text()
        assert 'plotly' in content.lower()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_results(self):
        """Test with empty results list."""
        with pytest.raises(ValueError):
            plot_metric_comparison([], metric='MRR')
    
    def test_single_result(self):
        """Test with single result."""
        results = [{'_filename': 'exp1', 'MRR': 0.85}]
        fig = plot_metric_comparison(results, metric='MRR')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_missing_filename(self):
        """Test results without _filename field."""
        results = [
            {'asr': 'Whisper', 'MRR': 0.85},
            {'asr': 'Wav2Vec2', 'MRR': 0.80},
        ]
        fig = plot_metric_comparison(results, metric='MRR')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_partial_metrics(self):
        """Test results with partial metrics."""
        results = [
            {'_filename': 'exp1', 'MRR': 0.85, 'Recall@5': 0.90},
            {'_filename': 'exp2', 'MRR': 0.80},  # Missing Recall@5
            {'_filename': 'exp3', 'Recall@5': 0.95},  # Missing MRR
        ]
        fig = plot_multi_metric_comparison(results, metrics=['MRR', 'Recall@5'])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestIntegration:
    """Integration tests with real-like scenarios."""
    
    def test_full_workflow(self, sample_results, tmp_path):
        """Test complete visualization workflow."""
        # Create all plot types
        fig1 = plot_metric_comparison(sample_results, metric='MRR')
        fig2 = plot_multi_metric_comparison(sample_results)
        fig3 = plot_metric_heatmap(sample_results)
        fig4 = create_interactive_bar_chart(sample_results, metric='MRR')
        
        # Save outputs
        (tmp_path / "static").mkdir()
        (tmp_path / "interactive").mkdir()
        
        fig1.savefig(tmp_path / "static" / "comparison.png")
        fig2.savefig(tmp_path / "static" / "multi_metric.png")
        fig3.savefig(tmp_path / "static" / "heatmap.png")
        save_interactive_html(fig4, tmp_path / "interactive" / "bar_chart.html")
        
        # Verify files exist
        assert (tmp_path / "static" / "comparison.png").exists()
        assert (tmp_path / "static" / "multi_metric.png").exists()
        assert (tmp_path / "static" / "heatmap.png").exists()
        assert (tmp_path / "interactive" / "bar_chart.html").exists()
        
        plt.close('all')
    
    def test_with_result_files(self, tmp_path):
        """Test loading from JSON files and visualizing."""
        # Create mock result files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        results = [
            {
                'result_format_version': '1.0',
                '_filename': 'exp1',
                'MRR': 0.85,
                'Recall@5': 0.90,
            },
            {
                'result_format_version': '1.0',
                '_filename': 'exp2',
                'MRR': 0.80,
                'Recall@5': 0.85,
            },
        ]
        
        for i, result in enumerate(results):
            with open(data_dir / f"result_{i}.json", 'w') as f:
                json.dump(result, f)
        
        # Load and visualize
        loaded_results = []
        for file_path in data_dir.glob("*.json"):
            with open(file_path) as f:
                loaded_results.append(json.load(f))
        
        fig = plot_metric_comparison(loaded_results, metric='MRR')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
