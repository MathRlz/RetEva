"""Tests for statistical significance testing module."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from evaluator.analysis.significance import (
    paired_ttest,
    wilcoxon_test,
    bootstrap_confidence_interval,
    compare_experiments,
    load_results,
    compare_result_files,
    format_comparison_report,
    _find_common_numeric_metrics,
    _extract_per_sample_scores,
)


class TestPairedTTest:
    """Tests for paired_ttest function."""

    def test_identical_scores_returns_zero_tstat(self):
        """Identical scores should have t-stat of NaN (undefined)."""
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        t_stat, p_value = paired_ttest(scores, scores)
        # For identical arrays, t-stat is NaN (0/0) and p-value is NaN
        assert np.isnan(t_stat) or t_stat == 0.0
        assert np.isnan(p_value) or p_value == 1.0

    def test_significantly_different_scores(self):
        """Clearly different distributions should have low p-value."""
        scores_a = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]
        scores_b = [0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8]
        t_stat, p_value = paired_ttest(scores_a, scores_b)
        assert p_value < 0.05

    def test_returns_tuple_of_floats(self):
        """Should return tuple of floats."""
        scores_a = [0.5, 0.6, 0.7]
        scores_b = [0.6, 0.7, 0.8]
        result = paired_ttest(scores_a, scores_b)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_different_lengths_raises_error(self):
        """Should raise ValueError for different length arrays."""
        with pytest.raises(ValueError, match="same length"):
            paired_ttest([0.1, 0.2, 0.3], [0.1, 0.2])

    def test_insufficient_samples_raises_error(self):
        """Should raise ValueError for fewer than 2 samples."""
        with pytest.raises(ValueError, match="at least 2"):
            paired_ttest([0.5], [0.6])


class TestWilcoxonTest:
    """Tests for wilcoxon_test function."""

    def test_identical_scores_returns_high_pvalue(self):
        """Identical scores should have p-value of 1.0."""
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        stat, p_value = wilcoxon_test(scores, scores)
        assert stat == 0.0
        assert p_value == 1.0

    def test_significantly_different_scores(self):
        """Clearly different distributions should have low p-value."""
        scores_a = [0.1, 0.15, 0.12, 0.18, 0.11, 0.14, 0.13, 0.16]
        scores_b = [0.9, 0.85, 0.92, 0.88, 0.91, 0.84, 0.93, 0.86]
        stat, p_value = wilcoxon_test(scores_a, scores_b)
        assert p_value < 0.05

    def test_returns_tuple_of_floats(self):
        """Should return tuple of floats."""
        scores_a = [0.5, 0.6, 0.7, 0.8]
        scores_b = [0.6, 0.7, 0.8, 0.9]
        result = wilcoxon_test(scores_a, scores_b)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_different_lengths_raises_error(self):
        """Should raise ValueError for different length arrays."""
        with pytest.raises(ValueError, match="same length"):
            wilcoxon_test([0.1, 0.2, 0.3], [0.1, 0.2])

    def test_insufficient_samples_raises_error(self):
        """Should raise ValueError for fewer than 2 samples."""
        with pytest.raises(ValueError, match="at least 2"):
            wilcoxon_test([0.5], [0.6])


class TestBootstrapConfidenceInterval:
    """Tests for bootstrap_confidence_interval function."""

    def test_ci_contains_mean(self):
        """CI should typically contain the sample mean."""
        np.random.seed(42)
        scores = list(np.random.normal(0.5, 0.1, 100))
        lower, upper = bootstrap_confidence_interval(scores, random_state=42)
        mean = np.mean(scores)
        assert lower <= mean <= upper

    def test_ci_narrows_with_lower_alpha(self):
        """Lower alpha should give wider CI."""
        scores = list(np.random.normal(0.5, 0.1, 100))
        ci_90 = bootstrap_confidence_interval(scores, alpha=0.10, random_state=42)
        ci_95 = bootstrap_confidence_interval(scores, alpha=0.05, random_state=42)
        # 95% CI should be wider than 90% CI
        assert (ci_95[1] - ci_95[0]) >= (ci_90[1] - ci_90[0])

    def test_returns_tuple_of_floats(self):
        """Should return tuple of floats."""
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = bootstrap_confidence_interval(scores, random_state=42)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_lower_less_than_upper(self):
        """Lower bound should be less than or equal to upper."""
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        lower, upper = bootstrap_confidence_interval(scores, random_state=42)
        assert lower <= upper

    def test_empty_scores_raises_error(self):
        """Should raise ValueError for empty scores."""
        with pytest.raises(ValueError, match="empty"):
            bootstrap_confidence_interval([])

    def test_invalid_alpha_raises_error(self):
        """Should raise ValueError for alpha outside (0, 1)."""
        with pytest.raises(ValueError, match="Alpha"):
            bootstrap_confidence_interval([0.5, 0.6], alpha=0.0)
        with pytest.raises(ValueError, match="Alpha"):
            bootstrap_confidence_interval([0.5, 0.6], alpha=1.0)

    def test_reproducibility_with_random_state(self):
        """Same random_state should give same results."""
        scores = list(np.random.normal(0.5, 0.1, 50))
        ci1 = bootstrap_confidence_interval(scores, random_state=123)
        ci2 = bootstrap_confidence_interval(scores, random_state=123)
        assert ci1 == ci2


class TestCompareExperiments:
    """Tests for compare_experiments function."""

    def test_basic_comparison(self):
        """Should compare aggregate metrics."""
        results_a = {"MRR": 0.8, "Recall@5": 0.9}
        results_b = {"MRR": 0.85, "Recall@5": 0.92}
        
        comparison = compare_experiments(results_a, results_b)
        
        assert "MRR" in comparison
        assert "Recall@5" in comparison
        assert comparison["MRR"]["mean_a"] == 0.8
        assert comparison["MRR"]["mean_b"] == 0.85
        assert abs(comparison["MRR"]["diff"] - 0.05) < 1e-10

    def test_with_per_sample_scores(self):
        """Should perform statistical tests with per-sample scores."""
        results_a = {
            "MRR": 0.5,
            "per_sample": {"MRR": [0.4, 0.5, 0.6, 0.5, 0.5]}
        }
        results_b = {
            "MRR": 0.8,
            "per_sample": {"MRR": [0.7, 0.8, 0.9, 0.8, 0.8]}
        }
        
        comparison = compare_experiments(results_a, results_b)
        
        assert comparison["MRR"]["ttest"] is not None
        assert comparison["MRR"]["wilcoxon"] is not None
        assert "t_stat" in comparison["MRR"]["ttest"]
        assert "p_value" in comparison["MRR"]["ttest"]

    def test_with_specific_metrics(self):
        """Should only compare specified metrics."""
        results_a = {"MRR": 0.8, "Recall@5": 0.9, "WER": 0.1}
        results_b = {"MRR": 0.85, "Recall@5": 0.92, "WER": 0.08}
        
        comparison = compare_experiments(
            results_a, results_b, metric_names=["MRR"]
        )
        
        assert "MRR" in comparison
        assert "Recall@5" not in comparison
        assert "WER" not in comparison

    def test_skips_non_numeric_fields(self):
        """Should skip non-numeric fields like model names."""
        results_a = {"asr": "whisper", "MRR": 0.8}
        results_b = {"asr": "wav2vec", "MRR": 0.85}
        
        comparison = compare_experiments(results_a, results_b)
        
        assert "asr" not in comparison
        assert "MRR" in comparison


class TestFindCommonNumericMetrics:
    """Tests for _find_common_numeric_metrics helper."""

    def test_finds_common_numeric_keys(self):
        """Should find keys that are numeric in both dicts."""
        a = {"MRR": 0.8, "WER": 0.1, "name": "exp_a"}
        b = {"MRR": 0.85, "WER": 0.08, "name": "exp_b"}
        
        metrics = _find_common_numeric_metrics(a, b)
        
        assert "MRR" in metrics
        assert "WER" in metrics
        assert "name" not in metrics

    def test_skips_reserved_keys(self):
        """Should skip reserved keys like 'per_sample'."""
        a = {"MRR": 0.8, "per_sample": {"x": [1, 2]}}
        b = {"MRR": 0.85, "per_sample": {"x": [2, 3]}}
        
        metrics = _find_common_numeric_metrics(a, b)
        
        assert "per_sample" not in metrics


class TestExtractPerSampleScores:
    """Tests for _extract_per_sample_scores helper."""

    def test_extracts_from_per_sample_key(self):
        """Should extract scores from 'per_sample' key."""
        results = {
            "MRR": 0.8,
            "per_sample": {"MRR": [0.7, 0.8, 0.9]}
        }
        
        scores = _extract_per_sample_scores(results)
        
        assert "MRR" in scores
        assert scores["MRR"] == [0.7, 0.8, 0.9]

    def test_extracts_from_details_list(self):
        """Should extract scores from 'details' list format."""
        results = {
            "MRR": 0.8,
            "details": [
                {"MRR": 0.7, "query": "q1"},
                {"MRR": 0.8, "query": "q2"},
                {"MRR": 0.9, "query": "q3"},
            ]
        }
        
        scores = _extract_per_sample_scores(results)
        
        assert "MRR" in scores
        assert scores["MRR"] == [0.7, 0.8, 0.9]

    def test_returns_empty_when_no_per_sample(self):
        """Should return empty dict when no per-sample data."""
        results = {"MRR": 0.8}
        
        scores = _extract_per_sample_scores(results)
        
        assert scores == {}


class TestLoadResults:
    """Tests for load_results function."""

    def test_loads_valid_json(self):
        """Should load valid JSON file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"MRR": 0.85}, f)
            path = f.name
        
        try:
            results = load_results(path)
            assert results == {"MRR": 0.85}
        finally:
            Path(path).unlink()

    def test_raises_on_missing_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_results("/nonexistent/path.json")

    def test_raises_on_invalid_json(self):
        """Should raise JSONDecodeError for invalid JSON."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json")
            path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_results(path)
        finally:
            Path(path).unlink()


class TestCompareResultFiles:
    """Tests for compare_result_files function."""

    def test_compares_two_files(self):
        """Should compare two result files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "results_a.json"
            path_b = Path(tmpdir) / "results_b.json"
            
            path_a.write_text(json.dumps({
                "asr": "whisper",
                "MRR": 0.8,
                "Recall@5": 0.9
            }))
            path_b.write_text(json.dumps({
                "asr": "wav2vec",
                "MRR": 0.85,
                "Recall@5": 0.92
            }))
            
            comparison = compare_result_files(path_a, path_b)
            
            assert comparison["experiment_a"]["asr"] == "whisper"
            assert comparison["experiment_b"]["asr"] == "wav2vec"
            assert "MRR" in comparison["metrics"]
            assert "Recall@5" in comparison["metrics"]


class TestFormatComparisonReport:
    """Tests for format_comparison_report function."""

    def test_formats_basic_comparison(self):
        """Should format comparison as readable text."""
        comparison = {
            "experiment_a": {"path": "a.json", "asr": "whisper"},
            "experiment_b": {"path": "b.json", "asr": "wav2vec"},
            "metrics": {
                "MRR": {
                    "mean_a": 0.8,
                    "mean_b": 0.85,
                    "diff": 0.05,
                    "ttest": {"t_stat": -2.5, "p_value": 0.03},
                    "wilcoxon": {"stat": 5.0, "p_value": 0.04},
                    "significant_ttest": True,
                    "significant_wilcoxon": True,
                    "ci_a": (0.75, 0.85),
                    "ci_b": (0.80, 0.90),
                }
            }
        }
        
        report = format_comparison_report(comparison)
        
        assert "EXPERIMENT COMPARISON REPORT" in report
        assert "MRR" in report
        assert "whisper" in report
        assert "wav2vec" in report
        assert "t-test" in report
        assert "Wilcoxon" in report
        assert "✓" in report  # Significant marker

    def test_handles_missing_tests(self):
        """Should handle comparison without statistical tests."""
        comparison = {
            "experiment_a": {"path": "a.json"},
            "experiment_b": {"path": "b.json"},
            "metrics": {
                "MRR": {
                    "mean_a": 0.8,
                    "mean_b": 0.85,
                    "diff": 0.05,
                    "ttest": None,
                    "wilcoxon": None,
                    "significant_ttest": False,
                    "significant_wilcoxon": False,
                    "ci_a": None,
                    "ci_b": None,
                }
            }
        }
        
        report = format_comparison_report(comparison)
        
        assert "MRR" in report
        assert "0.8000" in report
        assert "0.8500" in report


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_comparison_workflow(self):
        """Test complete workflow from files to formatted report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files with per-sample data
            path_a = Path(tmpdir) / "exp_a.json"
            path_b = Path(tmpdir) / "exp_b.json"
            
            # Experiment A: lower MRR
            np.random.seed(42)
            scores_a = list(np.random.normal(0.7, 0.1, 20))
            path_a.write_text(json.dumps({
                "asr": "WhisperModel - openai/whisper-large",
                "embedder": "LaBSE",
                "MRR": float(np.mean(scores_a)),
                "Recall@5": 0.85,
                "per_sample": {"MRR": scores_a}
            }))
            
            # Experiment B: higher MRR
            scores_b = list(np.random.normal(0.85, 0.1, 20))
            path_b.write_text(json.dumps({
                "asr": "WhisperModel - openai/whisper-large",
                "embedder": "Jina-v4",
                "MRR": float(np.mean(scores_b)),
                "Recall@5": 0.92,
                "per_sample": {"MRR": scores_b}
            }))
            
            # Run comparison
            comparison = compare_result_files(path_a, path_b)
            
            # Verify structure
            assert "experiment_a" in comparison
            assert "experiment_b" in comparison
            assert "metrics" in comparison
            
            # Verify metrics comparison
            mrr_cmp = comparison["metrics"]["MRR"]
            assert mrr_cmp["mean_a"] is not None
            assert mrr_cmp["mean_b"] is not None
            assert mrr_cmp["ttest"] is not None
            assert mrr_cmp["ci_a"] is not None
            
            # Format report
            report = format_comparison_report(comparison)
            assert "MRR" in report
            assert "LaBSE" in report
            assert "Jina-v4" in report
