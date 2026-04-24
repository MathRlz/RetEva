"""Statistical significance testing for metric comparisons.

Provides functions for comparing evaluation results between experiments using
parametric (t-test) and non-parametric (Wilcoxon) tests, as well as bootstrap
confidence intervals.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


def paired_ttest(
    scores_a: List[float], scores_b: List[float]
) -> Tuple[float, float]:
    """Perform paired t-test between two sets of scores.
    
    Tests whether the mean difference between paired observations is
    significantly different from zero. Assumes normally distributed differences.
    
    Args:
        scores_a: First set of scores (per-sample metrics from experiment A).
        scores_b: Second set of scores (per-sample metrics from experiment B).
        
    Returns:
        Tuple of (t_statistic, p_value).
        
    Raises:
        ValueError: If inputs have different lengths or fewer than 2 samples.
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score arrays must have same length: {len(scores_a)} vs {len(scores_b)}"
        )
    if len(scores_a) < 2:
        raise ValueError("Need at least 2 samples for paired t-test")
    
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return float(t_stat), float(p_value)


def wilcoxon_test(
    scores_a: List[float], scores_b: List[float]
) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test between two sets of scores.
    
    Non-parametric alternative to paired t-test. Does not assume normal
    distribution of differences. Tests whether the distribution of differences
    is symmetric around zero.
    
    Args:
        scores_a: First set of scores (per-sample metrics from experiment A).
        scores_b: Second set of scores (per-sample metrics from experiment B).
        
    Returns:
        Tuple of (statistic, p_value).
        
    Raises:
        ValueError: If inputs have different lengths or fewer than 2 samples.
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score arrays must have same length: {len(scores_a)} vs {len(scores_b)}"
        )
    if len(scores_a) < 2:
        raise ValueError("Need at least 2 samples for Wilcoxon test")
    
    # Handle case where all differences are zero
    differences = scores_a - scores_b
    if np.allclose(differences, 0):
        return 0.0, 1.0
    
    try:
        stat, p_value = stats.wilcoxon(scores_a, scores_b, zero_method='wilcox')
    except ValueError:
        # Wilcoxon can fail if all differences are zero after rounding
        return 0.0, 1.0
    
    return float(stat), float(p_value)


def bootstrap_confidence_interval(
    scores: List[float],
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for mean of scores.
    
    Uses percentile bootstrap method to estimate confidence interval.
    
    Args:
        scores: Sample scores to compute CI for.
        alpha: Significance level (default 0.05 for 95% CI).
        n_bootstrap: Number of bootstrap iterations.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval.
        
    Raises:
        ValueError: If scores is empty or alpha not in (0, 1).
    """
    scores = np.asarray(scores)
    
    if len(scores) == 0:
        raise ValueError("Cannot compute CI for empty score array")
    if not 0 < alpha < 1:
        raise ValueError(f"Alpha must be in (0, 1), got {alpha}")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(scores)
    bootstrap_means = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        resample_indices = np.random.randint(0, n_samples, size=n_samples)
        bootstrap_means[i] = np.mean(scores[resample_indices])
    
    # Compute percentile confidence interval
    lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    
    return float(lower), float(upper)


def compare_experiments(
    results_a: Dict[str, Any],
    results_b: Dict[str, Any],
    metric_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Compare two experiment results across multiple metrics.
    
    For each metric, performs paired t-test and Wilcoxon test if per-sample
    scores are available, otherwise compares aggregate values.
    
    Args:
        results_a: Results dictionary from first experiment. Expected format:
            - Aggregate metrics as top-level keys (e.g., "MRR": 0.85)
            - Optional "per_sample" or "details" key with per-query scores
        results_b: Results dictionary from second experiment.
        metric_names: List of metric names to compare. If None, compares all
            common numeric metrics.
            
    Returns:
        Dictionary mapping metric names to comparison results:
        {
            "metric_name": {
                "mean_a": float,
                "mean_b": float,
                "diff": float,
                "ttest": {"t_stat": float, "p_value": float} or None,
                "wilcoxon": {"stat": float, "p_value": float} or None,
                "significant_ttest": bool,
                "significant_wilcoxon": bool,
                "ci_a": (lower, upper) or None,
                "ci_b": (lower, upper) or None,
            }
        }
    """
    comparison = {}
    
    # Find common metrics if not specified
    if metric_names is None:
        metric_names = _find_common_numeric_metrics(results_a, results_b)
    
    # Extract per-sample scores if available
    per_sample_a = _extract_per_sample_scores(results_a)
    per_sample_b = _extract_per_sample_scores(results_b)
    
    for metric in metric_names:
        metric_result: Dict[str, Any] = {
            "mean_a": None,
            "mean_b": None,
            "diff": None,
            "ttest": None,
            "wilcoxon": None,
            "significant_ttest": False,
            "significant_wilcoxon": False,
            "ci_a": None,
            "ci_b": None,
        }
        
        # Get aggregate values
        val_a = results_a.get(metric)
        val_b = results_b.get(metric)
        
        if val_a is not None and val_b is not None:
            metric_result["mean_a"] = float(val_a)
            metric_result["mean_b"] = float(val_b)
            metric_result["diff"] = float(val_b) - float(val_a)
        
        # Check for per-sample scores
        scores_a = per_sample_a.get(metric)
        scores_b = per_sample_b.get(metric)
        
        if scores_a is not None and scores_b is not None:
            if len(scores_a) == len(scores_b) and len(scores_a) >= 2:
                # Perform statistical tests
                try:
                    t_stat, t_pval = paired_ttest(scores_a, scores_b)
                    metric_result["ttest"] = {"t_stat": t_stat, "p_value": t_pval}
                    metric_result["significant_ttest"] = t_pval < 0.05
                except Exception:
                    pass
                
                try:
                    w_stat, w_pval = wilcoxon_test(scores_a, scores_b)
                    metric_result["wilcoxon"] = {"stat": w_stat, "p_value": w_pval}
                    metric_result["significant_wilcoxon"] = w_pval < 0.05
                except Exception:
                    pass
            
            # Compute confidence intervals
            if len(scores_a) >= 2:
                try:
                    metric_result["ci_a"] = bootstrap_confidence_interval(
                        scores_a, random_state=42
                    )
                except Exception:
                    pass
            
            if len(scores_b) >= 2:
                try:
                    metric_result["ci_b"] = bootstrap_confidence_interval(
                        scores_b, random_state=42
                    )
                except Exception:
                    pass
        
        comparison[metric] = metric_result
    
    return comparison


def _find_common_numeric_metrics(
    results_a: Dict[str, Any], results_b: Dict[str, Any]
) -> List[str]:
    """Find common numeric metrics between two result dictionaries."""
    skip_keys = {"asr", "embedder", "per_sample", "details", "config", "metadata"}
    
    metrics = []
    for key in results_a:
        if key in skip_keys:
            continue
        val_a = results_a.get(key)
        val_b = results_b.get(key)
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            metrics.append(key)
    
    return metrics


def _extract_per_sample_scores(
    results: Dict[str, Any]
) -> Dict[str, List[float]]:
    """Extract per-sample scores from results dictionary.
    
    Looks for scores in common formats:
    - results["per_sample"][metric] = [scores...]
    - results["details"][i][metric] = score
    """
    per_sample: Dict[str, List[float]] = {}
    
    # Check for explicit per_sample key
    if "per_sample" in results and isinstance(results["per_sample"], dict):
        for metric, scores in results["per_sample"].items():
            if isinstance(scores, list):
                per_sample[metric] = [float(s) for s in scores if s is not None]
    
    # Check for details list
    elif "details" in results and isinstance(results["details"], list):
        details = results["details"]
        if details and isinstance(details[0], dict):
            # Infer metrics from first item
            sample_keys = [
                k for k, v in details[0].items() 
                if isinstance(v, (int, float))
            ]
            for metric in sample_keys:
                scores = []
                for item in details:
                    if metric in item and item[metric] is not None:
                        scores.append(float(item[metric]))
                if scores:
                    per_sample[metric] = scores
    
    return per_sample


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """Load evaluation results from JSON file.
    
    Args:
        path: Path to JSON results file.
        
    Returns:
        Results dictionary.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_result_files(
    path_a: Union[str, Path],
    path_b: Union[str, Path],
    metric_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compare two result files and generate comparison report.
    
    Args:
        path_a: Path to first results file.
        path_b: Path to second results file.
        metric_names: Optional list of metrics to compare.
        
    Returns:
        Comparison report dictionary with experiment metadata and metrics.
    """
    results_a = load_results(path_a)
    results_b = load_results(path_b)
    
    comparison = compare_experiments(results_a, results_b, metric_names)
    
    return {
        "experiment_a": {
            "path": str(path_a),
            "asr": results_a.get("asr"),
            "embedder": results_a.get("embedder"),
        },
        "experiment_b": {
            "path": str(path_b),
            "asr": results_b.get("asr"),
            "embedder": results_b.get("embedder"),
        },
        "metrics": comparison,
    }


def format_comparison_report(comparison: Dict[str, Any]) -> str:
    """Format comparison results as human-readable text report.
    
    Args:
        comparison: Comparison dictionary from compare_result_files.
        
    Returns:
        Formatted string report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Experiment info
    exp_a = comparison.get("experiment_a", {})
    exp_b = comparison.get("experiment_b", {})
    
    lines.append("Experiment A:")
    lines.append(f"  Path: {exp_a.get('path', 'N/A')}")
    lines.append(f"  ASR: {exp_a.get('asr', 'N/A')}")
    lines.append(f"  Embedder: {exp_a.get('embedder', 'N/A')}")
    lines.append("")
    
    lines.append("Experiment B:")
    lines.append(f"  Path: {exp_b.get('path', 'N/A')}")
    lines.append(f"  ASR: {exp_b.get('asr', 'N/A')}")
    lines.append(f"  Embedder: {exp_b.get('embedder', 'N/A')}")
    lines.append("")
    
    # Metrics comparison
    lines.append("-" * 70)
    lines.append("METRIC COMPARISON")
    lines.append("-" * 70)
    lines.append("")
    
    metrics = comparison.get("metrics", {})
    
    for metric_name, metric_data in metrics.items():
        lines.append(f"📊 {metric_name}")
        
        mean_a = metric_data.get("mean_a")
        mean_b = metric_data.get("mean_b")
        diff = metric_data.get("diff")
        
        if mean_a is not None and mean_b is not None:
            lines.append(f"   A: {mean_a:.4f}  |  B: {mean_b:.4f}  |  Δ: {diff:+.4f}")
            
            # Confidence intervals
            ci_a = metric_data.get("ci_a")
            ci_b = metric_data.get("ci_b")
            if ci_a:
                lines.append(f"   95% CI A: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
            if ci_b:
                lines.append(f"   95% CI B: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
        
        # Statistical tests
        ttest = metric_data.get("ttest")
        wilcoxon = metric_data.get("wilcoxon")
        
        if ttest:
            sig = "✓" if metric_data.get("significant_ttest") else "✗"
            lines.append(
                f"   t-test: t={ttest['t_stat']:.3f}, p={ttest['p_value']:.4f} [{sig}]"
            )
        
        if wilcoxon:
            sig = "✓" if metric_data.get("significant_wilcoxon") else "✗"
            lines.append(
                f"   Wilcoxon: W={wilcoxon['stat']:.3f}, p={wilcoxon['p_value']:.4f} [{sig}]"
            )
        
        lines.append("")
    
    lines.append("=" * 70)
    lines.append("Legend: ✓ = significant at α=0.05, ✗ = not significant")
    lines.append("=" * 70)
    
    return "\n".join(lines)
