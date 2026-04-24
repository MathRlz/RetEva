"""Calibration utilities for LLM judge scoring.

This module provides utilities for calibrating and normalizing LLM judge
scores to improve consistency and interpretability.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

from ..logging_config import get_logger

logger = get_logger(__name__)


class JudgeCalibrator:
    """Calibrates LLM judge scores using reference data.
    
    Learns score distribution and applies normalization to improve
    consistency across different queries and models.
    """
    
    def __init__(self):
        """Initialize calibrator."""
        self.score_history: List[float] = []
        self.aspect_scores: Dict[str, List[float]] = defaultdict(list)
        self.is_fitted = False
        
        # Calibration parameters
        self.mean_score: float = 3.0
        self.std_score: float = 1.0
        self.aspect_means: Dict[str, float] = {}
        self.aspect_stds: Dict[str, float] = {}
    
    def add_score(self, score: float, aspect: str = "overall"):
        """Add a score to the calibration history.
        
        Args:
            score: Judge score (typically 1-5).
            aspect: Aspect being scored.
        """
        self.score_history.append(score)
        self.aspect_scores[aspect].append(score)
    
    def fit(self, min_samples: int = 10):
        """Fit calibration parameters from score history.
        
        Args:
            min_samples: Minimum number of samples needed to fit.
            
        Returns:
            True if successfully fitted, False otherwise.
        """
        if len(self.score_history) < min_samples:
            logger.warning(
                f"Insufficient samples for calibration: {len(self.score_history)} < {min_samples}"
            )
            return False
        
        # Compute overall statistics
        self.mean_score = np.mean(self.score_history)
        self.std_score = np.std(self.score_history)
        
        if self.std_score < 0.1:
            self.std_score = 1.0  # Avoid division by zero
        
        # Compute aspect-specific statistics
        for aspect, scores in self.aspect_scores.items():
            if len(scores) >= min_samples:
                self.aspect_means[aspect] = np.mean(scores)
                self.aspect_stds[aspect] = np.std(scores)
                
                if self.aspect_stds[aspect] < 0.1:
                    self.aspect_stds[aspect] = 1.0
        
        self.is_fitted = True
        logger.info(
            f"Calibrator fitted with {len(self.score_history)} samples: "
            f"mean={self.mean_score:.2f}, std={self.std_score:.2f}"
        )
        return True
    
    def calibrate_score(
        self,
        score: float,
        aspect: str = "overall",
        target_mean: float = 3.0,
        target_std: float = 1.0
    ) -> float:
        """Calibrate a score using learned statistics.
        
        Args:
            score: Raw score to calibrate.
            aspect: Aspect being scored.
            target_mean: Target mean for calibrated scores.
            target_std: Target standard deviation.
            
        Returns:
            Calibrated score.
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning original score")
            return score
        
        # Use aspect-specific or overall statistics
        if aspect in self.aspect_means:
            mean = self.aspect_means[aspect]
            std = self.aspect_stds[aspect]
        else:
            mean = self.mean_score
            std = self.std_score
        
        # Z-score normalization
        z_score = (score - mean) / std
        
        # Scale to target distribution
        calibrated = target_mean + z_score * target_std
        
        # Clip to valid range [1, 5]
        calibrated = np.clip(calibrated, 1.0, 5.0)
        
        return float(calibrated)
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get calibration statistics.
        
        Returns:
            Dictionary of statistics per aspect.
        """
        stats = {
            "overall": {
                "mean": self.mean_score,
                "std": self.std_score,
                "count": len(self.score_history),
            }
        }
        
        for aspect, scores in self.aspect_scores.items():
            stats[aspect] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "count": len(scores),
            }
        
        return stats


def detect_score_bias(
    scores: List[float],
    expected_mean: float = 3.0,
    threshold: float = 0.5
) -> Tuple[bool, float]:
    """Detect if scores show systematic bias.
    
    Args:
        scores: List of judge scores.
        expected_mean: Expected mean score.
        threshold: Threshold for detecting bias.
        
    Returns:
        Tuple of (is_biased, bias_amount).
    """
    if not scores:
        return False, 0.0
    
    actual_mean = np.mean(scores)
    bias = actual_mean - expected_mean
    
    is_biased = abs(bias) > threshold
    
    if is_biased:
        logger.warning(
            f"Score bias detected: mean={actual_mean:.2f}, "
            f"expected={expected_mean:.2f}, bias={bias:+.2f}"
        )
    
    return is_biased, bias


def compute_inter_rater_agreement(
    scores_a: List[float],
    scores_b: List[float]
) -> Dict[str, float]:
    """Compute agreement between two sets of scores.
    
    Args:
        scores_a: First set of scores.
        scores_b: Second set of scores.
        
    Returns:
        Dictionary with agreement metrics.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length")
    
    if not scores_a:
        return {"correlation": 0.0, "mae": 0.0, "exact_agreement": 0.0}
    
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    
    # Pearson correlation
    correlation = np.corrcoef(scores_a, scores_b)[0, 1]
    
    # Mean absolute error
    mae = np.mean(np.abs(scores_a - scores_b))
    
    # Exact agreement rate (same integer score)
    exact_matches = np.sum(np.round(scores_a) == np.round(scores_b))
    exact_agreement = exact_matches / len(scores_a)
    
    return {
        "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
        "mae": float(mae),
        "exact_agreement": float(exact_agreement),
    }


def normalize_scores_min_max(
    scores: List[float],
    target_min: float = 1.0,
    target_max: float = 5.0
) -> List[float]:
    """Normalize scores to target range using min-max scaling.
    
    Args:
        scores: List of scores to normalize.
        target_min: Target minimum value.
        target_max: Target maximum value.
        
    Returns:
        Normalized scores.
    """
    if not scores:
        return []
    
    scores_array = np.array(scores)
    
    min_score = scores_array.min()
    max_score = scores_array.max()
    
    if max_score - min_score < 1e-6:
        # All scores are the same, return target middle
        return [target_min + (target_max - target_min) / 2] * len(scores)
    
    # Min-max normalization
    normalized = (scores_array - min_score) / (max_score - min_score)
    normalized = normalized * (target_max - target_min) + target_min
    
    return normalized.tolist()


def apply_temperature_scaling(
    scores: List[float],
    temperature: float = 1.0
) -> List[float]:
    """Apply temperature scaling to soften or sharpen score distribution.
    
    Args:
        scores: List of scores.
        temperature: Temperature parameter (>1 = softer, <1 = sharper).
        
    Returns:
        Temperature-scaled scores.
    """
    if not scores or temperature <= 0:
        return scores
    
    scores_array = np.array(scores)
    
    # Center scores around mean
    mean_score = scores_array.mean()
    centered = scores_array - mean_score
    
    # Apply temperature
    scaled = centered / temperature
    
    # Re-center and clip
    result = scaled + mean_score
    result = np.clip(result, 1.0, 5.0)
    
    return result.tolist()


def ensemble_judge_scores(
    score_sets: List[List[float]],
    method: str = "mean",
    weights: Optional[List[float]] = None
) -> List[float]:
    """Ensemble multiple sets of judge scores.
    
    Combines scores from multiple judges or models.
    
    Args:
        score_sets: List of score lists (one per judge/model).
        method: Combination method. Options: "mean", "median", "weighted".
        weights: Optional weights for weighted combination.
        
    Returns:
        Ensembled scores.
    """
    if not score_sets:
        return []
    
    # Check all sets have same length
    n_scores = len(score_sets[0])
    if not all(len(s) == n_scores for s in score_sets):
        raise ValueError("All score sets must have same length")
    
    # Convert to array
    scores_array = np.array(score_sets)  # Shape: (n_judges, n_scores)
    
    if method == "mean":
        return np.mean(scores_array, axis=0).tolist()
    
    elif method == "median":
        return np.median(scores_array, axis=0).tolist()
    
    elif method == "weighted":
        if weights is None:
            raise ValueError("weights must be provided for weighted ensemble")
        
        if len(weights) != len(score_sets):
            raise ValueError("Number of weights must match number of score sets")
        
        weights_array = np.array(weights).reshape(-1, 1)
        weighted_scores = np.sum(scores_array * weights_array, axis=0)
        
        return weighted_scores.tolist()
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


class ScoreTracker:
    """Tracks and analyzes judge scores over time."""
    
    def __init__(self):
        """Initialize tracker."""
        self.scores: List[float] = []
        self.aspects: List[str] = []
        self.queries: List[str] = []
        self.timestamps: List[float] = []
    
    def add(
        self,
        score: float,
        aspect: str = "overall",
        query: str = "",
        timestamp: Optional[float] = None
    ):
        """Add a score to the tracker.
        
        Args:
            score: Judge score.
            aspect: Aspect being scored.
            query: Query associated with score.
            timestamp: Optional timestamp (defaults to current time).
        """
        import time
        
        self.scores.append(score)
        self.aspects.append(aspect)
        self.queries.append(query)
        self.timestamps.append(timestamp or time.time())
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary statistics.
        
        Returns:
            Dictionary of summary statistics.
        """
        if not self.scores:
            return {}
        
        scores_array = np.array(self.scores)
        
        summary = {
            "count": len(self.scores),
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "median": float(np.median(scores_array)),
        }
        
        # Per-aspect statistics
        aspect_stats = {}
        for aspect in set(self.aspects):
            aspect_scores = [s for s, a in zip(self.scores, self.aspects) if a == aspect]
            if aspect_scores:
                aspect_stats[aspect] = {
                    "count": len(aspect_scores),
                    "mean": float(np.mean(aspect_scores)),
                    "std": float(np.std(aspect_scores)),
                }
        
        summary["aspects"] = aspect_stats
        
        return summary
