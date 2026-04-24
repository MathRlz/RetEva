"""Grid search utilities for hyperparameter exploration.

This module provides utilities for running experiments across multiple
configuration combinations to find optimal settings.
"""

from typing import List, Dict, Any, Iterator, Callable, Optional
from itertools import product
from dataclasses import replace
import json

from ..config import EvaluationConfig
from ..logging_config import get_logger

logger = get_logger(__name__)


class GridSearch:
    """Grid search over configuration parameters.

    Generates configuration combinations for systematic hyperparameter tuning.
    """

    def __init__(self, base_config: EvaluationConfig):
        """Initialize grid search with base configuration.

        Args:
            base_config: Base configuration to modify.
        """
        self.base_config = base_config
        self.param_grid: Dict[str, List[Any]] = {}

    def add_param(self, param_path: str, values: List[Any]) -> 'GridSearch':
        """Add a parameter to grid search.

        Args:
            param_path: Dot-separated path to parameter (e.g., "vector_db.k").
            values: List of values to try for this parameter.

        Returns:
            Self for method chaining.

        Examples:
            >>> grid = GridSearch(base_config)
            >>> grid.add_param("vector_db.k", [5, 10, 20])
            >>> grid.add_param("vector_db.hybrid_dense_weight", [0.3, 0.5, 0.7])
        """
        self.param_grid[param_path] = values
        return self

    def generate_configs(self) -> Iterator[tuple[Dict[str, Any], EvaluationConfig]]:
        """Generate all configuration combinations.

        Yields:
            Tuples of (param_dict, config) for each combination.
        """
        if not self.param_grid:
            logger.warning("No parameters added to grid search")
            yield {}, self.base_config
            return

        # Get all parameter names and value lists
        param_names = list(self.param_grid.keys())
        value_lists = [self.param_grid[name] for name in param_names]

        # Generate all combinations
        for values in product(*value_lists):
            param_dict = dict(zip(param_names, values))

            # Apply parameters to config
            config = self._apply_params(self.base_config, param_dict)

            yield param_dict, config

    def _apply_params(
        self,
        config: EvaluationConfig,
        param_dict: Dict[str, Any]
    ) -> EvaluationConfig:
        """Apply parameter dictionary to configuration.

        Args:
            config: Base configuration.
            param_dict: Parameters to apply.

        Returns:
            New configuration with parameters applied.
        """
        new_config = config

        for param_path, value in param_dict.items():
            new_config = self._set_nested_param(new_config, param_path, value)

        return new_config

    def _set_nested_param(
        self,
        config: EvaluationConfig,
        param_path: str,
        value: Any
    ) -> EvaluationConfig:
        """Set a nested parameter in configuration.

        Args:
            config: Configuration to modify.
            param_path: Dot-separated path to parameter.
            value: Value to set.

        Returns:
            New configuration with parameter set.
        """
        parts = param_path.split(".")

        if len(parts) == 1:
            # Top-level parameter
            return replace(config, **{parts[0]: value})

        # Nested parameter
        parent_attr = parts[0]
        child_path = ".".join(parts[1:])

        # Get parent object
        parent = getattr(config, parent_attr)

        # Recursively set in parent
        if len(parts) == 2:
            # Direct child
            new_parent = replace(parent, **{parts[1]: value})
        else:
            # Deeper nesting
            new_parent = self._set_nested_param_recursive(parent, parts[1:], value)

        # Return config with updated parent
        return replace(config, **{parent_attr: new_parent})

    def _set_nested_param_recursive(
        self,
        obj: Any,
        parts: List[str],
        value: Any
    ) -> Any:
        """Recursively set nested parameter.

        Args:
            obj: Current object.
            parts: Remaining path parts.
            value: Value to set.

        Returns:
            New object with parameter set.
        """
        if len(parts) == 1:
            return replace(obj, **{parts[0]: value})

        child = getattr(obj, parts[0])
        new_child = self._set_nested_param_recursive(child, parts[1:], value)
        return replace(obj, **{parts[0]: new_child})

    def get_size(self) -> int:
        """Get total number of configurations in grid.

        Returns:
            Number of configuration combinations.
        """
        if not self.param_grid:
            return 1

        size = 1
        for values in self.param_grid.values():
            size *= len(values)

        return size

    def summary(self) -> Dict[str, Any]:
        """Get summary of grid search configuration.

        Returns:
            Dictionary with grid search details.
        """
        return {
            "total_configs": self.get_size(),
            "parameters": {
                name: len(values)
                for name, values in self.param_grid.items()
            },
            "param_grid": self.param_grid,
        }


def create_fusion_grid(base_config: EvaluationConfig) -> GridSearch:
    """Create grid search for embedding fusion parameters.

    Args:
        base_config: Base configuration.

    Returns:
        Configured GridSearch instance.
    """
    grid = GridSearch(base_config)
    grid.add_param("embedding_fusion.audio_weight", [0.3, 0.5, 0.7])
    grid.add_param("embedding_fusion.fusion_method", ["weighted", "max_pool", "average"])
    return grid


def create_retrieval_grid(base_config: EvaluationConfig) -> GridSearch:
    """Create grid search for retrieval parameters.

    Args:
        base_config: Base configuration.

    Returns:
        Configured GridSearch instance.
    """
    grid = GridSearch(base_config)
    grid.add_param("vector_db.k", [5, 10, 20])
    grid.add_param("vector_db.hybrid_dense_weight", [0.3, 0.5, 0.7])
    grid.add_param("vector_db.hybrid_fusion_method", ["weighted", "rrf"])
    return grid


def create_advanced_rag_grid(base_config: EvaluationConfig) -> GridSearch:
    """Create grid search for advanced RAG parameters.

    Args:
        base_config: Base configuration.

    Returns:
        Configured GridSearch instance.
    """
    grid = GridSearch(base_config)
    grid.add_param("vector_db.use_mmr", [False, True])
    grid.add_param("vector_db.mmr_lambda", [0.5, 0.6, 0.7])
    grid.add_param("vector_db.query_expansion_enabled", [False, True])
    grid.add_param("vector_db.pseudo_feedback_enabled", [False, True])
    return grid


def run_grid_search(
    grid: GridSearch,
    evaluation_fn: Callable[[EvaluationConfig], Dict[str, Any]],
    save_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Run grid search and collect results.

    Args:
        grid: GridSearch instance.
        evaluation_fn: Function that takes config and returns metrics dict.
        save_path: Optional path to save results.

    Returns:
        List of result dictionaries with params and metrics.

    Examples:
        >>> def my_eval(config):
        ...     # Run evaluation
        ...     return {"recall@5": 0.8, "mrr": 0.7}
        >>>
        >>> grid = create_retrieval_grid(base_config)
        >>> results = run_grid_search(grid, my_eval, "grid_results.json")
    """
    results = []

    total = grid.get_size()
    logger.info(f"Starting grid search with {total} configurations")

    for i, (params, config) in enumerate(grid.generate_configs(), 1):
        logger.info(f"Evaluating configuration {i}/{total}: {params}")

        try:
            metrics = evaluation_fn(config)

            result = {
                "config_id": i,
                "params": params,
                "metrics": metrics,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Configuration {i} failed: {e}")
            result = {
                "config_id": i,
                "params": params,
                "error": str(e),
                "success": False,
            }

        results.append(result)

    # Save results if path provided
    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {save_path}")

    return results


def analyze_grid_results(
    results: List[Dict[str, Any]],
    metric_name: str = "recall@5",
    top_k: int = 5
) -> Dict[str, Any]:
    """Analyze grid search results to find best configurations.

    Args:
        results: Results from run_grid_search.
        metric_name: Metric to optimize.
        top_k: Number of top configurations to return.

    Returns:
        Dictionary with analysis results.
    """
    # Filter successful runs
    successful = [r for r in results if r.get("success", False)]

    if not successful:
        return {
            "best_config": None,
            "top_configs": [],
            "summary": {
                "total_runs": len(results),
                "successful_runs": 0,
                "failed_runs": len(results),
            }
        }

    # Sort by metric
    sorted_results = sorted(
        successful,
        key=lambda r: r["metrics"].get(metric_name, 0),
        reverse=True
    )

    best = sorted_results[0]
    top_configs = sorted_results[:top_k]

    # Compute statistics
    metric_values = [r["metrics"].get(metric_name, 0) for r in successful]

    import numpy as np

    return {
        "best_config": {
            "params": best["params"],
            "metrics": best["metrics"],
            metric_name: best["metrics"].get(metric_name, 0),
        },
        "top_configs": [
            {
                "rank": i + 1,
                "params": r["params"],
                metric_name: r["metrics"].get(metric_name, 0),
            }
            for i, r in enumerate(top_configs)
        ],
        "summary": {
            "total_runs": len(results),
            "successful_runs": len(successful),
            "failed_runs": len(results) - len(successful),
            "metric_stats": {
                "mean": float(np.mean(metric_values)),
                "std": float(np.std(metric_values)),
                "min": float(np.min(metric_values)),
                "max": float(np.max(metric_values)),
            }
        }
    }


def export_best_config(
    results: List[Dict[str, Any]],
    base_config: EvaluationConfig,
    metric_name: str = "recall@5",
    output_path: str = "best_config.yaml"
) -> EvaluationConfig:
    """Export best configuration from grid search results.

    Args:
        results: Results from run_grid_search.
        base_config: Base configuration used in grid search.
        metric_name: Metric to optimize.
        output_path: Path to save configuration YAML.

    Returns:
        Best configuration.
    """
    analysis = analyze_grid_results(results, metric_name=metric_name, top_k=1)

    if not analysis["best_config"]:
        logger.error("No successful configurations found")
        return base_config

    best_params = analysis["best_config"]["params"]

    # Apply best params to base config
    grid = GridSearch(base_config)
    best_config = grid._apply_params(base_config, best_params)

    # Save to YAML
    best_config.to_yaml(output_path)
    logger.info(f"Best configuration saved to {output_path}")
    logger.info(f"Best {metric_name}: {analysis['best_config'][metric_name]:.4f}")

    return best_config
