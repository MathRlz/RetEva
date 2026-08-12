"""Grid search utilities for hyperparameter exploration.

This module provides utilities for running experiments across multiple
configuration combinations to find optimal settings.
"""

from typing import List, Dict, Any, Iterator
from itertools import product
from dataclasses import replace

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
        """Set a dot-separated nested parameter, rebuilding the dataclass chain."""
        return self._set_nested_param_recursive(config, param_path.split("."), value)

    def _set_nested_param_recursive(self, obj: Any, parts: List[str], value: Any) -> Any:
        if len(parts) == 1:
            return replace(obj, **{parts[0]: value})
        child = getattr(obj, parts[0])
        return replace(obj, **{parts[0]: self._set_nested_param_recursive(child, parts[1:], value)})

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
