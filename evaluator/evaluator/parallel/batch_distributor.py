"""Batch distribution utilities for parallel evaluation.

This module provides classes for distributing dataset items across
multiple workers and running evaluation on specific GPU devices.
"""

from typing import List, Any, Dict, Optional, Callable
from dataclasses import dataclass
import math


class BatchDistributor:
    """Distributes items evenly across workers.
    
    Handles the division of dataset items into balanced chunks for
    parallel processing across multiple GPU workers.
    
    Examples:
        Basic distribution::
        
            >>> distributor = BatchDistributor()
            >>> items = list(range(100))
            >>> batches = distributor.distribute(items, num_workers=4)
            >>> len(batches)
            4
            >>> [len(b) for b in batches]
            [25, 25, 25, 25]
        
        Uneven distribution::
        
            >>> items = list(range(10))
            >>> batches = distributor.distribute(items, num_workers=3)
            >>> [len(b) for b in batches]
            [4, 3, 3]
    """
    
    def distribute(self, items: List[Any], num_workers: int) -> List[List[Any]]:
        """Distribute items evenly across workers.
        
        Divides the items into approximately equal chunks. If the number of
        items is not perfectly divisible, earlier workers get one extra item.
        
        Args:
            items: List of items to distribute.
            num_workers: Number of workers to distribute across.
            
        Returns:
            List of lists, one per worker, containing their assigned items.
            
        Raises:
            ValueError: If num_workers <= 0.
        """
        if num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {num_workers}")
        
        if not items:
            return [[] for _ in range(num_workers)]
        
        n = len(items)
        base_size = n // num_workers
        remainder = n % num_workers
        
        batches = []
        start = 0
        for i in range(num_workers):
            # Earlier workers get one extra item if there's a remainder
            size = base_size + (1 if i < remainder else 0)
            batches.append(items[start:start + size])
            start += size
        
        return batches
    
    def distribute_indices(self, total_items: int, num_workers: int) -> List[List[int]]:
        """Distribute item indices evenly across workers.
        
        Similar to distribute(), but operates on indices instead of items.
        Useful when you don't want to copy the actual data.
        
        Args:
            total_items: Total number of items.
            num_workers: Number of workers to distribute across.
            
        Returns:
            List of lists of indices, one per worker.
        """
        return self.distribute(list(range(total_items)), num_workers)


@dataclass
class WorkerResult:
    """Result from a single worker's evaluation.
    
    Attributes:
        worker_id: Identifier of the worker that produced this result.
        device: Device string the worker used (e.g., "cuda:0").
        results: Dictionary of evaluation results.
        num_samples: Number of samples processed by this worker.
        error: Error message if the worker failed, None otherwise.
    """
    worker_id: int
    device: str
    results: Optional[Dict[str, Any]] = None
    num_samples: int = 0
    error: Optional[str] = None


class Worker:
    """Worker that runs evaluation on a specific GPU device.
    
    Encapsulates the logic for running evaluation on a subset of the
    dataset using a specific GPU device.
    
    Attributes:
        worker_id: Unique identifier for this worker.
        device: Device string (e.g., "cuda:0", "cuda:1").
        
    Examples:
        Creating and running a worker::
        
            >>> worker = Worker(worker_id=0, device="cuda:0")
            >>> result = worker.run(
            ...     indices=[0, 1, 2, 3],
            ...     dataset=my_dataset,
            ...     evaluate_fn=my_eval_function,
            ...     **eval_kwargs
            ... )
    """
    
    def __init__(self, worker_id: int, device: str):
        """Initialize worker.
        
        Args:
            worker_id: Unique identifier for this worker.
            device: Device string (e.g., "cuda:0", "cuda:1").
        """
        self.worker_id = worker_id
        self.device = device
    
    def run(
        self,
        indices: List[int],
        dataset: Any,
        evaluate_fn: Callable[..., Dict[str, Any]],
        **kwargs
    ) -> WorkerResult:
        """Run evaluation on assigned indices.
        
        Creates a subset of the dataset using the assigned indices and
        runs the evaluation function on it.
        
        Args:
            indices: List of dataset indices to process.
            dataset: The full dataset (must support __getitem__ and Subset).
            evaluate_fn: Evaluation function to call.
            **kwargs: Additional arguments to pass to evaluate_fn.
            
        Returns:
            WorkerResult containing the evaluation results.
        """
        try:
            from torch.utils.data import Subset
            
            # Create subset of dataset for this worker
            subset = Subset(dataset, indices)
            
            # Run evaluation
            results = evaluate_fn(
                dataset=subset,
                **kwargs
            )
            
            return WorkerResult(
                worker_id=self.worker_id,
                device=self.device,
                results=results,
                num_samples=len(indices),
                error=None
            )
            
        except Exception as e:
            return WorkerResult(
                worker_id=self.worker_id,
                device=self.device,
                results=None,
                num_samples=0,
                error=str(e)
            )


def aggregate_results(worker_results: List[WorkerResult]) -> Dict[str, Any]:
    """Aggregate results from multiple workers.
    
    Combines evaluation results from multiple workers into a single
    result dictionary with weighted averages for metrics.
    
    Args:
        worker_results: List of WorkerResult objects from all workers.
        
    Returns:
        Aggregated evaluation results dictionary.
        
    Raises:
        RuntimeError: If all workers failed.
    """
    successful = [r for r in worker_results if r.error is None and r.results]
    
    if not successful:
        errors = [f"Worker {r.worker_id}: {r.error}" for r in worker_results if r.error]
        raise RuntimeError(f"All workers failed:\n" + "\n".join(errors))
    
    total_samples = sum(r.num_samples for r in successful)
    
    if total_samples == 0:
        return {"total_samples": 0, "parallel": True}
    
    # Aggregate metrics using weighted averages
    aggregated: Dict[str, Any] = {}
    
    # Get all metric keys from first successful result
    first_results = successful[0].results
    
    for key, value in first_results.items():
        if isinstance(value, (int, float)):
            # Weighted average for numeric metrics
            weighted_sum = sum(
                r.results.get(key, 0) * r.num_samples
                for r in successful
                if isinstance(r.results.get(key), (int, float))
            )
            aggregated[key] = weighted_sum / total_samples
        elif key == "pipeline_mode":
            # Keep the pipeline mode from first worker
            aggregated[key] = value
        elif key == "phased":
            aggregated[key] = value
        elif key == "query_traces":
            # Concatenate query traces
            aggregated[key] = []
            for r in successful:
                if "query_traces" in r.results:
                    aggregated[key].extend(r.results["query_traces"])
    
    aggregated["total_samples"] = total_samples
    aggregated["parallel"] = True
    aggregated["num_workers"] = len(successful)
    
    # Report any failed workers
    failed = [r for r in worker_results if r.error]
    if failed:
        aggregated["failed_workers"] = [
            {"worker_id": r.worker_id, "device": r.device, "error": r.error}
            for r in failed
        ]
    
    return aggregated
