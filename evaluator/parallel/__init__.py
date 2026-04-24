"""Multi-GPU parallel evaluation module.

This module provides utilities for distributing evaluation workloads
across multiple GPUs for improved throughput.

Classes:
    ParallelEvaluator: Main class for parallel evaluation across GPUs.
    BatchDistributor: Distributes dataset items evenly across workers.
    Worker: Runs evaluation on a specific GPU device.
"""

from .data_parallel import ParallelEvaluator
from .batch_distributor import BatchDistributor, Worker

__all__ = [
    "ParallelEvaluator",
    "BatchDistributor",
    "Worker",
]
