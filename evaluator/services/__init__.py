"""Service layer for high-level API orchestration."""

from .evaluation_service import run_evaluation, load_dataset_and_build_index

__all__ = [
    "run_evaluation",
    "load_dataset_and_build_index",
]
