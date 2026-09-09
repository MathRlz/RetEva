"""Dataset registration: bind a descriptor, and the loader it carries, into the registry.

The :class:`~evaluator.config.dataset_descriptors.DatasetDescriptor` dataclass and the
id → descriptor lookups live in :mod:`evaluator.config.dataset_descriptors`, since config/
resolves a dataset while translating and validating a config. Registration stays here, with
the loaders it binds; :mod:`.builtins` registers through :func:`.types.register_eval_dataset`.
"""
from __future__ import annotations

from ..config.dataset_descriptors import DESCRIPTOR_REGISTRY, DatasetDescriptor


def register_dataset(descriptor: DatasetDescriptor) -> DatasetDescriptor:
    """Register *descriptor* under its id.  Returns the descriptor unchanged."""
    DESCRIPTOR_REGISTRY[descriptor.id] = descriptor
    return descriptor
