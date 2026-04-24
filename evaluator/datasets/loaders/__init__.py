"""Dataset loaders for various data sources.

This module provides loaders for:
- Local audio directories
- HuggingFace Hub datasets
- Custom dataset formats
"""

from .base import AudioSample, DatasetLoaderProtocol, GenericDatasetLoader
from .huggingface import HuggingFaceDatasetLoader
from .local import LocalAudioDatasetLoader
from .factory import create_dataset_loader

__all__ = [
    "AudioSample",
    "DatasetLoaderProtocol",
    "GenericDatasetLoader",
    "HuggingFaceDatasetLoader",
    "LocalAudioDatasetLoader",
    "create_dataset_loader",
]
