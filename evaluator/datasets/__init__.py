"""Dataset modules for the evaluator framework.

This package provides:
- The ``QueryDataset`` base + the per-type ABCs / descriptor registry (the extension API)
- Dataset loaders (local, HuggingFace, custom)
"""

from .core import (
    QueryDataset,
    get_data_dir,
    get_admed_voice_path,
    load_admed_voice_corpus,
)

# New dataset loaders
from .loaders import (
    AudioSample,
    DatasetLoaderProtocol,
    GenericDatasetLoader,
    HuggingFaceDatasetLoader,
    LocalAudioDatasetLoader,
    create_dataset_loader,
)
from .profiles import DatasetCapabilityProfile, resolve_dataset_profile, list_known_dataset_names
from .descriptor import (
    DatasetDescriptor,
    METRICS_BY_MODE,
    register_dataset,
    get_descriptor,
    list_registered_datasets,
    resolve_dataset_descriptor,
)
from .runtime import (
    AudioSamplesQueryDataset,
    LazyAudioQueryDataset,
    validate_dataset_runtime_config,
    load_runtime_dataset,
    load_runtime_datasets,
    validate_dataset_join,
)
from .types import (
    EvalDataset,
    AudioTranscriptionDataset,
    AudioRetrievalDataset,
    TextRetrievalDataset,
    MultimodalQADataset,
    register_eval_dataset,
)
from . import builtins  # noqa: F401  — registers the built-in datasets (side-effect)

__all__ = [
    "QueryDataset",
    "get_data_dir",
    "get_admed_voice_path",
    "load_admed_voice_corpus",
    # Loaders
    "AudioSample",
    "DatasetLoaderProtocol",
    "GenericDatasetLoader",
    "HuggingFaceDatasetLoader",
    "LocalAudioDatasetLoader",
    "create_dataset_loader",
    # Profiles (backward compat)
    "DatasetCapabilityProfile",
    "resolve_dataset_profile",
    "list_known_dataset_names",
    # Per-type ABCs (new extension API)
    "EvalDataset",
    "AudioTranscriptionDataset",
    "AudioRetrievalDataset",
    "TextRetrievalDataset",
    "MultimodalQADataset",
    "register_eval_dataset",
    # Unified descriptor registry (new API)
    "DatasetDescriptor",
    "METRICS_BY_MODE",
    "register_dataset",
    "get_descriptor",
    "list_registered_datasets",
    "resolve_dataset_descriptor",
    # Runtime
    "AudioSamplesQueryDataset",
    "LazyAudioQueryDataset",
    "validate_dataset_runtime_config",
    "load_runtime_dataset",
    "load_runtime_datasets",
    "validate_dataset_join",
]
