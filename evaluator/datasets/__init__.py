"""Dataset modules for the evaluator framework.

This package provides:
- Legacy dataset classes (QueryDataset, AdmedQueryDataset, PubMedQADataset)
- New dataset loaders (local, HuggingFace, custom)
"""

# Re-export all legacy dataset classes and functions for backward compatibility
from .core import (
    QueryDataset,
    AdmedQueryDataset,
    PubMedQADataset,
    get_data_dir,
    get_admed_voice_path,
    load_admed_voice_corpus,
    load_pubmed_qa_dataset,
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
from .profiles import DatasetCapabilityProfile, register_dataset_profile, resolve_dataset_profile, list_known_dataset_names
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
    DatasetRuntimeSpec,
    list_dataset_runtime_specs,
    resolve_dataset_runtime_spec,
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
    # Legacy classes (backward compatibility)
    "QueryDataset",
    "AdmedQueryDataset",
    "PubMedQADataset",
    "get_data_dir",
    "get_admed_voice_path",
    "load_admed_voice_corpus",
    "load_pubmed_qa_dataset",
    # New loaders
    "AudioSample",
    "DatasetLoaderProtocol",
    "GenericDatasetLoader",
    "HuggingFaceDatasetLoader",
    "LocalAudioDatasetLoader",
    "create_dataset_loader",
    # Profiles (backward compat)
    "DatasetCapabilityProfile",
    "register_dataset_profile",
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
    "DatasetRuntimeSpec",
    "list_dataset_runtime_specs",
    "resolve_dataset_runtime_spec",
    "validate_dataset_runtime_config",
    "load_runtime_dataset",
    "load_runtime_datasets",
    "validate_dataset_join",
]
