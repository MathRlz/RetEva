"""D3/F8: per-source loader builders + the create_dataset_loader dispatch shim.

The flat 15-param factory is now a thin dispatcher over create_local_loader /
create_huggingface_loader / create_custom_loader. Pins the dispatch, the per-source
required-arg errors, and that the dispatcher delegates to the focused builders.
"""

import pytest

from evaluator.datasets.loaders import factory as F
from evaluator.datasets.loaders.base import GenericDatasetLoader
from evaluator.datasets.loaders.huggingface import HuggingFaceDatasetLoader
from evaluator.datasets.loaders.local import LocalAudioDatasetLoader


def test_local_requires_audio_dir():
    with pytest.raises(ValueError, match="audio_dir is required"):
        F.create_local_loader(None)
    with pytest.raises(ValueError, match="audio_dir is required"):
        F.create_dataset_loader(source="local")


def test_huggingface_requires_dataset():
    with pytest.raises(ValueError, match="huggingface_dataset is required"):
        F.create_huggingface_loader(None)
    with pytest.raises(ValueError, match="huggingface_dataset is required"):
        F.create_dataset_loader(source="huggingface")


def test_custom_requires_data_and_map_fn():
    with pytest.raises(ValueError, match="'data' and 'map_fn'"):
        F.create_custom_loader(None, None)


def test_unknown_source_rejected():
    with pytest.raises(ValueError, match="Unknown dataset source"):
        F.create_dataset_loader(source="nope")


def test_local_builder_and_dispatch_agree(tmp_path):
    direct = F.create_local_loader(str(tmp_path), recursive=True)
    viashim = F.create_dataset_loader(source="local", audio_dir=str(tmp_path), recursive=True)
    assert isinstance(direct, LocalAudioDatasetLoader)
    assert isinstance(viashim, LocalAudioDatasetLoader)
    assert direct.recursive is viashim.recursive is True


def test_huggingface_builder_passes_options():
    loader = F.create_huggingface_loader(
        "google/fleurs", huggingface_subset="en_us", huggingface_split="train",
        max_samples=10,
    )
    assert isinstance(loader, HuggingFaceDatasetLoader)
    assert loader.dataset_name == "google/fleurs"
    assert loader.split == "train"
    assert loader.max_samples == 10


def test_custom_builder_returns_generic_loader():
    loader = F.create_custom_loader([1, 2, 3], map_fn=lambda x: x, filter_fn=lambda x: True)
    assert isinstance(loader, GenericDatasetLoader)
