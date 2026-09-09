"""Configuration for the ``dataset_sink`` node (persist generated/synthesized data)."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetSinkConfig:
    """Persist the run's (possibly TTS-augmented / answer-augmented) dataset to disk: e.g.
    ``dataset_source → tts → dataset_sink`` writes a prepared dataset with ``audio_path``.
    """

    enabled: bool = False
    path: Optional[str] = None  # None -> <output_dir>/prepared_<experiment_name>.jsonl at run time
    format: str = "jsonl"
    include_audio: bool = True
    include_generated: bool = True
