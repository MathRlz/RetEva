"""Configuration for the ``dataset_sink`` node (persist generated/synthesized data)."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetSinkConfig:
    """Persist the run's (possibly TTS-augmented / answer-augmented) dataset to disk.

    Enables benchmark-prep + synthetic-data graphs: ``dataset_source → tts →
    dataset_sink`` writes a prepared dataset (questions with ``audio_path``); with
    answer generation it also records the generated answers.

    Attributes:
        enabled: Add the ``dataset_sink`` node to the graph.
        path: Output file path. When None, defaults to
            ``<output_dir>/prepared_<experiment_name>.jsonl`` at run time.
        format: Output format (currently ``jsonl``).
        include_audio: Write each question's ``audio_path`` (the synthesized clip).
        include_generated: Write generated answers when answer generation ran.
    """

    enabled: bool = False
    path: Optional[str] = None
    format: str = "jsonl"
    include_audio: bool = True
    include_generated: bool = True
