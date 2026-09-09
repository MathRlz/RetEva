"""Streaming / windowed-execution configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StreamingConfig:
    """Opt-in windowed query-side execution for corpus-scale runs.

    ``window_size`` slices the query set into windows the executor runs one at a time —
    the corpus is embedded/indexed once and shared, so only one window's data is resident
    at a time. ``None`` (default) keeps whole-dataset execution. See
    ``evaluation/executor/streaming.py`` for the driver.
    """

    window_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.window_size is not None and self.window_size <= 0:
            raise ValueError(
                f"streaming.window_size must be a positive int or None, got {self.window_size}"
            )

    @property
    def enabled(self) -> bool:
        """Whether windowed execution is requested."""
        return self.window_size is not None
