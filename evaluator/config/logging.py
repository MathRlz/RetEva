"""Logging configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class LoggingConfig:
    """Configuration for logging output and levels: separate console/file verbosity."""
    log_dir: str = "logs"
    # default (lifecycle+warnings) | verbose (+node/timing) | debug (everything)
    verbosity: str = "default"
    console_level: Optional[str] = None  # overrides the verbosity profile's level when set
    file_level: str = "DEBUG"  # file log keeps the full record

    def get_console_level(self) -> Optional[int]:
        import logging
        return None if self.console_level is None else getattr(logging, self.console_level.upper())

    def get_file_level(self) -> int:
        import logging
        return getattr(logging, self.file_level.upper())
