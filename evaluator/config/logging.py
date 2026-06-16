"""Logging configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class LoggingConfig:
    """
    Configuration for logging output and levels.

    Controls where logs are written and at what verbosity level. Separate
    levels can be set for console and file output.

    Attributes:
        log_dir: Directory for log files. Default: "logs".
        verbosity: Console verbosity profile — "default" (lifecycle + warnings only),
            "verbose" (+ node/timing/summaries), or "debug" (everything). Default: "default".
        console_level: Explicit console level override (else the profile's base level).
            Optional. Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        file_level: File logging level (the file log keeps the full record). Default: "DEBUG".

    Examples:
        >>> config = LoggingConfig(verbosity="verbose")
        >>> config = LoggingConfig(console_level="WARNING")  # explicit override
    """
    log_dir: str = "logs"
    verbosity: str = "default"
    console_level: Optional[str] = None
    file_level: str = "DEBUG"

    def get_console_level(self) -> Optional[int]:
        import logging
        return None if self.console_level is None else getattr(logging, self.console_level.upper())

    def get_file_level(self) -> int:
        import logging
        return getattr(logging, self.file_level.upper())
