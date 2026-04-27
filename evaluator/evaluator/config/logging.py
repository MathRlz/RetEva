"""Logging configuration."""
from dataclasses import dataclass


@dataclass
class LoggingConfig:
    """
    Configuration for logging output and levels.
    
    Controls where logs are written and at what verbosity level. Separate
    levels can be set for console and file output.
    
    Attributes:
        log_dir: Directory for log files. Default: "logs".
        console_level: Console logging level. Default: "INFO".
            Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        file_level: File logging level. Default: "DEBUG".
            Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
    
    Examples:
        >>> config = LoggingConfig(console_level="WARNING", file_level="DEBUG")
        >>> level = config.get_console_level()  # Returns logging.WARNING
    """
    log_dir: str = "logs"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    def get_console_level(self) -> int:
        import logging
        return getattr(logging, self.console_level.upper())
    
    def get_file_level(self) -> int:
        import logging
        return getattr(logging, self.file_level.upper())
