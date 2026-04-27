"""Structured logging setup for evaluation framework."""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import time
from functools import wraps


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    experiment_name: str = "evaluation",
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Setup structured logging with both file and console handlers.
    
    Args:
        experiment_name: Name for this experiment run
        log_dir: Directory to store log files
        console_level: Logging level for console output
        file_level: Logging level for file output
    
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_path / f"{timestamp}_{experiment_name}.log"
    
    logger = logging.getLogger("evaluator")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_format = ColoredFormatter(
        '%(levelname)s | %(name)s | %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # File handler with detailed format
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def get_logger(name: str = "evaluator") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


def log_timing(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time."""
    if logger is None:
        logger = get_logger()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator


class TimingContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or get_logger()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"{self.name} completed in {elapsed:.2f}s")
        else:
            self.logger.error(f"{self.name} failed after {elapsed:.2f}s")
        return False


def log_cache_stats(cache_manager, logger: Optional[logging.Logger] = None):
    """Log cache statistics."""
    if logger is None:
        logger = get_logger()
    
    stats = cache_manager.get_cache_stats()
    logger.info("=" * 60)
    logger.info("Cache Statistics:")

    if "sizes_human" in stats:
        logger.info(f"  Total size: {stats['sizes_human']['total']}")
        logger.info("  Breakdown:")
        for cache_type, size in stats['sizes_human'].items():
            if cache_type != 'total':
                count = stats.get('file_counts', {}).get(cache_type, 0)
                logger.info(f"    {cache_type}: {size} ({count} files)")
    else:
        logger.info(f"  Total size: {stats.get('total_size_human', '0 B')}")
        logger.info("  Breakdown:")
        for cache_type, info in stats.get('by_category', {}).items():
            logger.info(f"    {cache_type}: {info.get('size_human', '0 B')} ({info.get('file_count', 0)} files)")

    logger.info("=" * 60)
