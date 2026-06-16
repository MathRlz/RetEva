"""Structured logging setup for the evaluation framework (audit Batch F).

Console verbosity is a **named profile** (`default` / `verbose` / `debug`) rather than a raw
level. Messages are tagged with a **category** by emitting them through a child logger
(`evaluator.node` / `.timing` / `.runtime` / `.cache` / `.diag`); a profile decides which
categories reach the console. The **file** handler always records everything at DEBUG, so a
quiet console never loses post-hoc detail.

Categories (everything not under one of these child loggers is *lifecycle* — always shown):
    node     ``▶ node …``, offload, resume-from-level
    timing   ``… completed in Xs`` (TimingContext / log_timing), ``STAGE … DONE``
    runtime  the ``RUNTIME …`` line, the full ``Configuration`` dump
    cache    the ``Cache Statistics`` breakdown
    diag     retrieval-debug + per-query internals (debug only)

Profiles:
    default  lifecycle + warnings/errors only (third-party at WARNING)
    verbose  + node + timing + one-line runtime/cache summaries
    debug    everything, incl. diag + third-party at INFO + DEBUG on the console
"""
import logging
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional

# Category child loggers — import and use these to tag a message's category.
node_logger = logging.getLogger("evaluator.node")
timing_logger = logging.getLogger("evaluator.timing")
runtime_logger = logging.getLogger("evaluator.runtime")
cache_logger = logging.getLogger("evaluator.cache")
diag_logger = logging.getLogger("evaluator.diag")

_CATEGORIES = ("node", "timing", "runtime", "cache", "diag")

# profile -> (console categories shown, console base level, third-party level)
VERBOSITY_PROFILES = {
    "default": (frozenset({"lifecycle"}), logging.INFO, logging.WARNING),
    "verbose": (
        frozenset({"lifecycle", "node", "timing", "runtime", "cache"}),
        logging.INFO,
        logging.WARNING,
    ),
    "debug": (frozenset({"lifecycle", *_CATEGORIES}), logging.DEBUG, logging.INFO),
}

# Noisy libraries whose loggers we cap so they don't drown the console.
_THIRD_PARTY = (
    "transformers", "torch", "datasets", "httpx", "httpcore", "urllib3",
    "sentence_transformers", "filelock", "huggingface_hub", "fsspec",
)


def _category_of(record_name: str) -> str:
    """The category for a record's logger name (lifecycle for anything uncategorised)."""
    for cat in _CATEGORIES:
        if record_name == f"evaluator.{cat}" or record_name.startswith(f"evaluator.{cat}."):
            return cat
    return "lifecycle"


class _ConsoleCategoryFilter(logging.Filter):
    """Pass a record to the console only if its category is enabled by the profile.
    Warnings and errors always pass — a quiet profile never hides a problem."""

    def __init__(self, allowed: frozenset):
        super().__init__()
        self.allowed = allowed

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return _category_of(record.name) in self.allowed


class ColoredFormatter(logging.Formatter):
    """Console formatter with per-level colors (TTY only)."""

    COLORS = {
        "DEBUG": "\033[36m", "INFO": "\033[32m", "WARNING": "\033[33m",
        "ERROR": "\033[31m", "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record):
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def _apply_verbosity(
    logger: logging.Logger, verbosity: str, console_level: Optional[int]
) -> None:
    """(Re)apply a verbosity profile to the console handler + third-party loggers."""
    allowed, base_level, third_party_level = VERBOSITY_PROFILES.get(
        verbosity, VERBOSITY_PROFILES["default"]
    )
    level = console_level if console_level is not None else base_level
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(level)
            handler.filters = [_ConsoleCategoryFilter(allowed)]
    for name in _THIRD_PARTY:
        logging.getLogger(name).setLevel(third_party_level)


def setup_logging(
    experiment_name: str = "evaluation",
    log_dir: str = "logs",
    console_level: Optional[int] = None,
    file_level: int = logging.DEBUG,
    verbosity: str = "default",
) -> logging.Logger:
    """Set up file + console logging for the ``evaluator`` logger tree.

    Args:
        experiment_name / log_dir: where the DEBUG file log is written.
        console_level: explicit console level override (else the profile's base level).
        file_level: file handler level (DEBUG — the full record, always).
        verbosity: ``default`` / ``verbose`` / ``debug`` — gates console categories.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("evaluator")

    # Handlers are created once; the verbosity profile is (re)applied on every call so a
    # later call with a different profile takes effect.
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = log_path / f"{timestamp}_{experiment_name}.log"

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter("%(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
        logger.info("Logging initialized. Log file: %s", log_file)

    _apply_verbosity(logger, verbosity, console_level)
    return logger


def get_logger(name: str = "evaluator") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def log_timing(logger: Optional[logging.Logger] = None):
    """Decorator: log a function's execution time under the ``timing`` category."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            timing_logger.debug("Starting %s", func.__name__)
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                timing_logger.info("%s completed in %.2fs", func.__name__, elapsed)
                return result
            except Exception:
                elapsed = time.time() - start_time
                timing_logger.error("%s failed after %.2fs", func.__name__, elapsed)
                raise

        return wrapper

    return decorator


class TimingContext:
    """Context manager that times a code block under the ``timing`` category."""

    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        timing_logger.debug("Starting: %s", self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type is None:
            timing_logger.info("%s completed in %.2fs", self.name, elapsed)
        else:
            timing_logger.error("%s failed after %.2fs", self.name, elapsed)
        return False


def log_cache_stats(cache_manager, logger: Optional[logging.Logger] = None):
    """Log cache stats under the ``cache`` category: a one-line summary (verbose) plus a
    per-category breakdown (debug only)."""
    stats = cache_manager.get_cache_stats()
    by_cat = stats.get("by_category", {})
    cache_logger.info(
        "cache: %s total across %d categories",
        stats.get("total_size_human", "0 B"),
        len(by_cat),
    )
    for cache_type, info in by_cat.items():
        cache_logger.debug(
            "  %s: %s (%d files)",
            cache_type, info.get("size_human", "0 B"), info.get("file_count", 0),
        )
