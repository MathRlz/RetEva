"""Shared utilities for evaluator WebAPI."""

from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def with_provider(provider_factory: Callable[[], Any], fn: Callable[[Any], T]) -> T:
    """Run ``fn`` with a freshly built ModelServiceProvider, always shutting it down."""
    provider = provider_factory()
    try:
        return fn(provider)
    finally:
        provider.shutdown()
