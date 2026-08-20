"""Model service wrappers with lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar
import gc
import logging
import time

if TYPE_CHECKING:
    from ..devices import GPUPool

import torch

T = TypeVar("T")
logger = logging.getLogger(__name__)


def _offload_and_clear(model: object) -> None:
    """Best-effort release of GPU memory for a model-like object."""
    to_method = getattr(model, "to", None)
    if callable(to_method):
        # Most model wrappers in this codebase support .to(torch.device)
        to_method(torch.device("cpu"))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class FactoryModelService(Generic[T]):
    """Lazy service over existing model factory functions."""

    factory: Callable[[], T]
    label: str = "model"
    pool: Optional["GPUPool"] = field(default=None, repr=False)
    model_type: Optional[str] = None
    memory_gb: float = 0.0
    _instance: Optional[T] = field(default=None, init=False, repr=False)
    _on_gpu: bool = field(default=False, init=False, repr=False)

    def start(self) -> None:
        if self._instance is None:
            t0 = time.perf_counter()
            self._instance = self.factory()
            elapsed = time.perf_counter() - t0
            logger.info("service.start label=%s load_time=%.2fs", self.label, elapsed)
            if self.pool is not None and self.model_type is not None:
                self.pool.register_eviction_callback(self.model_type, self._evict_to_cpu)
                self._on_gpu = True

    def _evict_to_cpu(self) -> None:
        """Called by GPUPool when this model is evicted from GPU."""
        if self._instance is not None:
            to_method = getattr(self._instance, "to", None)
            if callable(to_method):
                to_method(torch.device("cpu"))
        self._on_gpu = False
        logger.info("service.evicted label=%s", self.label)

    def _ensure_on_gpu(self) -> None:
        """Re-promote model to GPU if it was evicted."""
        if self._on_gpu or self.pool is None or self.model_type is None:
            return
        device = self.pool.allocate(self.model_type, self.memory_gb)
        if device == "cpu":
            return
        to_method = getattr(self._instance, "to", None)
        if callable(to_method):
            to_method(torch.device(device))
        self._on_gpu = True
        self.pool.register_eviction_callback(self.model_type, self._evict_to_cpu)
        logger.info("service.promoted label=%s device=%s", self.label, device)

    def stop(self) -> None:
        if self._instance is None:
            return
        _offload_and_clear(self._instance)
        self._instance = None
        self._on_gpu = False
        logger.info("service.stop label=%s", self.label)

    def get(self) -> T:
        if self._instance is None:
            self.start()
        else:
            self._ensure_on_gpu()
        assert self._instance is not None
        if self.pool is not None and self.model_type is not None:
            self.pool.touch(self.model_type)
        return self._instance

    def move_to_device(self, device: str) -> None:
        model = self.get()
        to_method = getattr(model, "to", None)
        if callable(to_method):
            to_method(torch.device(device))
            self._on_gpu = device != "cpu"
            logger.info("service.move label=%s device=%s", self.label, device)


class LLMServerService:
    """Lifecycle wrapper for local LLM server backends."""

    def __init__(
        self,
        factory: Callable[[], Any],
        label: str = "llm_server",
        auto_start: bool = True,
    ) -> None:
        self.factory = factory
        self.label = label
        self.auto_start = auto_start
        self._instance: Optional[Any] = None
        self._owns_process: bool = False

    def start(self) -> None:
        if self._instance is None:
            self._instance = self.factory()
        health_before = self._instance.health_check()
        if health_before.is_healthy:
            self._owns_process = False
            logger.info("service.reuse label=%s", self.label)
            return

        if not self.auto_start:
            raise RuntimeError(
                f"Local LLM server unhealthy for {self.label} and auto_start is disabled"
            )
        t0 = time.perf_counter()
        started = self._instance.start()
        if not started:
            raise RuntimeError(f"Failed to start local LLM server for {self.label}")
        self._owns_process = True
        health_after = self._instance.health_check()
        if not health_after.is_healthy:
            raise RuntimeError(
                f"Local LLM server unhealthy after start for {self.label}"
            )
        elapsed = time.perf_counter() - t0
        logger.info("service.start label=%s load_time=%.2fs", self.label, elapsed)

    def stop(self, owned_only: bool = True) -> None:
        if self._instance is None:
            return
        if not owned_only or self._owns_process:
            self._instance.stop()
            logger.info("service.stop label=%s", self.label)
        self._instance = None
        self._owns_process = False

    def get(self) -> Any:
        self.start()
        assert self._instance is not None
        return self._instance

    def get_api_url(self) -> str:
        return self.get().get_api_url()
