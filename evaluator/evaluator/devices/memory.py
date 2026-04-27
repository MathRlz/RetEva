"""Memory management and optimization utilities for GPU operations."""

import gc
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Generator, Any, Callable

from .monitor import GPUMonitor, get_monitor, MemoryInfo


logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state at a point in time."""
    device: str
    timestamp: float
    total_mb: float
    used_mb: float
    free_mb: float
    utilization_percent: float


class MemoryManager:
    """Manages GPU memory optimization and monitoring.
    
    Provides utilities for:
    - Clearing GPU cache periodically vs. on-demand
    - Monitoring peak memory usage
    - Batched cleanup operations
    - Memory-efficient context managers
    """
    
    def __init__(
        self,
        monitor: Optional[GPUMonitor] = None,
        auto_cleanup: bool = False,
        cleanup_interval: int = 10,
    ):
        """Initialize the memory manager.
        
        Args:
            monitor: GPU monitor instance. Uses default if not provided.
            auto_cleanup: Whether to automatically clear cache after operations.
            cleanup_interval: Interval (in operations) for automatic cleanup.
        """
        self._monitor = monitor or get_monitor()
        self._auto_cleanup = auto_cleanup
        self._cleanup_interval = cleanup_interval
        self._operation_count = 0
        self._peak_memory: dict[str, float] = {}
        self._memory_snapshots: list[MemorySnapshot] = []
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU cache on all CUDA devices.
        
        This is a no-op if CUDA is not available or torch is not installed.
        """
        if not self._monitor.cuda_available:
            return
        
        try:
            import torch
            torch.cuda.empty_cache()
            logger.debug("Cleared GPU cache")
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")
    
    def collect_garbage(self) -> None:
        """Trigger Python garbage collection to free memory.
        
        Useful after processing large batches to free Python object memory.
        """
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")
    
    def cleanup_all(self) -> None:
        """Perform full cleanup: garbage collection and GPU cache clearing."""
        self.collect_garbage()
        self.clear_gpu_cache()
    
    def record_operation(self) -> None:
        """Record an operation for tracking periodic cleanup."""
        self._operation_count += 1
        if self._auto_cleanup and self._cleanup_interval > 0 and self._operation_count % self._cleanup_interval == 0:
            self.clear_gpu_cache()
    
    def get_memory_snapshot(self, device_idx: int = 0) -> Optional[MemorySnapshot]:
        """Get a snapshot of current GPU memory state.
        
        Args:
            device_idx: CUDA device index.
            
        Returns:
            MemorySnapshot or None if device unavailable.
        """
        import time
        
        device_str = f"cuda:{device_idx}"
        memory_info = self._monitor.get_memory_usage(device_idx)
        
        if memory_info is None:
            return None
        
        # Convert to MB
        total_mb = memory_info.total * 1024
        used_mb = memory_info.used * 1024
        free_mb = memory_info.free * 1024
        
        snapshot = MemorySnapshot(
            device=device_str,
            timestamp=time.time(),
            total_mb=total_mb,
            used_mb=used_mb,
            free_mb=free_mb,
            utilization_percent=min(100.0, (used_mb / total_mb) * 100) if total_mb > 0 else 0.0
        )
        
        # Update peak memory tracking
        key = f"peak_used_{device_idx}"
        current_peak = self._peak_memory.get(key, 0.0)
        if used_mb > current_peak:
            self._peak_memory[key] = used_mb
        
        self._memory_snapshots.append(snapshot)
        return snapshot
    
    def get_peak_memory_usage(self, device_idx: int = 0) -> Optional[float]:
        """Get peak memory usage for a device in MB.
        
        Args:
            device_idx: CUDA device index.
            
        Returns:
            Peak memory used in MB, or None if never recorded.
        """
        key = f"peak_used_{device_idx}"
        return self._peak_memory.get(key)
    
    def get_memory_stats(self) -> dict[str, Any]:
        """Get comprehensive memory statistics.
        
        Returns:
            Dictionary with memory stats for all tracked devices.
        """
        if not self._memory_snapshots:
            return {"error": "No memory snapshots recorded"}
        
        stats = {}
        
        # Group by device
        by_device = {}
        for snapshot in self._memory_snapshots:
            if snapshot.device not in by_device:
                by_device[snapshot.device] = []
            by_device[snapshot.device].append(snapshot)
        
        # Calculate stats per device
        for device, snapshots in by_device.items():
            used_values = [s.used_mb for s in snapshots]
            util_values = [s.utilization_percent for s in snapshots]
            
            stats[device] = {
                "peak_used_mb": max(used_values),
                "avg_used_mb": sum(used_values) / len(used_values),
                "min_used_mb": min(used_values),
                "peak_util_percent": max(util_values),
                "avg_util_percent": sum(util_values) / len(util_values),
                "snapshots": len(snapshots),
            }
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset all collected statistics."""
        self._peak_memory.clear()
        self._memory_snapshots.clear()
        self._operation_count = 0
    
    @contextmanager
    def device_memory_scope(self, cleanup_on_exit: bool = True):
        """Context manager for device memory management.
        
        Optionally clears GPU cache and collects garbage on exit.
        
        Args:
            cleanup_on_exit: Whether to cleanup after exiting the context.
            
        Yields:
            This memory manager instance.
        """
        try:
            yield self
        finally:
            if cleanup_on_exit:
                self.cleanup_all()
    
    @contextmanager
    def batch_processing_scope(
        self,
        cleanup_interval: int = 10,
        monitor_memory: bool = False
    ):
        """Context manager optimized for batch processing loops.
        
        Provides periodic cleanup during batch processing and optional
        memory monitoring.
        
        Args:
            cleanup_interval: Clear cache every N operations.
            monitor_memory: Whether to monitor memory during processing.
            
        Yields:
            A BatchProcessor helper object.
        """
        saved_interval = self._cleanup_interval
        saved_auto_cleanup = self._auto_cleanup
        saved_operation_count = self._operation_count
        
        self._cleanup_interval = cleanup_interval
        self._auto_cleanup = True
        self._operation_count = 0
        
        try:
            yield BatchProcessor(self, monitor_memory=monitor_memory)
        finally:
            self._cleanup_interval = saved_interval
            self._auto_cleanup = saved_auto_cleanup
            self._operation_count = saved_operation_count
            self.cleanup_all()


class BatchProcessor:
    """Helper for batch processing with periodic memory management.
    
    Tracks batch operations and triggers cleanup at specified intervals.
    """
    
    def __init__(self, memory_manager: MemoryManager, monitor_memory: bool = False):
        """Initialize the batch processor.
        
        Args:
            memory_manager: The MemoryManager instance.
            monitor_memory: Whether to monitor memory for each batch.
        """
        self.memory_manager = memory_manager
        self.monitor_memory = monitor_memory
        self.batch_count = 0
        self.snapshots = []
    
    def record_batch(self, batch_size: Optional[int] = None) -> Optional[MemorySnapshot]:
        """Record processing of a batch.
        
        Args:
            batch_size: Size of the batch (for logging).
            
        Returns:
            Memory snapshot if monitoring, None otherwise.
        """
        self.memory_manager.record_operation()
        self.batch_count += 1
        
        snapshot = None
        if self.monitor_memory:
            snapshot = self.memory_manager.get_memory_snapshot()
            if snapshot:
                self.snapshots.append(snapshot)
                if batch_size:
                    logger.debug(
                        f"Batch {self.batch_count} (size {batch_size}): "
                        f"Memory {snapshot.used_mb:.0f}MB / {snapshot.total_mb:.0f}MB "
                        f"({snapshot.utilization_percent:.1f}%)"
                    )
        
        return snapshot
    
    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        if not self.snapshots:
            return {"batches_processed": self.batch_count}
        
        used_values = [s.used_mb for s in self.snapshots]
        return {
            "batches_processed": self.batch_count,
            "peak_memory_mb": max(used_values),
            "avg_memory_mb": sum(used_values) / len(used_values),
            "min_memory_mb": min(used_values),
        }


# Global memory manager instance for convenience
_default_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the default memory manager instance."""
    global _default_memory_manager
    if _default_memory_manager is None:
        _default_memory_manager = MemoryManager()
    return _default_memory_manager


@contextmanager
def memory_managed(cleanup_on_exit: bool = True) -> Generator[MemoryManager, None, None]:
    """Convenient context manager for memory management.
    
    Args:
        cleanup_on_exit: Whether to cleanup after exiting the context.
        
    Yields:
        MemoryManager instance.
    """
    manager = get_memory_manager()
    with manager.device_memory_scope(cleanup_on_exit=cleanup_on_exit) as mgr:
        yield mgr


@contextmanager
def batch_processing(
    cleanup_interval: int = 10,
    monitor_memory: bool = False
) -> Generator[BatchProcessor, None, None]:
    """Convenient context manager for batch processing.
    
    Args:
        cleanup_interval: Clear cache every N operations.
        monitor_memory: Whether to monitor memory during processing.
        
    Yields:
        BatchProcessor instance.
    """
    manager = get_memory_manager()
    with manager.batch_processing_scope(
        cleanup_interval=cleanup_interval,
        monitor_memory=monitor_memory
    ) as processor:
        yield processor
