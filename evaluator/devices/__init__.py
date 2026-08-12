"""GPU device management and allocation."""

from .pool import GPUPool, DeviceUsage, pool_from_config
from .strategy import MemoryAwareStrategy, ManualStrategy, pick_strategy
from .monitor import GPUMonitor, MemoryInfo, GPUInfo
from .memory import (
    MemoryManager,
    MemorySnapshot,
    BatchProcessor,
    get_memory_manager,
)

__all__ = [
    # Pool
    "GPUPool",
    "DeviceUsage",
    "pool_from_config",
    # Strategies
    "MemoryAwareStrategy",
    "ManualStrategy",
    "pick_strategy",
    # Monitor
    "GPUMonitor",
    "MemoryInfo",
    "GPUInfo",
    # Memory Management
    "MemoryManager",
    "MemorySnapshot",
    "BatchProcessor",
    "get_memory_manager",
]
