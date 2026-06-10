"""GPU device management and allocation."""

from .pool import GPUPool, DeviceUsage
from .strategy import (
    AllocationStrategy,
    RoundRobinStrategy,
    MemoryAwareStrategy,
    PackingStrategy,
    ManualStrategy,
)
from .monitor import GPUMonitor, MemoryInfo, GPUInfo
from .capability import (
    usable_gpu_indices,
    usable_gpu_count,
    is_usable_index,
    reset_cache as reset_gpu_capability_cache,
)
from .memory import (
    MemoryManager,
    MemorySnapshot,
    BatchProcessor,
    get_memory_manager,
    memory_managed,
    batch_processing,
)

__all__ = [
    # Pool
    "GPUPool",
    "DeviceUsage",
    # Strategies
    "AllocationStrategy",
    "RoundRobinStrategy",
    "MemoryAwareStrategy",
    "PackingStrategy",
    "ManualStrategy",
    # Monitor
    "GPUMonitor",
    "MemoryInfo",
    "GPUInfo",
    # Memory Management
    "MemoryManager",
    "MemorySnapshot",
    "BatchProcessor",
    "get_memory_manager",
    "memory_managed",
    "batch_processing",
    # Capability
    "usable_gpu_indices",
    "usable_gpu_count",
    "is_usable_index",
    "reset_gpu_capability_cache",
]
