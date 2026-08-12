"""GPU allocation strategies: memory-aware (default) or manual overrides.

A strategy is duck-typed: ``allocate(pool, model_type, memory_gb) -> Optional[str]``.
``pick_strategy(device_pool_cfg)`` chooses — manual iff ``model_device_overrides`` is set.
"""

from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .pool import GPUPool


class MemoryAwareStrategy:
    """Allocate to the GPU with the most free memory (good balance for mixed model sizes)."""

    def allocate(self, pool: "GPUPool", model_type: str, memory_gb: float) -> Optional[str]:
        usage = pool.get_usage()
        best_device = None
        best_free = -1.0
        for device, device_usage in usage.items():
            if device == "cpu":
                continue
            if device_usage.free_memory_gb >= memory_gb:
                if device_usage.free_memory_gb > best_free:
                    best_device = device
                    best_free = device_usage.free_memory_gb
        return best_device


class ManualStrategy:
    """User-specified model_type → device mapping; unmapped models fall back to the pool."""

    def __init__(self, overrides: Dict[str, str]):
        self._overrides = overrides

    def allocate(self, pool: "GPUPool", model_type: str, memory_gb: float) -> Optional[str]:
        if model_type in self._overrides:
            # An override device outside the pool is honored as-is (deliberate escape hatch).
            return self._overrides[model_type]
        return None


def pick_strategy(device_pool_cfg):
    """Manual when the config maps models to devices, else memory-aware."""
    overrides = getattr(device_pool_cfg, "model_device_overrides", None)
    return ManualStrategy(dict(overrides)) if overrides else MemoryAwareStrategy()
