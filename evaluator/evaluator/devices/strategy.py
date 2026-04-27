"""Allocation strategies for GPU pool."""

from abc import abstractmethod
from typing import Dict, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .pool import GPUPool


class AllocationStrategy(Protocol):
    """Protocol for GPU allocation strategies."""
    
    @abstractmethod
    def allocate(self, pool: "GPUPool", model_type: str, memory_gb: float) -> Optional[str]:
        """Allocate a device for a model.
        
        Args:
            pool: The GPU pool to allocate from.
            model_type: Identifier for the model.
            memory_gb: Estimated memory requirement in GB.
            
        Returns:
            Device string if allocation succeeded, None to let pool handle it.
        """
        ...


class RoundRobinStrategy:
    """Distribute models sequentially across GPUs.
    
    Each new allocation goes to the next GPU in order, regardless
    of current memory usage. This provides even distribution when
    models have similar memory requirements.
    """
    
    def allocate(self, pool: "GPUPool", model_type: str, memory_gb: float) -> Optional[str]:
        """Allocate using round-robin distribution."""
        gpu_devices = pool.gpu_devices
        if not gpu_devices:
            return None
        
        device = pool.get_next_round_robin_device()
        usage = pool.get_usage()
        
        # Check if device has enough memory
        if device in usage:
            if usage[device].free_memory_gb >= memory_gb:
                return device
        
        # Round-robin device doesn't have enough memory, find any that does
        for dev in gpu_devices:
            if dev in usage and usage[dev].free_memory_gb >= memory_gb:
                return dev
        
        return None


class MemoryAwareStrategy:
    """Allocate to GPU with most free memory.
    
    This strategy picks the GPU with the most available memory,
    providing good load balancing when model sizes vary.
    """
    
    def allocate(self, pool: "GPUPool", model_type: str, memory_gb: float) -> Optional[str]:
        """Allocate to GPU with most free memory."""
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


class PackingStrategy:
    """Maximize GPU utilization before moving to next.
    
    This strategy fills up each GPU as much as possible before
    allocating to the next one. Useful when you want to leave
    some GPUs completely free for other tasks.
    """
    
    def allocate(self, pool: "GPUPool", model_type: str, memory_gb: float) -> Optional[str]:
        """Allocate by packing models onto fewest GPUs."""
        usage = pool.get_usage()
        
        # Sort GPUs by utilization (most utilized first)
        gpu_devices = sorted(
            [d for d in pool.gpu_devices if d in usage],
            key=lambda d: usage[d].utilization,
            reverse=True
        )
        
        # Find the most utilized GPU that still has enough space
        for device in gpu_devices:
            if usage[device].free_memory_gb >= memory_gb:
                return device
        
        return None


class ManualStrategy:
    """User-specified device mapping.
    
    Allows explicit control over which models go to which devices.
    Falls back to pool default for unmapped models.
    
    Example:
        >>> strategy = ManualStrategy({
        ...     "asr": "cuda:0",
        ...     "text_embedding": "cuda:1",
        ... })
    """
    
    def __init__(self, overrides: Dict[str, str]):
        """Initialize with device overrides.
        
        Args:
            overrides: Mapping from model_type to device string.
        """
        self._overrides = overrides
    
    def allocate(self, pool: "GPUPool", model_type: str, memory_gb: float) -> Optional[str]:
        """Allocate using manual overrides."""
        if model_type in self._overrides:
            device = self._overrides[model_type]
            # Verify device is in pool
            if device in pool.devices:
                return device
            # If override device not in pool, try to add it
            # (this allows specifying devices not originally in the pool)
            return device
        
        # No override, let pool handle it
        return None
    
    @property
    def overrides(self) -> Dict[str, str]:
        """Get the current overrides mapping."""
        return dict(self._overrides)


class AutoStrategy:
    """Intelligent automatic device allocation based on GPU memory and model requirements.
    
    This strategy analyzes available GPU memory and model memory requirements
    to make optimal allocation decisions. It considers:
    - GPU memory capacity
    - Expected model memory requirements
    - Current GPU utilization
    - Memory safety margins
    
    Falls back to CPU when GPU memory is insufficient with proper logging.
    """
    
    def __init__(self, memory_safety_margin: float = 0.2):
        """Initialize auto allocation strategy.
        
        Args:
            memory_safety_margin: Safety margin to reserve (0.0-1.0).
                                 e.g., 0.2 means keep 20% of GPU memory free.
        """
        self._memory_safety_margin = memory_safety_margin
        self._model_type_priority = {
            # Higher priority models get better GPUs
            "asr": 3,  # ASR models typically largest
            "audio_embedding": 2,
            "text_embedding": 1,
        }
    
    def allocate(self, pool: "GPUPool", model_type: str, memory_gb: float) -> Optional[str]:
        """Allocate device intelligently based on memory and model type."""
        import logging
        logger = logging.getLogger(__name__)
        
        usage = pool.get_usage()
        gpu_devices = [d for d in pool.gpu_devices if d in usage]
        
        if not gpu_devices:
            logger.info(f"No GPUs available for {model_type}, using CPU fallback")
            return None  # Let pool handle CPU fallback
        
        # Adjust memory requirement with safety margin
        required_memory = memory_gb * (1.0 + self._memory_safety_margin)
        
        # Find GPUs that can fit this model
        candidates = []
        for device in gpu_devices:
            device_usage = usage[device]
            if device_usage.free_memory_gb >= required_memory:
                candidates.append((device, device_usage))
        
        if not candidates:
            logger.warning(
                f"No GPU has sufficient memory for {model_type} "
                f"(requires {required_memory:.2f}GB with {self._memory_safety_margin*100:.0f}% margin). "
                f"Best available: {max((u.free_memory_gb for _, u in usage.items() if _ in gpu_devices), default=0):.2f}GB"
            )
            return None  # Let pool handle fallback
        
        # Sort by free memory (most free first)
        candidates.sort(key=lambda x: x[1].free_memory_gb, reverse=True)
        
        # Select best device
        best_device = candidates[0][0]
        
        logger.debug(
            f"AutoStrategy allocated {best_device} for {model_type} "
            f"({memory_gb:.2f}GB required, {usage[best_device].free_memory_gb:.2f}GB available)"
        )
        
        return best_device


def create_strategy(name: str, **kwargs) -> AllocationStrategy:
    """Create an allocation strategy by name.
    
    Args:
        name: Strategy name ("auto", "round_robin", "memory_aware", "packing", "manual").
        **kwargs: Additional arguments for the strategy.
        
    Returns:
        An AllocationStrategy instance.
        
    Raises:
        ValueError: If the strategy name is unknown.
    """
    strategies = {
        "auto": AutoStrategy,
        "round_robin": RoundRobinStrategy,
        "memory_aware": MemoryAwareStrategy,
        "packing": PackingStrategy,
        "manual": ManualStrategy,
    }
    
    if name not in strategies:
        raise ValueError(
            f"Unknown allocation strategy: '{name}'. "
            f"Available strategies: {list(strategies.keys())}"
        )
    
    strategy_class = strategies[name]
    
    # Manual strategy requires 'overrides' argument
    if name == "manual" and "overrides" not in kwargs:
        raise ValueError("ManualStrategy requires 'overrides' argument")
    
    return strategy_class(**kwargs)
    
    if name not in strategies:
        available = ", ".join(sorted(strategies.keys()))
        raise ValueError(f"Unknown allocation strategy: '{name}'. Available: {available}")
    
    strategy_class = strategies[name]
    
    if name == "manual":
        overrides = kwargs.get("overrides", {})
        return strategy_class(overrides)
    
    return strategy_class()
