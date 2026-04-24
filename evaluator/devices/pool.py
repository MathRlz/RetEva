"""GPU pool allocation and tracking."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING, Generator

from .monitor import GPUMonitor, get_monitor

if TYPE_CHECKING:
    from .strategy import AllocationStrategy

logger = logging.getLogger(__name__)


@dataclass
class DeviceUsage:
    """Tracks memory usage and allocations for a device."""
    device: str
    total_memory_gb: float
    reserved_memory_gb: float = 0.0
    allocations: Dict[str, float] = field(default_factory=dict)
    
    @property
    def free_memory_gb(self) -> float:
        """Calculate free memory after reservations."""
        return max(0.0, self.total_memory_gb - self.reserved_memory_gb)
    
    @property
    def utilization(self) -> float:
        """Calculate utilization as a fraction (0.0 to 1.0)."""
        if self.total_memory_gb <= 0:
            return 0.0
        return min(1.0, self.reserved_memory_gb / self.total_memory_gb)


class GPUPool:
    """Manages GPU device allocation for models.
    
    The GPUPool tracks which devices are available and how much memory
    is reserved on each. It supports different allocation strategies
    and can fall back to CPU when GPU memory is exhausted.
    
    Example:
        >>> pool = GPUPool(["cuda:0", "cuda:1"])
        >>> device = pool.allocate("asr", memory_gb=2.5)
        >>> print(device)
        'cuda:0'
        >>> pool.release("asr")
    """
    
    def __init__(
        self,
        devices: List[str],
        monitor: Optional[GPUMonitor] = None,
        memory_buffer_percent: float = 0.1,
        allow_cpu_fallback: bool = True,
    ):
        """Initialize the GPU pool.
        
        Args:
            devices: List of device strings (e.g., ["cuda:0", "cuda:1"]).
                    Use ["auto"] to auto-detect available GPUs.
            monitor: GPU monitor instance. Uses default if not provided.
            memory_buffer_percent: Buffer to keep free on each GPU (0.0 to 1.0).
            allow_cpu_fallback: If True, fall back to CPU when GPUs are full.
        """
        self._monitor = monitor or get_monitor()
        self._memory_buffer_percent = memory_buffer_percent
        self._allow_cpu_fallback = allow_cpu_fallback
        self._strategy: Optional["AllocationStrategy"] = None
        self._round_robin_index = 0
        
        # Resolve "auto" to actual devices
        if devices == ["auto"] or (len(devices) == 1 and devices[0] == "auto"):
            devices = self._auto_detect_devices()
        
        # Initialize device usage tracking
        self._devices: Dict[str, DeviceUsage] = {}
        for device in devices:
            self._devices[device] = self._create_device_usage(device)
    
    def _auto_detect_devices(self) -> List[str]:
        """Auto-detect available GPU devices."""
        gpu_count = self._monitor.get_device_count()
        if gpu_count == 0:
            logger.info("No GPUs detected, using CPU")
            return ["cpu"]
        
        devices = [f"cuda:{i}" for i in range(gpu_count)]
        logger.info(f"Auto-detected {gpu_count} GPU(s): {devices}")
        return devices
    
    def _create_device_usage(self, device: str) -> DeviceUsage:
        """Create a DeviceUsage object for a device."""
        if device == "cpu":
            # CPU has "unlimited" memory for our purposes
            return DeviceUsage(device=device, total_memory_gb=float('inf'))
        
        # Parse device index
        if ":" in device:
            try:
                device_idx = int(device.split(":")[1])
            except (ValueError, IndexError):
                device_idx = 0
        else:
            device_idx = 0
        
        # Get actual memory from monitor
        memory_info = self._monitor.get_memory_usage(device_idx)
        if memory_info is not None:
            # Apply buffer
            total_usable = memory_info.total * (1 - self._memory_buffer_percent)
            return DeviceUsage(device=device, total_memory_gb=total_usable)
        
        # Fallback: assume 8GB if we can't query
        return DeviceUsage(device=device, total_memory_gb=8.0)
    
    def set_strategy(self, strategy: "AllocationStrategy") -> None:
        """Set the allocation strategy.
        
        Args:
            strategy: The allocation strategy to use.
        """
        self._strategy = strategy
    
    def allocate(self, model_type: str, memory_gb: float) -> str:
        """Allocate a device for a model.
        
        Args:
            model_type: Identifier for the model (e.g., "asr", "text_embedding").
            memory_gb: Estimated memory requirement in GB.
            
        Returns:
            Device string (e.g., "cuda:0", "cpu").
            
        Raises:
            RuntimeError: If no device can accommodate the model.
        """
        logger.debug(f"Allocating device for '{model_type}' (requires {memory_gb:.2f}GB)")
        
        # Use strategy if set
        if self._strategy is not None:
            device = self._strategy.allocate(self, model_type, memory_gb)
            if device is not None:
                self._reserve(device, model_type, memory_gb)
                logger.info(
                    f"Allocated '{device}' for '{model_type}' using {self._strategy.__class__.__name__} "
                    f"({memory_gb:.2f}GB reserved)"
                )
                return device
        
        # Default: find device with most free memory
        device = self._find_best_device(memory_gb)
        if device is not None:
            self._reserve(device, model_type, memory_gb)
            logger.info(
                f"Allocated '{device}' for '{model_type}' "
                f"({memory_gb:.2f}GB reserved, {self._devices[device].free_memory_gb:.2f}GB free)"
            )
            return device
        
        # Try CPU fallback
        if self._allow_cpu_fallback and "cpu" not in self._devices:
            logger.warning(
                f"No GPU available for '{model_type}' requiring {memory_gb:.2f}GB. "
                f"Falling back to CPU. Available GPUs: {self.gpu_devices}"
            )
            self._devices["cpu"] = DeviceUsage(device="cpu", total_memory_gb=float('inf'))
            self._reserve("cpu", model_type, memory_gb)
            return "cpu"
        
        if self._allow_cpu_fallback and "cpu" in self._devices:
            logger.warning(
                f"No GPU available for '{model_type}' requiring {memory_gb:.2f}GB. "
                f"Using CPU fallback."
            )
            self._reserve("cpu", model_type, memory_gb)
            return "cpu"
        
        # Build detailed error message
        usage_info = {dev: f"{usage.reserved_memory_gb:.2f}/{usage.total_memory_gb:.2f}GB" 
                      for dev, usage in self._devices.items()}
        logger.error(
            f"Failed to allocate device for '{model_type}' requiring {memory_gb:.2f}GB. "
            f"Current usage: {usage_info}"
        )
        raise RuntimeError(
            f"No device available for model '{model_type}' requiring {memory_gb:.1f}GB. "
            f"Available devices: {list(self._devices.keys())}, Current usage: {usage_info}"
        )
    
    def _find_best_device(self, memory_gb: float) -> Optional[str]:
        """Find the device with the most free memory that can fit the model."""
        best_device = None
        best_free = -1.0
        
        for device, usage in self._devices.items():
            if device == "cpu":
                continue  # Prefer GPUs
            if usage.free_memory_gb >= memory_gb and usage.free_memory_gb > best_free:
                best_device = device
                best_free = usage.free_memory_gb
        
        return best_device
    
    def _reserve(self, device: str, model_type: str, memory_gb: float) -> None:
        """Reserve memory on a device for a model."""
        if device not in self._devices:
            return
        
        usage = self._devices[device]
        usage.reserved_memory_gb += memory_gb
        usage.allocations[model_type] = memory_gb
    
    def release(self, model_type: str) -> None:
        """Release a model's allocation.
        
        Args:
            model_type: Identifier for the model to release.
        """
        for device, usage in self._devices.items():
            if model_type in usage.allocations:
                memory_gb = usage.allocations.pop(model_type)
                usage.reserved_memory_gb -= memory_gb
                # Prevent negative values due to floating point errors
                usage.reserved_memory_gb = max(0.0, usage.reserved_memory_gb)
                logger.debug(
                    f"Released '{model_type}' from '{device}' "
                    f"({memory_gb:.2f}GB freed, {usage.free_memory_gb:.2f}GB now free)"
                )
                return
        
        logger.warning(f"Attempted to release '{model_type}' but no allocation found")
    
    def get_usage(self) -> Dict[str, DeviceUsage]:
        """Get current allocation status for all devices.
        
        Returns:
            Dictionary mapping device strings to DeviceUsage objects.
        """
        return dict(self._devices)
    
    def get_device_for_model(self, model_type: str) -> Optional[str]:
        """Get the device currently allocated to a model.
        
        Args:
            model_type: Identifier for the model.
            
        Returns:
            Device string if allocated, None otherwise.
        """
        for device, usage in self._devices.items():
            if model_type in usage.allocations:
                return device
        return None
    
    @property
    def devices(self) -> List[str]:
        """List of devices in the pool."""
        return list(self._devices.keys())
    
    @property
    def gpu_devices(self) -> List[str]:
        """List of GPU devices in the pool (excludes CPU)."""
        return [d for d in self._devices.keys() if d != "cpu"]
    
    def get_next_round_robin_device(self) -> str:
        """Get the next GPU device in round-robin order."""
        gpu_devices = self.gpu_devices
        if not gpu_devices:
            return "cpu"
        
        device = gpu_devices[self._round_robin_index % len(gpu_devices)]
        self._round_robin_index += 1
        return device
    
    @contextmanager
    def managed_allocation(
        self,
        model_type: str,
        memory_gb: float,
        cleanup_on_exit: bool = False
    ) -> Generator[str, None, None]:
        """Context manager for model allocation with automatic cleanup.
        
        Args:
            model_type: Identifier for the model.
            memory_gb: Estimated memory requirement in GB.
            cleanup_on_exit: Whether to clear GPU cache on exit.
            
        Yields:
            Device string allocated for the model.
        """
        device = self.allocate(model_type, memory_gb)
        try:
            yield device
        finally:
            self.release(model_type)
            if cleanup_on_exit:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except (RuntimeError, ImportError) as exc:
                    logger.debug("GPU cache clear failed during cleanup: %s", exc)
