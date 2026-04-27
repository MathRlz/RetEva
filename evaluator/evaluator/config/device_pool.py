"""Device pool configuration."""
from dataclasses import dataclass, field
from typing import List, Dict, Union

from ..config.types import AllocationStrategy, to_enum


@dataclass
class DevicePoolConfig:
    """Configuration for GPU pool allocation.
    
    Attributes:
        available_devices: List of devices to use. ["auto"] auto-detects GPUs.
        allocation_strategy: Strategy for allocating models to devices.
            Options: "memory_aware", "round_robin", "packing", "manual".
        memory_buffer_percent: Fraction of GPU memory to keep free (0.0-1.0).
        allow_cpu_fallback: If True, fall back to CPU when GPUs are full.
        model_device_overrides: Manual device assignments per model type.
    """
    available_devices: List[str] = field(default_factory=lambda: ["auto"])
    allocation_strategy: Union[str, AllocationStrategy] = "memory_aware"
    memory_buffer_percent: float = 0.1
    allow_cpu_fallback: bool = True
    model_device_overrides: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Normalize allocation_strategy to enum
        if isinstance(self.allocation_strategy, str):
            self.allocation_strategy = to_enum(self.allocation_strategy, AllocationStrategy)
        
        if not 0.0 <= self.memory_buffer_percent <= 1.0:
            raise ValueError(
                f"memory_buffer_percent must be between 0.0 and 1.0, "
                f"got: {self.memory_buffer_percent}"
            )
