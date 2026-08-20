"""Device pool configuration."""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class DevicePoolConfig:
    """Configuration for GPU pool allocation.

    Attributes:
        available_devices: List of devices to use. ["auto"] auto-detects GPUs.
        memory_buffer_percent: Fraction of GPU memory to keep free (0.0-1.0).
        allow_cpu_fallback: If True, fall back to CPU when GPUs are full.
        model_device_overrides: Manual device assignments per model type
            (their presence selects the manual allocation strategy;
            memory-aware otherwise).
    """
    available_devices: List[str] = field(default_factory=lambda: ["auto"])
    memory_buffer_percent: float = 0.1
    allow_cpu_fallback: bool = True
    allow_gpu_eviction: bool = True
    model_device_overrides: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.memory_buffer_percent <= 1.0:
            raise ValueError(
                f"memory_buffer_percent must be between 0.0 and 1.0, "
                f"got: {self.memory_buffer_percent}"
            )
