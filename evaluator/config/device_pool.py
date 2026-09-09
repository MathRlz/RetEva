"""Device pool configuration."""
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class DevicePoolConfig:
    """Configuration for GPU pool allocation."""
    # ["auto"] = auto-detect GPUs
    available_devices: List[str] = field(default_factory=lambda: ["auto"])
    memory_buffer_percent: float = 0.1  # 0.0-1.0
    allow_cpu_fallback: bool = True
    allow_gpu_eviction: bool = True
    # Presence selects the manual per-model allocation strategy (else memory-aware).
    model_device_overrides: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.memory_buffer_percent <= 1.0:
            raise ValueError(
                f"memory_buffer_percent must be between 0.0 and 1.0, "
                f"got: {self.memory_buffer_percent}"
            )
