"""Service runtime policy configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ServiceRuntimeConfig:
    """Configuration for model service startup/offload behavior."""

    startup_mode: str = "lazy"  # lazy | eager
    # on_finish (free after last use) | never (keep resident) | on_finish_soft_cpu (park warm
    # on host RAM after last use → fast CPU↔device reuse; bounded by soft_offload_max_warm).
    offload_policy: str = "on_finish"
    # Soft-CPU warm-pool bounds (Roadmap 2c); only consulted under on_finish_soft_cpu.
    soft_offload_max_warm: int = 2  # max models kept warm on CPU (LRU-evicted past this)
    soft_offload_ttl_s: Optional[float] = None  # evict a warm model older than this (s)

    def __post_init__(self) -> None:
        from .types import SERVICE_STARTUP_MODES as valid_startup
        from .types import SERVICE_OFFLOAD_POLICIES as valid_offload

        if self.startup_mode not in valid_startup:
            raise ValueError(
                f"Invalid service startup_mode: '{self.startup_mode}'. "
                f"Expected one of: {sorted(valid_startup)}"
            )
        if self.offload_policy not in valid_offload:
            raise ValueError(
                f"Invalid service offload_policy: '{self.offload_policy}'. "
                f"Expected one of: {sorted(valid_offload)}"
            )
