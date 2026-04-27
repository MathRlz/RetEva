"""Service runtime policy configuration."""

from dataclasses import dataclass


@dataclass
class ServiceRuntimeConfig:
    """Configuration for model service startup/offload behavior."""

    startup_mode: str = "lazy"  # lazy | eager
    offload_policy: str = "on_finish"  # on_finish | never

    def __post_init__(self) -> None:
        valid_startup = {"lazy", "eager"}
        valid_offload = {"on_finish", "never"}
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
