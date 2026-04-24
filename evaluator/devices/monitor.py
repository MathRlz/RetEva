"""GPU monitoring utilities wrapping torch.cuda."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MemoryInfo:
    """GPU memory information in GB."""
    total: float
    used: float
    free: float


@dataclass
class GPUInfo:
    """GPU device information."""
    index: int
    name: str
    total_memory_gb: float


class GPUMonitor:
    """Monitor GPU status and memory usage.
    
    Wraps torch.cuda functionality to provide a clean interface
    for querying GPU availability and memory usage.
    """
    
    def __init__(self):
        """Initialize the GPU monitor."""
        self._cuda_available: Optional[bool] = None
    
    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        if self._cuda_available is None:
            try:
                import torch
                self._cuda_available = torch.cuda.is_available()
            except ImportError:
                self._cuda_available = False
        return self._cuda_available
    
    def get_device_count(self) -> int:
        """Return the number of available CUDA devices."""
        if not self.cuda_available:
            return 0
        import torch
        return torch.cuda.device_count()
    
    def get_memory_usage(self, device_idx: int) -> Optional[MemoryInfo]:
        """Get memory usage for a specific GPU.
        
        Args:
            device_idx: CUDA device index.
            
        Returns:
            MemoryInfo with total, used, and free memory in GB,
            or None if the device is unavailable.
        """
        if not self.cuda_available:
            return None
        
        try:
            import torch
            if device_idx >= torch.cuda.device_count():
                return None
            
            props = torch.cuda.get_device_properties(device_idx)
            total_gb = props.total_memory / (1024 ** 3)
            
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            free_gb = free_bytes / (1024 ** 3)
            used_gb = total_gb - free_gb
            
            return MemoryInfo(total=total_gb, used=used_gb, free=free_gb)
        except (ImportError, RuntimeError, ValueError, OSError):
            return None
    
    def get_all_gpus(self) -> List[GPUInfo]:
        """Get information about all available GPUs.
        
        Returns:
            List of GPUInfo objects for each available GPU.
        """
        gpus = []
        if not self.cuda_available:
            return gpus
        
        try:
            import torch
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpus.append(GPUInfo(
                    index=idx,
                    name=props.name,
                    total_memory_gb=props.total_memory / (1024 ** 3),
                ))
        except (ImportError, RuntimeError, ValueError, OSError):
            return gpus
        
        return gpus
    
    def is_available(self, device: str) -> bool:
        """Check if a device string is available.
        
        Args:
            device: Device string (e.g., "cuda:0", "cuda:1", "cpu").
            
        Returns:
            True if the device is available, False otherwise.
        """
        if device == "cpu":
            return True
        
        if not device.startswith("cuda"):
            return False
        
        if not self.cuda_available:
            return False
        
        # Parse device index
        if ":" in device:
            try:
                device_idx = int(device.split(":")[1])
            except (ValueError, IndexError):
                return False
        else:
            device_idx = 0
        
        import torch
        return device_idx < torch.cuda.device_count()


# Global monitor instance for convenience
_default_monitor: Optional[GPUMonitor] = None


def get_monitor() -> GPUMonitor:
    """Get the default GPU monitor instance."""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = GPUMonitor()
    return _default_monitor
