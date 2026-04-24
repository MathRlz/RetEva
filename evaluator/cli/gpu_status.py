"""GPU status display utility."""

import sys
from typing import Optional


def format_memory(gb: float) -> str:
    """Format memory in GB to a readable string."""
    if gb == float('inf'):
        return "unlimited"
    return f"{gb:.2f} GB"


def format_utilization(fraction: float) -> str:
    """Format utilization fraction as percentage with bar."""
    pct = fraction * 100
    bar_width = 20
    filled = int(fraction * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    return f"[{bar}] {pct:5.1f}%"


def show_gpu_status(pool=None) -> None:
    """Display current GPU status and memory usage.
    
    Args:
        pool: Optional GPUPool to show allocation status.
    """
    from ..devices import GPUMonitor
    
    monitor = GPUMonitor()
    
    print("\n" + "=" * 60)
    print("GPU Status")
    print("=" * 60)
    
    if not monitor.cuda_available:
        print("CUDA is not available. Running on CPU only.")
        print("=" * 60 + "\n")
        return
    
    gpus = monitor.get_all_gpus()
    
    if not gpus:
        print("No GPUs detected.")
        print("=" * 60 + "\n")
        return
    
    print(f"Found {len(gpus)} GPU(s):\n")
    
    for gpu in gpus:
        mem_info = monitor.get_memory_usage(gpu.index)
        
        print(f"  GPU {gpu.index}: {gpu.name}")
        print(f"    Total Memory: {format_memory(gpu.total_memory_gb)}")
        
        if mem_info:
            print(f"    Used:         {format_memory(mem_info.used)}")
            print(f"    Free:         {format_memory(mem_info.free)}")
            utilization = mem_info.used / mem_info.total if mem_info.total > 0 else 0
            print(f"    Utilization:  {format_utilization(utilization)}")
        print()
    
    # Show pool allocations if provided
    if pool is not None:
        print("-" * 60)
        print("Pool Allocations:")
        print("-" * 60)
        
        usage = pool.get_usage()
        for device, device_usage in sorted(usage.items()):
            print(f"\n  {device}:")
            print(f"    Reserved:   {format_memory(device_usage.reserved_memory_gb)}")
            print(f"    Available:  {format_memory(device_usage.free_memory_gb)}")
            if device_usage.allocations:
                print(f"    Models:")
                for model_type, mem in device_usage.allocations.items():
                    print(f"      - {model_type}: {format_memory(mem)}")
    
    print("=" * 60 + "\n")


def print_gpu_summary() -> str:
    """Return a one-line GPU summary for logging."""
    from ..devices import GPUMonitor
    
    monitor = GPUMonitor()
    
    if not monitor.cuda_available:
        return "GPU: None (CPU only)"
    
    gpus = monitor.get_all_gpus()
    if not gpus:
        return "GPU: None detected"
    
    summaries = []
    for gpu in gpus:
        mem_info = monitor.get_memory_usage(gpu.index)
        if mem_info:
            free_pct = (mem_info.free / mem_info.total * 100) if mem_info.total > 0 else 0
            summaries.append(f"GPU{gpu.index}: {free_pct:.0f}% free")
        else:
            summaries.append(f"GPU{gpu.index}: unknown")
    
    return "GPU: " + ", ".join(summaries)


def main():
    """CLI entry point for GPU status."""
    show_gpu_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
