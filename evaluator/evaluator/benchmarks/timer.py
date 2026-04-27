"""Timer utilities for measuring execution time."""

import time
from dataclasses import dataclass, field
from typing import List, Optional
from contextlib import contextmanager


@dataclass
class PerformanceStats:
    """Statistics computed from timing samples."""
    
    mean: float
    std: float
    min: float
    max: float
    samples: int
    
    def __str__(self) -> str:
        return (
            f"mean={self.mean:.4f}s, std={self.std:.4f}s, "
            f"min={self.min:.4f}s, max={self.max:.4f}s, n={self.samples}"
        )


def aggregate_timings(timings: List[float]) -> PerformanceStats:
    """Compute statistics from a list of timing measurements.
    
    Args:
        timings: List of timing measurements in seconds.
        
    Returns:
        PerformanceStats with computed mean, std, min, max, and sample count.
        
    Raises:
        ValueError: If timings list is empty.
    """
    if not timings:
        raise ValueError("Cannot aggregate empty timings list")
    
    n = len(timings)
    mean = sum(timings) / n
    
    if n > 1:
        variance = sum((t - mean) ** 2 for t in timings) / (n - 1)
        std = variance ** 0.5
    else:
        std = 0.0
    
    return PerformanceStats(
        mean=mean,
        std=std,
        min=min(timings),
        max=max(timings),
        samples=n
    )


@dataclass
class Timer:
    """Context manager for measuring execution time.
    
    Can be used as a context manager or manually via start()/stop().
    Supports multiple measurements and automatic statistics computation.
    
    Example:
        # As context manager
        timer = Timer("encoding")
        with timer:
            model.encode(texts)
        print(f"Elapsed: {timer.elapsed:.4f}s")
        
        # Multiple measurements
        timer = Timer("batch_processing")
        for batch in batches:
            with timer:
                process(batch)
        print(timer.stats)
    """
    
    name: str = "timer"
    timings: List[float] = field(default_factory=list)
    _start_time: Optional[float] = field(default=None, repr=False)
    _elapsed: Optional[float] = field(default=None, repr=False)
    
    def start(self) -> "Timer":
        """Start the timer."""
        self._start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """Stop the timer and record the elapsed time.
        
        Returns:
            Elapsed time in seconds.
            
        Raises:
            RuntimeError: If timer was not started.
        """
        if self._start_time is None:
            raise RuntimeError("Timer was not started. Call start() first.")
        
        self._elapsed = time.perf_counter() - self._start_time
        self.timings.append(self._elapsed)
        self._start_time = None
        return self._elapsed
    
    def __enter__(self) -> "Timer":
        """Enter context manager and start timer."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and stop timer."""
        self.stop()
    
    @property
    def elapsed(self) -> float:
        """Get the most recent elapsed time.
        
        Returns:
            Most recent elapsed time, or 0.0 if no measurements.
        """
        if self._elapsed is not None:
            return self._elapsed
        if self.timings:
            return self.timings[-1]
        return 0.0
    
    @property
    def stats(self) -> PerformanceStats:
        """Compute statistics from all recorded timings.
        
        Returns:
            PerformanceStats computed from all timings.
            
        Raises:
            ValueError: If no timings have been recorded.
        """
        return aggregate_timings(self.timings)
    
    def reset(self) -> None:
        """Clear all recorded timings."""
        self.timings.clear()
        self._start_time = None
        self._elapsed = None
    
    @property
    def total_time(self) -> float:
        """Get total time across all measurements."""
        return sum(self.timings)


@contextmanager
def timed(name: str = "operation"):
    """Simple context manager that yields a Timer object.
    
    Example:
        with timed("encoding") as t:
            model.encode(texts)
        print(f"{t.name}: {t.elapsed:.4f}s")
    """
    timer = Timer(name)
    with timer:
        yield timer
