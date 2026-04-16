# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

"""Timer utility for measuring and analyzing code execution time."""

import time
from typing import Callable

import numpy as np


class Timer:
    """Timer for measuring code execution time with statistics.

    Uses MATLAB-style tic/toc naming.

    Example:
        >>> timer = Timer("inference")
        >>> for _ in range(100):
        ...     timer.tic()
        ...     model.forward(x)
        ...     timer.toc()
        >>> timer.print_stats()
        [inference:] mean=1.234ms, std=0.123ms, ..., total=123.400ms (n=100)

        # Or use as context manager:
        >>> with timer:
        ...     model.forward(x)
    """

    def __init__(self, name: str = "Timer", unit: str = "ms") -> None:
        """Initialize the timer.

        Args:
            name: Name of the timer for display purposes.
            unit: Time unit for display ("s", "ms", "us", "ns"). Default is "ms".
        """
        self.name = name
        self.unit = unit
        self._tik: float | None = None
        self._times: list[float] = []

    def tic(self) -> None:
        """Start the timer (MATLAB-style)."""
        self._tik = time.perf_counter()

    def toc(self) -> float:
        """Stop the timer and record the elapsed time (MATLAB-style).

        Returns:
            The elapsed time in seconds.

        Raises:
            RuntimeError: If the timer was not started.
        """
        if self._tik is None:
            raise RuntimeError("Timer was not started. Call tic() first.")
        elapsed = time.perf_counter() - self._tik
        self._times.append(elapsed)
        self._tik = None
        return elapsed

    # Aliases for backwards compatibility
    start = tic
    stop = toc

    def __enter__(self) -> "Timer":
        """Context manager entry - starts the timer."""
        self.tic()
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit - stops the timer."""
        self.toc()

    def reset(self) -> None:
        """Reset all recorded times."""
        self._times = []
        self._tik = None

    @property
    def times(self) -> list[float]:
        """Get all recorded times in seconds."""
        return self._times

    @property
    def count(self) -> int:
        """Get the number of recorded times."""
        return len(self._times)

    def _convert_time(self, seconds: float) -> float:
        """Convert time from seconds to the configured unit."""
        multipliers = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
        return seconds * multipliers.get(self.unit, 1e3)

    @property
    def mean(self) -> float:
        """Get the mean time in seconds."""
        if not self._times:
            return 0.0
        return float(np.mean(self._times))

    @property
    def std(self) -> float:
        """Get the standard deviation of times in seconds."""
        if len(self._times) < 2:
            return 0.0
        return float(np.std(self._times))

    @property
    def median(self) -> float:
        """Get the median time in seconds."""
        if not self._times:
            return 0.0
        return float(np.median(self._times))

    @property
    def min(self) -> float:
        """Get the minimum time in seconds."""
        if not self._times:
            return 0.0
        return float(np.min(self._times))

    @property
    def max(self) -> float:
        """Get the maximum time in seconds."""
        if not self._times:
            return 0.0
        return float(np.max(self._times))

    @property
    def total(self) -> float:
        """Get the total accumulated time in seconds."""
        return sum(self._times)

    def get_stats(self) -> dict[str, float]:
        """Get all statistics as a dictionary.

        Returns:
            Dictionary with keys: mean, std, median, min, max, total, count.
            All time values are in seconds.
        """
        return {
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "min": self.min,
            "max": self.max,
            "total": self.total,
            "count": self.count,
        }

    def print_stats(self, print_fn: Callable[[str], None] = print) -> None:
        """Print timing statistics.

        Args:
            print_fn: Function to use for printing (default: print).
                     Can be logging.info or any callable that takes a string.
        """
        if not self._times:
            print_fn(f"{self.name}: no measurements recorded")
            return

        stats = (
            f"[{self.name}:] "
            f"mean={self._convert_time(self.mean):.3f}{self.unit}, "
            f"std={self._convert_time(self.std):.3f}{self.unit}, "
            f"median={self._convert_time(self.median):.3f}{self.unit}, "
            f"min={self._convert_time(self.min):.3f}{self.unit}, "
            f"max={self._convert_time(self.max):.3f}{self.unit}, "
            f"total={self._convert_time(self.total):.3f}{self.unit} "
            f"(n={self.count})"
        )
        print_fn(stats)

    def __repr__(self) -> str:
        """Return a string representation of the timer."""
        return f"Timer(name={self.name!r}, count={self.count}, mean={self.mean:.6f}s)"
