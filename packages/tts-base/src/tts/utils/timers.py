"""Timing and performance measurement utilities."""

import time
from typing import Any


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str) -> None:
        """Initialize timer.

        Args:
            name: Timer name for logging
        """
        self.name = name
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timing and record elapsed time."""
        self.elapsed = time.perf_counter() - self.start_time
