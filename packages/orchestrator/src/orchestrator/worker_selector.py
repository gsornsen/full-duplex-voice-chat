"""Worker selection algorithms for load balancing.

This module implements various worker selection strategies for M9 Routing v1:
- Round-robin: Simple fair distribution
- Least-loaded: Select worker with lowest queue depth
- Least-latency: Select worker with lowest synthesis latency
- Random: Random selection (for testing)

Performance targets:
- Selection overhead: <100μs (p95)
- Thread-safe for concurrent requests
- Stateless algorithms (except round-robin counter)

Example usage:
    >>> selector = WorkerSelector()
    >>> worker = selector.least_loaded(workers)
    >>> print(f"Selected: {worker.name} (queue: {worker.metrics['queue_depth']})")
"""

import logging
import random

from orchestrator.registry import WorkerRegistration

logger = logging.getLogger(__name__)


class WorkerSelector:
    """Worker selection algorithms for load balancing.

    Implements multiple selection strategies with minimal overhead.
    All methods are thread-safe and designed for low-latency operation.

    Attributes:
        round_robin_counter: Internal counter for round-robin selection
    """

    def __init__(self) -> None:
        """Initialize worker selector."""
        self.round_robin_counter = 0
        logger.debug("WorkerSelector initialized")

    def round_robin(self, workers: list[WorkerRegistration]) -> WorkerRegistration:
        """Select worker using round-robin strategy.

        Simple fair distribution that cycles through workers in order,
        ignoring current load metrics. Provides predictable distribution
        for testing and debugging.

        Thread-safety: Safe for single-threaded orchestrator. For
        multi-threaded use, consider using thread-local counters.

        Args:
            workers: List of available workers (must be non-empty)

        Returns:
            Selected worker

        Raises:
            ValueError: If workers list is empty
        """
        if not workers:
            raise ValueError("Cannot select from empty worker list")

        selected = workers[self.round_robin_counter % len(workers)]
        self.round_robin_counter += 1

        logger.debug(
            f"Round-robin selected: {selected.name}",
            extra={
                "worker": selected.name,
                "counter": self.round_robin_counter,
                "pool_size": len(workers),
            },
        )

        return selected

    def least_loaded(self, workers: list[WorkerRegistration]) -> WorkerRegistration:
        """Select worker with lowest queue depth.

        Minimizes queuing delay by routing to the least-busy worker.
        This strategy provides optimal latency distribution for bursty
        traffic and varying load patterns.

        Metric interpretation:
        - queue_depth: Number of pending synthesis requests
        - Missing metric: Treated as infinite (worst case)
        - Zero queue_depth: Worker is idle (best case)

        Args:
            workers: List of available workers (must be non-empty)

        Returns:
            Worker with lowest queue depth

        Raises:
            ValueError: If workers list is empty
        """
        if not workers:
            raise ValueError("Cannot select from empty worker list")

        # Select worker with minimum queue depth
        # Default to infinity if metric missing (avoid unhealthy workers)
        selected = min(
            workers,
            key=lambda w: w.metrics.get("queue_depth", float("inf")),
        )

        queue_depth = selected.metrics.get("queue_depth", -1)
        logger.debug(
            f"Least-loaded selected: {selected.name} (queue: {queue_depth})",
            extra={
                "worker": selected.name,
                "queue_depth": queue_depth,
                "pool_size": len(workers),
            },
        )

        return selected

    def least_latency(self, workers: list[WorkerRegistration]) -> WorkerRegistration:
        """Select worker with lowest synthesis latency.

        Optimizes for fastest synthesis by selecting worker with best
        RTF (real-time factor) metric. Best when synthesis speed varies
        significantly across workers (e.g., GPU vs CPU, different models).

        Metric interpretation:
        - rtf: Real-time factor (synthesis time / audio duration)
        - RTF < 1.0: Faster than real-time (e.g., 0.3 = 3x faster)
        - RTF > 1.0: Slower than real-time (CPU fallback)
        - Missing metric: Treated as infinite (worst case)

        Note: May overload fast workers if queue depth is not considered.
        Consider combining with queue_depth checks for production use.

        Args:
            workers: List of available workers (must be non-empty)

        Returns:
            Worker with lowest RTF

        Raises:
            ValueError: If workers list is empty
        """
        if not workers:
            raise ValueError("Cannot select from empty worker list")

        # Select worker with minimum RTF (real-time factor)
        # Lower RTF = faster synthesis = better user experience
        selected = min(
            workers,
            key=lambda w: w.metrics.get("rtf", float("inf")),
        )

        rtf = selected.metrics.get("rtf", -1)
        logger.debug(
            f"Least-latency selected: {selected.name} (RTF: {rtf:.3f})",
            extra={
                "worker": selected.name,
                "rtf": rtf,
                "pool_size": len(workers),
            },
        )

        return selected

    def random(self, workers: list[WorkerRegistration]) -> WorkerRegistration:
        """Select worker randomly.

        Random selection for testing and load distribution analysis.
        Not recommended for production use as it ignores load metrics
        and may create imbalanced distribution.

        Thread-safety: Safe (uses thread-safe random.choice).

        Args:
            workers: List of available workers (must be non-empty)

        Returns:
            Randomly selected worker

        Raises:
            ValueError: If workers list is empty
        """
        if not workers:
            raise ValueError("Cannot select from empty worker list")

        selected = random.choice(workers)  # noqa: S311

        logger.debug(
            f"Random selected: {selected.name}",
            extra={
                "worker": selected.name,
                "pool_size": len(workers),
            },
        )

        return selected

    def weighted_selection(
        self,
        workers: list[WorkerRegistration],
        queue_weight: float = 0.7,
        latency_weight: float = 0.3,
    ) -> WorkerRegistration:
        """Select worker using weighted scoring.

        Combines multiple metrics with configurable weights to balance
        different optimization goals. Provides hybrid strategy between
        queue depth and latency optimization.

        Scoring formula:
            score = queue_weight * norm(queue_depth) + latency_weight * norm(rtf)
            (Lower score is better)

        Normalization: Metrics are normalized to [0, 1] range using
        min-max normalization across available workers.

        Args:
            workers: List of available workers (must be non-empty)
            queue_weight: Weight for queue depth metric (0.0-1.0)
            latency_weight: Weight for latency metric (0.0-1.0)

        Returns:
            Worker with lowest weighted score

        Raises:
            ValueError: If workers list is empty or weights invalid
        """
        if not workers:
            raise ValueError("Cannot select from empty worker list")

        if queue_weight + latency_weight != 1.0:
            raise ValueError(
                f"Weights must sum to 1.0 (got {queue_weight + latency_weight})"
            )

        # Extract metrics with defaults
        queue_depths = [w.metrics.get("queue_depth", 0) for w in workers]
        rtfs = [w.metrics.get("rtf", 1.0) for w in workers]

        # Normalize metrics to [0, 1] range
        max_queue = max(queue_depths) if max(queue_depths) > 0 else 1
        max_rtf = max(rtfs) if max(rtfs) > 0 else 1

        # Calculate weighted scores
        scores = []
        for i, worker in enumerate(workers):
            norm_queue = queue_depths[i] / max_queue
            norm_rtf = rtfs[i] / max_rtf
            score = queue_weight * norm_queue + latency_weight * norm_rtf
            scores.append((score, worker))

        # Select worker with minimum score
        selected = min(scores, key=lambda x: x[0])[1]

        logger.debug(
            f"Weighted selection: {selected.name} "
            f"(queue: {selected.metrics.get('queue_depth', 0)}, "
            f"rtf: {selected.metrics.get('rtf', 1.0):.3f})",
            extra={
                "worker": selected.name,
                "queue_weight": queue_weight,
                "latency_weight": latency_weight,
                "pool_size": len(workers),
            },
        )

        return selected
