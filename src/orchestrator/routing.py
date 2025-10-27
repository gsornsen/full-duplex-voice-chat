"""Worker selection and routing logic.

This module provides worker selection for TTS synthesis requests.

M2 scope: Static routing to single configured worker.
M9 scope: Dynamic routing with capability matching, load balancing, and session affinity.

M9 Enhancements:
- Capability-based routing (match model requirements to worker capabilities)
- Load balancing (distribute across least-busy workers)
- Session affinity (same session → same worker for voice consistency)
- Graceful fallback (handle worker failures)
- Health-based filtering (exclude unhealthy workers)
- Metrics integration (track routing decisions)

Performance targets:
- Routing decision overhead: <1ms (p95)
- Session affinity hit rate: >95%
- Load distribution variance: <10%
"""

import logging
import time
from enum import Enum
from typing import Any

from redis import asyncio as aioredis

from src.orchestrator.registry import WorkerRegistration, WorkerRegistry
from src.orchestrator.worker_selector import WorkerSelector

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategy for load balancing.

    Attributes:
        ROUND_ROBIN: Simple round-robin selection
        LEAST_LOADED: Select worker with lowest queue depth
        LEAST_LATENCY: Select worker with lowest synthesis latency
        RANDOM: Random selection (for testing)
    """

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    LEAST_LATENCY = "least_latency"
    RANDOM = "random"

    @classmethod
    def from_string(cls, value: str) -> "RoutingStrategy":
        """Convert string to RoutingStrategy enum.

        Args:
            value: Strategy name string

        Returns:
            RoutingStrategy enum value

        Raises:
            ValueError: If value is not a valid strategy
        """
        try:
            return cls(value.lower())
        except ValueError as e:
            raise ValueError(
                f"Invalid routing strategy: {value}. "
                f"Valid options: {', '.join([s.value for s in cls])}"
            ) from e


class Router:
    """Capability-aware worker routing with load balancing and session affinity.

    For M2: Static routing to single pre-configured worker address.
    For M9: Dynamic routing with:
        - Capability matching (language, streaming, zero-shot, etc.)
        - Resident model preference (avoid model load latency)
        - Load balancing (queue depth, RTF, round-robin)
        - Session affinity (voice consistency across messages)
        - Health filtering (exclude unhealthy workers)

    Attributes:
        registry: WorkerRegistry for service discovery
        selector: WorkerSelector for load balancing algorithms
        redis: Redis client for session affinity (M9)
        static_worker_addr: Fallback static worker address (M2)
        prefer_resident_models: Prefer workers with model loaded (M9)
        load_balance_strategy: Default load balancing strategy (M9)
        affinity_enabled: Enable session affinity (M9)
        affinity_ttl_seconds: Session affinity TTL (M9)
        health_check_interval: Worker health check interval (M9)
        metrics: Routing metrics dictionary
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        redis_url: str | None = None,
        static_worker_addr: str | None = None,
        prefer_resident_models: bool = True,
        load_balance_strategy: str = "least_loaded",
        affinity_enabled: bool = True,
        affinity_ttl_seconds: int = 3600,
        health_check_interval: int = 30,
        redis_key_prefix: str = "routing:affinity:",
    ) -> None:
        """Initialize router with worker registry.

        Args:
            registry: Redis-based worker registry client
            redis_url: Redis connection URL for session affinity (M9)
            static_worker_addr: Static worker address for M2 (fallback)
            prefer_resident_models: Prefer workers with model already loaded (M9)
            load_balance_strategy: Load balancing strategy (M9):
                - "queue_depth": Prefer workers with lowest queue
                - "latency": Prefer workers with best RTF
                - "round_robin": Simple round-robin selection
            affinity_enabled: Enable session affinity (M9)
            affinity_ttl_seconds: Session affinity TTL in seconds (M9)
            health_check_interval: Worker health check interval in seconds (M9)
            redis_key_prefix: Redis key prefix for session affinity
        """
        self.registry = registry
        self.redis_url = redis_url
        self.static_worker_addr = static_worker_addr
        self.prefer_resident_models = prefer_resident_models

        # Parse load balance strategy to enum
        try:
            self.load_balance_strategy = RoutingStrategy.from_string(load_balance_strategy)
        except ValueError as e:
            logger.warning(f"{e}. Using LEAST_LOADED as fallback.")
            self.load_balance_strategy = RoutingStrategy.LEAST_LOADED

        # M9 session affinity configuration
        self.affinity_enabled = affinity_enabled
        self.affinity_ttl_seconds = affinity_ttl_seconds
        self.health_check_interval = health_check_interval
        self.redis_key_prefix = redis_key_prefix

        # Worker selector for load balancing algorithms (M9)
        self.selector = WorkerSelector()

        # Redis client for session affinity (lazy initialization)
        self._redis: aioredis.Redis[Any] | None = None
        self._redis_connected = False

        # Round-robin state (M2/M9)
        self._round_robin_index = 0

        # M9 Metrics
        self.metrics = {
            "total_decisions": 0,
            "affinity_hits": 0,
            "affinity_misses": 0,
            "no_workers_errors": 0,
            "total_latency_ms": 0.0,
            "capability_filters": 0,
            "health_filters": 0,
        }

        logger.info(
            "Router initialized",
            extra={
                "strategy": self.load_balance_strategy.value,
                "affinity_enabled": affinity_enabled,
                "static_worker": static_worker_addr is not None,
            },
        )

    async def initialize(self) -> None:
        """Initialize router resources (M9).

        Connects to Redis for session affinity if enabled.
        This is a no-op for M2 static routing.

        Raises:
            ConnectionError: If Redis connection fails (M9 only)
        """
        # M9: Connect to Redis for session affinity
        if self.affinity_enabled and self.redis_url:
            try:
                self._redis = aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,
                )
                await self._redis.ping()
                self._redis_connected = True
                logger.info("Router connected to Redis for session affinity")
            except Exception as e:
                logger.warning(
                    f"Failed to connect to Redis for session affinity: {e}. "
                    "Continuing without affinity support."
                )
                # Don't raise - degrade gracefully
                self._redis_connected = False

        logger.info("Router initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown router and cleanup resources (M9).

        Closes Redis connection gracefully.
        """
        if self._redis and self._redis_connected:
            await self._redis.close()
            self._redis_connected = False
            logger.info("Router disconnected from Redis")

    async def select_worker(
        self,
        language: str | None = None,
        model_id: str | None = None,
        capabilities: dict[str, Any] | None = None,
        session_id: str | None = None,
        strategy: RoutingStrategy | None = None,
    ) -> str:
        """Select best available worker for synthesis request.

        M2 behavior: Always returns static_worker_addr if configured,
        otherwise returns first discovered worker.

        M9 behavior: Filters workers by capabilities, checks session
        affinity, and selects based on load balancing strategy.

        Args:
            language: Target language code (e.g., "en", "zh")
            model_id: Optional specific model ID to prefer
            capabilities: Optional required capabilities dict:
                - streaming: bool
                - zero_shot: bool
                - lora: bool
                - cpu_ok: bool
            session_id: Session identifier for affinity tracking (M9)
            strategy: Routing strategy override (M9)

        Returns:
            Worker gRPC address (e.g., "grpc://localhost:7001")

        Raises:
            RuntimeError: If no suitable worker found
        """
        start_time = time.monotonic()

        try:
            # M2: Static routing
            # When static_worker_addr is configured, bypass all discovery and
            # capability matching logic. This provides a simple, reliable routing
            # mechanism for single-worker deployments and testing scenarios.
            if self.static_worker_addr:
                logger.debug(f"Using static worker address: {self.static_worker_addr}")
                self._record_metrics(0.0, affinity_hit=False)
                return self.static_worker_addr

            # M9: Dynamic routing with capability matching and affinity
            worker = await self.select_worker_dynamic(
                language=language,
                model_id=model_id,
                capabilities=capabilities,
                session_id=session_id,
                strategy=strategy,
            )

            # Record routing decision latency
            latency_ms = (time.monotonic() - start_time) * 1000
            self._record_metrics(latency_ms, affinity_hit=False)

            return worker.addr

        except Exception as e:
            logger.error(
                f"Worker selection failed: {e}",
                exc_info=True,
                extra={
                    "language": language,
                    "model_id": model_id,
                    "session_id": session_id,
                },
            )
            raise

    async def select_worker_dynamic(
        self,
        language: str | None = None,
        model_id: str | None = None,
        capabilities: dict[str, Any] | None = None,
        session_id: str | None = None,
        strategy: RoutingStrategy | None = None,
    ) -> WorkerRegistration:
        """Select worker with full capability matching and load balancing (M9).

        This is the M9 implementation for dynamic routing with:
        1. Filter by language support
        2. Filter by capabilities
        3. Filter by health status
        4. Check session affinity
        5. Prefer workers with resident model
        6. Apply load balancing strategy

        Args:
            language: Target language code
            model_id: Optional specific model ID to prefer
            capabilities: Optional required capabilities
            session_id: Session identifier for affinity (M9)
            strategy: Routing strategy override (M9)

        Returns:
            Selected worker registration

        Raises:
            RuntimeError: If no suitable worker found
        """
        start_time = time.monotonic()

        # Use provided strategy or config default
        selected_strategy = strategy or self.load_balance_strategy

        try:
            # Get all registered workers
            workers = await self.registry.get_workers()

            if not workers:
                self.metrics["no_workers_errors"] += 1
                error_msg = "No workers available in registry"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.debug(
                f"Found {len(workers)} registered workers",
                extra={"worker_count": len(workers)},
            )

            # STEP 1: Filter by language support
            if language:
                initial_count = len(workers)
                workers = [w for w in workers if language in w.capabilities.get("languages", [])]

                if not workers:
                    self.metrics["capability_filters"] += 1
                    raise RuntimeError(f"No workers support language '{language}'")

                logger.debug(
                    f"Language filter: {initial_count} → {len(workers)} workers",
                    extra={"language": language, "remaining": len(workers)},
                )

            # STEP 2: Filter by capabilities
            if capabilities:
                initial_count = len(workers)
                workers = self._filter_by_capabilities(workers, capabilities)

                if not workers:
                    self.metrics["capability_filters"] += 1
                    raise RuntimeError(
                        f"No workers match required capabilities: {capabilities}"
                    )

                logger.debug(
                    f"Capability filter: {initial_count} → {len(workers)} workers",
                    extra={"capabilities": capabilities, "remaining": len(workers)},
                )

            # STEP 3: Filter by health status (M9)
            initial_count = len(workers)
            workers = self._filter_by_health(workers)

            if not workers:
                self.metrics["health_filters"] += 1
                self.metrics["no_workers_errors"] += 1
                raise RuntimeError("All workers are unhealthy")

            if len(workers) < initial_count:
                logger.debug(
                    f"Health filter: {initial_count} → {len(workers)} workers",
                    extra={"remaining": len(workers)},
                )

            # STEP 4: Check session affinity (M9)
            if session_id and self.affinity_enabled:
                affinity_worker = await self._check_session_affinity(session_id, workers)

                if affinity_worker:
                    logger.info(
                        f"Session affinity hit: {affinity_worker.name}",
                        extra={"session_id": session_id, "worker": affinity_worker.name},
                    )
                    self.metrics["affinity_hits"] += 1

                    # Record metrics and return
                    latency_ms = (time.monotonic() - start_time) * 1000
                    self._record_metrics(latency_ms, affinity_hit=True)

                    return affinity_worker
                else:
                    self.metrics["affinity_misses"] += 1

            # STEP 5: Prefer workers with resident model
            if model_id and self.prefer_resident_models:
                resident_workers = [w for w in workers if model_id in w.resident_models]

                if resident_workers:
                    workers = resident_workers
                    logger.debug(
                        f"Filtered to {len(workers)} workers with resident model '{model_id}'"
                    )

            # STEP 6: Apply load balancing strategy
            selected = self._apply_load_balancing(workers, selected_strategy)

            # STEP 7: Update session affinity (M9)
            if session_id and self.affinity_enabled:
                await self._update_session_affinity(session_id, selected.name)

            logger.info(
                f"Selected worker '{selected.name}' at {selected.addr}",
                extra={
                    "worker": selected.name,
                    "strategy": selected_strategy.value,
                    "language": language,
                    "model_id": model_id,
                    "session_id": session_id,
                },
            )

            return selected

        except RuntimeError:
            # Re-raise known errors
            raise
        except Exception as e:
            logger.error(
                f"Dynamic worker selection failed: {e}",
                exc_info=True,
                extra={
                    "language": language,
                    "model_id": model_id,
                    "session_id": session_id,
                },
            )
            raise RuntimeError(f"Dynamic worker selection failed: {e}") from e

    def _filter_by_capabilities(
        self,
        workers: list[WorkerRegistration],
        required_capabilities: dict[str, Any],
    ) -> list[WorkerRegistration]:
        """Filter workers by required capabilities.

        Capability Matching Strategy:
        1. Boolean flags: Exact match required
           - If streaming=True is required, worker MUST have streaming=True
           - Workers can have additional True flags (superset matching allowed)

        2. Language lists: "Any match" strategy
           - At least ONE required language must be in worker's language list
           - Example: Required ["en"] matches worker with ["en", "zh"]
           - Example: Required ["en", "zh"] matches worker with ["en", "zh", "fr"]
           - Example: Required ["fr"] does NOT match worker with ["en", "zh"]

        3. Other types: Exact equality required
           - Used for numeric constraints or string identifiers

        Edge Cases:
        - Worker with MORE capabilities than required: MATCH (superset allowed)
        - Worker with FEWER capabilities: NO MATCH
        - Missing capability key in worker data: NO MATCH (defensive)
        - Empty language list in requirements: NO MATCH (at least one required)

        Args:
            workers: List of worker registrations to filter
            required_capabilities: Required capability flags and constraints

        Returns:
            Filtered list of workers matching ALL requirements

        Example:
            >>> required = {"streaming": True, "languages": ["en"]}
            >>> filtered = filter_by_capabilities(workers, required)
            # Returns only workers with streaming=True AND supporting English
        """
        filtered: list[WorkerRegistration] = []

        for worker in workers:
            matches = True

            # Check each required capability against worker's capabilities
            for key, required_value in required_capabilities.items():
                worker_value = worker.capabilities.get(key)

                # Boolean capability check (exact match)
                if isinstance(required_value, bool):
                    if worker_value != required_value:
                        matches = False
                        break

                # List membership check (any match for languages)
                elif isinstance(required_value, list):
                    worker_list = worker_value if isinstance(worker_value, list) else []
                    if not any(item in worker_list for item in required_value):
                        matches = False
                        break

                # Exact match for other types
                else:
                    if worker_value != required_value:
                        matches = False
                        break

            # Worker matched all requirements - include in results
            if matches:
                filtered.append(worker)

        return filtered

    def _filter_by_health(self, workers: list[WorkerRegistration]) -> list[WorkerRegistration]:
        """Filter workers by health status (M9).

        Checks worker heartbeat timestamp to determine health.
        Workers are considered unhealthy if their last heartbeat
        is older than 2x the health check interval.

        Args:
            workers: List of workers to filter

        Returns:
            Filtered list of healthy workers
        """
        current_time = time.time()
        health_threshold = self.health_check_interval * 2  # 2x TTL

        filtered = []
        for worker in workers:
            time_since_heartbeat = current_time - worker.last_heartbeat_ts

            if time_since_heartbeat < health_threshold:
                filtered.append(worker)
            else:
                logger.warning(
                    f"Worker {worker.name} is unhealthy "
                    f"(last heartbeat: {time_since_heartbeat:.1f}s ago)",
                    extra={
                        "worker": worker.name,
                        "heartbeat_age": time_since_heartbeat,
                    },
                )

        return filtered

    async def _check_session_affinity(
        self,
        session_id: str,
        available_workers: list[WorkerRegistration],
    ) -> WorkerRegistration | None:
        """Check session affinity and return assigned worker if available (M9).

        Args:
            session_id: Session identifier
            available_workers: List of available workers

        Returns:
            Worker assigned to session, or None if no affinity
        """
        if not self._redis_connected or not self._redis:
            return None

        try:
            # Lookup session affinity in Redis
            key = f"{self.redis_key_prefix}{session_id}"
            worker_name = await self._redis.get(key)

            if not worker_name:
                logger.debug(
                    f"No session affinity for session '{session_id}'",
                    extra={"session_id": session_id},
                )
                return None

            # Find worker in available list
            for worker in available_workers:
                if worker.name == worker_name:
                    return worker

            logger.warning(
                f"Session affinity worker '{worker_name}' not available",
                extra={"session_id": session_id, "worker": worker_name},
            )
            return None

        except Exception as e:
            logger.warning(
                f"Failed to check session affinity: {e}",
                extra={"session_id": session_id},
            )
            return None

    async def _update_session_affinity(self, session_id: str, worker_name: str) -> None:
        """Update session affinity in Redis (M9).

        Args:
            session_id: Session identifier
            worker_name: Worker name to assign
        """
        if not self._redis_connected or not self._redis:
            return

        try:
            key = f"{self.redis_key_prefix}{session_id}"
            await self._redis.setex(
                key,
                self.affinity_ttl_seconds,
                worker_name,
            )

            logger.debug(
                f"Updated session affinity: {session_id} → {worker_name}",
                extra={"session_id": session_id, "worker": worker_name},
            )

        except Exception as e:
            logger.warning(
                f"Failed to update session affinity: {e}",
                extra={"session_id": session_id, "worker": worker_name},
            )

    def _apply_load_balancing(
        self,
        workers: list[WorkerRegistration],
        strategy: RoutingStrategy,
    ) -> WorkerRegistration:
        """Apply load balancing strategy to select worker.

        Load Balancing Strategies:

        1. "queue_depth" (default): Minimize queuing delay
           - Selects worker with lowest queue_depth metric
           - Best for latency-sensitive applications

        2. "latency": Optimize for fastest synthesis
           - Selects worker with lowest RTF (real-time factor)
           - Best when synthesis speed varies across workers

        3. "round_robin": Fair distribution
           - Cycles through workers in order
           - Simple and predictable for testing

        4. "random": Random selection
           - For testing and load distribution analysis

        Args:
            workers: Non-empty list of candidate workers
            strategy: Routing strategy to apply

        Returns:
            Selected worker based on load balancing strategy

        Raises:
            ValueError: If workers list is empty
        """
        if not workers:
            raise ValueError("Cannot apply load balancing to empty worker list")

        # Fast path: Only one worker available
        if len(workers) == 1:
            return workers[0]

        # Apply strategy using WorkerSelector
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return self.selector.round_robin(workers)
        elif strategy == RoutingStrategy.LEAST_LOADED:
            return self.selector.least_loaded(workers)
        elif strategy == RoutingStrategy.LEAST_LATENCY:
            return self.selector.least_latency(workers)
        elif strategy == RoutingStrategy.RANDOM:
            return self.selector.random(workers)
        else:
            logger.warning(
                f"Unknown routing strategy '{strategy}', using first worker"
            )
            return workers[0]

    def _record_metrics(self, latency_ms: float, affinity_hit: bool) -> None:
        """Record routing metrics (M9).

        Args:
            latency_ms: Routing decision latency in milliseconds
            affinity_hit: Whether session affinity was used
        """
        self.metrics["total_decisions"] += 1
        self.metrics["total_latency_ms"] += latency_ms

        logger.debug(
            f"Routing decision took {latency_ms:.2f}ms",
            extra={
                "latency_ms": latency_ms,
                "affinity_hit": affinity_hit,
            },
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get routing metrics (M9).

        Returns:
            Dictionary with routing metrics
        """
        total_decisions = self.metrics["total_decisions"]

        if total_decisions == 0:
            avg_latency_ms = 0.0
            affinity_hit_rate = 0.0
        else:
            avg_latency_ms = self.metrics["total_latency_ms"] / total_decisions
            affinity_hit_rate = (
                self.metrics["affinity_hits"] / total_decisions * 100.0
            )

        return {
            "total_decisions": total_decisions,
            "affinity_hits": self.metrics["affinity_hits"],
            "affinity_misses": self.metrics["affinity_misses"],
            "affinity_hit_rate_percent": affinity_hit_rate,
            "no_workers_errors": self.metrics["no_workers_errors"],
            "capability_filters": self.metrics["capability_filters"],
            "health_filters": self.metrics["health_filters"],
            "avg_latency_ms": avg_latency_ms,
        }

    async def get_worker_info(self, worker_name: str) -> WorkerRegistration | None:
        """Retrieve information about specific worker.

        Args:
            worker_name: Worker name identifier

        Returns:
            Worker registration if found, None otherwise
        """
        try:
            return await self.registry.get_worker_by_name(worker_name)
        except ConnectionError as e:
            logger.error(f"Failed to retrieve worker info for '{worker_name}': {e}")
            return None

    async def health_check(self) -> bool:
        """Check if router can access worker registry.

        Returns:
            True if registry is accessible, False otherwise
        """
        # If using static routing, always healthy
        if self.static_worker_addr:
            return True

        # Check Redis health for dynamic routing
        return await self.registry.health_check()
