"""Worker selection and routing logic.

This module provides worker selection for TTS synthesis requests.

M2 scope: Static routing to single configured worker.
M9+ scope: Dynamic routing with capability matching and load balancing.
"""

import logging
from typing import Any

from src.orchestrator.registry import WorkerRegistration, WorkerRegistry

logger = logging.getLogger(__name__)


class Router:
    """Capability-aware worker routing with load balancing.

    For M2: Static routing to single pre-configured worker address.
    For M9+: Dynamic routing with:
        - Capability matching (language, streaming, zero-shot, etc.)
        - Resident model preference (avoid model load latency)
        - Load balancing (queue depth, RTF, round-robin)
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        static_worker_addr: str | None = None,
        prefer_resident_models: bool = True,
        load_balance_strategy: str = "queue_depth",
    ) -> None:
        """Initialize router with worker registry.

        Args:
            registry: Redis-based worker registry client
            static_worker_addr: Static worker address for M2 (fallback)
            prefer_resident_models: Prefer workers with model already loaded (M9+)
            load_balance_strategy: Load balancing strategy (M9+):
                - "queue_depth": Prefer workers with lowest queue
                - "latency": Prefer workers with best RTF
                - "round_robin": Simple round-robin selection
        """
        self.registry = registry
        self.static_worker_addr = static_worker_addr
        self.prefer_resident_models = prefer_resident_models
        self.load_balance_strategy = load_balance_strategy

        # Round-robin state (M9+)
        self._round_robin_index = 0

    async def select_worker(
        self,
        language: str | None = None,
        model_id: str | None = None,
        capabilities: dict[str, Any] | None = None,
    ) -> str:
        """Select best available worker for synthesis request.

        M2 behavior: Always returns static_worker_addr if configured,
        otherwise returns first discovered worker.

        M9+ behavior: Filters workers by capabilities and selects based on
        load balancing strategy.

        Args:
            language: Target language code (e.g., "en", "zh")
            model_id: Optional specific model ID to prefer
            capabilities: Optional required capabilities dict:
                - streaming: bool
                - zero_shot: bool
                - lora: bool
                - cpu_ok: bool

        Returns:
            Worker gRPC address (e.g., "grpc://localhost:7001")

        Raises:
            RuntimeError: If no suitable worker found
        """
        # M2: Static routing
        # When static_worker_addr is configured, bypass all discovery and
        # capability matching logic. This provides a simple, reliable routing
        # mechanism for single-worker deployments and testing scenarios.
        if self.static_worker_addr:
            logger.debug(f"Using static worker address: {self.static_worker_addr}")
            return self.static_worker_addr

        # M2+: Dynamic discovery fallback
        # When no static worker is configured, attempt to discover workers
        # from Redis. This enables multi-worker deployments with automatic
        # failover if Redis is available.
        try:
            workers = await self.registry.get_workers()

            if not workers:
                error_msg = (
                    "No workers available in registry and no static worker configured"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # M2: Return first worker (no filtering yet)
            # In M2, we don't perform capability matching or load balancing.
            # We simply return the first available worker. This is sufficient
            # for the initial implementation where we assume all workers are
            # compatible and capable of handling any request.
            selected = workers[0]
            logger.info(
                f"Selected worker '{selected.name}' at {selected.addr} "
                f"(dynamic discovery)"
            )
            return selected.addr

        except ConnectionError as e:
            # Redis unavailable - fail if no static fallback
            # If Redis is down and we don't have a static worker configured,
            # we cannot proceed. This is a hard failure that requires operator
            # intervention (either fix Redis or configure static worker).
            if not self.static_worker_addr:
                error_msg = (
                    f"Redis unavailable and no static worker configured: {e}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

            # Fall back to static worker if Redis is unavailable
            # This provides resilience in case Redis fails temporarily.
            # The static worker acts as a reliable fallback path.
            logger.warning(
                f"Redis unavailable, using static worker fallback: {e}"
            )
            return self.static_worker_addr

    async def select_worker_dynamic(
        self,
        language: str | None = None,
        model_id: str | None = None,
        capabilities: dict[str, Any] | None = None,
    ) -> WorkerRegistration:
        """Select worker with full capability matching and load balancing.

        This is the M9+ implementation for dynamic routing.
        Not used in M2, but interface provided for future implementation.

        Routing Algorithm (M9+):
        1. Filter by language support (must match at least one required language)
        2. Filter by capabilities (all required capabilities must be satisfied)
        3. Prefer workers with model already resident (avoid load latency)
        4. Apply load balancing strategy (queue_depth, latency, or round_robin)

        Args:
            language: Target language code
            model_id: Optional specific model ID to prefer
            capabilities: Optional required capabilities

        Returns:
            Selected worker registration

        Raises:
            RuntimeError: If no suitable worker found
        """
        workers = await self.registry.get_workers()

        if not workers:
            raise RuntimeError("No workers available in registry")

        # STEP 1: Filter by language support
        # Language filtering uses an "any match" strategy: if the request requires
        # English ("en"), any worker supporting English will match, even if it also
        # supports other languages like Chinese ("zh"). This allows multi-language
        # workers to serve requests for any of their supported languages.
        if language:
            workers = [
                w
                for w in workers
                if language in w.capabilities.get("languages", [])
            ]
            if not workers:
                # No workers support this language - hard failure
                # This prevents routing requests to incompatible workers
                raise RuntimeError(f"No workers support language '{language}'")

        # STEP 2: Filter by capabilities
        # Capability filtering is strict: ALL required capabilities must match.
        # Workers can have MORE capabilities than required (superset matching),
        # but they cannot have FEWER. This ensures we only route to workers
        # that can actually fulfill the request requirements.
        if capabilities:
            workers = self._filter_by_capabilities(workers, capabilities)
            if not workers:
                raise RuntimeError(
                    f"No workers match required capabilities: {capabilities}"
                )

        # STEP 3: Prefer workers with resident model
        # Model residency optimization: If a specific model is requested and
        # prefer_resident_models is enabled, we filter to workers that already
        # have the model loaded in VRAM. This avoids 2-5 second model load
        # latency that would delay first audio frame.
        #
        # Important: This is a PREFERENCE, not a requirement. If no workers
        # have the model resident, we fall back to the full worker list and
        # trigger an async model load. This provides resilience while still
        # optimizing the common case where models are already loaded.
        if model_id and self.prefer_resident_models:
            resident_workers = [w for w in workers if model_id in w.resident_models]
            if resident_workers:
                # Found workers with resident model - use them exclusively
                workers = resident_workers
                logger.debug(
                    f"Filtered to {len(workers)} workers with resident model '{model_id}'"
                )
            # else: No workers have model resident - continue with all workers

        # STEP 4: Apply load balancing strategy
        # Load balancing selects the best worker from the filtered candidates
        # based on real-time metrics. Different strategies optimize for
        # different goals (lowest queue vs. fastest synthesis vs. fair distribution).
        selected = self._apply_load_balancing(workers)

        logger.info(
            f"Selected worker '{selected.name}' at {selected.addr} "
            f"(language={language}, model={model_id}, "
            f"strategy={self.load_balance_strategy})"
        )

        return selected

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
                # If we require streaming=True, worker must have streaming=True.
                # If we require zero_shot=False, worker must have zero_shot=False.
                if isinstance(required_value, bool):
                    if worker_value != required_value:
                        matches = False
                        break

                # List membership check (any match for languages)
                # At least one required item must be in worker's list.
                # This handles language lists where we want workers that support
                # ANY of the required languages (OR logic, not AND).
                elif isinstance(required_value, list):
                    worker_list = worker_value if isinstance(worker_value, list) else []
                    if not any(item in worker_list for item in required_value):
                        matches = False
                        break

                # Exact match for other types (numbers, strings, etc.)
                # Used for constraints like max_concurrent_sessions >= 3
                else:
                    if worker_value != required_value:
                        matches = False
                        break

            # Worker matched all requirements - include in results
            if matches:
                filtered.append(worker)

        return filtered

    def _apply_load_balancing(
        self, workers: list[WorkerRegistration]
    ) -> WorkerRegistration:
        """Apply load balancing strategy to select worker.

        Load Balancing Strategies:

        1. "queue_depth" (default): Minimize queuing delay
           - Selects worker with lowest queue_depth metric
           - Best for latency-sensitive applications
           - Automatically distributes load to least-busy workers
           - Handles bursty traffic well

        2. "latency": Optimize for fastest synthesis
           - Selects worker with lowest RTF (real-time factor)
           - RTF < 1.0 means faster than real-time (0.3 = 3x faster)
           - Best when synthesis speed varies across workers (GPU vs CPU)
           - May overload fast workers if not combined with queue_depth

        3. "round_robin": Fair distribution
           - Cycles through workers in order
           - Ignores current load and performance metrics
           - Simple and predictable for testing
           - May send requests to overloaded workers

        Metric Handling:
        - Missing metrics default to float("inf") to avoid selection
        - This ensures workers with no metrics aren't preferred
        - Workers should report metrics via heartbeat for accurate routing

        Edge Cases:
        - Single worker: Return immediately (no comparison needed)
        - Empty list: Raises ValueError (caller should prevent this)
        - Unknown strategy: Log warning and return first worker (safe fallback)

        Args:
            workers: Non-empty list of candidate workers

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

        if self.load_balance_strategy == "queue_depth":
            # Select worker with lowest queue depth
            # Lower queue depth = less waiting time for this request
            # Default to infinity if metric missing to avoid selecting workers
            # that don't report metrics (likely unhealthy)
            return min(
                workers,
                key=lambda w: w.metrics.get("queue_depth", float("inf")),
            )

        elif self.load_balance_strategy == "latency":
            # Select worker with best (lowest) RTF
            # RTF = real-time factor: time to synthesize / duration of audio
            # RTF of 0.3 means 10 seconds of audio generated in 3 seconds
            # Lower RTF = faster synthesis = better user experience
            return min(
                workers,
                key=lambda w: w.metrics.get("rtf", float("inf")),
            )

        elif self.load_balance_strategy == "round_robin":
            # Simple round-robin selection
            # Cycles through workers in order, ignoring current load
            # Uses modulo to wrap around when reaching end of list
            # Thread-safe for single orchestrator instance (no locking needed)
            selected = workers[self._round_robin_index % len(workers)]
            self._round_robin_index += 1
            return selected

        else:
            # Unknown strategy - log warning and use safe fallback
            # This prevents crashes from configuration errors while still
            # allowing requests to be processed
            logger.warning(
                f"Unknown load balance strategy '{self.load_balance_strategy}', "
                f"using first worker"
            )
            return workers[0]

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
        # Static routing doesn't depend on Redis, so it can't fail
        # (assuming the static worker itself is reachable)
        if self.static_worker_addr:
            return True

        # Check Redis health for dynamic routing
        # This verifies we can query worker registry for routing decisions
        return await self.registry.health_check()
