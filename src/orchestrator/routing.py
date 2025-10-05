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
        if self.static_worker_addr:
            logger.debug(f"Using static worker address: {self.static_worker_addr}")
            return self.static_worker_addr

        # M2+: Dynamic discovery fallback
        try:
            workers = await self.registry.get_workers()

            if not workers:
                error_msg = (
                    "No workers available in registry and no static worker configured"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # M2: Return first worker (no filtering yet)
            selected = workers[0]
            logger.info(
                f"Selected worker '{selected.name}' at {selected.addr} "
                f"(dynamic discovery)"
            )
            return selected.addr

        except ConnectionError as e:
            # Redis unavailable - fail if no static fallback
            if not self.static_worker_addr:
                error_msg = (
                    f"Redis unavailable and no static worker configured: {e}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

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

        # Filter by language
        if language:
            workers = [
                w
                for w in workers
                if language in w.capabilities.get("languages", [])
            ]
            if not workers:
                raise RuntimeError(f"No workers support language '{language}'")

        # Filter by capabilities
        if capabilities:
            workers = self._filter_by_capabilities(workers, capabilities)
            if not workers:
                raise RuntimeError(
                    f"No workers match required capabilities: {capabilities}"
                )

        # Prefer workers with resident model
        if model_id and self.prefer_resident_models:
            resident_workers = [w for w in workers if model_id in w.resident_models]
            if resident_workers:
                workers = resident_workers
                logger.debug(
                    f"Filtered to {len(workers)} workers with resident model '{model_id}'"
                )

        # Apply load balancing strategy
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

        Args:
            workers: List of worker registrations
            required_capabilities: Required capabilities dict

        Returns:
            Filtered list of workers matching all requirements
        """
        filtered: list[WorkerRegistration] = []

        for worker in workers:
            matches = True
            for key, required_value in required_capabilities.items():
                worker_value = worker.capabilities.get(key)

                # Boolean capability check
                if isinstance(required_value, bool):
                    if worker_value != required_value:
                        matches = False
                        break

                # List membership check (e.g., languages)
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

            if matches:
                filtered.append(worker)

        return filtered

    def _apply_load_balancing(
        self, workers: list[WorkerRegistration]
    ) -> WorkerRegistration:
        """Apply load balancing strategy to select worker.

        Args:
            workers: List of candidate workers (non-empty)

        Returns:
            Selected worker based on load balancing strategy
        """
        if not workers:
            raise ValueError("Cannot apply load balancing to empty worker list")

        if len(workers) == 1:
            return workers[0]

        if self.load_balance_strategy == "queue_depth":
            # Select worker with lowest queue depth
            return min(
                workers,
                key=lambda w: w.metrics.get("queue_depth", float("inf")),
            )

        elif self.load_balance_strategy == "latency":
            # Select worker with best (lowest) RTF
            return min(
                workers,
                key=lambda w: w.metrics.get("rtf", float("inf")),
            )

        elif self.load_balance_strategy == "round_robin":
            # Simple round-robin
            selected = workers[self._round_robin_index % len(workers)]
            self._round_robin_index += 1
            return selected

        else:
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
        if self.static_worker_addr:
            return True

        # Check Redis health
        return await self.registry.health_check()
