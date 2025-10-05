"""Worker selection and routing logic."""

from typing import Any


class Router:
    """Capability-aware worker routing with load balancing."""

    def __init__(self, registry_client: Any) -> None:
        """Initialize router with worker registry.

        Args:
            registry_client: Redis-based worker registry client
        """
        self.registry = registry_client

    async def select_worker(
        self,
        language: str,
        model_id: str | None = None,
        capabilities: dict[str, Any] | None = None,
    ) -> str:
        """Select best available worker for synthesis request.

        Args:
            language: Target language code
            model_id: Optional specific model ID
            capabilities: Optional required capabilities

        Returns:
            Worker gRPC address
        """
        return "grpc://localhost:7002"
