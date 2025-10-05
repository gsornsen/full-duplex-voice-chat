"""WebSocket CLI client for testing."""

from typing import Any


async def run_client(server_url: str, config: dict[str, Any] | None = None) -> None:
    """Run CLI client connecting to orchestrator via WebSocket.

    Args:
        server_url: WebSocket server URL
        config: Optional client configuration
    """
    pass
