"""Base transport abstraction for client connections.

Defines the interface that all transport implementations (WebSocket, LiveKit)
must implement to provide a unified session management layer.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class TransportSession(ABC):
    """Base class for transport-specific sessions.

    Each transport implementation provides a concrete session type that handles
    the specifics of audio/text transmission while conforming to this interface.
    """

    @abstractmethod
    async def send_audio_frame(self, frame: bytes) -> None:
        """Send a 20ms PCM audio frame to the client.

        Args:
            frame: PCM audio data (960 samples @ 48kHz, 16-bit LE)
                  Expected size: 1920 bytes (960 samples * 2 bytes)

        Raises:
            ConnectionError: If the connection is closed or broken
            ValueError: If frame size is invalid
        """
        pass

    @abstractmethod
    async def receive_text(self) -> AsyncIterator[str]:
        """Receive text chunks from the client.

        Yields text chunks as they arrive from the client. Each chunk may be
        partial or complete depending on the transport implementation.

        Yields:
            str: Text chunk from client

        Raises:
            ConnectionError: If the connection is closed or broken
        """
        # Using yield to make this an async generator
        if False:
            yield ""

    @abstractmethod
    async def close(self) -> None:
        """Clean session shutdown.

        Performs graceful cleanup of transport resources, closes connections,
        and ensures any pending data is flushed or discarded appropriately.
        """
        pass

    @property
    @abstractmethod
    def session_id(self) -> str:
        """Unique session identifier for logging and tracking."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the session connection is still active."""
        pass


class Transport(ABC):
    """Base transport implementation.

    Manages the lifecycle of a specific transport type (e.g., WebSocket server,
    LiveKit agent) and creates sessions for incoming client connections.
    """

    @abstractmethod
    async def start(self) -> None:
        """Start the transport server.

        Initialize and bind the transport server to begin accepting connections.
        This should be non-blocking and return once the server is ready.

        Raises:
            RuntimeError: If the transport fails to start
            OSError: If port binding fails (for network transports)
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport server.

        Gracefully shut down the transport, closing all active sessions and
        releasing resources.
        """
        pass

    @abstractmethod
    async def accept_session(self) -> TransportSession:
        """Accept a new client session.

        Blocks until a new client connection is established, then returns
        a TransportSession instance for that client.

        Returns:
            TransportSession: New client session

        Raises:
            RuntimeError: If the transport is not running
            ConnectionError: If session acceptance fails
        """
        pass

    @property
    @abstractmethod
    def transport_type(self) -> str:
        """Transport type identifier (e.g., 'websocket', 'livekit')."""
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the transport server is currently running."""
        pass
