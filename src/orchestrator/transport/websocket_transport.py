"""WebSocket transport implementation.

Provides WebSocket-based client connections for the orchestrator, enabling
CLI and other clients to connect without LiveKit/WebRTC.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection
from websockets.protocol import State

from src.orchestrator.audio.packetizer import AudioFramePacketizer
from src.orchestrator.transport.base import Transport, TransportSession
from src.orchestrator.transport.websocket_protocol import (
    ControlMessage,
    ErrorMessage,
    ServerMessage,
    SessionEndMessage,
    SessionStartMessage,
    TextMessage,
)

logger = logging.getLogger(__name__)


class WebSocketSession(TransportSession):
    """WebSocket-based transport session.

    Implements the TransportSession interface for WebSocket connections,
    handling JSON message serialization and PCM audio frame encoding.
    """

    def __init__(self, websocket: ServerConnection, session_id: str) -> None:
        """Initialize WebSocket session.

        Args:
            websocket: WebSocket connection
            session_id: Unique session identifier
        """
        self._websocket = websocket
        self._session_id = session_id
        self._connected = True
        self._packetizer = AudioFramePacketizer()

        # Control message queue (for PAUSE/RESUME/STOP)
        self._control_queue: asyncio.Queue[str] = asyncio.Queue()

        logger.info(
            "WebSocket session initialized",
            extra={"session_id": session_id, "remote": websocket.remote_address},
        )

    @property
    def session_id(self) -> str:
        """Get unique session identifier."""
        return self._session_id

    @property
    def is_connected(self) -> bool:
        """Check if the session connection is still active."""
        return self._connected and self._websocket.state == State.OPEN

    async def send_audio_frame(self, frame: bytes) -> None:
        """Send a 20ms PCM audio frame to the client.

        Args:
            frame: PCM audio data (960 samples @ 48kHz, 16-bit LE)
                  Expected size: 1920 bytes (960 samples * 2 bytes)

        Raises:
            ConnectionError: If the connection is closed or broken
            ValueError: If frame size is invalid
        """
        if not self.is_connected:
            raise ConnectionError("WebSocket connection is closed")

        try:
            # Create audio message with frame metadata
            frame_metadata = self._packetizer.create_frame_metadata(frame)
            message = {
                "type": "audio",
                **frame_metadata,
            }

            # Send as JSON
            await self._websocket.send(json.dumps(message))

            logger.debug(
                "Audio frame sent",
                extra={
                    "session_id": self._session_id,
                    "sequence": frame_metadata["sequence"],
                    "size": len(frame),
                },
            )

        except websockets.exceptions.ConnectionClosed as e:
            self._connected = False
            raise ConnectionError(f"WebSocket connection closed: {e}") from e
        except Exception as e:
            logger.error(
                "Failed to send audio frame",
                extra={"session_id": self._session_id, "error": str(e)},
            )
            raise

    async def receive_text(self) -> AsyncIterator[str]:
        """Receive text chunks from the client.

        Yields text chunks as they arrive from the client. Each chunk may be
        partial or complete depending on the transport implementation.

        Yields:
            str: Text chunk from client

        Raises:
            ConnectionError: If the connection is closed or broken
        """
        try:
            async for raw_message in self._websocket:
                if not isinstance(raw_message, str):
                    logger.warning(
                        "Received non-text WebSocket message, skipping",
                        extra={"session_id": self._session_id},
                    )
                    continue

                try:
                    # Parse JSON message
                    data = json.loads(raw_message)
                    message_type = data.get("type")

                    if message_type == "text":
                        # Validate and extract text
                        text_msg = TextMessage.model_validate(data)
                        logger.debug(
                            "Text message received",
                            extra={
                                "session_id": self._session_id,
                                "text_length": len(text_msg.text),
                                "is_final": text_msg.is_final,
                            },
                        )
                        yield text_msg.text

                    elif message_type == "control":
                        # Handle control messages
                        control_msg = ControlMessage.model_validate(data)
                        logger.info(
                            "Control message received",
                            extra={
                                "session_id": self._session_id,
                                "command": control_msg.command,
                            },
                        )
                        # Queue control command for processing
                        await self._control_queue.put(control_msg.command)

                    else:
                        logger.warning(
                            "Unknown message type",
                            extra={"session_id": self._session_id, "type": message_type},
                        )

                except json.JSONDecodeError as e:
                    logger.error(
                        "Invalid JSON message",
                        extra={"session_id": self._session_id, "error": str(e)},
                    )
                    await self._send_error(f"Invalid JSON: {e}")
                except Exception as e:
                    logger.error(
                        "Error processing message",
                        extra={"session_id": self._session_id, "error": str(e)},
                    )
                    await self._send_error(f"Message processing error: {e}")

        except websockets.exceptions.ConnectionClosed:
            self._connected = False
            logger.info(
                "WebSocket connection closed by client",
                extra={"session_id": self._session_id},
            )
        except Exception as e:
            self._connected = False
            logger.error(
                "Error in receive_text",
                extra={"session_id": self._session_id, "error": str(e)},
            )
            raise ConnectionError(f"WebSocket receive error: {e}") from e

    async def get_control_command(self) -> str | None:
        """Get next control command if available.

        Returns:
            Control command string (PAUSE/RESUME/STOP) or None if queue is empty
        """
        try:
            return self._control_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def send_session_start(self) -> None:
        """Send session start notification to client."""
        message = SessionStartMessage(session_id=self._session_id)
        await self._send_message(message)

    async def send_session_end(self, reason: str = "completed") -> None:
        """Send session end notification to client."""
        message = SessionEndMessage(session_id=self._session_id, reason=reason)
        await self._send_message(message)

    async def _send_error(self, error_msg: str, code: str = "INTERNAL_ERROR") -> None:
        """Send error message to client."""
        message = ErrorMessage(message=error_msg, code=code)
        await self._send_message(message)

    async def _send_message(self, message: ServerMessage) -> None:
        """Send a server message to the client."""
        if not self.is_connected:
            return

        try:
            await self._websocket.send(message.model_dump_json())
        except Exception as e:
            logger.error(
                "Failed to send message",
                extra={"session_id": self._session_id, "error": str(e)},
            )

    async def close(self) -> None:
        """Clean session shutdown.

        Performs graceful cleanup of transport resources, closes connections,
        and ensures any pending data is flushed or discarded appropriately.
        """
        if not self._connected:
            return

        logger.info("Closing WebSocket session", extra={"session_id": self._session_id})

        try:
            # Send session end notification
            await self.send_session_end(reason="closed")

            # Close WebSocket connection
            await self._websocket.close()
        except Exception as e:
            logger.warning(
                "Error during session close",
                extra={"session_id": self._session_id, "error": str(e)},
            )
        finally:
            self._connected = False


class WebSocketTransport(Transport):
    """WebSocket transport server.

    Manages WebSocket server lifecycle and creates WebSocketSession instances
    for incoming client connections.
    """

    def __init__(
        self, host: str = "0.0.0.0",  # noqa: S104
        port: int = 8080, max_connections: int = 100
    ) -> None:
        """Initialize WebSocket transport.

        Args:
            host: Bind host address
            port: Bind port
            max_connections: Maximum concurrent connections
        """
        self._host = host
        self._port = port
        self._max_connections = max_connections
        self._server: Any = None  # websockets.Server type
        self._running = False
        self._session_queue: asyncio.Queue[WebSocketSession] = asyncio.Queue()

        logger.info(
            "WebSocket transport initialized",
            extra={"host": host, "port": port, "max_connections": max_connections},
        )

    @property
    def transport_type(self) -> str:
        """Transport type identifier."""
        return "websocket"

    @property
    def is_running(self) -> bool:
        """Check if the transport server is currently running."""
        return self._running

    async def start(self) -> None:
        """Start the WebSocket server.

        Initialize and bind the transport server to begin accepting connections.
        This should be non-blocking and return once the server is ready.

        Raises:
            RuntimeError: If the transport fails to start
            OSError: If port binding fails
        """
        if self._running:
            raise RuntimeError("WebSocket transport is already running")

        logger.info(
            "Starting WebSocket server", extra={"host": self._host, "port": self._port}
        )

        try:
            self._server = await websockets.serve(
                self._handle_connection,
                self._host,
                self._port,
                max_size=2**20,  # 1MB max message size
            )
            self._running = True

            logger.info(
                "WebSocket server started",
                extra={"host": self._host, "port": self._port},
            )

        except OSError as e:
            logger.error(
                "Failed to bind WebSocket server",
                extra={"host": self._host, "port": self._port, "error": str(e)},
            )
            raise
        except Exception as e:
            logger.error(
                "Failed to start WebSocket server",
                extra={"error": str(e)},
            )
            raise RuntimeError(f"Failed to start WebSocket transport: {e}") from e

    async def stop(self) -> None:
        """Stop the WebSocket server.

        Gracefully shut down the transport, closing all active sessions and
        releasing resources.
        """
        if not self._running:
            return

        logger.info("Stopping WebSocket server")

        self._running = False

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        logger.info("WebSocket server stopped")

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
        if not self._running:
            raise RuntimeError("WebSocket transport is not running")

        # Wait for next session from the queue
        session = await self._session_queue.get()
        return session

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        """Handle incoming WebSocket connection.

        Args:
            websocket: WebSocket connection
        """
        session_id = f"ws-{uuid.uuid4().hex[:12]}"

        logger.info(
            "New WebSocket connection",
            extra={
                "session_id": session_id,
                "remote": websocket.remote_address,
            },
        )

        # Create session
        session = WebSocketSession(websocket, session_id)

        # Send session start notification
        await session.send_session_start()

        # Queue session for orchestrator to accept
        await self._session_queue.put(session)

        # Keep connection alive until closed
        try:
            # Wait for the connection to close
            await websocket.wait_closed()
        except Exception as e:
            logger.error(
                "Error in connection handler",
                extra={"session_id": session_id, "error": str(e)},
            )
        finally:
            logger.info(
                "WebSocket connection closed",
                extra={"session_id": session_id},
            )
