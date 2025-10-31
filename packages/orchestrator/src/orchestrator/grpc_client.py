"""gRPC client for communicating with TTS workers.

This module provides a high-level async interface for the orchestrator to
interact with TTS worker services via gRPC. It handles connection management,
session lifecycle, streaming synthesis, control commands, and model operations.

Example usage:
    Basic session with synthesis:
        >>> client = TTSWorkerClient("localhost:7001")
        >>> await client.connect()
        >>> await client.start_session("session-123", "mock-440hz")
        >>> async for frame in client.synthesize(["Hello, world!"]):
        ...     # Process audio frame
        ...     audio_data = frame.audio_data
        >>> await client.end_session()
        >>> await client.disconnect()

    Using control commands for barge-in:
        >>> await client.start_session("session-456", "cosyvoice2-en-base")
        >>> # Start synthesis in background
        >>> synthesis_task = asyncio.create_task(
        ...     client.synthesize(["Long text to synthesize..."])
        ... )
        >>> # Pause synthesis on user interrupt
        >>> await client.control("PAUSE")
        >>> # Resume when user stops speaking
        >>> await client.control("RESUME")
        >>> # Stop synthesis completely
        >>> await client.control("STOP")

    Model management:
        >>> # List available models
        >>> models = await client.list_models()
        >>> for model in models:
        ...     print(f"{model.model_id}: {model.family}")
        >>> # Load a model dynamically
        >>> await client.load_model("xtts-v2-en")
        >>> # Query capabilities
        >>> caps = await client.get_capabilities()
        >>> print(f"Streaming: {caps.capabilities.streaming}")
        >>> print(f"Resident models: {caps.resident_models}")
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import cast

import grpc

from rpc.generated import tts_pb2, tts_pb2_grpc

logger = logging.getLogger(__name__)


class TTSWorkerClient:
    """Async gRPC client for TTS worker communication.

    This client manages the connection to a TTS worker and provides methods
    for session management, streaming synthesis, and control commands. It
    encapsulates all gRPC communication details and provides a clean async API.

    Attributes:
        address: Worker address (e.g., "localhost:7001")
        channel: gRPC async channel
        stub: TTSService stub
        session_id: Active session ID (if any)
        is_connected: Connection status flag
    """

    def __init__(self, address: str) -> None:
        """Initialize TTS worker client.

        Args:
            address: Worker gRPC address (host:port)
        """
        self.address = address
        self.channel: grpc.aio.Channel | None = None
        self.stub: tts_pb2_grpc.TTSServiceStub | None = None
        self.session_id: str | None = None
        self.is_connected: bool = False
        logger.info("TTSWorkerClient initialized", extra={"address": address})

    async def connect(
        self,
        max_retries: int = 3,
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 10.0,
        timeout_s: float = 5.0,
    ) -> None:
        """Establish connection to worker with retry logic.

        Creates gRPC channel and stub for communication. Tests the connection
        by calling GetCapabilities to ensure the worker is reachable. Uses
        exponential backoff for retries.

        Args:
            max_retries: Maximum number of connection attempts
            initial_backoff_s: Initial backoff delay in seconds
            max_backoff_s: Maximum backoff delay in seconds
            timeout_s: Timeout for each connection attempt

        Raises:
            grpc.RpcError: If connection fails after all retries
        """
        logger.info(
            "Connecting to worker with retry logic",
            extra={
                "address": self.address,
                "max_retries": max_retries,
                "initial_backoff_s": initial_backoff_s,
            },
        )

        self.channel = grpc.aio.insecure_channel(self.address)
        # Generated gRPC stubs are untyped
        self.stub = tts_pb2_grpc.TTSServiceStub(self.channel)  # type: ignore[no-untyped-call]

        # Test connection with retries
        backoff = initial_backoff_s
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                # Test connection with timeout
                await asyncio.wait_for(self.get_capabilities(), timeout=timeout_s)
                self.is_connected = True
                logger.info(
                    "Connected to worker successfully",
                    extra={"address": self.address, "attempt": attempt + 1},
                )
                return
            except (TimeoutError, grpc.RpcError) as e:
                last_error = e
                self.is_connected = False

                if attempt < max_retries - 1:
                    logger.warning(
                        "Connection attempt failed, retrying...",
                        extra={
                            "address": self.address,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "backoff_s": backoff,
                            "error": str(e),
                        },
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff_s)
                else:
                    logger.error(
                        "Failed to connect to worker after all retries",
                        extra={
                            "address": self.address,
                            "attempts": max_retries,
                            "error": str(e),
                        },
                    )

        # All retries failed
        if last_error:
            raise last_error

    async def check_connection(self) -> bool:
        """Check if connection to worker is healthy.

        Attempts to call GetCapabilities to verify the worker is reachable.
        Updates the is_connected flag based on the result.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            await asyncio.wait_for(self.get_capabilities(), timeout=3.0)
            if not self.is_connected:
                logger.info("Worker connection restored", extra={"address": self.address})
            self.is_connected = True
            return True
        except (TimeoutError, grpc.RpcError) as e:
            if self.is_connected:
                logger.warning(
                    "Worker connection lost",
                    extra={"address": self.address, "error": str(e)},
                )
            self.is_connected = False
            return False

    async def wait_for_connection(
        self,
        max_wait_s: float = 60.0,
        check_interval_s: float = 2.0,
    ) -> bool:
        """Wait for worker to become available.

        Polls the worker connection status until it becomes available or timeout.
        Useful for startup when worker may not be ready immediately.

        Args:
            max_wait_s: Maximum time to wait in seconds
            check_interval_s: Interval between connection checks

        Returns:
            True if worker became available, False if timeout
        """
        logger.info(
            "Waiting for worker to become available",
            extra={
                "address": self.address,
                "max_wait_s": max_wait_s,
                "check_interval_s": check_interval_s,
            },
        )

        start_time = asyncio.get_event_loop().time()
        attempt = 0

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= max_wait_s:
                logger.warning(
                    "Timeout waiting for worker",
                    extra={
                        "address": self.address,
                        "elapsed_s": elapsed,
                        "attempts": attempt,
                    },
                )
                return False

            attempt += 1
            if await self.check_connection():
                logger.info(
                    "Worker is now available",
                    extra={
                        "address": self.address,
                        "elapsed_s": elapsed,
                        "attempts": attempt,
                    },
                )
                return True

            # Log progress every 10 seconds
            if attempt % 5 == 0:
                logger.info(
                    "Still waiting for worker...",
                    extra={
                        "address": self.address,
                        "elapsed_s": elapsed,
                        "attempts": attempt,
                    },
                )

            await asyncio.sleep(check_interval_s)

    async def disconnect(self) -> None:
        """Close connection to worker.

        Ends any active session and closes the gRPC channel gracefully.
        Safe to call multiple times.
        """
        if self.session_id:
            try:
                await self.end_session()
            except Exception as e:
                logger.warning("Error ending session during disconnect", extra={"error": str(e)})

        if self.channel:
            logger.info("Disconnecting from worker", extra={"address": self.address})
            await self.channel.close()
            self.channel = None
            self.stub = None
            self.is_connected = False

    async def start_session(
        self, session_id: str, model_id: str = "mock-440hz", options: dict[str, str] | None = None
    ) -> bool:
        """Start a new TTS session.

        Creates a new session with the worker for the specified model. Each
        session is isolated and can be controlled independently.

        Args:
            session_id: Unique session identifier
            model_id: Model to use for synthesis
            options: Additional session options (key-value pairs)

        Returns:
            True if session started successfully, False otherwise

        Raises:
            RuntimeError: If not connected to worker
        """
        if not self.stub:
            raise RuntimeError("Not connected to worker")

        if not self.is_connected:
            logger.warning(
                "Attempting to start session while disconnected",
                extra={"session_id": session_id, "address": self.address},
            )
            # Attempt to reconnect
            if not await self.check_connection():
                raise RuntimeError(f"Worker not available at {self.address}")

        logger.info(
            "Starting session",
            extra={"session_id": session_id, "model_id": model_id},
        )

        request = tts_pb2.StartSessionRequest(
            session_id=session_id,
            model_id=model_id,
            options=options or {},
        )

        response = await self.stub.StartSession(request)

        if response.success:
            self.session_id = session_id
            logger.info("Session started", extra={"session_id": session_id})
        else:
            logger.error(
                "Failed to start session",
                extra={"session_id": session_id, "message": response.message},
            )

        return bool(response.success)

    async def end_session(self) -> bool:
        """End the active session.

        Terminates the current session and cleans up resources on the worker.
        Should be called when synthesis is complete or on error.

        Returns:
            True if session ended successfully, False otherwise

        Raises:
            RuntimeError: If no active session
        """
        if not self.session_id:
            raise RuntimeError("No active session")

        if not self.stub:
            raise RuntimeError("Not connected to worker")

        logger.info("Ending session", extra={"session_id": self.session_id})

        request = tts_pb2.EndSessionRequest(session_id=self.session_id)
        response = await self.stub.EndSession(request)

        if response.success:
            logger.info("Session ended", extra={"session_id": self.session_id})
            self.session_id = None
        else:
            logger.warning("Failed to end session", extra={"session_id": self.session_id})

        return bool(response.success)

    async def synthesize(self, text_chunks: list[str]) -> AsyncIterator[tts_pb2.AudioFrame]:
        """Synthesize text to audio frames.

        Sends text chunks to the worker and streams back 20ms audio frames at
        48kHz. This is the main synthesis endpoint used for TTS generation.

        Args:
            text_chunks: List of text strings to synthesize

        Yields:
            AudioFrame messages with 20ms PCM audio at 48kHz

        Raises:
            RuntimeError: If no active session or not connected
        """
        if not self.session_id:
            raise RuntimeError("No active session")

        if not self.stub:
            raise RuntimeError("Not connected to worker")

        logger.info(
            "Starting synthesis",
            extra={"session_id": self.session_id, "chunks": len(text_chunks)},
        )

        async def text_stream() -> AsyncIterator[tts_pb2.TextChunk]:
            """Generate TextChunk stream from text list."""
            for i, text in enumerate(text_chunks):
                chunk = tts_pb2.TextChunk(
                    session_id=self.session_id or "",
                    text=text,
                    is_final=(i == len(text_chunks) - 1),
                    sequence_number=i + 1,
                )
                yield chunk

        frame_count = 0
        async for frame in self.stub.Synthesize(text_stream()):
            frame_count += 1
            logger.debug(
                "Received audio frame",
                extra={
                    "session_id": self.session_id,
                    "sequence": frame.sequence_number,
                    "size": len(frame.audio_data),
                    "is_final": frame.is_final,
                },
            )
            yield frame

        logger.info(
            "Synthesis completed",
            extra={"session_id": self.session_id, "frames": frame_count},
        )

    async def control(self, command: str) -> bool:
        """Send control command to worker.

        Sends a runtime control command to the active session. Commands are
        executed within 50ms to meet barge-in latency requirements.

        Args:
            command: Control command (PAUSE, RESUME, STOP, RELOAD)

        Returns:
            True if command executed successfully, False otherwise

        Raises:
            RuntimeError: If no active session or not connected
            ValueError: If command is invalid
        """
        if not self.session_id:
            raise RuntimeError("No active session")

        if not self.stub:
            raise RuntimeError("Not connected to worker")

        # Map command string to enum
        command_map = {
            "PAUSE": tts_pb2.PAUSE,
            "RESUME": tts_pb2.RESUME,
            "STOP": tts_pb2.STOP,
            "RELOAD": tts_pb2.RELOAD,
        }

        if command not in command_map:
            raise ValueError(f"Invalid command: {command}")

        logger.info(
            "Sending control command",
            extra={"session_id": self.session_id, "command": command},
        )

        request = tts_pb2.ControlRequest(
            session_id=self.session_id,
            command=command_map[command],
        )

        response = await self.stub.Control(request)

        if response.success:
            logger.info(
                "Control command executed",
                extra={
                    "session_id": self.session_id,
                    "command": command,
                    "timestamp": response.timestamp_ms,
                },
            )
        else:
            logger.error(
                "Control command failed",
                extra={
                    "session_id": self.session_id,
                    "command": command,
                    "message": response.message,
                },
            )

        return bool(response.success)

    async def list_models(self) -> list[tts_pb2.ModelInfo]:
        """List available models.

        Queries the worker for all available TTS models. Returns metadata
        about each model including language support and capabilities.

        Returns:
            List of ModelInfo objects

        Raises:
            RuntimeError: If not connected to worker
        """
        if not self.stub:
            raise RuntimeError("Not connected to worker")

        logger.debug("Listing models")

        request = tts_pb2.ListModelsRequest()
        response = await self.stub.ListModels(request)

        logger.info("Listed models", extra={"count": len(response.models)})
        return list(response.models)

    async def load_model(self, model_id: str, preload_only: bool = False) -> bool:
        """Load a model.

        Dynamically loads a TTS model into memory. Can optionally preload
        without activating for faster session start times.

        Args:
            model_id: Model to load
            preload_only: If True, only preload without activation

        Returns:
            True if model loaded successfully, False otherwise

        Raises:
            RuntimeError: If not connected to worker
        """
        if not self.stub:
            raise RuntimeError("Not connected to worker")

        logger.info("Loading model", extra={"model_id": model_id})

        request = tts_pb2.LoadModelRequest(
            model_id=model_id,
            preload_only=preload_only,
        )

        response = await self.stub.LoadModel(request)

        if response.success:
            logger.info(
                "Model loaded",
                extra={"model_id": model_id, "duration_ms": response.load_duration_ms},
            )
        else:
            logger.error(
                "Failed to load model",
                extra={"model_id": model_id, "message": response.message},
            )

        return bool(response.success)

    async def unload_model(self, model_id: str) -> bool:
        """Unload a model.

        Unloads a TTS model from memory to free resources. Model will need
        to be reloaded before it can be used again.

        Args:
            model_id: Model to unload

        Returns:
            True if model unloaded successfully, False otherwise

        Raises:
            RuntimeError: If not connected to worker
        """
        if not self.stub:
            raise RuntimeError("Not connected to worker")

        logger.info("Unloading model", extra={"model_id": model_id})

        request = tts_pb2.UnloadModelRequest(model_id=model_id)
        response = await self.stub.UnloadModel(request)

        if response.success:
            logger.info("Model unloaded", extra={"model_id": model_id})
        else:
            logger.error(
                "Failed to unload model",
                extra={"model_id": model_id, "message": response.message},
            )

        return bool(response.success)

    async def get_capabilities(self) -> tts_pb2.GetCapabilitiesResponse:
        """Get worker capabilities.

        Queries the worker for its capabilities, resident models, and current
        performance metrics. Used for routing decisions and health checks.

        Returns:
            GetCapabilitiesResponse with capabilities, models, and metrics

        Raises:
            RuntimeError: If not connected to worker
        """
        if not self.stub:
            raise RuntimeError("Not connected to worker")

        logger.debug("Getting capabilities")

        request = tts_pb2.GetCapabilitiesRequest()
        response = await self.stub.GetCapabilities(request)

        logger.debug(
            "Got capabilities",
            extra={
                "streaming": response.capabilities.streaming,
                "resident_models": list(response.resident_models),
            },
        )

        return cast(tts_pb2.GetCapabilitiesResponse, response)
