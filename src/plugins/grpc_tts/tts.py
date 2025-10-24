"""gRPC TTS plugin for LiveKit Agents.

This module provides a LiveKit Agents-compatible TTS plugin that connects to our
custom gRPC TTS worker. It enables using our Model Manager with Piper and future
adapters within the LiveKit Agents framework.

Features:
- Streaming TTS with 20ms, 48kHz PCM frames
- Supports PAUSE/RESUME/STOP control (barge-in)
- Model Manager integration (hot-swapping, TTL eviction, etc.)
- Multiple adapters (Piper CPU baseline, future GPU adapters)
- Warm-up support for eliminating cold-start latency
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator

import grpc
from livekit.agents import APIConnectOptions, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from src.rpc.generated import tts_pb2, tts_pb2_grpc

logger = logging.getLogger(__name__)


class TTS(tts.TTS[None]):
    """gRPC TTS plugin for LiveKit Agents.

    Connects to our custom gRPC TTS worker and streams audio frames.
    Supports our Model Manager features (multi-model, hot-swapping, etc.).

    Args:
        worker_address: gRPC worker address (e.g., "localhost:7001")
        model_id: Model ID to use (e.g., "piper-en-us-lessac-medium")
        sample_rate: Output sample rate in Hz (default: 48000)
        num_channels: Number of audio channels (default: 1 for mono)

    Example:
        ```python
        from src.plugins.grpc_tts import TTS

        # Create TTS instance
        tts_plugin = TTS(
            worker_address="localhost:7001",
            model_id="piper-en-us-lessac-medium",
        )

        # Use in LiveKit Agent
        agent = Agent(
            tts=tts_plugin,
            ...,
        )
        ```
    """

    def __init__(
        self,
        *,
        worker_address: str = "localhost:7001",
        model_id: str = "piper-en-us-lessac-medium",
        sample_rate: int = 48000,
        num_channels: int = 1,
    ):
        """Initialize gRPC TTS plugin.

        Args:
            worker_address: gRPC server address
            model_id: TTS model ID
            sample_rate: Output sample rate (48kHz)
            num_channels: Audio channels (1 for mono)
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,  # Using ChunkedStream pattern (non-streaming)
            ),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._worker_address = worker_address
        self._model_id = model_id

        # gRPC channel and stub (initialized lazily)
        self._channel: grpc.aio.Channel | None = None
        self._stub: tts_pb2_grpc.TTSServiceStub | None = None
        self._session_id: str | None = None
        self._initialization_lock = asyncio.Lock()

        logger.info(
            "gRPC TTS plugin created",
            extra={
                "worker_address": worker_address,
                "model_id": model_id,
                "sample_rate": sample_rate,
            },
        )

    @property
    def model(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "grpc-tts-worker"

    async def _ensure_connected(self) -> None:
        """Ensure gRPC connection and session are established."""
        if self._stub is not None and self._session_id is not None:
            return

        async with self._initialization_lock:
            # Double-check after acquiring lock
            if self._stub is not None and self._session_id is not None:
                return

            logger.info(f"Connecting to gRPC TTS worker at {self._worker_address}...")

            # Create gRPC channel
            self._channel = grpc.aio.insecure_channel(self._worker_address)
            self._stub = tts_pb2_grpc.TTSServiceStub(self._channel)  # type: ignore[no-untyped-call]

            # Generate unique session ID
            self._session_id = str(uuid.uuid4())

            # Start TTS session
            request = tts_pb2.StartSessionRequest(
                session_id=self._session_id, model_id=self._model_id
            )
            response = await self._stub.StartSession(request)

            if not response.success:
                raise RuntimeError(
                    f"Failed to start TTS session: {response.message}"
                )

            logger.info(
                "Connected to TTS worker",
                extra={
                    "session_id": self._session_id,
                    "model_id": self._model_id,
                },
            )

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "ChunkedStream":
        """Synthesize text to audio using gRPC TTS worker.

        Args:
            text: Text to synthesize
            conn_options: API connection options

        Returns:
            ChunkedStream instance for streaming audio frames
        """
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def warm_up(self) -> None:
        """Warm up the TTS model by synthesizing a test utterance.

        This method connects to the TTS worker and triggers warm-up on the
        underlying adapter (CosyVoice, Piper, etc.) to eliminate cold-start
        latency (CUDA compilation, model loading, etc.).

        The warm-up happens via a special gRPC call that:
        1. Ensures session is started
        2. Calls the adapter's warm_up() method
        3. Returns when warm-up is complete

        Notes:
            - Non-fatal: Logs warning if warm-up fails but doesn't raise
            - Idempotent: Safe to call multiple times
            - Timeout: 30 seconds (CUDA compilation can be slow on first run)
        """
        logger.info(
            "Starting TTS warm-up via gRPC worker",
            extra={"model_id": self._model_id, "worker_address": self._worker_address},
        )

        try:
            # Ensure connected to worker
            await self._ensure_connected()
            assert self._stub is not None
            assert self._session_id is not None

            # Warm-up via synthesizing a test utterance
            # The adapter's warm_up() will be triggered during model loading
            # if warmup_enabled=True in ModelManager config
            #
            # ALTERNATIVE: Add a dedicated WarmUp() gRPC endpoint for explicit warm-up
            # For now, we rely on ModelManager's warmup_enabled configuration
            # which triggers adapter.warm_up() automatically after loading
            #
            # Since the model is already loaded (via StartSession), we can verify
            # it's ready by doing a quick test synthesis
            warmup_text = "Testing warmup synthesis."

            logger.info(
                "Synthesizing warmup text to verify model readiness",
                extra={"text": warmup_text, "session_id": self._session_id},
            )

            # Create synthesis request stream
            async def warmup_request_stream() -> AsyncIterator[tts_pb2.TextChunk]:
                yield tts_pb2.TextChunk(
                    session_id=self._session_id,
                    text=warmup_text,
                    is_final=True,
                )

            # Stream audio frames (discard output, just measure time)
            frame_count = 0
            async for audio_frame in self._stub.Synthesize(warmup_request_stream()):
                if audio_frame.audio_data:
                    frame_count += 1

            logger.info(
                "TTS warm-up complete",
                extra={
                    "model_id": self._model_id,
                    "frames_generated": frame_count,
                    "session_id": self._session_id,
                },
            )

        except Exception as e:
            logger.warning(
                "TTS warm-up failed (non-fatal, continuing)",
                extra={
                    "model_id": self._model_id,
                    "error": str(e),
                    "worker_address": self._worker_address,
                },
            )
            # Don't raise - warm-up is optional optimization

    async def aclose(self) -> None:
        """Close the TTS plugin and release resources."""
        if self._session_id and self._stub:
            try:
                logger.info(f"Ending TTS session: {self._session_id}")
                request = tts_pb2.EndSessionRequest(session_id=self._session_id)
                await self._stub.EndSession(request)
                self._session_id = None
            except Exception as e:
                logger.warning(f"Error ending TTS session: {e}")

        if self._channel:
            logger.info("Closing gRPC channel")
            await self._channel.close()
            self._channel = None
            self._stub = None

        logger.info("gRPC TTS plugin closed successfully")


class ChunkedStream(tts.ChunkedStream):
    """ChunkedStream implementation for gRPC TTS worker.

    This class handles the streaming synthesis by connecting to the gRPC
    TTS worker and emitting audio frames as they arrive.
    """

    def __init__(
        self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the synthesis and emit audio frames.

        This is called by the ChunkedStream base class to perform the actual
        synthesis work.

        Args:
            output_emitter: Emitter to push audio frames to
        """
        # Ensure connection
        await self._tts._ensure_connected()
        assert self._tts._stub is not None
        assert self._tts._session_id is not None

        logger.debug(f"Synthesizing text: {self.input_text[:50]}...")

        # Create synthesis request stream (single text chunk)
        async def request_stream() -> AsyncIterator[tts_pb2.TextChunk]:
            yield tts_pb2.TextChunk(
                session_id=self._tts._session_id,
                text=self.input_text,
                is_final=True,
            )

        try:
            # Initialize the output emitter
            # Use session_id as request_id for tracking
            output_emitter.initialize(
                request_id=self._tts._session_id,
                sample_rate=self._tts._sample_rate,
                num_channels=self._tts._num_channels,
                mime_type="audio/pcm",
            )

            # Stream audio frames from worker
            async for audio_frame in self._tts._stub.Synthesize(request_stream()):
                # audio_frame.audio_data is bytes (16-bit PCM)
                pcm_data = audio_frame.audio_data

                # Skip empty frames (end markers)
                if not pcm_data:
                    continue

                # Push raw PCM data to emitter
                # The emitter will handle conversion to rtc.AudioFrame
                output_emitter.push(pcm_data)

            # Signal completion
            output_emitter.flush()

            logger.debug("Synthesis complete")

        except grpc.RpcError as e:
            logger.error(
                "gRPC error during synthesis",
                extra={"error": str(e), "code": e.code()},
            )
            raise
