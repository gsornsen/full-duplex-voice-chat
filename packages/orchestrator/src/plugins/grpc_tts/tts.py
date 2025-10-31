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
- Optional parallel synthesis for 2-3x throughput improvement
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import grpc
from livekit.agents import APIConnectOptions, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from rpc.generated import tts_pb2, tts_pb2_grpc

logger = logging.getLogger(__name__)


class TTS(tts.TTS[None]):
    """gRPC TTS plugin for LiveKit Agents.

    Connects to our custom gRPC TTS worker and streams audio frames.
    Supports our Model Manager features (multi-model, hot-swapping, etc.).

    Optionally supports parallel synthesis mode for improved throughput by
    delegating to ParallelTTSWrapper for persistent worker pool management.

    Args:
        worker_address: gRPC worker address (e.g., "localhost:7001")
        model_id: Model ID to use (e.g., "piper-en-us-lessac-medium")
        sample_rate: Output sample rate in Hz (default: 48000)
        num_channels: Number of audio channels (default: 1 for mono)
        parallel_enabled: Enable parallel synthesis mode
        parallel_num_workers: Number of parallel workers (default: 2)
        parallel_max_queue: Max buffered sentences (default: 10)
        parallel_gpu_limit: Max concurrent GPU operations (default: None)

    Example:
        Sequential mode (default):
            ```python
            from plugins.grpc_tts import TTS

            tts_plugin = TTS(
                worker_address="localhost:7001",
                model_id="piper-en-us-lessac-medium",
            )
            ```

        Parallel mode (2-3x throughput):
            ```python
            from plugins.grpc_tts import TTS

            tts_plugin = TTS(
                worker_address="localhost:7001",
                model_id="piper-en-us-lessac-medium",
                parallel_enabled=True,
                parallel_num_workers=2,
                parallel_gpu_limit=2,
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
        parallel_enabled: bool = False,
        parallel_num_workers: int = 2,
        parallel_max_queue: int = 10,
        parallel_gpu_limit: int | None = None,
        # DEPRECATED: Old API for backward compatibility
        parallel_pipeline: Any = None,
    ):
        """Initialize gRPC TTS plugin.

        Args:
            worker_address: gRPC server address
            model_id: TTS model ID
            sample_rate: Output sample rate (48kHz)
            num_channels: Audio channels (1 for mono)
            parallel_enabled: Enable parallel synthesis
            parallel_num_workers: Number of parallel workers
            parallel_max_queue: Max buffered sentences
            parallel_gpu_limit: Max concurrent GPU operations
            parallel_pipeline: DEPRECATED - use parallel_enabled instead
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

        # Parallel synthesis support
        self._parallel_enabled = parallel_enabled or (parallel_pipeline is not None)
        self._parallel_wrapper: Any = None  # ParallelTTSWrapper | None (avoid circular import)

        # Initialize parallel wrapper if enabled
        if self._parallel_enabled:
            from plugins.grpc_tts.parallel_wrapper import ParallelTTSWrapper

            self._parallel_wrapper = ParallelTTSWrapper(
                grpc_client=self,
                num_workers=parallel_num_workers,
                max_sentence_queue=parallel_max_queue,
                max_gpu_concurrent=parallel_gpu_limit,
            )
            logger.info(
                "gRPC TTS plugin created with PARALLEL synthesis",
                extra={
                    "worker_address": worker_address,
                    "model_id": model_id,
                    "sample_rate": sample_rate,
                    "num_workers": parallel_num_workers,
                    "max_queue": parallel_max_queue,
                    "gpu_limit": parallel_gpu_limit,
                },
            )
        else:
            logger.info(
                "gRPC TTS plugin created with SEQUENTIAL synthesis",
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

        If parallel mode is enabled, delegates to ParallelTTSWrapper for
        improved throughput via persistent worker pool. Otherwise, performs
        sequential synthesis.

        Args:
            text: Text to synthesize
            conn_options: API connection options

        Returns:
            ChunkedStream instance for streaming audio frames
        """
        # Use parallel wrapper if enabled
        if self._parallel_wrapper is not None:
            logger.debug(
                f"Using PARALLEL synthesis for: '{text[:50]}...'",
                extra={"mode": "parallel"},
            )
            return self._parallel_wrapper.synthesize(text, conn_options=conn_options)  # type: ignore[no-any-return]

        # Fall back to sequential synthesis
        logger.debug(
            f"Using SEQUENTIAL synthesis for: '{text[:50]}...'",
            extra={"mode": "sequential"},
        )
        return self._synthesize_sequential(text, conn_options=conn_options)

    def _synthesize_sequential(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "ChunkedStream":
        """Synthesize text using sequential synthesis path (bypasses parallel wrapper).

        This method is used internally by the parallel wrapper's workers to perform
        actual synthesis without routing back through the parallel wrapper (which
        would cause infinite recursion).

        Args:
            text: Text to synthesize
            conn_options: API connection options

        Returns:
            ChunkedStream instance for streaming audio frames
        """
        logger.debug(
            f"Using SEQUENTIAL synthesis (direct) for: '{text[:50]}...'",
            extra={"mode": "sequential-direct"},
        )
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def start_parallel_mode(self) -> None:
        """Start parallel synthesis background worker.

        This method should be called after the TTS plugin is initialized and
        before synthesis begins. It starts the persistent worker pool that
        processes sentences in parallel.

        Only has effect if parallel_enabled=True during initialization.
        Idempotent: Safe to call multiple times.
        """
        if self._parallel_wrapper is not None:
            await self._parallel_wrapper.start()
            logger.info("Parallel synthesis mode started")
        else:
            logger.debug(
                "start_parallel_mode() called but parallel mode not enabled, skipping"
            )

    async def stop_parallel_mode(self) -> None:
        """Stop parallel synthesis background worker.

        Gracefully shuts down the persistent worker pool and drains any
        pending sentences. Should be called during cleanup.

        Only has effect if parallel_enabled=True during initialization.
        Idempotent: Safe to call multiple times.
        """
        if self._parallel_wrapper is not None:
            await self._parallel_wrapper.stop()
            logger.info("Parallel synthesis mode stopped")
        else:
            logger.debug(
                "stop_parallel_mode() called but parallel mode not enabled, skipping"
            )

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
        # Stop parallel mode if active
        if self._parallel_wrapper is not None:
            await self.stop_parallel_mode()

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

    Used for sequential synthesis mode (default).
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
