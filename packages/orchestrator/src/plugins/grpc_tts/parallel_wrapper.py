"""Parallel TTS wrapper for LiveKit Agent integration.

This module bridges the LiveKit Agent's sequential TTS API with persistent
worker pool for parallel synthesis, enabling 2-3x throughput improvement while
maintaining compatibility with the Agent framework.

Architecture:
    LiveKit Agent → ParallelTTSWrapper → Persistent Worker Pool → gRPC TTS
                         ↓                         ↓
                    sentence buffer          per-sentence audio queue
                         ↓                         ↓
                    BufferedChunkedStream ← audio frames

Key Features:
    - Persistent worker pool across sentences (no restart overhead)
    - Buffered sentence queue for parallel processing
    - Per-sentence audio queues for delivery
    - ChunkedStream compatibility for LiveKit Agent
    - Graceful fallback on errors
    - Session lifecycle management (start/stop/cleanup)

Design reference: Task coordination for parallel synthesis integration
Performance fix: /tmp/PARALLEL_SYNTHESIS_ENDTOEND_FIX.md
"""

import asyncio
import logging
import time
from typing import Any

from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger(__name__)


class ParallelTTSWrapper:
    """Wrapper that manages persistent worker pool for parallel synthesis.

    This class accepts individual sentence synthesis requests from the LiveKit
    Agent and coordinates them through a persistent worker pool for improved
    throughput. Workers synthesize sentences in parallel and deliver audio
    via per-sentence queues.

    Unlike the previous design which restarted workers for each sentence,
    this implementation maintains persistent workers across the entire session,
    eliminating startup overhead and enabling true parallel processing.

    Attributes:
        grpc_client: Underlying gRPC TTS client for synthesis
        num_workers: Number of parallel synthesis workers
        sentence_queue: Buffer for incoming synthesis requests
        worker_tasks: List of worker task handles
        gpu_semaphore: Optional semaphore for GPU concurrency control
    """

    def __init__(
        self,
        grpc_client: Any,  # GrpcTTSPlugin instance
        num_workers: int = 2,
        max_sentence_queue: int = 10,
        max_gpu_concurrent: int | None = None,
    ) -> None:
        """Initialize parallel TTS wrapper.

        Args:
            grpc_client: gRPC TTS client for synthesis
            num_workers: Number of parallel workers (2-3 recommended)
            max_sentence_queue: Max buffered sentences (backpressure threshold)
            max_gpu_concurrent: Max concurrent GPU operations (None = unlimited)
        """
        self.grpc_client = grpc_client
        self.num_workers = num_workers
        self.max_sentence_queue = max_sentence_queue

        # Sentence buffer for queuing synthesis requests
        # Use bounded queue for backpressure (prevents memory exhaustion)
        self.sentence_queue: asyncio.Queue[
            tuple[str, asyncio.Queue[bytes | None]] | None
        ] = asyncio.Queue(maxsize=max_sentence_queue)

        # Worker pool
        self.worker_tasks: list[asyncio.Task[None]] = []

        # GPU concurrency control (optional)
        self.gpu_semaphore = (
            asyncio.Semaphore(max_gpu_concurrent) if max_gpu_concurrent else None
        )

        # Session state
        self._started = False
        self._shutdown = False

        logger.info(
            "ParallelTTSWrapper initialized",
            extra={
                "num_workers": num_workers,
                "queue_size": max_sentence_queue,
                "gpu_limit": max_gpu_concurrent,
            },
        )

    async def start(self) -> None:
        """Start persistent worker pool.

        Spawns worker tasks that continuously dequeue sentences from the
        sentence buffer and synthesize them in parallel. Workers persist
        across sentences until stop() is called.

        Idempotent: Safe to call multiple times (no-op if already started).
        """
        if self._started:
            logger.debug("ParallelTTSWrapper already started, skipping")
            return

        logger.info(f"Starting {self.num_workers} persistent TTS workers")

        # Spawn worker tasks
        for worker_id in range(self.num_workers):
            task = asyncio.create_task(self._worker_loop(worker_id))
            self.worker_tasks.append(task)

        self._started = True
        logger.info(
            f"ParallelTTSWrapper started successfully with {len(self.worker_tasks)} workers"
        )

    async def stop(self) -> None:
        """Stop persistent worker pool and cleanup resources.

        Gracefully shuts down all workers and clears any pending sentences
        from the buffer. Workers drain the queue before exiting.

        Idempotent: Safe to call multiple times.
        """
        if self._shutdown:
            logger.debug("ParallelTTSWrapper already stopped, skipping")
            return

        logger.info("Stopping ParallelTTSWrapper")
        self._shutdown = True

        # Signal workers to shutdown by sending sentinel values
        for _ in range(self.num_workers):
            try:
                await self.sentence_queue.put(None)
            except asyncio.QueueFull:
                logger.warning("Queue full during shutdown, forcing sentinel")
                # Force-add sentinel (bypass maxsize)
                self.sentence_queue._queue.append(None)  # type: ignore[attr-defined]

        # Wait for workers to finish processing
        if self.worker_tasks:
            logger.debug("Waiting for workers to complete...")
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
            self.worker_tasks.clear()

        # Clear any remaining items from queue
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("ParallelTTSWrapper stopped successfully")

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "BufferedChunkedStream":
        """Synthesize text using persistent worker pool.

        This method is called by the LiveKit Agent for each sentence. Instead
        of synthesizing immediately, we queue the sentence for parallel
        processing and return a stream that reads from the sentence's
        dedicated audio queue.

        Args:
            text: Text to synthesize
            conn_options: API connection options (for compatibility)

        Returns:
            BufferedChunkedStream that yields audio from worker pool
        """
        if not self._started:
            logger.warning(
                "ParallelTTSWrapper not started, auto-starting now "
                "(consider calling start() explicitly)"
            )
            # Start in background (don't block synthesis request)
            asyncio.create_task(self.start())

        logger.debug(f"Queueing sentence for parallel synthesis: '{text[:50]}...'")

        # Create output queue for this sentence
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        # Enqueue sentence with its output queue
        # This may block if queue is full (backpressure)
        try:
            self.sentence_queue.put_nowait((text, audio_queue))
        except asyncio.QueueFull:
            logger.warning(
                f"Sentence queue full ({self.max_sentence_queue}), "
                "applying backpressure (blocking until space available)"
            )
            # Fall back to blocking put (backpressure)
            asyncio.create_task(self.sentence_queue.put((text, audio_queue)))

        # Return stream that reads from this sentence's output queue
        return BufferedChunkedStream(
            wrapper=self,
            input_text=text,
            audio_queue=audio_queue,
            conn_options=conn_options,
        )

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop that continuously synthesizes sentences.

        Each worker:
        1. Dequeues sentence and its output queue from sentence_queue
        2. Synthesizes via gRPC client (respects GPU semaphore)
        3. Streams audio frames to the sentence's output queue
        4. Signals completion with None sentinel
        5. Repeats until shutdown sentinel (None) received

        Args:
            worker_id: Unique worker identifier for logging
        """
        logger.info(f"Worker {worker_id} started (persistent)")

        synthesis_count = 0
        error_count = 0

        try:
            while not self._shutdown:
                # Get next sentence (blocking)
                item = await self.sentence_queue.get()

                # Check for shutdown sentinel
                if item is None:
                    logger.info(
                        f"Worker {worker_id} received shutdown signal "
                        f"(synthesized {synthesis_count}, errors {error_count})"
                    )
                    break

                sentence, audio_queue = item

                logger.debug(
                    f"Worker {worker_id} processing: '{sentence[:50]}...'",
                    extra={"worker_id": worker_id, "queue_size": self.sentence_queue.qsize()},
                )

                try:
                    # Synthesize via gRPC client
                    # Apply GPU semaphore if configured (limits concurrent GPU operations)
                    if self.gpu_semaphore:
                        # Track GPU semaphore wait time for performance monitoring
                        semaphore_start = time.perf_counter()
                        async with self.gpu_semaphore:
                            semaphore_wait = time.perf_counter() - semaphore_start
                            if semaphore_wait > 0.01:  # Log if wait > 10ms (GPU contention)
                                logger.debug(
                                    f"Worker {worker_id} waited {semaphore_wait:.3f}s "
                                    f"for GPU semaphore",
                                    extra={
                                        "worker_id": worker_id,
                                        "wait_time_s": semaphore_wait,
                                        "sentence_preview": sentence[:50],
                                    },
                                )
                            await self._synthesize_sentence(sentence, audio_queue, worker_id)
                    else:
                        await self._synthesize_sentence(sentence, audio_queue, worker_id)

                    synthesis_count += 1

                except Exception as e:
                    logger.error(
                        f"Worker {worker_id} synthesis error: {e}",
                        exc_info=True,
                        extra={"worker_id": worker_id, "sentence": sentence[:50]},
                    )
                    error_count += 1

                    # Signal error to waiting stream (None sentinel)
                    try:
                        await audio_queue.put(None)
                    except Exception:  # noqa: S110
                        pass  # Queue might be closed, nothing to do

                    # Continue processing other sentences (error isolation)

        except asyncio.CancelledError:
            logger.info(f"Worker {worker_id} cancelled")
            raise

        except Exception as e:
            logger.error(
                f"Worker {worker_id} fatal error: {e}",
                exc_info=True,
            )
            raise

        finally:
            logger.info(
                f"Worker {worker_id} stopped "
                f"(synthesized {synthesis_count}, errors {error_count})"
            )

    async def _synthesize_sentence(
        self,
        sentence: str,
        audio_queue: asyncio.Queue[bytes | None],
        worker_id: int,
    ) -> None:
        """Synthesize a single sentence and stream audio chunks to its queue.

        Streams audio frames incrementally as they become available from the
        gRPC worker, enabling smooth playback without waiting for full sentence
        completion. This reduces perceived latency and eliminates pauses.

        Args:
            sentence: Text to synthesize
            audio_queue: Queue to stream audio frames to
            worker_id: Worker ID for logging

        Raises:
            Exception: If synthesis fails (caller handles error isolation)
        """
        # Track synthesis performance (RTF calculation)
        synthesis_start = time.perf_counter()
        total_audio_bytes = 0

        # Call gRPC client's SEQUENTIAL synthesis method (bypasses parallel wrapper)
        # This prevents infinite recursion and performs a single gRPC Synthesize() call
        stream = self.grpc_client._synthesize_sequential(sentence)

        # Stream audio frames incrementally as they become available
        # This enables smooth playback without waiting for full sentence completion
        try:
            # Create a simple emitter that pushes frames to our queue
            # Frames are streamed incrementally as they arrive from gRPC worker
            class QueueEmitter:
                def __init__(
                    self, queue: asyncio.Queue[bytes | None], audio_bytes_ref: list[int]
                ) -> None:
                    self.queue = queue
                    self.audio_bytes_ref = audio_bytes_ref
                    self._loop = asyncio.get_event_loop()
                    self._pending_puts: list[asyncio.Task[None]] = []

                def initialize(self, **kwargs: Any) -> None:
                    pass

                def push(self, audio_data: bytes) -> None:
                    # Track audio bytes for RTF calculation
                    self.audio_bytes_ref[0] += len(audio_data)
                    # Schedule put operation (non-blocking for synthesis thread)
                    # Use ensure_future to handle case where we're in a sync context
                    # All tasks will be awaited before signaling completion
                    task = self._loop.create_task(self.queue.put(audio_data))
                    self._pending_puts.append(task)

                async def wait_for_pending(self) -> None:
                    """Wait for all pending queue operations to complete."""
                    if self._pending_puts:
                        # Wait for all puts to complete
                        # Ignoring exceptions (errors handled by caller)
                        await asyncio.gather(
                            *self._pending_puts, return_exceptions=True
                        )
                        self._pending_puts.clear()

                def flush(self) -> None:
                    pass

            audio_bytes_ref = [0]  # Use list for mutable reference
            emitter = QueueEmitter(audio_queue, audio_bytes_ref)

            # Run the stream and emit frames as they arrive
            # This streams 20ms frames incrementally instead of waiting for full sentence
            await stream._run(emitter)
            
            # Wait for all pending queue operations to complete before signaling end
            # This ensures all frames are queued before we signal completion
            await emitter.wait_for_pending()
            
            # Flush any remaining frames
            emitter.flush()

            # Calculate RTF (Real-Time Factor) for performance monitoring
            synthesis_time = time.perf_counter() - synthesis_start
            total_audio_bytes = audio_bytes_ref[0]

            # Calculate audio duration (48kHz, 16-bit mono = 2 bytes per sample)
            # samples = bytes / 2, duration = samples / 48000
            if total_audio_bytes > 0:
                audio_samples = total_audio_bytes // 2
                audio_duration_s = audio_samples / 48000.0
                rtf = synthesis_time / audio_duration_s if audio_duration_s > 0 else 0.0

                # Log RTF at INFO level for visibility
                logger.info(
                    f"Worker {worker_id} synthesis RTF",
                    extra={
                        "worker_id": worker_id,
                        "sentence_preview": sentence[:50],
                        "synthesis_time_s": round(synthesis_time, 3),
                        "audio_duration_s": round(audio_duration_s, 3),
                        "audio_bytes": total_audio_bytes,
                        "rtf": round(rtf, 3),
                        "realtime_status": "faster" if rtf < 1.0 else "slower",
                    },
                )
            else:
                logger.warning(
                    f"Worker {worker_id} synthesized 0 bytes",
                    extra={"worker_id": worker_id, "sentence_preview": sentence[:50]},
                )

        except Exception as e:
            logger.error(
                f"Worker {worker_id} failed to synthesize: {e}",
                exc_info=True,
            )
            raise  # Re-raise for caller's error handling

        finally:
            # Signal completion (None sentinel)
            await audio_queue.put(None)


class BufferedChunkedStream(tts.ChunkedStream):
    """ChunkedStream that reads from worker pool output.

    This class implements the LiveKit ChunkedStream interface but instead of
    performing synthesis directly, it reads pre-synthesized audio frames from
    the worker pool's per-sentence output queue.

    Attributes:
        wrapper: ParallelTTSWrapper instance
        audio_queue: Queue containing audio frames for this sentence
    """

    def __init__(
        self,
        *,
        wrapper: ParallelTTSWrapper,
        input_text: str,
        audio_queue: asyncio.Queue[bytes | None],
        conn_options: APIConnectOptions,
    ) -> None:
        """Initialize buffered chunked stream.

        Args:
            wrapper: ParallelTTSWrapper instance
            input_text: Original input text (for logging)
            audio_queue: Queue to read audio frames from
            conn_options: API connection options
        """
        # Get TTS capabilities from underlying gRPC client
        super().__init__(
            tts=wrapper.grpc_client,
            input_text=input_text,
            conn_options=conn_options,
        )

        self._wrapper = wrapper
        self._audio_queue = audio_queue

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Stream audio frames from worker pool output.

        This is called by the ChunkedStream base class to perform the actual
        audio streaming work. It reads from the per-sentence audio queue
        populated by the worker pool.

        Args:
            output_emitter: Emitter to push audio frames to
        """
        logger.debug(f"Streaming buffered audio for: '{self.input_text[:50]}...'")

        try:
            # Initialize emitter
            output_emitter.initialize(
                request_id=str(id(self)),  # Use object ID as request identifier
                sample_rate=self._wrapper.grpc_client._sample_rate,
                num_channels=self._wrapper.grpc_client._num_channels,
                mime_type="audio/pcm",
            )

            # Stream frames from audio queue
            frame_count = 0
            total_bytes = 0

            while True:
                # Get next audio frame (with timeout for error detection)
                try:
                    audio_data = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=30.0,  # 30s timeout for synthesis
                    )
                except TimeoutError:
                    logger.error(
                        f"Timeout waiting for audio frames for: '{self.input_text[:50]}...'"
                    )
                    raise

                # Check for sentinel (None = end of stream or error)
                if audio_data is None:
                    logger.debug(
                        f"End of audio stream for: '{self.input_text[:50]}...' "
                        f"({frame_count} frames, {total_bytes} bytes)"
                    )
                    break

                # Push frame to emitter
                output_emitter.push(audio_data)
                frame_count += 1
                total_bytes += len(audio_data)

            # Flush remaining frames
            output_emitter.flush()

            logger.debug(
                f"Completed streaming for: '{self.input_text[:50]}...' "
                f"({frame_count} frames, {total_bytes} bytes)"
            )

        except Exception as e:
            logger.error(
                f"Error streaming buffered audio: {e}",
                exc_info=True,
            )
            raise
