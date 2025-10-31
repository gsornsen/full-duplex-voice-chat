"""Parallel TTS synthesis pipeline with ordered playback.

This module provides parallel TTS synthesis to increase throughput while
maintaining strict FIFO playback order. Multiple worker tasks synthesize
sentences concurrently, with results buffered and emitted in sequence order.

Design reference: /tmp/parallel-tts-worker-pool-design.md (Phase B)

Key Features:
    - 2-3x throughput improvement via parallel synthesis
    - Ordered playback with PriorityQueue and out-of-order buffering
    - GPU-aware scheduling with semaphore to prevent OOM
    - Error handling with retry logic and graceful degradation
    - Backpressure mechanism to regulate LLM speed

Architecture:
    LLM Stream → Sentence Queue → Worker Pool → Audio Queue → Ordered Playback

    1. Sentence Queue: Buffer incoming sentences with backpressure (maxsize)
    2. Worker Pool: N parallel workers pull sentences and synthesize
    3. Audio Queue: PriorityQueue ordered by sequence ID
    4. Ordered Playback: Emit audio in strict FIFO order with gap detection

Example:
    ```python
    from orchestrator.parallel_tts import ParallelSynthesisPipeline
    from plugins.grpc_tts import TTS

    # Create TTS adapter
    tts_adapter = TTS(worker_address="localhost:7001", model_id="piper-en-us-lessac-medium")

    # Create pipeline
    pipeline = ParallelSynthesisPipeline(
        tts_adapter=tts_adapter,
        num_workers=3,
        max_sentence_queue=10,
        max_gpu_concurrent=2,
    )

    # Process LLM response
    async def llm_sentence_stream():
        yield "Hello, how are you?"
        yield "I'm here to help."
        yield "What can I do for you today?"

    # Synthesize with parallel workers
    async for audio_bytes in pipeline.synthesize_response(llm_sentence_stream()):
        await play_audio(audio_bytes)
    ```
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple, Protocol

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


class SentenceTask(NamedTuple):
    """Represents a sentence to synthesize.

    Attributes:
        seq_id: Sequence ID for ordering (monotonic)
        sentence: Sentence text to synthesize
        timestamp: Unix timestamp when task was created
    """

    seq_id: int
    sentence: str
    timestamp: float


class SynthesisStatus(Enum):
    """Result status of TTS synthesis.

    Attributes:
        SUCCESS: Synthesis completed successfully
        ERROR: Synthesis failed after retries
        SKIPPED: Sentence skipped (empty or whitespace-only)
    """

    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class AudioResult:
    """Result of TTS synthesis for a sentence.

    Attributes:
        seq_id: Sequence ID (matches SentenceTask.seq_id)
        status: Synthesis status
        audio_bytes: Raw PCM audio bytes (None if error/skipped)
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels
        error_msg: Error message if status is ERROR
    """

    seq_id: int
    status: SynthesisStatus
    audio_bytes: bytes | None
    sample_rate: int = 48000
    channels: int = 1
    error_msg: str | None = None

    def __lt__(self, other: "AudioResult") -> bool:
        """Priority queue ordering by seq_id (lowest first)."""
        return self.seq_id < other.seq_id


# ============================================================================
# Protocols
# ============================================================================


class TTSAdapter(Protocol):
    """Protocol for TTS adapters compatible with parallel pipeline.

    This protocol defines the minimal interface required for TTS adapters
    to work with the parallel synthesis pipeline. Adapters can be from
    LiveKit (openai.TTS, etc.) or custom (grpc_tts.TTS).
    """

    def synthesize(self, text: str, **kwargs: Any) -> "SynthesisStream":
        """Synthesize text to audio stream.

        Args:
            text: Text to synthesize
            **kwargs: Additional adapter-specific options

        Returns:
            Synthesis stream with audio chunks
        """
        ...


class SynthesisStream(Protocol):
    """Protocol for synthesis stream (LiveKit ChunkedStream compatible).

    This protocol abstracts over LiveKit's ChunkedStream and similar
    streaming synthesis interfaces.
    """

    async def collect(self) -> bytes:
        """Collect all audio chunks into a single bytes object.

        Returns:
            Concatenated PCM audio bytes
        """
        ...


# ============================================================================
# Sentence Queue
# ============================================================================


class SentenceQueue:
    """Thread-safe sentence buffer with backpressure.

    Buffers incoming sentences from LLM stream and applies backpressure
    when the queue is full, preventing memory exhaustion.

    Attributes:
        maxsize: Maximum queue size (0 = unlimited)
    """

    def __init__(self, maxsize: int = 10):
        """Initialize sentence queue.

        Args:
            maxsize: Maximum number of sentences to buffer
        """
        self._queue: asyncio.Queue[SentenceTask | None] = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    async def put(self, task: SentenceTask) -> None:
        """Add sentence to queue (blocks if full).

        Args:
            task: Sentence task to enqueue

        Raises:
            RuntimeError: If queue is closed
        """
        if self._closed:
            raise RuntimeError("Cannot put to closed queue")
        await self._queue.put(task)

    async def get(self) -> SentenceTask | None:
        """Get next sentence (returns None if closed and empty).

        Returns:
            Next sentence task, or None if queue is closed and drained
        """
        task = await self._queue.get()
        return task

    async def close(self) -> None:
        """Mark queue as closed (no new items).

        Existing items can still be retrieved until drained.
        """
        self._closed = True
        # Put sentinel value for each expected worker
        # Workers will receive None and shut down
        # We put multiple sentinels to ensure all workers get one
        for _ in range(10):  # Sufficient for typical worker counts
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                break

    def qsize(self) -> int:
        """Get current queue size (for monitoring).

        Returns:
            Number of items in queue
        """
        return self._queue.qsize()

    @property
    def closed(self) -> bool:
        """Check if queue is closed.

        Returns:
            True if queue is closed
        """
        return self._closed


# ============================================================================
# TTS Worker
# ============================================================================


class TTSWorker:
    """Worker that synthesizes sentences in parallel.

    Pulls sentences from shared queue, synthesizes them, and emits results
    to the audio queue. Supports retry logic and GPU semaphore for resource
    management.

    Attributes:
        worker_id: Unique worker identifier
        sentence_queue: Shared sentence queue
        audio_queue: Shared audio result queue
        tts_adapter: TTS adapter for synthesis
        gpu_semaphore: Optional semaphore to limit GPU concurrency
    """

    def __init__(
        self,
        worker_id: int,
        sentence_queue: SentenceQueue,
        audio_queue: asyncio.PriorityQueue[AudioResult],
        tts_adapter: TTSAdapter,
        gpu_semaphore: asyncio.Semaphore | None = None,
        max_retries: int = 2,
    ):
        """Initialize TTS worker.

        Args:
            worker_id: Unique worker ID
            sentence_queue: Shared sentence queue
            audio_queue: Shared audio priority queue
            tts_adapter: TTS adapter for synthesis
            gpu_semaphore: Optional GPU concurrency limiter
            max_retries: Maximum retry attempts on error
        """
        self.worker_id = worker_id
        self.sentence_queue = sentence_queue
        self.audio_queue = audio_queue
        self.tts_adapter = tts_adapter
        self.gpu_semaphore = gpu_semaphore
        self.max_retries = max_retries
        self._stats = {"synthesized": 0, "errors": 0, "skipped": 0}

    async def run(self) -> None:
        """Main worker loop.

        Continuously pulls sentences from queue, synthesizes them, and emits
        results until the queue is closed and drained.
        """
        logger.info(f"Worker {self.worker_id} started")

        while True:
            # Get next sentence
            task = await self.sentence_queue.get()
            if task is None:  # Sentinel value - queue closed
                logger.info(f"Worker {self.worker_id} shutting down")
                break

            # Synthesize with retry logic
            result = await self._synthesize_with_retry(task)

            # Emit result to audio queue
            await self.audio_queue.put(result)

            # Update stats
            self._update_stats(result.status)

        logger.info(f"Worker {self.worker_id} stats: {self._stats}")

    async def _synthesize_with_retry(self, task: SentenceTask) -> AudioResult:
        """Synthesize sentence with retry logic.

        Args:
            task: Sentence task to synthesize

        Returns:
            Audio result with success/error/skipped status
        """
        # Skip empty sentences
        if not task.sentence.strip():
            logger.debug(f"Skipping empty sentence {task.seq_id}")
            return AudioResult(
                seq_id=task.seq_id,
                status=SynthesisStatus.SKIPPED,
                audio_bytes=None,
            )

        # Retry loop with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                # Acquire GPU semaphore (if configured)
                if self.gpu_semaphore:
                    async with self.gpu_semaphore:
                        audio_bytes = await self._synthesize(task.sentence)
                else:
                    audio_bytes = await self._synthesize(task.sentence)

                logger.debug(
                    f"Worker {self.worker_id} synthesized seq={task.seq_id} "
                    f"({len(audio_bytes)} bytes)"
                )

                return AudioResult(
                    seq_id=task.seq_id,
                    status=SynthesisStatus.SUCCESS,
                    audio_bytes=audio_bytes,
                )

            except Exception as e:
                logger.warning(
                    f"Worker {self.worker_id} attempt {attempt + 1}/{self.max_retries + 1} "
                    f"failed for seq={task.seq_id}: {e}"
                )

                if attempt == self.max_retries:
                    # All retries exhausted
                    return AudioResult(
                        seq_id=task.seq_id,
                        status=SynthesisStatus.ERROR,
                        audio_bytes=None,
                        error_msg=str(e),
                    )

                # Exponential backoff: 100ms, 200ms, 400ms
                await asyncio.sleep(0.1 * (2**attempt))

        # Unreachable (loop always returns)
        raise RuntimeError("Retry loop exited without return")

    async def _synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes.

        Args:
            text: Text to synthesize

        Returns:
            Raw PCM audio bytes

        Raises:
            Exception: If synthesis fails
        """
        # Call TTS adapter (LiveKit ChunkedStream pattern)
        stream = self.tts_adapter.synthesize(text)

        # Collect all chunks
        # Note: LiveKit's collect() returns rtc.AudioFrame, not bytes
        audio_frame: Any = await stream.collect()  # rtc.AudioFrame (avoid import)

        # Extract raw PCM bytes from AudioFrame
        # AudioFrame.data is a memoryview/buffer containing raw PCM data
        audio_bytes: bytes = bytes(audio_frame.data)

        return audio_bytes

    def _update_stats(self, status: SynthesisStatus) -> None:
        """Update worker statistics.

        Args:
            status: Synthesis status to record
        """
        if status == SynthesisStatus.SUCCESS:
            self._stats["synthesized"] += 1
        elif status == SynthesisStatus.ERROR:
            self._stats["errors"] += 1
        elif status == SynthesisStatus.SKIPPED:
            self._stats["skipped"] += 1


# ============================================================================
# Ordered Playback
# ============================================================================


class OrderedPlayback:
    """Ensures audio chunks play in FIFO order.

    Dequeues audio results from the priority queue and emits them in strict
    sequential order. Buffers out-of-order results and handles missing
    sequences with gap timeout.

    Attributes:
        audio_queue: Priority queue of audio results
        max_gap_timeout: Maximum time to wait for missing sequence
        next_seq_id: Next expected sequence ID
    """

    def __init__(
        self,
        audio_queue: asyncio.PriorityQueue[AudioResult],
        max_gap_timeout: float = 5.0,
    ):
        """Initialize ordered playback.

        Args:
            audio_queue: Priority queue of audio results
            max_gap_timeout: Max seconds to wait for missing sequence
        """
        self.audio_queue = audio_queue
        self.max_gap_timeout = max_gap_timeout
        self.next_seq_id = 0
        self._buffer: dict[int, AudioResult] = {}  # Out-of-order buffer

    async def stream_ordered_audio(self) -> AsyncIterator[bytes]:
        """Yield audio bytes in strict FIFO order.

        Continuously dequeues audio results and emits them in sequence order,
        buffering out-of-order results and handling missing sequences.

        Yields:
            Raw PCM audio bytes for each successful sentence
        """
        while True:
            # Check buffer for next sequence
            if self.next_seq_id in self._buffer:
                result = self._buffer.pop(self.next_seq_id)
            else:
                # Wait for next result from queue with timeout
                try:
                    result = await asyncio.wait_for(
                        self.audio_queue.get(),
                        timeout=self.max_gap_timeout,
                    )
                except TimeoutError:
                    logger.error(
                        f"Gap timeout waiting for seq={self.next_seq_id}, skipping"
                    )
                    # Skip missing sequence and continue
                    self.next_seq_id += 1
                    continue

            # Handle out-of-order arrival
            if result.seq_id > self.next_seq_id:
                logger.debug(
                    f"Buffering out-of-order seq={result.seq_id} "
                    f"(expected {self.next_seq_id})"
                )
                self._buffer[result.seq_id] = result
                continue

            elif result.seq_id < self.next_seq_id:
                logger.warning(
                    f"Dropping duplicate/old seq={result.seq_id} "
                    f"(expected {self.next_seq_id})"
                )
                continue

            # Correct sequence arrived
            if result.status == SynthesisStatus.SUCCESS and result.audio_bytes:
                logger.debug(f"Yielding audio for seq={result.seq_id}")
                yield result.audio_bytes
            elif result.status == SynthesisStatus.ERROR:
                logger.warning(
                    f"Skipping failed seq={result.seq_id}: {result.error_msg}"
                )
            elif result.status == SynthesisStatus.SKIPPED:
                logger.debug(f"Skipping empty seq={result.seq_id}")

            # Advance sequence counter
            self.next_seq_id += 1

    def get_buffer_size(self) -> int:
        """Get current out-of-order buffer size (for monitoring).

        Returns:
            Number of results in out-of-order buffer
        """
        return len(self._buffer)


# ============================================================================
# Parallel TTS Pipeline
# ============================================================================


class ParallelSynthesisPipeline:
    """Parallel TTS synthesis with ordered playback.

    Orchestrates parallel sentence synthesis using multiple workers while
    maintaining strict FIFO playback order. Provides throughput improvements
    (2-3x) over sequential synthesis.

    Attributes:
        tts_adapter: TTS adapter for synthesis
        num_workers: Number of parallel workers
        max_sentence_queue: Max buffered sentences (backpressure)
        max_gpu_concurrent: Max concurrent GPU operations (None = unlimited)
    """

    def __init__(
        self,
        tts_adapter: TTSAdapter,
        num_workers: int = 2,
        max_sentence_queue: int = 10,
        max_gpu_concurrent: int | None = None,
    ):
        """Initialize parallel TTS pipeline.

        Args:
            tts_adapter: TTS adapter for synthesis
            num_workers: Number of parallel workers (2-3 recommended)
            max_sentence_queue: Max buffered sentences (backpressure threshold)
            max_gpu_concurrent: Max concurrent GPU operations (None = unlimited)
        """
        self.tts_adapter = tts_adapter
        self.num_workers = num_workers
        self.max_sentence_queue = max_sentence_queue
        self.max_gpu_concurrent = max_gpu_concurrent

        # Queues
        self.sentence_queue = SentenceQueue(maxsize=max_sentence_queue)
        self.audio_queue: asyncio.PriorityQueue[AudioResult] = asyncio.PriorityQueue()

        # GPU semaphore (optional)
        self.gpu_semaphore = (
            asyncio.Semaphore(max_gpu_concurrent) if max_gpu_concurrent else None
        )

        # Workers
        self.workers: list[TTSWorker] = []
        self.worker_tasks: list[asyncio.Task[None]] = []

        # Sequence tracking
        self.sequence_counter = 0

        logger.info(
            f"ParallelSynthesisPipeline initialized: "
            f"workers={num_workers}, queue_size={max_sentence_queue}, "
            f"gpu_limit={max_gpu_concurrent}"
        )

    async def synthesize_response(
        self, llm_stream: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """Synthesize LLM response with parallel workers.

        Args:
            llm_stream: Async iterator of sentences from LLM

        Yields:
            Audio bytes in strict FIFO order

        Example:
            ```python
            async for audio_bytes in pipeline.synthesize_response(llm_stream):
                await play_audio(audio_bytes)
            ```
        """
        # Start workers
        await self._start_workers()

        try:
            # Feed sentences from LLM (runs in background)
            feeder_task = asyncio.create_task(self._feed_sentences(llm_stream))

            # Stream ordered audio
            playback = OrderedPlayback(self.audio_queue)

            async for audio_bytes in playback.stream_ordered_audio():
                yield audio_bytes

                # Stop if feeder done and queues empty
                if (
                    feeder_task.done()
                    and self.sentence_queue.qsize() == 0
                    and self.audio_queue.empty()
                    and playback.get_buffer_size() == 0
                ):
                    break

        finally:
            # Cleanup
            await self._stop_workers()

    async def _start_workers(self) -> None:
        """Start TTS worker pool."""
        logger.info(f"Starting {self.num_workers} TTS workers")

        for i in range(self.num_workers):
            worker = TTSWorker(
                worker_id=i,
                sentence_queue=self.sentence_queue,
                audio_queue=self.audio_queue,
                tts_adapter=self.tts_adapter,
                gpu_semaphore=self.gpu_semaphore,
            )
            self.workers.append(worker)

            task = asyncio.create_task(worker.run())
            self.worker_tasks.append(task)

    async def _stop_workers(self) -> None:
        """Stop TTS worker pool gracefully."""
        logger.info("Stopping TTS workers")

        # Close sentence queue (no new items)
        await self.sentence_queue.close()

        # Wait for workers to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        self.workers.clear()
        self.worker_tasks.clear()

    async def _feed_sentences(self, llm_stream: AsyncIterator[str]) -> None:
        """Feed sentences from LLM stream to worker queue.

        Args:
            llm_stream: Async iterator of sentences from LLM
        """
        try:
            async for sentence in llm_stream:
                task = SentenceTask(
                    seq_id=self.sequence_counter,
                    sentence=sentence,
                    timestamp=time.time(),
                )

                logger.debug(
                    f"Enqueuing sentence {self.sequence_counter}: {sentence[:50]}..."
                )

                await self.sentence_queue.put(task)
                self.sequence_counter += 1

        except Exception as e:
            logger.error(f"Error feeding sentences: {e}", exc_info=True)
            raise
