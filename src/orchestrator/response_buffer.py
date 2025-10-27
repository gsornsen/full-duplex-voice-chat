"""Intelligent response buffer for filler-to-full transitions.

This module provides buffering functionality for seamless audio transitions from
filler phrases ("Let me think about that...") to full LLM-generated responses.

Key Features:
- Parallel stream coordination (filler LLM, filler TTS, full LLM, buffered TTS)
- Accurate timing coordination with adaptive lead time
- Bounded buffering with overflow strategies
- Seamless audio transition with crossfade to eliminate clicks
- Comprehensive edge case handling
- Production-ready metrics and monitoring

Example:
    Basic usage with buffering:
        >>> buffer = ResponseBuffer(
        ...     config=ResponseBufferConfig(),
        ...     tts_client=tts_worker_client,
        ... )
        >>> filler_text = await buffer.generate_filler(question="Complex question?")
        >>> async def full_response_stream():
        ...     async for sentence in llm.stream_response(question):
        ...         yield sentence
        >>> async for frame in buffer.buffer_during_filler(
        ...     filler_text=filler_text,
        ...     full_response_stream=full_response_stream(),
        ... ):
        ...     await audio_output.put(frame)
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.tts.audio.processing import crossfade_buffers

logger = logging.getLogger(__name__)


class BufferState(Enum):
    """Buffer state machine states.

    State Transitions:
    - IDLE → FILLER_PENDING (on complex question detected)
    - FILLER_PENDING → FILLER_SYNTH (on filler text ready)
    - FILLER_PENDING → DIRECT_SPEAK (on filler LLM failed)
    - FILLER_SYNTH → FILLER_PLAYING (on filler audio ready)
    - FILLER_PLAYING → BUFFERED_READY (on filler complete OR buffer ready)
    - FILLER_PLAYING → DIRECT_SPEAK (on filler failed)
    - BUFFERED_READY → PLAYING_BUFFER (on start buffered)
    - PLAYING_BUFFER → COMPLETED (on playback complete)
    - DIRECT_SPEAK → COMPLETED (on direct playback complete)
    - * → COMPLETED (on shutdown or error)

    States:
    - IDLE: Initial state, no activity
    - FILLER_PENDING: Waiting for filler LLM to generate phrase
    - FILLER_SYNTH: Synthesizing filler audio
    - FILLER_PLAYING: Playing filler + buffering response
    - BUFFERED_READY: Buffer has content, filler ending
    - PLAYING_BUFFER: Playing buffered response
    - DIRECT_SPEAK: Skip filler, speak directly (fallback)
    - COMPLETED: All playback finished
    """

    IDLE = "idle"
    FILLER_PENDING = "filler_pending"
    FILLER_SYNTH = "filler_synth"
    FILLER_PLAYING = "filler_playing"
    BUFFERED_READY = "buffered_ready"
    PLAYING_BUFFER = "playing_buffer"
    DIRECT_SPEAK = "direct_speak"
    COMPLETED = "completed"


class OverflowStrategy(Enum):
    """Strategy for handling sentence queue overflow.

    Strategies:
    - PAUSE: Block LLM until queue has space (backpressure)
    - DROP: Drop the sentence and continue
    - EXTEND_FILLER: Try to extend filler audio (if not yet complete)
    """

    PAUSE = "pause"
    DROP = "drop"
    EXTEND_FILLER = "extend_filler"


@dataclass
class FillerMetrics:
    """Metrics for filler and buffering operations.

    Tracks timing and performance metrics for monitoring and optimization.
    """

    # Filler timing
    filler_llm_latency_ms: float = 0.0
    filler_tts_latency_ms: float = 0.0
    filler_duration_ms: float = 0.0

    # Buffering timing
    buffer_start_ts: float = 0.0
    buffer_end_ts: float = 0.0
    buffered_sentences: int = 0
    buffered_audio_frames: int = 0

    # Transition timing
    transition_gap_ms: float = 0.0  # Gap between filler end and buffer start
    transition_overlap_ms: float = 0.0  # Overlap (negative gap)
    crossfade_applied: bool = False  # Whether crossfade was applied

    # Full response timing
    full_llm_start_ts: float = 0.0
    full_llm_end_ts: float = 0.0
    full_llm_duration_ms: float = 0.0

    # Edge case tracking
    overflow_events: int = 0
    slow_llm_fallback: bool = False
    fast_filler_fallback: bool = False


@dataclass
class ResponseBufferConfig:
    """Configuration for response buffering behavior.

    Controls buffer sizes, timing parameters, and overflow strategies.
    """

    # Buffer limits
    max_buffered_sentences: int = 20  # Max sentences to buffer
    max_buffered_audio_frames: int = 500  # Max audio frames (~10 seconds @ 20ms)

    # Timing parameters
    filler_duration_estimate_ms: float = 3000.0  # Default filler duration estimate
    buffer_tts_lead_time_ms: float = 800.0  # Start buffered TTS this early
    transition_crossfade_ms: float = 50.0  # Crossfade duration for smooth transition

    # Overflow behavior
    overflow_strategy: OverflowStrategy = OverflowStrategy.PAUSE

    # Fallback settings
    max_filler_extension_ms: float = 2000.0  # Max extension if LLM slow
    silence_gap_ms: float = 200.0  # Acceptable silence gap fallback

    # Frame parameters (for duration calculation)
    frame_duration_ms: float = 20.0  # Duration of each audio frame
    frame_size_bytes: int = 1920  # Expected frame size (48kHz, 16-bit, mono, 20ms)


class ResponseBuffer:
    """Buffer full LLM response sentences during filler playback.

    This class coordinates three parallel streams:
    1. Filler generation and playback
    2. Full LLM response generation
    3. Buffered TTS synthesis of full response

    Key Features:
    - Accurate filler duration tracking
    - Bounded sentence and audio buffering
    - Seamless audio transition with crossfade (eliminates clicks)
    - Overflow handling with backpressure
    - Edge case handling (slow LLM, fast filler, barge-in)

    Thread-safety: NOT thread-safe. Use from a single async task.

    Example:
        Basic buffering workflow:
            >>> buffer = ResponseBuffer(config=ResponseBufferConfig(), tts_client=client)
            >>> filler_text = await buffer.generate_filler(question="Complex question?")
            >>> async def full_response_generator():
            ...     async for sentence in llm.stream_response(question):
            ...         yield sentence
            >>> buffered_audio = buffer.buffer_during_filler(
            ...     filler_text=filler_text,
            ...     full_response_stream=full_response_generator(),
            ... )
            >>> async for frame in buffered_audio:
            ...     await audio_queue.put(frame)
    """

    def __init__(
        self,
        config: ResponseBufferConfig,
        tts_client: Any,  # TTSWorkerClient or compatible interface
    ):
        """Initialize response buffer.

        Args:
            config: Buffer configuration
            tts_client: TTS client for synthesizing audio (must have synthesize method)
        """
        self.config = config
        self.tts = tts_client

        # State
        self.state = BufferState.IDLE
        self.metrics = FillerMetrics()

        # Queues (bounded to prevent memory overflow)
        self.sentence_queue: asyncio.Queue[str] = asyncio.Queue(
            maxsize=config.max_buffered_sentences
        )
        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=config.max_buffered_audio_frames
        )

        # Timing tracking
        self._filler_start_ts: float | None = None
        self._filler_end_ts: float | None = None
        self._actual_filler_duration_ms: float | None = None

        # Coordination events
        self._filler_complete_event = asyncio.Event()
        self._buffer_ready_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()

        # Background tasks
        self._filler_task: asyncio.Task[None] | None = None
        self._buffer_task: asyncio.Task[None] | None = None
        self._tts_task: asyncio.Task[None] | None = None

        # Filler frames storage
        self._filler_frames: list[bytes] = []

        logger.info(
            "ResponseBuffer initialized",
            extra={
                "max_buffered_sentences": config.max_buffered_sentences,
                "max_buffered_audio_frames": config.max_buffered_audio_frames,
                "overflow_strategy": config.overflow_strategy.value,
                "crossfade_ms": config.transition_crossfade_ms,
            },
        )

    async def generate_filler(self, question: str) -> str:
        """Generate contextual filler phrase for complex question.

        Uses a lightweight template selection (future: LLM) to generate natural
        filler phrases like "That's a great question, let me think..." based on
        the question content.

        Args:
            question: User's question text

        Returns:
            Filler phrase text

        Example:
            >>> filler = await buffer.generate_filler("What is quantum physics?")
            >>> print(filler)
            "That's a fascinating question. Let me think about that for a moment..."
        """
        start_ts = time.monotonic()

        # Template-based filler generation (future: replace with LLM)
        templates = [
            "That's a great question. Let me think about that for a moment...",
            "Interesting. Give me a second to consider that...",
            "Let me think through that carefully...",
            "That's worth thinking about. One moment...",
            "Good question. Let me gather my thoughts...",
        ]

        # Deterministic selection based on question hash (not for crypto)
        # S324: MD5 used only for template selection, not security
        idx = (
            int(question.encode("utf-8").__hash__() & 0x7FFFFFFF) % len(templates)
        )  # noqa: S324
        filler_text = templates[idx]

        self.metrics.filler_llm_latency_ms = (time.monotonic() - start_ts) * 1000

        logger.info(
            "Generated filler phrase",
            extra={
                "filler_text": filler_text,
                "latency_ms": self.metrics.filler_llm_latency_ms,
            },
        )

        return filler_text

    async def buffer_during_filler(
        self,
        filler_text: str,
        full_response_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        """Buffer full LLM response during filler playback.

        This is the main coordination method. It:
        1. Starts filler TTS synthesis and playback
        2. Starts buffering sentences from full_response_stream
        3. Monitors timing to determine when to start buffered TTS
        4. Handles seamless transition from filler to buffered audio with crossfade
        5. Streams buffered audio frames to caller

        Args:
            filler_text: Filler phrase to speak first
            full_response_stream: Async iterator yielding response sentences

        Yields:
            Audio frames (bytes) for complete response (filler + buffered)

        Raises:
            asyncio.CancelledError: If cancelled (e.g., barge-in)
            Exception: On synthesis or buffering errors

        Example:
            >>> async for frame in buffer.buffer_during_filler(
            ...     filler_text="Let me think...",
            ...     full_response_stream=llm.stream_response(question),
            ... ):
            ...     await audio_output.put(frame)
        """
        try:
            # Transition state
            self.state = BufferState.FILLER_SYNTH

            # Start three parallel tasks:
            # 1. Filler synthesis and playback
            self._filler_task = asyncio.create_task(
                self._filler_playback_task(filler_text)
            )

            # 2. Full response buffering
            self._buffer_task = asyncio.create_task(
                self._buffer_response_task(full_response_stream)
            )

            # 3. Buffered TTS synthesis (starts when timing is right)
            self._tts_task = asyncio.create_task(self._buffered_tts_task())

            # Stream filler audio frames first
            async for frame in self._stream_filler_frames():
                yield frame

            # Transition to buffered audio
            logger.info("Transitioning from filler to buffered response")
            self.state = BufferState.PLAYING_BUFFER

            # Stream buffered audio frames (with crossfade on first frame)
            async for frame in self._stream_buffered_frames():
                yield frame

            # Complete
            self.state = BufferState.COMPLETED
            logger.info(
                "Response buffering completed",
                extra={"metrics": self._get_metrics_summary()},
            )

        except asyncio.CancelledError:
            logger.warning("Response buffering cancelled (barge-in?)")
            await self._cleanup_tasks()
            raise
        except Exception as e:
            logger.error(f"Response buffering failed: {e}", exc_info=True)
            await self._cleanup_tasks()
            raise
        finally:
            await self._cleanup_tasks()

    async def _filler_playback_task(self, filler_text: str) -> None:
        """Synthesize and track filler audio playback.

        This task:
        1. Requests TTS synthesis of filler phrase
        2. Measures actual synthesis time
        3. Tracks audio duration from frame count
        4. Signals completion via _filler_complete_event

        Args:
            filler_text: Filler phrase to synthesize

        Raises:
            Exception: On synthesis errors
        """
        try:
            self._filler_start_ts = time.monotonic()
            start_ts = self._filler_start_ts

            logger.info(f"Starting filler TTS synthesis: '{filler_text}'")

            # Synthesize filler (accumulate to measure duration)
            filler_frames: list[bytes] = []
            async for frame in self.tts.synthesize([filler_text]):
                # Extract audio data from frame (handle both gRPC and dict formats)
                if hasattr(frame, "audio_data"):
                    audio_data = frame.audio_data
                elif isinstance(frame, dict) and "audio_data" in frame:
                    audio_data = frame["audio_data"]
                else:
                    audio_data = frame

                filler_frames.append(audio_data)

            # Calculate metrics
            end_ts = time.monotonic()
            self.metrics.filler_tts_latency_ms = (end_ts - start_ts) * 1000

            # Calculate audio duration from frame count
            frame_duration_ms = self.config.frame_duration_ms
            self._actual_filler_duration_ms = len(filler_frames) * frame_duration_ms
            self.metrics.filler_duration_ms = self._actual_filler_duration_ms

            logger.info(
                "Filler audio synthesized",
                extra={
                    "synthesis_latency_ms": self.metrics.filler_tts_latency_ms,
                    "audio_duration_ms": self._actual_filler_duration_ms,
                    "frame_count": len(filler_frames),
                },
            )

            # Store frames for streaming
            self._filler_frames = filler_frames

            # Transition state
            self.state = BufferState.FILLER_PLAYING

            # Wait for actual playback to complete (simulate)
            # In real implementation, this would track actual audio output timing
            await asyncio.sleep(self._actual_filler_duration_ms / 1000.0)

            # Mark filler complete
            self._filler_end_ts = time.monotonic()
            self._filler_complete_event.set()

            logger.info("Filler playback completed")

        except Exception as e:
            logger.error(f"Filler playback task failed: {e}", exc_info=True)
            self._filler_complete_event.set()  # Signal completion even on error
            raise

    async def _buffer_response_task(
        self,
        full_response_stream: AsyncIterator[str],
    ) -> None:
        """Buffer sentences from full LLM response stream.

        This task:
        1. Accumulates sentences from LLM stream
        2. Queues them in sentence_queue (bounded)
        3. Handles overflow based on config.overflow_strategy
        4. Tracks buffering metrics

        Args:
            full_response_stream: Async iterator yielding response sentences

        Raises:
            Exception: On buffering errors
        """
        try:
            self.metrics.buffer_start_ts = time.monotonic()
            self.metrics.full_llm_start_ts = time.monotonic()

            logger.info("Starting full response buffering")

            sentence_count = 0

            async for sentence in full_response_stream:
                if self._shutdown_event.is_set():
                    break

                # Queue sentence with overflow handling
                try:
                    await asyncio.wait_for(
                        self.sentence_queue.put(sentence),
                        timeout=0.1,  # Non-blocking with short timeout
                    )
                    sentence_count += 1

                    logger.debug(
                        f"Buffered sentence {sentence_count}: '{sentence[:50]}...'"
                    )

                except TimeoutError:
                    # Queue full - apply overflow strategy
                    await self._handle_buffer_overflow(sentence)

            # Record metrics
            self.metrics.full_llm_end_ts = time.monotonic()
            self.metrics.full_llm_duration_ms = (
                self.metrics.full_llm_end_ts - self.metrics.full_llm_start_ts
            ) * 1000
            self.metrics.buffered_sentences = sentence_count
            self.metrics.buffer_end_ts = time.monotonic()

            logger.info(
                "Full response buffering completed",
                extra={
                    "sentence_count": sentence_count,
                    "duration_ms": self.metrics.full_llm_duration_ms,
                },
            )

            # Signal buffer ready
            self._buffer_ready_event.set()

        except Exception as e:
            logger.error(f"Response buffering task failed: {e}", exc_info=True)
            self._buffer_ready_event.set()  # Signal ready even on error
            raise

    async def _handle_buffer_overflow(self, sentence: str) -> None:
        """Handle sentence queue overflow based on configured strategy.

        Strategies:
        - PAUSE: Block LLM until queue has space (backpressure)
        - DROP: Drop the sentence and continue
        - EXTEND_FILLER: Try to extend filler audio (if not yet complete)

        Args:
            sentence: Sentence that couldn't be queued
        """
        strategy = self.config.overflow_strategy

        logger.warning(
            f"Sentence queue overflow, applying strategy: {strategy.value}",
            extra={"sentence_preview": sentence[:50]},
        )

        self.metrics.overflow_events += 1

        if strategy == OverflowStrategy.PAUSE:
            # Block until space available (backpressure to LLM)
            await self.sentence_queue.put(sentence)

        elif strategy == OverflowStrategy.DROP:
            # Drop sentence and continue
            logger.warning(f"Dropped sentence due to overflow: '{sentence[:50]}...'")

        elif strategy == OverflowStrategy.EXTEND_FILLER:
            # TODO: Implement filler extension
            # For now, fall back to pause
            logger.warning("Filler extension not implemented, falling back to pause")
            await self.sentence_queue.put(sentence)

        else:
            # Unknown strategy - default to pause
            logger.error(
                f"Unknown overflow strategy: {strategy.value}, defaulting to pause"
            )
            await self.sentence_queue.put(sentence)

    async def _buffered_tts_task(self) -> None:
        """Synthesize buffered sentences to audio frames.

        This task:
        1. Waits for optimal timing (filler ending soon)
        2. Dequeues sentences from sentence_queue
        3. Synthesizes to audio frames
        4. Queues frames in audio_queue
        5. Tracks timing for seamless transition

        Raises:
            Exception: On synthesis errors
        """
        try:
            # Wait for right timing to start
            await self._wait_for_buffer_tts_start()

            logger.info("Starting buffered TTS synthesis")

            frame_count = 0

            # Process buffered sentences
            while True:
                try:
                    # Get next sentence (with timeout to check for completion)
                    sentence = await asyncio.wait_for(
                        self.sentence_queue.get(),
                        timeout=0.5,
                    )

                    # Synthesize sentence to audio frames
                    async for frame in self.tts.synthesize([sentence]):
                        # Extract audio data from frame
                        if hasattr(frame, "audio_data"):
                            audio_data = frame.audio_data
                        elif isinstance(frame, dict) and "audio_data" in frame:
                            audio_data = frame["audio_data"]
                        else:
                            audio_data = frame

                        await self.audio_queue.put(audio_data)
                        frame_count += 1

                except TimeoutError:
                    # Check if buffering is complete
                    if (
                        self._buffer_ready_event.is_set()
                        and self.sentence_queue.empty()
                    ):
                        break
                    continue

            self.metrics.buffered_audio_frames = frame_count

            logger.info(
                "Buffered TTS synthesis completed",
                extra={"frame_count": frame_count},
            )

        except Exception as e:
            logger.error(f"Buffered TTS task failed: {e}", exc_info=True)
            raise

    async def _wait_for_buffer_tts_start(self) -> None:
        """Wait for optimal timing to start buffered TTS synthesis.

        Strategy:
        1. Estimate when filler will end (from synthesis metrics)
        2. Start buffered TTS at: filler_end_time - buffer_tts_lead_time
        3. This ensures buffered audio is ready right when filler ends

        Edge cases:
        - If filler already complete: start immediately
        - If estimate unavailable: wait for filler complete event
        - If filler shorter than lead time: start immediately

        Raises:
            None (handles edge cases gracefully)
        """
        # Wait for filler synthesis to complete (to get accurate duration)
        while self._actual_filler_duration_ms is None:
            await asyncio.sleep(0.1)
            if self._shutdown_event.is_set():
                return

        # Calculate when to start buffered TTS
        filler_duration_s = self._actual_filler_duration_ms / 1000.0
        lead_time_s = self.config.buffer_tts_lead_time_ms / 1000.0

        wait_duration_s = filler_duration_s - lead_time_s

        if wait_duration_s > 0:
            logger.info(
                f"Waiting {wait_duration_s:.2f}s before starting buffered TTS "
                f"(filler_duration={filler_duration_s:.2f}s, "
                f"lead_time={lead_time_s:.2f}s)"
            )
            await asyncio.sleep(wait_duration_s)
        else:
            logger.info(
                "Starting buffered TTS immediately "
                f"(filler_duration={filler_duration_s:.2f}s "
                f"shorter than lead_time={lead_time_s:.2f}s)"
            )
            self.metrics.fast_filler_fallback = True

    async def _stream_filler_frames(self) -> AsyncIterator[bytes]:
        """Stream filler audio frames.

        Yields:
            Audio frames (bytes) from filler synthesis
        """
        if not self._filler_frames:
            # Wait for filler synthesis to complete
            await self._filler_complete_event.wait()

        for frame in self._filler_frames:
            yield frame

    async def _stream_buffered_frames(self) -> AsyncIterator[bytes]:
        """Stream buffered audio frames with crossfade from filler.

        Applies crossfade between the last filler frame and first buffered frame
        to eliminate audio boundary clicks caused by DC offset discontinuities.

        Measures transition timing (gap/overlap) for monitoring.

        Yields:
            Audio frames (bytes) from buffered TTS synthesis

        Notes:
            - Crossfade only applies to first frame transition
            - Uses config.transition_crossfade_ms for crossfade duration
            - Handles PCM int16 ↔ float32 conversion for processing
            - Falls back to direct concatenation if buffers too short
        """
        # Calculate transition timing
        if self._filler_end_ts and self._actual_filler_duration_ms:
            # Measure actual gap/overlap
            current_ts = time.monotonic()
            gap_ms = (current_ts - self._filler_end_ts) * 1000

            self.metrics.transition_gap_ms = max(0, gap_ms)
            self.metrics.transition_overlap_ms = max(0, -gap_ms)

            logger.info(
                "Transitioning to buffered audio",
                extra={
                    "gap_ms": self.metrics.transition_gap_ms,
                    "overlap_ms": self.metrics.transition_overlap_ms,
                },
            )

            # Track slow LLM fallback
            if self.metrics.transition_gap_ms > self.config.silence_gap_ms:
                self.metrics.slow_llm_fallback = True

        # Get last filler frame for crossfade
        last_filler_frame: bytes | None = (
            self._filler_frames[-1] if self._filler_frames else None
        )

        first_frame = True

        # Stream buffered frames
        while True:
            try:
                frame = await asyncio.wait_for(self.audio_queue.get(), timeout=0.5)

                # Apply crossfade on first buffered frame
                if first_frame and last_filler_frame:
                    try:
                        # Convert int16 PCM to float32 for processing
                        filler_float = (
                            np.frombuffer(last_filler_frame, dtype=np.int16).astype(
                                np.float32
                            )
                            / 32768.0
                        )
                        buffered_float = (
                            np.frombuffer(frame, dtype=np.int16).astype(np.float32)
                            / 32768.0
                        )

                        # Apply crossfade to eliminate boundary clicks
                        merged = crossfade_buffers(
                            filler_float,
                            buffered_float,
                            crossfade_ms=self.config.transition_crossfade_ms,
                            sample_rate=48000,
                        )

                        # Convert back to int16 PCM
                        merged_int16 = (
                            (merged * 32768.0).clip(-32768, 32767).astype(np.int16)
                        )
                        frame = merged_int16.tobytes()

                        self.metrics.crossfade_applied = True
                        logger.info(
                            "Applied crossfade to filler→buffered transition",
                            extra={
                                "crossfade_ms": self.config.transition_crossfade_ms,
                                "filler_samples": len(filler_float),
                                "buffered_samples": len(buffered_float),
                                "merged_samples": len(merged),
                            },
                        )

                    except Exception as e:
                        logger.warning(
                            f"Failed to apply crossfade, using direct transition: {e}",
                            exc_info=True,
                        )
                        # Fall back to direct frame (no crossfade)

                    first_frame = False

                yield frame

            except TimeoutError:
                # Check if TTS task is complete
                if (
                    self._tts_task
                    and self._tts_task.done()
                    and self.audio_queue.empty()
                ):
                    break
                continue

    async def _cleanup_tasks(self) -> None:
        """Cancel and cleanup all running tasks.

        Ensures graceful shutdown of background tasks and clears queues.
        """
        self._shutdown_event.set()

        tasks = [
            self._filler_task,
            self._buffer_task,
            self._tts_task,
        ]

        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def _get_metrics_summary(self) -> dict[str, float | int | bool]:
        """Get summary of buffering metrics.

        Returns:
            Dictionary of key metrics for monitoring
        """
        return {
            "filler_llm_latency_ms": self.metrics.filler_llm_latency_ms,
            "filler_tts_latency_ms": self.metrics.filler_tts_latency_ms,
            "filler_duration_ms": self.metrics.filler_duration_ms,
            "full_llm_duration_ms": self.metrics.full_llm_duration_ms,
            "buffered_sentences": self.metrics.buffered_sentences,
            "buffered_audio_frames": self.metrics.buffered_audio_frames,
            "transition_gap_ms": self.metrics.transition_gap_ms,
            "transition_overlap_ms": self.metrics.transition_overlap_ms,
            "crossfade_applied": self.metrics.crossfade_applied,
            "overflow_events": self.metrics.overflow_events,
            "slow_llm_fallback": self.metrics.slow_llm_fallback,
            "fast_filler_fallback": self.metrics.fast_filler_fallback,
        }
