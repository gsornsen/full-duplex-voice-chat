"""Unit tests for ResponseBuffer.

Tests cover:
- Filler generation and template selection
- Parallel stream coordination
- Sentence queue buffering with overflow strategies
- Audio frame buffering and streaming
- Timing coordination and transition gap measurement
- Error handling and cleanup
- Metrics tracking
"""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from src.orchestrator.response_buffer import (
    BufferState,
    FillerMetrics,
    OverflowStrategy,
    ResponseBuffer,
    ResponseBufferConfig,
)


@pytest.fixture
def mock_tts_client() -> AsyncMock:
    """Create mock TTS client."""
    mock_client = AsyncMock()

    # Mock synthesize method that returns audio frames
    async def mock_synthesize(sentences: list[str]) -> AsyncIterator[bytes]:
        """Mock TTS synthesis that returns audio frames."""
        # Simulate realistic TTS: ~50ms per sentence, 20ms frames
        for _sentence in sentences:
            # Each sentence produces ~3 frames (60ms of audio)
            for _i in range(3):
                yield b"\x00" * 1920  # 1920 bytes = 20ms @ 48kHz mono 16-bit
                await asyncio.sleep(0.02)  # Simulate 20ms frame duration

    mock_client.synthesize = mock_synthesize
    return mock_client


@pytest.fixture
def config() -> ResponseBufferConfig:
    """Create default ResponseBufferConfig."""
    return ResponseBufferConfig(
        max_buffered_sentences=20,
        max_buffered_audio_frames=500,
        filler_duration_estimate_ms=3000.0,
        buffer_tts_lead_time_ms=800.0,
        overflow_strategy=OverflowStrategy.PAUSE,
    )


@pytest.fixture
def buffer(config: ResponseBufferConfig, mock_tts_client: AsyncMock) -> ResponseBuffer:
    """Create ResponseBuffer with mocked TTS client."""
    return ResponseBuffer(config=config, tts_client=mock_tts_client)


class TestInitialization:
    """Test ResponseBuffer initialization."""

    def test_init_with_default_config(self, mock_tts_client: AsyncMock) -> None:
        """Test initialization with default configuration."""
        config = ResponseBufferConfig()
        buffer = ResponseBuffer(config=config, tts_client=mock_tts_client)

        assert buffer.state == BufferState.IDLE
        assert buffer.config == config
        assert isinstance(buffer.metrics, FillerMetrics)

    def test_init_with_custom_config(self, mock_tts_client: AsyncMock) -> None:
        """Test initialization with custom configuration."""
        config = ResponseBufferConfig(
            max_buffered_sentences=10,
            overflow_strategy=OverflowStrategy.DROP,
        )
        buffer = ResponseBuffer(config=config, tts_client=mock_tts_client)

        assert buffer.config.max_buffered_sentences == 10
        assert buffer.config.overflow_strategy == OverflowStrategy.DROP

    def test_init_queue_sizes(self, buffer: ResponseBuffer) -> None:
        """Test queues are initialized with correct sizes."""
        assert buffer.sentence_queue.maxsize == buffer.config.max_buffered_sentences
        assert buffer.audio_queue.maxsize == buffer.config.max_buffered_audio_frames


class TestFillerGeneration:
    """Test filler phrase generation."""

    async def test_generate_filler_template_selection(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test filler template is selected deterministically."""
        filler1 = await buffer.generate_filler("What is Python?")
        filler2 = await buffer.generate_filler("What is Python?")

        # Same question should yield same filler (deterministic hashing)
        assert filler1 == filler2

    async def test_generate_filler_variety(self, buffer: ResponseBuffer) -> None:
        """Test different questions yield different fillers."""
        fillers = {
            await buffer.generate_filler(f"Question {i}?") for i in range(20)
        }

        # Should have some variety (not all identical)
        assert len(fillers) >= 3

    async def test_generate_filler_updates_metrics(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test filler generation updates metrics."""
        await buffer.generate_filler("Test question?")

        assert buffer.metrics.filler_llm_latency_ms >= 0
        assert buffer.metrics.filler_llm_latency_ms < 50  # Should be very fast

    async def test_generate_filler_returns_valid_text(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test filler generation returns valid text."""
        filler = await buffer.generate_filler("Complex question about quantum physics")

        assert isinstance(filler, str)
        assert len(filler) > 0
        assert filler in buffer.config.__class__.__dict__.get("_templates", []) or any(
            filler in templates
            for templates in [
                [
                    "That's a great question. Let me think about that for a moment...",
                    "Interesting. Give me a second to consider that...",
                    "Let me think through that carefully...",
                    "That's worth thinking about. One moment...",
                    "Good question. Let me gather my thoughts...",
                ]
            ]
        )


class TestFillerPlayback:
    """Test filler synthesis and playback."""

    async def test_filler_playback_task_synthesizes_audio(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test filler playback synthesizes audio frames."""
        filler_text = "Let me think about that..."

        # Run filler playback task
        await buffer._filler_playback_task(filler_text)

        # Verify filler frames were generated
        assert len(buffer._filler_frames) > 0
        assert buffer.state == BufferState.FILLER_PLAYING
        assert buffer._filler_complete_event.is_set()

    async def test_filler_playback_measures_duration(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test filler playback measures audio duration."""
        filler_text = "Quick filler"

        await buffer._filler_playback_task(filler_text)

        # Verify duration metrics
        assert buffer.metrics.filler_duration_ms > 0
        assert buffer._actual_filler_duration_ms is not None

        # Duration should match frame count * frame duration
        expected_duration = len(buffer._filler_frames) * buffer.config.frame_duration_ms
        assert abs(buffer.metrics.filler_duration_ms - expected_duration) < 1.0

    async def test_filler_playback_error_handling(
        self, config: ResponseBufferConfig
    ) -> None:
        """Test filler playback handles TTS errors gracefully."""
        # Mock TTS client that raises error
        mock_tts = AsyncMock()

        async def error_synthesize(*args, **kwargs) -> AsyncIterator[bytes]:  # type: ignore[misc]
            raise Exception("TTS synthesis failed")
            yield  # Make this a generator

        mock_tts.synthesize = error_synthesize

        buffer = ResponseBuffer(config=config, tts_client=mock_tts)

        # Should raise error but set completion event
        with pytest.raises(Exception, match="TTS synthesis failed"):
            await buffer._filler_playback_task("Test filler")

        assert buffer._filler_complete_event.is_set()


class TestResponseBuffering:
    """Test full response buffering."""

    async def test_buffer_response_task_queues_sentences(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test response buffering queues sentences."""

        async def sentence_stream() -> AsyncIterator[str]:
            for sentence in ["First sentence.", "Second sentence.", "Third sentence."]:
                yield sentence
                await asyncio.sleep(0.01)

        # Run buffering task
        await buffer._buffer_response_task(sentence_stream())

        # Verify sentences were queued
        assert buffer.sentence_queue.qsize() == 3
        assert buffer.metrics.buffered_sentences == 3
        assert buffer._buffer_ready_event.is_set()

    async def test_buffer_response_tracks_timing(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test response buffering tracks timing metrics."""

        async def sentence_stream() -> AsyncIterator[str]:
            yield "Test sentence."

        await buffer._buffer_response_task(sentence_stream())

        # Verify timing metrics
        assert buffer.metrics.buffer_start_ts > 0
        assert buffer.metrics.buffer_end_ts > 0
        assert buffer.metrics.full_llm_duration_ms >= 0

    async def test_buffer_response_overflow_pause_strategy(
        self, mock_tts_client: AsyncMock
    ) -> None:
        """Test PAUSE overflow strategy blocks until space available."""
        config = ResponseBufferConfig(
            max_buffered_sentences=2,
            overflow_strategy=OverflowStrategy.PAUSE,
        )
        buffer = ResponseBuffer(config=config, tts_client=mock_tts_client)

        async def sentence_stream() -> AsyncIterator[str]:
            # Generate more sentences than queue capacity
            for i in range(5):
                yield f"Sentence {i}."

        # Start buffering task
        buffer_task = asyncio.create_task(buffer._buffer_response_task(sentence_stream()))

        # Give it time to fill queue
        await asyncio.sleep(0.1)

        # Queue should be full (2 sentences)
        assert buffer.sentence_queue.qsize() == 2

        # Consume one sentence to free space
        await buffer.sentence_queue.get()

        # Wait for task to complete
        await buffer_task

        # Should have successfully queued all sentences (with blocking)
        assert buffer.metrics.buffered_sentences == 5

    async def test_buffer_response_overflow_drop_strategy(
        self, mock_tts_client: AsyncMock
    ) -> None:
        """Test DROP overflow strategy drops excess sentences."""
        config = ResponseBufferConfig(
            max_buffered_sentences=2,
            overflow_strategy=OverflowStrategy.DROP,
        )
        buffer = ResponseBuffer(config=config, tts_client=mock_tts_client)

        async def sentence_stream() -> AsyncIterator[str]:
            for i in range(5):
                yield f"Sentence {i}."
                await asyncio.sleep(0.01)

        await buffer._buffer_response_task(sentence_stream())

        # Queue should have at most 2 sentences
        assert buffer.sentence_queue.qsize() <= 2

        # Overflow events should be tracked
        assert buffer.metrics.overflow_events > 0


class TestBufferedTTS:
    """Test buffered TTS synthesis."""

    async def test_buffered_tts_waits_for_timing(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test buffered TTS waits for optimal timing."""
        # Simulate short filler (immediate start)
        buffer._actual_filler_duration_ms = 100.0  # 100ms filler

        # Start wait task
        start_time = asyncio.get_event_loop().time()
        await buffer._wait_for_buffer_tts_start()
        elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        # Should start immediately (filler shorter than lead time)
        assert elapsed_ms < 50
        assert buffer.metrics.fast_filler_fallback is True

    async def test_buffered_tts_synthesizes_queued_sentences(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test buffered TTS synthesizes queued sentences."""
        # Pre-populate sentence queue
        await buffer.sentence_queue.put("First sentence.")
        await buffer.sentence_queue.put("Second sentence.")
        buffer._buffer_ready_event.set()

        # Set filler duration to trigger immediate start
        buffer._actual_filler_duration_ms = 100.0

        # Run buffered TTS task
        await buffer._buffered_tts_task()

        # Verify audio frames were generated
        assert buffer.audio_queue.qsize() > 0
        assert buffer.metrics.buffered_audio_frames > 0

    async def test_buffered_tts_completes_when_queue_empty(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test buffered TTS completes when sentence queue is empty."""
        # Signal buffer ready immediately
        buffer._buffer_ready_event.set()
        buffer._actual_filler_duration_ms = 100.0

        # Run TTS task (should complete immediately with empty queue)
        await asyncio.wait_for(buffer._buffered_tts_task(), timeout=1.0)

        # Should complete without errors
        assert buffer.metrics.buffered_audio_frames == 0


class TestStreamingFrames:
    """Test audio frame streaming."""

    async def test_stream_filler_frames_yields_audio(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test filler frame streaming yields audio data."""
        # Pre-populate filler frames
        buffer._filler_frames = [b"\x00" * 1920 for _ in range(5)]

        frames = [f async for f in buffer._stream_filler_frames()]

        assert len(frames) == 5
        assert all(isinstance(f, bytes) for f in frames)
        assert all(len(f) == 1920 for f in frames)

    async def test_stream_filler_waits_for_completion(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test filler streaming waits for synthesis completion."""

        async def delayed_completion() -> None:
            await asyncio.sleep(0.1)
            buffer._filler_frames = [b"\x00" * 1920]
            buffer._filler_complete_event.set()

        # Start delayed completion
        asyncio.create_task(delayed_completion())

        # Stream should wait
        frames = [f async for f in buffer._stream_filler_frames()]

        assert len(frames) == 1

    async def test_stream_buffered_frames_measures_transition_gap(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test buffered frame streaming measures transition gap."""
        # Set filler end time
        buffer._filler_end_ts = asyncio.get_event_loop().time() - 0.05  # 50ms ago
        buffer._actual_filler_duration_ms = 1000.0

        # Pre-populate audio queue
        await buffer.audio_queue.put(b"\x00" * 1920)

        # Signal TTS task done
        buffer._tts_task = asyncio.create_task(asyncio.sleep(0))
        await buffer._tts_task

        # Stream frames
        frames = [f async for f in buffer._stream_buffered_frames()]

        # Verify gap metrics
        assert buffer.metrics.transition_gap_ms > 0
        assert frames[0] == b"\x00" * 1920


class TestCompleteWorkflow:
    """Test complete buffering workflow."""

    async def test_buffer_during_filler_complete_flow(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test complete buffer_during_filler workflow."""
        filler_text = "Let me think about that..."

        async def sentence_stream() -> AsyncIterator[str]:
            for sentence in ["First response.", "Second response."]:
                yield sentence
                await asyncio.sleep(0.01)

        # Execute complete workflow
        frames = []
        async for frame in buffer.buffer_during_filler(filler_text, sentence_stream()):
            frames.append(frame)

        # Verify state transitions
        assert buffer.state == BufferState.COMPLETED

        # Verify frames were generated
        assert len(frames) > 0

        # Verify metrics
        assert buffer.metrics.filler_duration_ms > 0
        assert buffer.metrics.buffered_sentences > 0

    async def test_buffer_during_filler_cancellation(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test buffer_during_filler handles cancellation gracefully."""
        filler_text = "Filler phrase"

        async def slow_stream() -> AsyncIterator[str]:
            yield "First sentence."
            await asyncio.sleep(10)  # Long delay
            yield "Second sentence."

        # Start buffering and cancel
        buffer_coro = buffer.buffer_during_filler(filler_text, slow_stream())
        task = asyncio.create_task(buffer_coro.__anext__())

        await asyncio.sleep(0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_buffer_during_filler_error_handling(
        self, config: ResponseBufferConfig
    ) -> None:
        """Test buffer_during_filler handles errors gracefully."""
        # Mock TTS that fails
        mock_tts = AsyncMock()

        async def error_synthesize(*args, **kwargs) -> AsyncIterator[bytes]:  # type: ignore[misc]
            raise Exception("Synthesis error")
            yield

        mock_tts.synthesize = error_synthesize

        buffer = ResponseBuffer(config=config, tts_client=mock_tts)

        async def sentence_stream() -> AsyncIterator[str]:
            yield "Test sentence."

        # Should raise error
        with pytest.raises(Exception, match="Synthesis error"):
            async for _ in buffer.buffer_during_filler("Filler", sentence_stream()):
                pass


class TestMetrics:
    """Test metrics tracking and reporting."""

    def test_metrics_summary_structure(self, buffer: ResponseBuffer) -> None:
        """Test metrics summary has expected structure."""
        summary = buffer._get_metrics_summary()

        expected_keys = {
            "filler_llm_latency_ms",
            "filler_tts_latency_ms",
            "filler_duration_ms",
            "full_llm_duration_ms",
            "buffered_sentences",
            "buffered_audio_frames",
            "transition_gap_ms",
            "transition_overlap_ms",
            "overflow_events",
            "slow_llm_fallback",
            "fast_filler_fallback",
        }

        assert set(summary.keys()) == expected_keys

    async def test_metrics_track_overflow_events(
        self, mock_tts_client: AsyncMock
    ) -> None:
        """Test overflow events are tracked in metrics."""
        config = ResponseBufferConfig(
            max_buffered_sentences=2,
            overflow_strategy=OverflowStrategy.DROP,
        )
        buffer = ResponseBuffer(config=config, tts_client=mock_tts_client)

        async def sentence_stream() -> AsyncIterator[str]:
            for i in range(10):
                yield f"Sentence {i}."
                await asyncio.sleep(0.01)

        await buffer._buffer_response_task(sentence_stream())

        assert buffer.metrics.overflow_events > 0

    async def test_metrics_track_slow_llm_fallback(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test slow LLM fallback flag is tracked."""
        # Simulate slow transition (large gap)
        buffer._filler_end_ts = asyncio.get_event_loop().time() - 0.5  # 500ms ago
        buffer._actual_filler_duration_ms = 1000.0

        await buffer.audio_queue.put(b"\x00" * 1920)
        buffer._tts_task = asyncio.create_task(asyncio.sleep(0))
        await buffer._tts_task

        # Stream should detect slow fallback
        async for _ in buffer._stream_buffered_frames():
            break

        assert buffer.metrics.slow_llm_fallback is True


class TestCleanup:
    """Test cleanup and resource management."""

    async def test_cleanup_tasks_cancels_running_tasks(
        self, buffer: ResponseBuffer
    ) -> None:
        """Test cleanup cancels all running tasks."""
        # Create mock tasks
        buffer._filler_task = asyncio.create_task(asyncio.sleep(10))
        buffer._buffer_task = asyncio.create_task(asyncio.sleep(10))
        buffer._tts_task = asyncio.create_task(asyncio.sleep(10))

        # Cleanup
        await buffer._cleanup_tasks()

        # All tasks should be cancelled
        assert buffer._filler_task.cancelled()
        assert buffer._buffer_task.cancelled()
        assert buffer._tts_task.cancelled()

    async def test_cleanup_sets_shutdown_event(self, buffer: ResponseBuffer) -> None:
        """Test cleanup sets shutdown event."""
        await buffer._cleanup_tasks()

        assert buffer._shutdown_event.is_set()
