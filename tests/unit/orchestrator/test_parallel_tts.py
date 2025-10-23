"""Unit tests for ParallelSynthesisPipeline.

Tests the core components of the parallel TTS synthesis pipeline:
- SentenceQueue: Backpressure and sentinel handling
- TTSWorker: Retry logic, error handling, empty sentence skipping
- OrderedPlayback: FIFO ordering, out-of-order buffering, gap timeout
- ParallelSynthesisPipeline: End-to-end parallel synthesis

Design reference: /tmp/parallel-tts-worker-pool-design.md (Phase B)
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from src.orchestrator.parallel_tts import (
    AudioResult,
    OrderedPlayback,
    ParallelSynthesisPipeline,
    SentenceQueue,
    SentenceTask,
    SynthesisStatus,
    TTSWorker,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_tts_adapter():
    """Create mock TTS adapter that synthesizes text to bytes."""

    class MockAdapter:
        def __init__(self):
            self.call_count = 0
            self.synthesis_delay = 0.1  # 100ms simulated synthesis

        def synthesize(self, text: str, **kwargs):
            """Return mock synthesis stream."""
            # Create a proper mock stream object
            stream = MagicMock()

            # Define the collect coroutine inline
            async def collect_audio():
                await asyncio.sleep(self.synthesis_delay)
                self.call_count += 1
                return text.encode("utf-8")

            stream.collect = collect_audio
            return stream

    return MockAdapter()


@pytest.fixture
def flaky_tts_adapter():
    """Create flaky TTS adapter that fails then succeeds."""

    class FlakyAdapter:
        def __init__(self):
            self.attempt_count = 0
            self.fail_count = 1  # Fail first N attempts

        def synthesize(self, text: str, **kwargs):
            """Return mock synthesis stream that fails initially."""
            stream = MagicMock()

            # Define the collect coroutine inline
            async def collect_audio():
                self.attempt_count += 1
                if self.attempt_count <= self.fail_count:
                    raise RuntimeError(f"Transient error (attempt {self.attempt_count})")
                return text.encode("utf-8")

            stream.collect = collect_audio
            return stream

    return FlakyAdapter()


# ============================================================================
# SentenceQueue Tests
# ============================================================================


@pytest.mark.asyncio
async def test_sentence_queue_basic_operations():
    """Test basic put/get operations."""
    queue = SentenceQueue(maxsize=5)

    task = SentenceTask(seq_id=0, sentence="Hello", timestamp=0.0)
    await queue.put(task)

    result = await queue.get()
    assert result == task
    assert result.seq_id == 0
    assert result.sentence == "Hello"


@pytest.mark.asyncio
async def test_sentence_queue_backpressure():
    """Test queue blocks when full (backpressure)."""
    queue = SentenceQueue(maxsize=2)

    # Fill queue
    await queue.put(SentenceTask(0, "A", 0.0))
    await queue.put(SentenceTask(1, "B", 0.0))

    # Next put should block
    put_task = asyncio.create_task(
        queue.put(SentenceTask(2, "C", 0.0))
    )

    await asyncio.sleep(0.05)
    assert not put_task.done()  # Still blocking

    # Consume one item -> put unblocks
    await queue.get()
    await asyncio.sleep(0.05)
    assert put_task.done()


@pytest.mark.asyncio
async def test_sentence_queue_close_with_sentinel():
    """Test queue close puts sentinel values."""
    queue = SentenceQueue(maxsize=5)

    await queue.put(SentenceTask(0, "A", 0.0))
    await queue.close()

    # Get real item
    task = await queue.get()
    assert task is not None
    assert task.sentence == "A"

    # Get sentinel
    sentinel = await queue.get()
    assert sentinel is None


@pytest.mark.asyncio
async def test_sentence_queue_cannot_put_after_close():
    """Test cannot put to closed queue."""
    queue = SentenceQueue(maxsize=5)
    await queue.close()

    with pytest.raises(RuntimeError, match="Cannot put to closed queue"):
        await queue.put(SentenceTask(0, "A", 0.0))


# ============================================================================
# TTSWorker Tests
# ============================================================================


@pytest.mark.asyncio
async def test_worker_synthesizes_sentences(mock_tts_adapter):
    """Test worker pulls from queue and synthesizes."""
    sentence_queue = SentenceQueue(maxsize=10)
    audio_queue = asyncio.PriorityQueue()

    worker = TTSWorker(
        worker_id=0,
        sentence_queue=sentence_queue,
        audio_queue=audio_queue,
        tts_adapter=mock_tts_adapter,
    )

    # Add sentences
    await sentence_queue.put(SentenceTask(0, "Hello", 0.0))
    await sentence_queue.put(SentenceTask(1, "World", 0.0))
    await sentence_queue.close()

    # Run worker
    await worker.run()

    # Check results
    assert mock_tts_adapter.call_count == 2
    assert worker._stats["synthesized"] == 2
    assert worker._stats["errors"] == 0

    # Verify audio queue has results (use tuple unpacking to get priority and result)
    result0 = await audio_queue.get()
    assert result0.seq_id == 0
    assert result0.status == SynthesisStatus.SUCCESS
    assert result0.audio_bytes == b"Hello"

    result1 = await audio_queue.get()
    assert result1.seq_id == 1
    assert result1.audio_bytes == b"World"


@pytest.mark.asyncio
async def test_worker_skips_empty_sentences(mock_tts_adapter):
    """Test worker skips whitespace-only sentences."""
    sentence_queue = SentenceQueue(maxsize=10)
    audio_queue = asyncio.PriorityQueue()

    worker = TTSWorker(
        worker_id=0,
        sentence_queue=sentence_queue,
        audio_queue=audio_queue,
        tts_adapter=mock_tts_adapter,
    )

    # Add sentences with empty ones
    await sentence_queue.put(SentenceTask(0, "Hello", 0.0))
    await sentence_queue.put(SentenceTask(1, "   ", 0.0))  # Whitespace
    await sentence_queue.put(SentenceTask(2, "", 0.0))  # Empty
    await sentence_queue.put(SentenceTask(3, "World", 0.0))
    await sentence_queue.close()

    # Run worker
    await worker.run()

    # Check stats
    assert worker._stats["synthesized"] == 2  # Only Hello and World
    assert worker._stats["skipped"] == 2  # Two empty
    assert worker._stats["errors"] == 0

    # Verify skipped results
    results = []
    while not audio_queue.empty():
        results.append(await audio_queue.get())

    assert len(results) == 4
    assert results[1].status == SynthesisStatus.SKIPPED
    assert results[2].status == SynthesisStatus.SKIPPED


@pytest.mark.asyncio
async def test_worker_retry_logic(flaky_tts_adapter):
    """Test worker retries transient errors."""
    sentence_queue = SentenceQueue(maxsize=10)
    audio_queue = asyncio.PriorityQueue()

    worker = TTSWorker(
        worker_id=0,
        sentence_queue=sentence_queue,
        audio_queue=audio_queue,
        tts_adapter=flaky_tts_adapter,
        max_retries=2,
    )

    # Add sentence
    await sentence_queue.put(SentenceTask(0, "Hello", 0.0))
    await sentence_queue.close()

    # Run worker
    await worker.run()

    # Should succeed after retry
    assert worker._stats["synthesized"] == 1
    assert worker._stats["errors"] == 0
    assert flaky_tts_adapter.attempt_count == 2  # Failed once, succeeded second


@pytest.mark.asyncio
async def test_worker_fails_after_max_retries():
    """Test worker gives up after max retries."""

    class AlwaysFailAdapter:
        def synthesize(self, text: str, **kwargs):
            stream = MagicMock()

            async def collect_fail():
                raise RuntimeError("Persistent error")

            stream.collect = collect_fail
            return stream

    sentence_queue = SentenceQueue(maxsize=10)
    audio_queue = asyncio.PriorityQueue()

    worker = TTSWorker(
        worker_id=0,
        sentence_queue=sentence_queue,
        audio_queue=audio_queue,
        tts_adapter=AlwaysFailAdapter(),
        max_retries=2,
    )

    # Add sentence
    await sentence_queue.put(SentenceTask(0, "Hello", 0.0))
    await sentence_queue.close()

    # Run worker
    await worker.run()

    # Should record error
    assert worker._stats["synthesized"] == 0
    assert worker._stats["errors"] == 1

    # Check error result
    result = await audio_queue.get()
    assert result.status == SynthesisStatus.ERROR
    assert result.audio_bytes is None
    assert "Persistent error" in result.error_msg


@pytest.mark.asyncio
async def test_worker_gpu_semaphore_limits_concurrency(mock_tts_adapter):
    """Test GPU semaphore limits concurrent synthesis."""
    sentence_queue = SentenceQueue(maxsize=10)
    audio_queue = asyncio.PriorityQueue()
    gpu_semaphore = asyncio.Semaphore(1)  # Only 1 concurrent

    # Create two workers
    worker1 = TTSWorker(
        worker_id=0,
        sentence_queue=sentence_queue,
        audio_queue=audio_queue,
        tts_adapter=mock_tts_adapter,
        gpu_semaphore=gpu_semaphore,
    )
    worker2 = TTSWorker(
        worker_id=1,
        sentence_queue=sentence_queue,
        audio_queue=audio_queue,
        tts_adapter=mock_tts_adapter,
        gpu_semaphore=gpu_semaphore,
    )

    # Add sentences
    await sentence_queue.put(SentenceTask(0, "A", 0.0))
    await sentence_queue.put(SentenceTask(1, "B", 0.0))
    await sentence_queue.close()

    # Run workers concurrently
    await asyncio.gather(worker1.run(), worker2.run())

    # Verify both sentences processed
    assert mock_tts_adapter.call_count == 2


# ============================================================================
# OrderedPlayback Tests
# ============================================================================


@pytest.mark.asyncio
async def test_ordered_playback_in_order():
    """Test ordered playback with in-order results."""
    audio_queue = asyncio.PriorityQueue()
    playback = OrderedPlayback(audio_queue)

    # Add results in order (using keyword arguments for dataclass)
    await audio_queue.put(AudioResult(seq_id=0, status=SynthesisStatus.SUCCESS, audio_bytes=b"A"))
    await audio_queue.put(AudioResult(seq_id=1, status=SynthesisStatus.SUCCESS, audio_bytes=b"B"))
    await audio_queue.put(AudioResult(seq_id=2, status=SynthesisStatus.SUCCESS, audio_bytes=b"C"))

    # Should yield in order
    results = []
    stream = playback.stream_ordered_audio()
    results.append(await stream.__anext__())
    results.append(await stream.__anext__())
    results.append(await stream.__anext__())

    assert results == [b"A", b"B", b"C"]
    assert playback.next_seq_id == 3


@pytest.mark.asyncio
async def test_ordered_playback_out_of_order():
    """Test ordered playback buffers out-of-order results."""
    audio_queue = asyncio.PriorityQueue()
    playback = OrderedPlayback(audio_queue, max_gap_timeout=1.0)

    # Add results out of order (using keyword arguments)
    await audio_queue.put(AudioResult(seq_id=2, status=SynthesisStatus.SUCCESS, audio_bytes=b"C"))
    await audio_queue.put(AudioResult(seq_id=0, status=SynthesisStatus.SUCCESS, audio_bytes=b"A"))
    await audio_queue.put(AudioResult(seq_id=1, status=SynthesisStatus.SUCCESS, audio_bytes=b"B"))

    # Should yield in order: A, B, C
    results = []
    stream = playback.stream_ordered_audio()
    results.append(await stream.__anext__())
    results.append(await stream.__anext__())
    results.append(await stream.__anext__())

    assert results == [b"A", b"B", b"C"]


@pytest.mark.asyncio
async def test_ordered_playback_skip_errors():
    """Test ordered playback skips error results."""
    audio_queue = asyncio.PriorityQueue()
    playback = OrderedPlayback(audio_queue)

    # Add results with error in middle (using keyword arguments)
    await audio_queue.put(AudioResult(seq_id=0, status=SynthesisStatus.SUCCESS, audio_bytes=b"A"))
    await audio_queue.put(
        AudioResult(seq_id=1, status=SynthesisStatus.ERROR, audio_bytes=None, error_msg="Failed")
    )
    await audio_queue.put(AudioResult(seq_id=2, status=SynthesisStatus.SUCCESS, audio_bytes=b"C"))

    # Should yield A and C, skip error
    results = []
    stream = playback.stream_ordered_audio()
    # Get A
    result = await stream.__anext__()
    if result:
        results.append(result)
    # Skip error (advances sequence)
    result = await stream.__anext__()
    if result:
        results.append(result)

    assert results == [b"A", b"C"]


@pytest.mark.asyncio
async def test_ordered_playback_skip_empty():
    """Test ordered playback skips skipped results."""
    audio_queue = asyncio.PriorityQueue()
    playback = OrderedPlayback(audio_queue)

    # Add results with skipped in middle (using keyword arguments)
    await audio_queue.put(AudioResult(seq_id=0, status=SynthesisStatus.SUCCESS, audio_bytes=b"A"))
    await audio_queue.put(AudioResult(seq_id=1, status=SynthesisStatus.SKIPPED, audio_bytes=None))
    await audio_queue.put(AudioResult(seq_id=2, status=SynthesisStatus.SUCCESS, audio_bytes=b"C"))

    # Should yield A and C
    results = []
    stream = playback.stream_ordered_audio()
    # Get A
    result = await stream.__anext__()
    if result:
        results.append(result)
    # Skip empty (advances sequence)
    result = await stream.__anext__()
    if result:
        results.append(result)

    assert results == [b"A", b"C"]


@pytest.mark.asyncio
async def test_ordered_playback_gap_timeout():
    """Test ordered playback skips missing sequence after timeout."""
    audio_queue = asyncio.PriorityQueue()
    playback = OrderedPlayback(audio_queue, max_gap_timeout=0.2)

    # Add result 0 and 2 (skip 1) - using keyword arguments
    await audio_queue.put(AudioResult(seq_id=0, status=SynthesisStatus.SUCCESS, audio_bytes=b"A"))
    await audio_queue.put(AudioResult(seq_id=2, status=SynthesisStatus.SUCCESS, audio_bytes=b"C"))

    # Should yield A, timeout waiting for 1, then yield C
    results = []
    stream = playback.stream_ordered_audio()

    # Get A
    results.append(await stream.__anext__())

    # Get next (should timeout and skip 1, then get C)
    results.append(await stream.__anext__())

    assert results == [b"A", b"C"]
    assert playback.next_seq_id == 3  # Skipped 1, advanced to 3


@pytest.mark.asyncio
async def test_ordered_playback_buffer_size():
    """Test get_buffer_size tracks out-of-order buffer."""
    audio_queue = asyncio.PriorityQueue()
    playback = OrderedPlayback(audio_queue)

    # Add results out of order (using keyword arguments)
    await audio_queue.put(AudioResult(seq_id=2, status=SynthesisStatus.SUCCESS, audio_bytes=b"C"))
    await audio_queue.put(AudioResult(seq_id=3, status=SynthesisStatus.SUCCESS, audio_bytes=b"D"))
    await audio_queue.put(AudioResult(seq_id=0, status=SynthesisStatus.SUCCESS, audio_bytes=b"A"))

    stream = playback.stream_ordered_audio()

    # Get A (0)
    await stream.__anext__()
    assert playback.get_buffer_size() == 2  # C and D buffered

    # Add 1
    await audio_queue.put(AudioResult(seq_id=1, status=SynthesisStatus.SUCCESS, audio_bytes=b"B"))

    # Get B (1)
    await stream.__anext__()
    assert playback.get_buffer_size() == 2  # C and D still buffered

    # Get C (2)
    await stream.__anext__()
    assert playback.get_buffer_size() == 1  # Only D buffered

    # Get D (3)
    await stream.__anext__()
    assert playback.get_buffer_size() == 0  # Empty


# ============================================================================
# ParallelSynthesisPipeline Tests
# ============================================================================


@pytest.mark.asyncio
async def test_pipeline_end_to_end(mock_tts_adapter):
    """Test complete pipeline with mock adapter."""

    async def sentence_stream():
        """Generate test sentences."""
        for i in range(5):
            yield f"Sentence {i}"
            await asyncio.sleep(0.05)  # Faster than TTS

    pipeline = ParallelSynthesisPipeline(
        tts_adapter=mock_tts_adapter,
        num_workers=2,
        max_sentence_queue=3,
    )

    results = []
    async for audio_bytes in pipeline.synthesize_response(sentence_stream()):
        results.append(audio_bytes)

    # Should have 5 results in order
    assert len(results) == 5
    assert results[0] == b"Sentence 0"
    assert results[4] == b"Sentence 4"


@pytest.mark.asyncio
async def test_pipeline_backpressure(mock_tts_adapter):
    """Test pipeline applies backpressure when queue full."""
    # Slow TTS adapter
    mock_tts_adapter.synthesis_delay = 0.3

    async def fast_sentence_stream():
        """Generate sentences faster than TTS can process."""
        for i in range(10):
            yield f"Sentence {i}"
            # No delay - instant production

    pipeline = ParallelSynthesisPipeline(
        tts_adapter=mock_tts_adapter,
        num_workers=2,
        max_sentence_queue=3,  # Small queue for backpressure
    )

    results = []
    async for audio_bytes in pipeline.synthesize_response(fast_sentence_stream()):
        results.append(audio_bytes)

    # All sentences should be processed despite backpressure
    assert len(results) == 10


@pytest.mark.asyncio
async def test_pipeline_parallel_speedup(mock_tts_adapter):
    """Test parallel pipeline is faster than sequential."""
    import time

    mock_tts_adapter.synthesis_delay = 0.1  # 100ms per sentence

    async def sentence_stream():
        for i in range(4):
            yield f"Sentence {i}"

    pipeline = ParallelSynthesisPipeline(
        tts_adapter=mock_tts_adapter,
        num_workers=2,  # 2 parallel workers
        max_sentence_queue=10,
    )

    start = time.monotonic()
    results = []
    async for audio_bytes in pipeline.synthesize_response(sentence_stream()):
        results.append(audio_bytes)
    duration = time.monotonic() - start

    # Sequential: 4 * 0.1 = 0.4s
    # Parallel (2 workers): ~0.2s (2x speedup)
    # Allow 0.3s for overhead
    assert len(results) == 4
    assert duration < 0.35  # Should be faster than sequential


@pytest.mark.asyncio
async def test_pipeline_handles_errors_gracefully():
    """Test pipeline continues after synthesis errors."""

    class MixedAdapter:
        """Adapter that fails on specific sentences."""

        def __init__(self):
            self.call_count = 0

        def synthesize(self, text: str, **kwargs):
            stream = MagicMock()

            async def collect_mixed():
                self.call_count += 1
                # Fail on sentence 1
                if "Sentence 1" in text:
                    raise RuntimeError("Intentional error")
                return text.encode("utf-8")

            stream.collect = collect_mixed
            return stream

    async def sentence_stream():
        for i in range(4):
            yield f"Sentence {i}"

    pipeline = ParallelSynthesisPipeline(
        tts_adapter=MixedAdapter(),
        num_workers=2,
        max_sentence_queue=10,
    )

    results = []
    async for audio_bytes in pipeline.synthesize_response(sentence_stream()):
        if audio_bytes:  # Skip None (errors)
            results.append(audio_bytes)

    # Should have 3 successful results (skip failed sentence 1)
    assert len(results) == 3
    assert b"Sentence 0" in results
    assert b"Sentence 2" in results
    assert b"Sentence 3" in results


@pytest.mark.asyncio
async def test_pipeline_worker_cleanup():
    """Test pipeline cleans up workers on completion."""
    mock_adapter = MagicMock()

    async def mock_collect():
        return b"audio"

    mock_adapter.synthesize = MagicMock(
        return_value=MagicMock(collect=mock_collect)
    )

    async def sentence_stream():
        yield "Hello"

    pipeline = ParallelSynthesisPipeline(
        tts_adapter=mock_adapter,
        num_workers=3,
    )

    async for _ in pipeline.synthesize_response(sentence_stream()):
        pass

    # Workers should be cleaned up
    assert len(pipeline.workers) == 0
    assert len(pipeline.worker_tasks) == 0


@pytest.mark.asyncio
async def test_pipeline_gpu_semaphore():
    """Test pipeline uses GPU semaphore to limit concurrency."""
    mock_adapter = MagicMock()

    async def mock_collect():
        return b"audio"

    mock_adapter.synthesize = MagicMock(
        return_value=MagicMock(collect=mock_collect)
    )

    async def sentence_stream():
        for i in range(3):
            yield f"Sentence {i}"

    pipeline = ParallelSynthesisPipeline(
        tts_adapter=mock_adapter,
        num_workers=3,
        max_gpu_concurrent=2,  # Limit to 2 concurrent GPU ops
    )

    # Verify semaphore created
    assert pipeline.gpu_semaphore is not None
    assert pipeline.gpu_semaphore._value == 2

    results = []
    async for audio_bytes in pipeline.synthesize_response(sentence_stream()):
        results.append(audio_bytes)

    assert len(results) == 3
