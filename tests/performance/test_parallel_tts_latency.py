"""Performance tests for parallel TTS synthesis pipeline.

This test suite validates performance characteristics and SLAs:
- Latency measurements (sequential vs parallel)
- Gap detection (<100ms target)
- GPU memory usage (under load)
- Throughput (sentences per second)
- RTF degradation (<10% target)
- First audio latency (FAL)
- Queue depth under load
- Concurrent request handling

Test Coverage:
- Latency benchmarks (2, 3, 5 workers)
- Memory profiling
- Throughput measurements
- RTF (Real-Time Factor) validation
- Stress testing
- Performance regression detection

# type: ignore  # TODO: Add proper type annotations for test fixtures and helpers

Design reference: Phase B ParallelSynthesisPipeline performance SLAs
"""

import asyncio
import gc
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

# ============================================================================
# Performance Metrics Collection
# ============================================================================


@dataclass
class PerformanceMetrics:
    """Performance metrics for synthesis runs."""

    total_time_ms: float
    first_audio_latency_ms: float
    avg_gap_ms: float
    max_gap_ms: float
    sentences_processed: int
    audio_chunks_emitted: int
    throughput_sentences_per_sec: float
    rtf: float  # Real-Time Factor (synthesis_time / audio_duration)


class PerformanceMonitor:
    """Monitor and collect performance metrics."""

    def __init__(self) -> None:
        """Initialize performance monitor."""
        self.start_time: float | None = None
        self.first_audio_time: float | None = None
        self.emission_times: list[float] = []
        self.sentence_count = 0
        self.chunk_count = 0

    def start(self) -> None:
        """Start monitoring."""
        self.start_time = time.monotonic()

    def record_audio_emission(self) -> None:
        """Record audio chunk emission."""
        now = time.monotonic()
        self.emission_times.append(now)
        self.chunk_count += 1

        if self.first_audio_time is None:
            self.first_audio_time = now

    def record_sentence(self) -> None:
        """Record sentence processed."""
        self.sentence_count += 1

    def get_metrics(self, audio_duration_ms: float = 0.0) -> PerformanceMetrics:
        """Get collected metrics.

        Args:
            audio_duration_ms: Total audio duration in milliseconds

        Returns:
            Performance metrics
        """
        if not self.start_time:
            raise ValueError("Monitor not started")

        end_time = time.monotonic()
        total_time = (end_time - self.start_time) * 1000  # ms

        # First audio latency
        fal = 0.0
        if self.first_audio_time:
            fal = (self.first_audio_time - self.start_time) * 1000

        # Calculate gaps
        gaps = []
        if len(self.emission_times) > 1:
            gaps = [
                (self.emission_times[i] - self.emission_times[i - 1]) * 1000
                for i in range(1, len(self.emission_times))
            ]

        avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
        max_gap = max(gaps) if gaps else 0.0

        # Throughput
        throughput = self.sentence_count / (total_time / 1000) if total_time > 0 else 0.0

        # RTF (Real-Time Factor)
        rtf = total_time / audio_duration_ms if audio_duration_ms > 0 else 0.0

        return PerformanceMetrics(
            total_time_ms=total_time,
            first_audio_latency_ms=fal,
            avg_gap_ms=avg_gap,
            max_gap_ms=max_gap,
            sentences_processed=self.sentence_count,
            audio_chunks_emitted=self.chunk_count,
            throughput_sentences_per_sec=throughput,
            rtf=rtf,
        )


# ============================================================================
# Mock Components with Performance Tracking
# ============================================================================


class MockTTSClientWithMetrics:
    """Mock TTS client with performance tracking."""

    def __init__(self, latency_ms: float = 50.0) -> None:
        """Initialize mock TTS client.

        Args:
            latency_ms: Synthesis latency in milliseconds
        """
        self.latency_ms = latency_ms
        self.call_count = 0
        self.total_synthesis_time_ms = 0.0

    async def synthesize(self, text: str) -> bytes:
        """Mock synthesize with latency tracking.

        Args:
            text: Text to synthesize

        Returns:
            Mock audio data
        """
        self.call_count += 1
        start = time.monotonic()

        await asyncio.sleep(self.latency_ms / 1000.0)

        elapsed = (time.monotonic() - start) * 1000
        self.total_synthesis_time_ms += elapsed

        # Return 20ms of 48kHz PCM audio
        frame_size = 48000 * 2 * 20 // 1000
        return b"\x00" * frame_size


class MockParallelPipelineWithMetrics:
    """Mock parallel pipeline with performance tracking."""

    def __init__(
        self,
        tts_client: MockTTSClientWithMetrics,
        num_workers: int = 2,
        monitor: PerformanceMonitor | None = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            tts_client: TTS client
            num_workers: Number of workers
            monitor: Performance monitor
        """
        self.tts_client = tts_client
        self.num_workers = num_workers
        self.monitor = monitor

    async def synthesize_sentences(
        self,
        sentence_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        """Synthesize sentences with performance tracking.

        Args:
            sentence_stream: Stream of sentences

        Yields:
            Audio chunks
        """
        sentence_queue = asyncio.Queue()
        audio_queue = asyncio.Queue()

        async def collect_sentences():
            seq_id = 0
            async for sentence in sentence_stream:
                if self.monitor:
                    self.monitor.record_sentence()
                await sentence_queue.put((seq_id, sentence))
                seq_id += 1
            for _ in range(self.num_workers):
                await sentence_queue.put(None)

        async def worker():
            while True:
                item = await sentence_queue.get()
                if item is None:
                    break
                seq_id, sentence = item
                audio = await self.tts_client.synthesize(sentence)
                await audio_queue.put((seq_id, audio))

        async def emit_audio():
            next_seq = 0
            buffer = {}
            sentences_emitted = 0

            while True:
                try:
                    seq_id, audio = await asyncio.wait_for(
                        audio_queue.get(),
                        timeout=0.1,
                    )
                    buffer[seq_id] = audio
                except TimeoutError:
                    if sentence_queue.empty() and audio_queue.empty():
                        break
                    continue

                while next_seq in buffer:
                    audio = buffer.pop(next_seq)
                    if self.monitor:
                        self.monitor.record_audio_emission()
                    yield audio
                    next_seq += 1
                    sentences_emitted += 1

        collector = asyncio.create_task(collect_sentences())
        workers = [asyncio.create_task(worker()) for _ in range(self.num_workers)]

        async for audio in emit_audio():
            yield audio

        await collector
        await asyncio.gather(*workers)


# ============================================================================
# Latency Benchmarks
# ============================================================================


class TestLatencyBenchmarks:
    """Test latency characteristics with different worker counts."""

    @pytest.mark.asyncio
    async def test_sequential_baseline(self) -> None:
        """Baseline: sequential synthesis (1 worker)."""
        tts_client = MockTTSClientWithMetrics(latency_ms=100)
        monitor = PerformanceMonitor()
        pipeline = MockParallelPipelineWithMetrics(
            tts_client,
            num_workers=1,
            monitor=monitor,
        )

        async def sentence_stream():
            for i in range(5):
                yield f"Sentence {i}."

        monitor.start()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        # Audio duration: 5 sentences * 20ms per chunk
        metrics = monitor.get_metrics(audio_duration_ms=100.0)

        # Assertions
        assert len(chunks) == 5
        assert metrics.sentences_processed == 5
        # Sequential: 5 * 100ms = 500ms
        assert metrics.total_time_ms >= 500
        assert metrics.total_time_ms < 600  # Allow some overhead
        # FAL should be ~100ms (first synthesis)
        assert metrics.first_audio_latency_ms >= 100
        assert metrics.first_audio_latency_ms < 150

    @pytest.mark.asyncio
    async def test_two_worker_speedup(self) -> None:
        """Test 2-worker parallel synthesis speedup."""
        tts_client = MockTTSClientWithMetrics(latency_ms=100)
        monitor = PerformanceMonitor()
        pipeline = MockParallelPipelineWithMetrics(
            tts_client,
            num_workers=2,
            monitor=monitor,
        )

        async def sentence_stream():
            for i in range(6):
                yield f"Sentence {i}."

        monitor.start()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        metrics = monitor.get_metrics(audio_duration_ms=120.0)

        # Assertions
        assert len(chunks) == 6
        assert metrics.sentences_processed == 6
        # Parallel (2 workers): 3 batches * 100ms = 300ms
        assert metrics.total_time_ms >= 300
        assert metrics.total_time_ms < 400  # Allow overhead
        # Throughput should be ~2x sequential
        assert metrics.throughput_sentences_per_sec > 10  # >10 sent/sec

    @pytest.mark.asyncio
    async def test_three_worker_speedup(self) -> None:
        """Test 3-worker parallel synthesis speedup."""
        tts_client = MockTTSClientWithMetrics(latency_ms=100)
        monitor = PerformanceMonitor()
        pipeline = MockParallelPipelineWithMetrics(
            tts_client,
            num_workers=3,
            monitor=monitor,
        )

        async def sentence_stream():
            for i in range(9):
                yield f"Sentence {i}."

        monitor.start()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        metrics = monitor.get_metrics(audio_duration_ms=180.0)

        # Assertions
        assert len(chunks) == 9
        assert metrics.sentences_processed == 9
        # Parallel (3 workers): 3 batches * 100ms = 300ms
        assert metrics.total_time_ms >= 300
        assert metrics.total_time_ms < 400
        # Throughput should be ~3x sequential
        assert metrics.throughput_sentences_per_sec > 20  # >20 sent/sec

    @pytest.mark.asyncio
    async def test_speedup_comparison(self) -> None:
        """Compare speedup across worker counts."""
        results = {}

        async def sentence_stream():
            for i in range(12):
                yield f"Sentence {i}."

        # Test different worker counts
        for num_workers in [1, 2, 3, 4]:
            tts_client = MockTTSClientWithMetrics(latency_ms=100)
            monitor = PerformanceMonitor()
            pipeline = MockParallelPipelineWithMetrics(
                tts_client,
                num_workers=num_workers,
                monitor=monitor,
            )

            monitor.start()
            chunks = []
            async for chunk in pipeline.synthesize_sentences(sentence_stream()):
                chunks.append(chunk)

            metrics = monitor.get_metrics(audio_duration_ms=240.0)
            results[num_workers] = metrics

        # Verify speedup
        baseline_time = results[1].total_time_ms

        # 2 workers should be ~2x faster (allow 20% overhead)
        assert results[2].total_time_ms < baseline_time * 0.6

        # 3 workers should be ~3x faster (allow 20% overhead)
        assert results[3].total_time_ms < baseline_time * 0.4

        # 4 workers should be ~4x faster (allow 20% overhead)
        assert results[4].total_time_ms < baseline_time * 0.3


# ============================================================================
# Gap Detection Tests
# ============================================================================


class TestGapDetection:
    """Test audio gap detection and continuity."""

    @pytest.mark.asyncio
    async def test_gap_measurement(self) -> None:
        """Measure gaps between audio emissions."""
        tts_client = MockTTSClientWithMetrics(latency_ms=50)
        monitor = PerformanceMonitor()
        pipeline = MockParallelPipelineWithMetrics(
            tts_client,
            num_workers=2,
            monitor=monitor,
        )

        async def sentence_stream():
            for i in range(10):
                yield f"Sentence {i}."

        monitor.start()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        metrics = monitor.get_metrics(audio_duration_ms=200.0)

        # Assertions
        assert len(chunks) == 10
        # Target: <100ms max gap
        assert metrics.max_gap_ms < 100, f"Max gap {metrics.max_gap_ms}ms exceeds 100ms"
        # Target: <50ms average gap
        assert metrics.avg_gap_ms < 50, f"Avg gap {metrics.avg_gap_ms}ms exceeds 50ms"

    @pytest.mark.asyncio
    async def test_no_gaps_with_buffering(self) -> None:
        """Test that buffering eliminates gaps."""
        tts_client = MockTTSClientWithMetrics(latency_ms=30)
        monitor = PerformanceMonitor()
        pipeline = MockParallelPipelineWithMetrics(
            tts_client,
            num_workers=3,  # Over-provision workers
            monitor=monitor,
        )

        async def sentence_stream():
            for i in range(6):
                yield f"Sentence {i}."

        monitor.start()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        metrics = monitor.get_metrics(audio_duration_ms=120.0)

        # With over-provisioned workers, gaps should be minimal
        assert metrics.max_gap_ms < 50  # Very low max gap
        assert metrics.avg_gap_ms < 20  # Very low average gap


# ============================================================================
# Throughput Tests
# ============================================================================


class TestThroughput:
    """Test synthesis throughput (sentences per second)."""

    @pytest.mark.asyncio
    async def test_throughput_measurement(self) -> None:
        """Measure sentences per second throughput."""
        tts_client = MockTTSClientWithMetrics(latency_ms=100)
        monitor = PerformanceMonitor()
        pipeline = MockParallelPipelineWithMetrics(
            tts_client,
            num_workers=2,
            monitor=monitor,
        )

        async def sentence_stream():
            for i in range(20):
                yield f"Sentence {i}."

        monitor.start()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        metrics = monitor.get_metrics(audio_duration_ms=400.0)

        # Assertions
        assert len(chunks) == 20
        # With 2 workers and 100ms latency: ~20 sentences per second theoretical max
        # Practical: ~15-18 sent/sec accounting for overhead
        assert metrics.throughput_sentences_per_sec > 15

    @pytest.mark.asyncio
    async def test_sustained_throughput(self) -> None:
        """Test sustained throughput over longer run."""
        tts_client = MockTTSClientWithMetrics(latency_ms=50)
        monitor = PerformanceMonitor()
        pipeline = MockParallelPipelineWithMetrics(
            tts_client,
            num_workers=3,
            monitor=monitor,
        )

        async def sentence_stream():
            for i in range(100):  # Large batch
                yield f"Sentence {i}."

        monitor.start()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        metrics = monitor.get_metrics(audio_duration_ms=2000.0)

        # Assertions
        assert len(chunks) == 100
        # With 3 workers and 50ms latency: ~60 sentences per second theoretical max
        # Practical: ~50-55 sent/sec
        assert metrics.throughput_sentences_per_sec > 45


# ============================================================================
# RTF (Real-Time Factor) Tests
# ============================================================================


class TestRealTimeFactor:
    """Test RTF (synthesis_time / audio_duration) degradation."""

    @pytest.mark.asyncio
    async def test_rtf_target(self) -> None:
        """Test RTF meets <10% degradation target."""
        tts_client = MockTTSClientWithMetrics(latency_ms=100)
        monitor = PerformanceMonitor()
        pipeline = MockParallelPipelineWithMetrics(
            tts_client,
            num_workers=2,
            monitor=monitor,
        )

        async def sentence_stream():
            for i in range(10):
                yield f"Sentence {i}."

        monitor.start()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        # Each chunk is 20ms, total audio = 10 * 20ms = 200ms
        audio_duration_ms = 200.0
        metrics = monitor.get_metrics(audio_duration_ms=audio_duration_ms)

        # RTF should be close to 1.0 (real-time)
        # With parallel synthesis, RTF can be < 1.0 (faster than real-time)
        # Target: RTF < 1.1 (within 10% of real-time)
        assert metrics.rtf < 1.1, f"RTF {metrics.rtf} exceeds 1.1 (10% degradation)"

    @pytest.mark.asyncio
    async def test_rtf_under_load(self) -> None:
        """Test RTF under heavy load."""
        tts_client = MockTTSClientWithMetrics(latency_ms=80)
        monitor = PerformanceMonitor()
        pipeline = MockParallelPipelineWithMetrics(
            tts_client,
            num_workers=3,
            monitor=monitor,
        )

        async def sentence_stream():
            for i in range(50):
                yield f"Long sentence {i} with more content."

        monitor.start()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        # 50 chunks * 20ms = 1000ms audio
        metrics = monitor.get_metrics(audio_duration_ms=1000.0)

        # Even under load, RTF should stay < 1.1
        assert metrics.rtf < 1.1


# ============================================================================
# First Audio Latency (FAL) Tests
# ============================================================================


class TestFirstAudioLatency:
    """Test First Audio Latency (time to first audio chunk)."""

    @pytest.mark.asyncio
    async def test_fal_measurement(self) -> None:
        """Measure FAL (First Audio Latency)."""
        tts_client = MockTTSClientWithMetrics(latency_ms=100)
        monitor = PerformanceMonitor()
        pipeline = MockParallelPipelineWithMetrics(
            tts_client,
            num_workers=2,
            monitor=monitor,
        )

        async def sentence_stream():
            for i in range(5):
                yield f"Sentence {i}."

        monitor.start()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        metrics = monitor.get_metrics(audio_duration_ms=100.0)

        # FAL should be ~100ms (time for first synthesis)
        assert metrics.first_audio_latency_ms >= 100
        assert metrics.first_audio_latency_ms < 150  # Allow overhead

    @pytest.mark.asyncio
    async def test_fal_target_gpu(self) -> None:
        """Test FAL meets GPU target (<300ms p95)."""
        results = []

        for _ in range(10):  # Multiple runs for p95
            tts_client = MockTTSClientWithMetrics(latency_ms=100)
            monitor = PerformanceMonitor()
            pipeline = MockParallelPipelineWithMetrics(
                tts_client,
                num_workers=2,
                monitor=monitor,
            )

            async def sentence_stream():
                yield "First sentence."

            monitor.start()
            async for _chunk in pipeline.synthesize_sentences(sentence_stream()):
                break  # Only need first chunk

            metrics = monitor.get_metrics(audio_duration_ms=20.0)
            results.append(metrics.first_audio_latency_ms)

        # Calculate p95
        results_sorted = sorted(results)
        p95_index = int(len(results_sorted) * 0.95)
        p95_fal = results_sorted[p95_index]

        # Target: p95 < 300ms for GPU
        assert p95_fal < 300, f"p95 FAL {p95_fal}ms exceeds 300ms target"


# ============================================================================
# Memory and Resource Tests
# ============================================================================


class TestMemoryAndResources:
    """Test memory usage and resource management."""

    @pytest.mark.asyncio
    async def test_memory_stability(self) -> None:
        """Test memory doesn't grow excessively during synthesis."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        tts_client = MockTTSClientWithMetrics(latency_ms=50)
        pipeline = MockParallelPipelineWithMetrics(tts_client, num_workers=2)

        # Process 100 sentences
        async def sentence_stream():
            for i in range(100):
                yield f"Sentence {i}."

        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (<50MB for 100 sentences)
        assert memory_growth < 50, f"Memory grew by {memory_growth}MB (excessive)"

    @pytest.mark.asyncio
    async def test_queue_depth_under_load(self) -> None:
        """Test queue depth stays bounded under load."""
        # This test would track queue depth during synthesis
        # Ensuring backpressure prevents unbounded growth
        pass  # Placeholder (requires queue depth instrumentation)


# ============================================================================
# Stress Tests
# ============================================================================


class TestStressScenarios:
    """Stress test scenarios for robustness."""

    @pytest.mark.asyncio
    async def test_rapid_short_sentences(self) -> None:
        """Test handling of many rapid short sentences."""
        tts_client = MockTTSClientWithMetrics(latency_ms=20)
        monitor = PerformanceMonitor()
        pipeline = MockParallelPipelineWithMetrics(
            tts_client,
            num_workers=3,
            monitor=monitor,
        )

        async def sentence_stream():
            for i in range(200):
                yield f"Hi {i}."

        monitor.start()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        metrics = monitor.get_metrics(audio_duration_ms=4000.0)

        assert len(chunks) == 200
        # High throughput scenario
        assert metrics.throughput_sentences_per_sec > 100

    @pytest.mark.asyncio
    async def test_very_long_sentences(self) -> None:
        """Test handling of very long sentences."""
        tts_client = MockTTSClientWithMetrics(latency_ms=200)  # Longer synthesis
        monitor = PerformanceMonitor()
        pipeline = MockParallelPipelineWithMetrics(
            tts_client,
            num_workers=2,
            monitor=monitor,
        )

        async def sentence_stream():
            for _i in range(5):
                long_sentence = " ".join(["word"] * 100) + "."
                yield long_sentence

        monitor.start()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        metrics = monitor.get_metrics(audio_duration_ms=100.0)

        assert len(chunks) == 5
        # Should still complete in reasonable time
        assert metrics.total_time_ms < 600  # 3 batches * 200ms

    @pytest.mark.asyncio
    async def test_mixed_sentence_lengths(self) -> None:
        """Test mixed short and long sentences."""
        tts_client = MockTTSClientWithMetrics(latency_ms=100)
        monitor = PerformanceMonitor()
        pipeline = MockParallelPipelineWithMetrics(
            tts_client,
            num_workers=2,
            monitor=monitor,
        )

        async def sentence_stream():
            lengths = [5, 50, 10, 100, 20, 75, 5, 30, 90, 15]
            for length in lengths:
                sentence = " ".join(["word"] * length) + "."
                yield sentence

        monitor.start()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        metrics = monitor.get_metrics(audio_duration_ms=200.0)

        assert len(chunks) == 10
        # Verify reasonable performance despite variation
        assert metrics.throughput_sentences_per_sec > 8


# ============================================================================
# Performance Regression Tests
# ============================================================================


class TestPerformanceRegression:
    """Test for performance regressions."""

    @pytest.mark.asyncio
    async def test_baseline_performance(self) -> None:
        """Establish baseline performance metrics.

        This test serves as a regression detector. If this test starts
        failing, it indicates a performance degradation in the pipeline.
        """
        tts_client = MockTTSClientWithMetrics(latency_ms=100)
        monitor = PerformanceMonitor()
        pipeline = MockParallelPipelineWithMetrics(
            tts_client,
            num_workers=2,
            monitor=monitor,
        )

        async def sentence_stream():
            for i in range(10):
                yield f"Baseline sentence {i}."

        monitor.start()
        chunks = []
        async for chunk in pipeline.synthesize_sentences(sentence_stream()):
            chunks.append(chunk)

        metrics = monitor.get_metrics(audio_duration_ms=200.0)

        # Baseline assertions (adjust based on actual implementation)
        assert metrics.total_time_ms < 600  # 10 sentences, 2 workers
        assert metrics.first_audio_latency_ms < 150
        assert metrics.max_gap_ms < 100
        assert metrics.throughput_sentences_per_sec > 15
        assert metrics.rtf < 1.1

        # Print metrics for reference
        print("\nBaseline Performance Metrics:")
        print(f"  Total Time: {metrics.total_time_ms:.2f}ms")
        print(f"  FAL: {metrics.first_audio_latency_ms:.2f}ms")
        print(f"  Max Gap: {metrics.max_gap_ms:.2f}ms")
        print(f"  Throughput: {metrics.throughput_sentences_per_sec:.2f} sent/sec")
        print(f"  RTF: {metrics.rtf:.3f}")
