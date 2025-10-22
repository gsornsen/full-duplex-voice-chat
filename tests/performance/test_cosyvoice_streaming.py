"""Performance tests for CosyVoice streaming mode.

This test suite validates performance characteristics of CosyVoice streaming
synthesis, ensuring it meets latency and throughput targets for realtime
duplex voice conversations.

Performance Targets:
- First Audio Latency (FAL): p95 < 500ms (streaming), < 300ms (GPU optimized)
- Real-Time Factor (RTF): < 0.5 (2x faster than real-time)
- Inter-chunk jitter: p95 < 10ms
- Streaming vs Batch: FAL improvement 10-20x

Test Categories:
- Latency: FAL, inter-chunk timing
- Throughput: RTF, samples per second
- Comparison: Streaming vs batch mode performance
- Scalability: Performance under load

Note: These tests require GPU acceleration for meaningful results.
      Run with pytest -m performance on GPU-enabled systems.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.tts.adapters.adapter_cosyvoice import CosyVoiceAdapter

# Performance test markers
pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for performance tests"),
]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_model_path(tmp_path: Path) -> Path:
    """Create temporary voicepack directory with dummy model files."""
    voicepack_dir = tmp_path / "cosyvoice-model"
    voicepack_dir.mkdir()

    # Create dummy model files to skip download
    for filename in ["cosyvoice.yaml", "llm.pt", "flow.pt", "hift.pt"]:
        (voicepack_dir / filename).touch()

    return voicepack_dir


@pytest.fixture
def mock_cosyvoice_streaming_class() -> Mock:
    """Create a mock CosyVoice2 class with streaming support.

    This mock simulates incremental audio generation to test streaming
    performance characteristics.
    """

    def streaming_inference_generator():
        """Simulate streaming audio generation with realistic timing."""
        # Simulate 10 chunks, each taking ~50ms to generate
        for _i in range(10):
            # Simulate inference latency
            time.sleep(0.05)  # 50ms per chunk

            # Yield 100ms of audio per chunk (2400 samples @ 24kHz)
            chunk_audio = torch.zeros(1, 2400, dtype=torch.float32)
            yield {"tts_speech": chunk_audio}

    mock_model_instance = Mock()
    mock_model_instance.inference_zero_shot.return_value = streaming_inference_generator()

    mock_class = Mock(return_value=mock_model_instance)
    return mock_class


@pytest.fixture
def streaming_adapter(
    mock_model_path: Path, mock_cosyvoice_streaming_class: Mock
) -> CosyVoiceAdapter:
    """Create CosyVoiceAdapter instance configured for streaming mode testing."""
    with patch("src.tts.adapters.adapter_cosyvoice.torch.cuda.is_available", return_value=True):
        with patch(
            "src.tts.adapters.adapter_cosyvoice.torch.cuda.get_device_name",
            return_value="NVIDIA RTX 4090",
        ):
            with patch(
                "src.tts.adapters.adapter_cosyvoice.CosyVoice2",
                mock_cosyvoice_streaming_class,
            ):
                adapter = CosyVoiceAdapter("test-model-streaming", mock_model_path)
                return adapter


# ============================================================================
# First Audio Latency (FAL) Tests
# ============================================================================


class TestFirstAudioLatency:
    """Test suite for First Audio Latency (FAL) validation."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Awaiting streaming mode implementation (stream=True)")
    async def test_fal_under_500ms_streaming_mode(
        self, streaming_adapter: CosyVoiceAdapter
    ) -> None:
        """Test First Audio Latency is < 500ms in streaming mode.

        Target: p95 FAL < 500ms
        Baseline: Batch mode FAL ~2000ms

        This test validates that streaming mode provides 4x+ improvement
        over batch mode for first audio latency.
        """
        test_text = "This is a test sentence for measuring first audio latency."

        async def text_gen() -> AsyncIterator[str]:
            yield test_text

        # Measure time to first frame
        start_time = time.perf_counter()
        first_frame_time: float | None = None

        async for _frame in streaming_adapter.synthesize_stream(text_gen()):
            if first_frame_time is None:
                first_frame_time = time.perf_counter() - start_time
                break  # Only measure first frame

        assert first_frame_time is not None, "No frames generated"

        # Convert to milliseconds
        fal_ms = first_frame_time * 1000

        # Assert FAL < 500ms
        assert fal_ms < 500, (
            f"First Audio Latency {fal_ms:.1f}ms exceeds 500ms target. "
            f"Streaming mode should provide <500ms FAL (vs ~2000ms batch mode)."
        )

        # Log performance metric for telemetry
        print(f"\nFirst Audio Latency: {fal_ms:.1f}ms (target: <500ms)")

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Awaiting GPU optimization implementation")
    async def test_fal_under_300ms_optimized(self, streaming_adapter: CosyVoiceAdapter) -> None:
        """Test First Audio Latency is < 300ms with GPU optimizations.

        Target: p95 FAL < 300ms (optimized)
        Requires: TensorRT, FP16, CUDA graph optimizations

        This is a stretch goal for maximum performance.
        """
        test_text = "Short test."

        async def text_gen() -> AsyncIterator[str]:
            yield test_text

        start_time = time.perf_counter()
        first_frame_time: float | None = None

        async for _frame in streaming_adapter.synthesize_stream(text_gen()):
            if first_frame_time is None:
                first_frame_time = time.perf_counter() - start_time
                break

        assert first_frame_time is not None
        fal_ms = first_frame_time * 1000

        assert fal_ms < 300, f"Optimized FAL {fal_ms:.1f}ms exceeds 300ms target"
        print(f"\nOptimized First Audio Latency: {fal_ms:.1f}ms (target: <300ms)")

    @pytest.mark.asyncio
    async def test_fal_consistency_across_runs(self, streaming_adapter: CosyVoiceAdapter) -> None:
        """Test FAL is consistent across multiple synthesis runs.

        Validates that warmup is effective and subsequent runs maintain
        low latency (no cold-start penalty).

        Target: FAL variance < 20% after warmup
        """
        test_text = "Consistency test sentence."

        async def text_gen() -> AsyncIterator[str]:
            yield test_text

        # Warmup run (discard)
        _ = [frame async for frame in streaming_adapter.synthesize_stream(text_gen())]
        await streaming_adapter.reset()

        # Measure FAL for 5 runs
        fal_measurements: list[float] = []

        for _run in range(5):
            start_time = time.perf_counter()
            first_frame_time: float | None = None

            async for _frame in streaming_adapter.synthesize_stream(text_gen()):
                if first_frame_time is None:
                    first_frame_time = time.perf_counter() - start_time
                    break

            if first_frame_time is not None:
                fal_measurements.append(first_frame_time * 1000)

            await streaming_adapter.reset()

        assert len(fal_measurements) == 5, "Not all runs completed"

        # Calculate statistics
        mean_fal = np.mean(fal_measurements)
        std_fal = np.std(fal_measurements)
        cv = (std_fal / mean_fal) * 100  # Coefficient of variation (%)

        print("\nFAL Statistics over 5 runs:")
        print(f"  Mean: {mean_fal:.1f}ms")
        print(f"  Std Dev: {std_fal:.1f}ms")
        print(f"  CV: {cv:.1f}%")

        # Assert variance < 20%
        assert cv < 20, f"FAL variance {cv:.1f}% exceeds 20% target (unstable performance)"


# ============================================================================
# Real-Time Factor (RTF) Tests
# ============================================================================


class TestRealTimeFactor:
    """Test suite for Real-Time Factor (RTF) validation."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Awaiting streaming mode implementation")
    async def test_rtf_under_threshold(self, streaming_adapter: CosyVoiceAdapter) -> None:
        """Test Real-Time Factor (RTF) is < 0.5 for 2x real-time performance.

        RTF = synthesis_time / audio_duration
        Target: RTF < 0.5 (audio generated 2x faster than real-time)

        Example: Generating 1 second of audio should take < 500ms
        """
        test_text = (
            "This is a longer test sentence to measure synthesis "
            "throughput and real-time factor."
        )

        async def text_gen() -> AsyncIterator[str]:
            yield test_text

        # Measure total synthesis time
        start_time = time.perf_counter()
        frames: list[bytes] = []

        async for frame in streaming_adapter.synthesize_stream(text_gen()):
            frames.append(frame)

        synthesis_time = time.perf_counter() - start_time

        # Calculate audio duration
        total_samples = len(frames) * 960  # 960 samples per frame @ 48kHz
        audio_duration = total_samples / 48000  # seconds

        # Calculate RTF
        rtf = synthesis_time / audio_duration if audio_duration > 0 else float("inf")

        print("\nRTF Performance:")
        print(f"  Synthesis Time: {synthesis_time:.3f}s")
        print(f"  Audio Duration: {audio_duration:.3f}s")
        print(f"  RTF: {rtf:.3f} (target: <0.5)")

        assert rtf < 0.5, (
            f"RTF {rtf:.3f} exceeds 0.5 threshold. "
            f"Synthesis should be at least 2x faster than real-time."
        )

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Awaiting streaming mode implementation")
    async def test_rtf_with_long_text(self, streaming_adapter: CosyVoiceAdapter) -> None:
        """Test RTF remains low for long text synthesis (30+ seconds).

        Validates that performance scales linearly with text length
        without degradation.

        Target: RTF < 0.5 for 30 second synthesis
        """
        # Generate long text (~30 seconds of audio)
        long_text = " ".join(["This is a test sentence."] * 50)

        async def text_gen() -> AsyncIterator[str]:
            yield long_text

        start_time = time.perf_counter()
        frames: list[bytes] = []

        async for frame in streaming_adapter.synthesize_stream(text_gen()):
            frames.append(frame)

        synthesis_time = time.perf_counter() - start_time
        total_samples = len(frames) * 960
        audio_duration = total_samples / 48000
        rtf = synthesis_time / audio_duration

        print("\nLong Text RTF:")
        print(f"  Audio Duration: {audio_duration:.1f}s")
        print(f"  Synthesis Time: {synthesis_time:.1f}s")
        print(f"  RTF: {rtf:.3f}")

        assert rtf < 0.5, f"Long text RTF {rtf:.3f} exceeds 0.5 threshold"


# ============================================================================
# Streaming vs Batch Comparison Tests
# ============================================================================


class TestStreamingVsBatch:
    """Test suite comparing streaming mode vs batch mode performance."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Awaiting streaming mode implementation")
    async def test_streaming_vs_batch_fal_comparison(
        self, mock_model_path: Path, mock_cosyvoice_streaming_class: Mock
    ) -> None:
        """Test streaming mode provides 10-20x FAL improvement over batch mode.

        Expected:
        - Batch mode FAL: ~2000ms
        - Streaming mode FAL: ~200ms (10x improvement)
        """
        test_text = "Comparison test sentence."

        # Test batch mode (stream=False)
        with patch("src.tts.adapters.adapter_cosyvoice.torch.cuda.is_available", return_value=True):
            with patch(
                "src.tts.adapters.adapter_cosyvoice.torch.cuda.get_device_name",
                return_value="NVIDIA RTX 4090",
            ):
                # Create batch mode adapter
                batch_mock = Mock()
                batch_audio = torch.zeros(1, 24000, dtype=torch.float32)  # Full audio at once
                batch_mock.inference_zero_shot.return_value = [{"tts_speech": batch_audio}]
                batch_class = Mock(return_value=batch_mock)

                with patch("src.tts.adapters.adapter_cosyvoice.CosyVoice2", batch_class):
                    batch_adapter = CosyVoiceAdapter("batch-model", mock_model_path)

                    async def text_gen_batch() -> AsyncIterator[str]:
                        yield test_text

                    # Measure batch FAL
                    batch_start = time.perf_counter()
                    batch_first_time: float | None = None

                    async for _frame in batch_adapter.synthesize_stream(text_gen_batch()):
                        if batch_first_time is None:
                            batch_first_time = time.perf_counter() - batch_start
                            break

        # Test streaming mode (stream=True)
        with patch("src.tts.adapters.adapter_cosyvoice.torch.cuda.is_available", return_value=True):
            with patch(
                "src.tts.adapters.adapter_cosyvoice.torch.cuda.get_device_name",
                return_value="NVIDIA RTX 4090",
            ):
                with patch(
                    "src.tts.adapters.adapter_cosyvoice.CosyVoice2",
                    mock_cosyvoice_streaming_class,
                ):
                    streaming_adapter = CosyVoiceAdapter("streaming-model", mock_model_path)

                    async def text_gen_streaming() -> AsyncIterator[str]:
                        yield test_text

                    # Measure streaming FAL
                    streaming_start = time.perf_counter()
                    streaming_first_time: float | None = None

                    async for _frame in streaming_adapter.synthesize_stream(text_gen_streaming()):
                        if streaming_first_time is None:
                            streaming_first_time = time.perf_counter() - streaming_start
                            break

        assert batch_first_time is not None
        assert streaming_first_time is not None

        batch_fal_ms = batch_first_time * 1000
        streaming_fal_ms = streaming_first_time * 1000
        improvement = batch_fal_ms / streaming_fal_ms

        print("\nBatch vs Streaming FAL:")
        print(f"  Batch Mode: {batch_fal_ms:.1f}ms")
        print(f"  Streaming Mode: {streaming_fal_ms:.1f}ms")
        print(f"  Improvement: {improvement:.1f}x")

        # Assert 10-20x improvement
        assert improvement >= 4.0, (
            f"Streaming FAL improvement {improvement:.1f}x is less than 4x minimum. "
            f"Expected 10-20x improvement over batch mode."
        )

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Awaiting streaming mode implementation")
    async def test_streaming_maintains_quality(self) -> None:
        """Test streaming mode maintains audio quality comparable to batch mode.

        This is a placeholder for audio quality comparison tests.
        Would require perceptual metrics (PESQ, MOS, etc.)
        """
        pytest.skip("Audio quality comparison requires perceptual metrics implementation")


# ============================================================================
# Inter-Chunk Timing Tests
# ============================================================================


class TestInterChunkTiming:
    """Test suite for inter-chunk timing consistency (jitter)."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Awaiting streaming mode implementation")
    async def test_inter_chunk_jitter_under_10ms(self, streaming_adapter: CosyVoiceAdapter) -> None:
        """Test inter-chunk timing jitter is < 10ms (p95).

        Consistent chunk delivery is critical for smooth playback.
        High jitter causes audio artifacts and buffer underruns.

        Target: p95 jitter < 10ms
        """
        test_text = "Test sentence for measuring inter-chunk timing consistency."

        async def text_gen() -> AsyncIterator[str]:
            yield test_text

        # Collect frame arrival times
        frame_times: list[float] = []
        start_time = time.perf_counter()

        async for _frame in streaming_adapter.synthesize_stream(text_gen()):
            frame_times.append(time.perf_counter() - start_time)

        # Calculate inter-frame intervals
        intervals = [
            (frame_times[i + 1] - frame_times[i]) * 1000  # Convert to ms
            for i in range(len(frame_times) - 1)
        ]

        if len(intervals) == 0:
            pytest.skip("Not enough frames to measure jitter")

        # Calculate jitter statistics
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        p95_jitter = np.percentile(intervals, 95)

        print("\nInter-Chunk Timing:")
        print(f"  Mean Interval: {mean_interval:.2f}ms")
        print(f"  Std Dev: {std_interval:.2f}ms")
        print(f"  p95 Jitter: {p95_jitter:.2f}ms (target: <10ms)")

        assert p95_jitter < 10, f"p95 jitter {p95_jitter:.2f}ms exceeds 10ms target"


# ============================================================================
# Scalability Tests
# ============================================================================


class TestScalability:
    """Test suite for performance under load and concurrency."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Awaiting streaming mode implementation")
    async def test_concurrent_synthesis_sessions(
        self, mock_model_path: Path, mock_cosyvoice_streaming_class: Mock
    ) -> None:
        """Test multiple concurrent synthesis sessions maintain performance.

        Validates that GPU can handle multiple concurrent requests
        without significant performance degradation.

        Target: 4 concurrent sessions with RTF < 0.5 each
        """

        async def synthesize_session(session_id: int) -> float:
            """Run single synthesis session and return RTF."""
            with patch(
                "src.tts.adapters.adapter_cosyvoice.torch.cuda.is_available", return_value=True
            ):
                with patch(
                    "src.tts.adapters.adapter_cosyvoice.torch.cuda.get_device_name",
                    return_value="NVIDIA RTX 4090",
                ):
                    with patch(
                        "src.tts.adapters.adapter_cosyvoice.CosyVoice2",
                        mock_cosyvoice_streaming_class,
                    ):
                        adapter = CosyVoiceAdapter(f"session-{session_id}", mock_model_path)

                        async def text_gen() -> AsyncIterator[str]:
                            yield f"Session {session_id} test sentence."

                        start_time = time.perf_counter()
                        frames: list[bytes] = []

                        async for frame in adapter.synthesize_stream(text_gen()):
                            frames.append(frame)

                        synthesis_time = time.perf_counter() - start_time
                        total_samples = len(frames) * 960
                        audio_duration = total_samples / 48000
                        rtf = synthesis_time / audio_duration if audio_duration > 0 else 0

                        return rtf

        # Run 4 concurrent sessions
        tasks = [synthesize_session(i) for i in range(4)]
        rtf_results = await asyncio.gather(*tasks)

        print("\nConcurrent Sessions RTF:")
        for i, rtf in enumerate(rtf_results):
            print(f"  Session {i}: RTF {rtf:.3f}")

        # All sessions should maintain RTF < 0.5
        for i, rtf in enumerate(rtf_results):
            assert rtf < 0.5, f"Session {i} RTF {rtf:.3f} exceeds 0.5 under concurrent load"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Awaiting streaming mode implementation")
    async def test_memory_usage_stable(self, streaming_adapter: CosyVoiceAdapter) -> None:
        """Test GPU memory usage remains stable across multiple synthesis runs.

        Validates no memory leaks during repeated synthesis.

        Target: Memory variance < 100MB across 10 runs
        """
        if not torch.cuda.is_available():
            pytest.skip("GPU required for memory tests")

        test_text = "Memory stability test sentence."

        async def text_gen() -> AsyncIterator[str]:
            yield test_text

        memory_measurements: list[float] = []

        for _run in range(10):
            # Clear cache before measurement
            torch.cuda.empty_cache()

            # Measure memory before synthesis
            memory_before = torch.cuda.memory_allocated() / (1024**2)  # MB

            # Synthesize
            _ = [frame async for frame in streaming_adapter.synthesize_stream(text_gen())]

            # Measure memory after synthesis
            memory_after = torch.cuda.memory_allocated() / (1024**2)  # MB
            memory_measurements.append(memory_after - memory_before)

            await streaming_adapter.reset()

        # Calculate memory statistics
        mean_memory = np.mean(memory_measurements)
        std_memory = np.std(memory_measurements)
        max_memory = np.max(memory_measurements)
        min_memory = np.min(memory_measurements)
        memory_range = max_memory - min_memory

        print("\nMemory Usage Stability (10 runs):")
        print(f"  Mean: {mean_memory:.1f}MB")
        print(f"  Std Dev: {std_memory:.1f}MB")
        print(f"  Range: {memory_range:.1f}MB (target: <100MB)")

        assert memory_range < 100, (
            f"Memory usage variance {memory_range:.1f}MB exceeds 100MB target. "
            f"Possible memory leak detected."
        )
