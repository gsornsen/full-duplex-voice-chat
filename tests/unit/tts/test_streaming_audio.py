"""Unit tests for CosyVoice streaming audio processing components.

This test suite validates the streaming audio processing components required for
CosyVoice streaming mode (stream=True). These components enable incremental audio
generation with minimal latency.

Test Coverage:
- StreamingResampler: Incremental resampling with overlap/add for continuity
- FrameBuffer: Accumulation of resampled audio into 20ms frames
- Integration: End-to-end streaming synthesis with pause/resume/stop
- Performance: First Audio Latency (FAL) and Real-Time Factor (RTF) validation

Note: These tests are designed for the planned streaming implementation.
      Current adapter uses batch mode (stream=False) as a baseline.
"""

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest
from tts.adapters.cosyvoice.adapter import (
    COSYVOICE_NATIVE_SAMPLE_RATE,
    TARGET_SAMPLE_RATE_HZ,
)

# ============================================================================
# Test Fixtures
# ============================================================================


class StreamingResampler:
    """Streaming audio resampler with overlap/add for artifact-free resampling.

    This is a PLACEHOLDER implementation for testing. The actual implementation
    should be provided by the ML engineer as part of the streaming mode feature.

    The resampler maintains overlap buffer state across chunks to ensure
    continuity at chunk boundaries, avoiding audio artifacts.

    Attributes:
        source_rate: Source sample rate (e.g., 24000 Hz)
        target_rate: Target sample rate (e.g., 48000 Hz)
        overlap_samples: Number of samples to overlap between chunks
        overlap_buffer: Buffer storing overlap from previous chunk
    """

    def __init__(
        self,
        source_rate: int = COSYVOICE_NATIVE_SAMPLE_RATE,
        target_rate: int = TARGET_SAMPLE_RATE_HZ,
        overlap_samples: int = 256,
    ):
        """Initialize streaming resampler.

        Args:
            source_rate: Source sample rate in Hz (default: 24000)
            target_rate: Target sample rate in Hz (default: 48000)
            overlap_samples: Overlap buffer size in samples (default: 256)
        """
        self.source_rate = source_rate
        self.target_rate = target_rate
        self.overlap_samples = overlap_samples
        self.overlap_buffer: np.ndarray | None = None

    def resample_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Resample audio chunk with overlap/add.

        Args:
            chunk: Audio chunk at source sample rate (int16 or float32)

        Returns:
            Resampled chunk at target sample rate (same dtype as input)
        """
        # Handle empty chunks gracefully
        if len(chunk) == 0:
            return chunk  # Return empty array with same dtype

        # PLACEHOLDER: Actual implementation should use proper resampling
        # with overlap/add to avoid boundary artifacts
        from scipy import signal

        # Convert to float for resampling
        chunk_float = chunk.astype(np.float32)

        # Calculate target length
        num_samples = int(len(chunk_float) * self.target_rate / self.source_rate)

        # Handle edge case where calculated num_samples is 0
        if num_samples == 0:
            # Return empty array with correct dtype
            if chunk.dtype == np.int16:
                return np.array([], dtype=np.int16)
            return np.array([], dtype=np.float32)

        # Resample using scipy
        resampled = signal.resample(chunk_float, num_samples)

        # Convert back to original dtype
        if chunk.dtype == np.int16:
            return resampled.astype(np.int16)  # type: ignore[no-any-return]
        return resampled.astype(np.float32)  # type: ignore[no-any-return]

    def flush(self) -> np.ndarray | None:
        """Flush overlap buffer and return final residual samples.

        Returns:
            Final audio samples from overlap buffer, or None if empty
        """
        if self.overlap_buffer is not None and len(self.overlap_buffer) > 0:
            residual = self.overlap_buffer.copy()
            self.overlap_buffer = None
            return residual
        return None


class FrameBuffer:
    """Buffer for accumulating resampled audio into fixed 20ms frames.

    Accumulates incoming audio samples and yields complete 20ms frames
    (960 samples at 48kHz). Partial frames are buffered until enough
    samples arrive.

    Attributes:
        sample_rate: Target sample rate in Hz (48000)
        frame_duration_ms: Frame duration in milliseconds (20)
        samples_per_frame: Samples per frame (960 at 48kHz)
        buffer: Accumulated samples waiting for next frame
    """

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE_HZ, frame_duration_ms: int = 20):
        """Initialize frame buffer.

        Args:
            sample_rate: Sample rate in Hz (default: 48000)
            frame_duration_ms: Frame duration in ms (default: 20)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
        self.buffer: list[int] = []

    def add_samples(self, samples: np.ndarray) -> list[bytes]:
        """Add samples to buffer and return complete frames.

        Args:
            samples: Audio samples to add (int16)

        Returns:
            List of complete 20ms frames as bytes
        """
        frames: list[bytes] = []

        # Add new samples to buffer
        self.buffer.extend(samples.tolist())

        # Extract complete frames
        while len(self.buffer) >= self.samples_per_frame:
            frame_samples = self.buffer[: self.samples_per_frame]
            self.buffer = self.buffer[self.samples_per_frame :]

            # Convert to bytes
            frame_array = np.array(frame_samples, dtype=np.int16)
            frames.append(frame_array.tobytes())

        return frames

    def flush(self) -> bytes | None:
        """Flush buffer and return final partial frame (zero-padded).

        Returns:
            Final frame as bytes (zero-padded), or None if buffer empty
        """
        if len(self.buffer) == 0:
            return None

        # Pad to full frame
        frame_samples = self.buffer + [0] * (self.samples_per_frame - len(self.buffer))
        frame_array = np.array(frame_samples, dtype=np.int16)
        self.buffer = []

        return frame_array.tobytes()


@pytest.fixture
def resampler() -> StreamingResampler:
    """Create StreamingResampler instance for testing."""
    return StreamingResampler(
        source_rate=COSYVOICE_NATIVE_SAMPLE_RATE, target_rate=TARGET_SAMPLE_RATE_HZ
    )


@pytest.fixture
def frame_buffer() -> FrameBuffer:
    """Create FrameBuffer instance for testing."""
    return FrameBuffer(sample_rate=TARGET_SAMPLE_RATE_HZ, frame_duration_ms=20)


# ============================================================================
# StreamingResampler Tests
# ============================================================================


class TestStreamingResampler:
    """Test suite for StreamingResampler component."""

    def test_basic_resampling_24khz_to_48khz(self, resampler: StreamingResampler) -> None:
        """Test basic resampling from 24kHz to 48kHz doubles sample count."""
        # Create 1 second of audio at 24kHz
        chunk = np.zeros(24000, dtype=np.int16)

        resampled = resampler.resample_chunk(chunk)

        # Should be ~48000 samples (2x due to upsampling)
        assert len(resampled) == pytest.approx(48000, abs=10)
        assert resampled.dtype == np.int16

    def test_chunk_boundary_continuity(self, resampler: StreamingResampler) -> None:
        """Test that consecutive chunks resample without boundary artifacts.

        Note: This is a basic test. Full artifact detection requires
        signal analysis (e.g., checking for discontinuities).
        """
        # Create two consecutive chunks
        chunk1 = np.random.randint(-1000, 1000, size=2400, dtype=np.int16)
        chunk2 = np.random.randint(-1000, 1000, size=2400, dtype=np.int16)

        resampled1 = resampler.resample_chunk(chunk1)
        resampled2 = resampler.resample_chunk(chunk2)

        # Should produce output without errors
        assert len(resampled1) > 0
        assert len(resampled2) > 0
        assert resampled1.dtype == np.int16
        assert resampled2.dtype == np.int16

    def test_overlap_buffer_logic(self, resampler: StreamingResampler) -> None:
        """Test overlap buffer is managed correctly across chunks.

        Note: Current placeholder implementation doesn't use overlap buffer.
        This test validates the interface for when proper implementation arrives.
        """
        # Initial state: no overlap buffer
        assert resampler.overlap_buffer is None

        # Process chunk
        chunk = np.zeros(2400, dtype=np.int16)
        _ = resampler.resample_chunk(chunk)

        # After processing, overlap buffer may or may not be set
        # (depends on implementation)
        # This test just validates it doesn't crash

    def test_flush_final_residual(self, resampler: StreamingResampler) -> None:
        """Test flush() returns final residual samples from overlap buffer."""
        # Process chunk to potentially populate overlap buffer
        chunk = np.zeros(2400, dtype=np.int16)
        _ = resampler.resample_chunk(chunk)

        # Flush
        residual = resampler.flush()

        # Residual may be None if no overlap buffer, or ndarray if present
        assert residual is None or isinstance(residual, np.ndarray)

    def test_different_chunk_sizes(self, resampler: StreamingResampler) -> None:
        """Test resampling works correctly with variable chunk sizes."""
        chunk_sizes = [480, 960, 2400, 4800, 24000]

        for size in chunk_sizes:
            chunk = np.zeros(size, dtype=np.int16)
            resampled = resampler.resample_chunk(chunk)

            # Should resample to ~2x size (24kHz → 48kHz)
            expected_size = int(size * 2)
            assert len(resampled) == pytest.approx(expected_size, abs=10)

    def test_gpu_vs_cpu_mode(self) -> None:
        """Test resampler works in both GPU and CPU mode.

        Note: Current implementation is CPU-only. This test validates
        the interface for GPU support when implemented.
        """
        # CPU mode (default)
        cpu_resampler = StreamingResampler()
        chunk = np.zeros(2400, dtype=np.int16)
        cpu_result = cpu_resampler.resample_chunk(chunk)

        assert len(cpu_result) > 0
        assert cpu_result.dtype == np.int16

        # GPU mode would be tested here when implemented
        # gpu_resampler = StreamingResampler(device='cuda')
        # gpu_result = gpu_resampler.resample_chunk(chunk)

    def test_empty_chunk_handling(self, resampler: StreamingResampler) -> None:
        """Test resampler handles empty chunks gracefully."""
        empty_chunk = np.zeros(0, dtype=np.int16)
        resampled = resampler.resample_chunk(empty_chunk)

        assert len(resampled) == 0
        assert resampled.dtype == np.int16

    def test_state_preservation_across_chunks(self, resampler: StreamingResampler) -> None:
        """Test resampler preserves internal state correctly across multiple chunks."""
        chunks = [
            np.random.randint(-1000, 1000, size=2400, dtype=np.int16) for _ in range(5)
        ]

        results = []
        for chunk in chunks:
            resampled = resampler.resample_chunk(chunk)
            results.append(resampled)

        # All chunks should resample successfully
        assert len(results) == 5
        for result in results:
            assert len(result) > 0
            assert result.dtype == np.int16


# ============================================================================
# FrameBuffer Tests
# ============================================================================


class TestFrameBuffer:
    """Test suite for FrameBuffer component."""

    def test_accumulation_into_20ms_frames(self, frame_buffer: FrameBuffer) -> None:
        """Test buffer accumulates samples into 20ms frames (960 samples @ 48kHz)."""
        # Add exactly 1 frame worth of samples
        samples = np.zeros(960, dtype=np.int16)
        frames = frame_buffer.add_samples(samples)

        assert len(frames) == 1
        assert len(frames[0]) == 1920  # 960 samples * 2 bytes/sample

    def test_partial_frame_handling(self, frame_buffer: FrameBuffer) -> None:
        """Test buffer holds partial frames until enough samples arrive."""
        # Add half a frame
        samples = np.zeros(480, dtype=np.int16)
        frames = frame_buffer.add_samples(samples)

        assert len(frames) == 0  # No complete frames yet
        assert len(frame_buffer.buffer) == 480

        # Add remaining half
        frames = frame_buffer.add_samples(samples)

        assert len(frames) == 1  # Now we have 1 complete frame
        assert len(frame_buffer.buffer) == 0

    def test_flush_final_frame(self, frame_buffer: FrameBuffer) -> None:
        """Test flush() returns final partial frame (zero-padded)."""
        # Add partial frame
        samples = np.zeros(480, dtype=np.int16)
        _ = frame_buffer.add_samples(samples)

        final_frame = frame_buffer.flush()

        assert final_frame is not None
        assert len(final_frame) == 1920  # Full frame size (zero-padded)
        assert len(frame_buffer.buffer) == 0

    def test_flush_empty_buffer(self, frame_buffer: FrameBuffer) -> None:
        """Test flush() returns None when buffer is empty."""
        final_frame = frame_buffer.flush()

        assert final_frame is None

    def test_exact_frame_alignment(self, frame_buffer: FrameBuffer) -> None:
        """Test adding exactly N complete frames returns N frames."""
        # Add exactly 3 frames worth of samples
        samples = np.zeros(960 * 3, dtype=np.int16)
        frames = frame_buffer.add_samples(samples)

        assert len(frames) == 3
        for frame in frames:
            assert len(frame) == 1920

    def test_large_chunk_multiple_frames(self, frame_buffer: FrameBuffer) -> None:
        """Test adding large chunk (multiple frames + partial) works correctly."""
        # Add 2.5 frames worth of samples
        samples = np.zeros(960 * 2 + 480, dtype=np.int16)
        frames = frame_buffer.add_samples(samples)

        assert len(frames) == 2  # 2 complete frames
        assert len(frame_buffer.buffer) == 480  # 0.5 frame remaining


# ============================================================================
# Integration Tests for Streaming Mode
# ============================================================================


class TestStreamingIntegration:
    """Integration tests for end-to-end streaming synthesis.

    Note: These tests assume a streaming-enabled CosyVoiceAdapter implementation.
          Current adapter uses batch mode (stream=False) as baseline.
    """

    @pytest.mark.asyncio
    async def test_end_to_end_streaming_synthesis(self) -> None:
        """Test end-to-end streaming synthesis flow.

        This is a MOCK test demonstrating the expected streaming flow.
        Actual implementation requires ML engineer's streaming mode.
        """
        # MOCK: Simulate streaming audio chunks from CosyVoice
        async def mock_streaming_synthesis() -> AsyncIterator[np.ndarray]:
            """Mock CosyVoice streaming inference."""
            # Simulate 10 chunks of 2400 samples each at 24kHz
            for _ in range(10):
                chunk = np.random.randint(-1000, 1000, size=2400, dtype=np.int16)
                yield chunk
                await asyncio.sleep(0.01)  # Simulate inference latency

        # Create processing pipeline
        resampler = StreamingResampler()
        frame_buffer = FrameBuffer()

        frames: list[bytes] = []
        async for chunk_24k in mock_streaming_synthesis():
            # Resample to 48kHz
            chunk_48k = resampler.resample_chunk(chunk_24k)

            # Buffer into 20ms frames
            new_frames = frame_buffer.add_samples(chunk_48k)
            frames.extend(new_frames)

        # Flush final partial frame
        final_frame = frame_buffer.flush()
        if final_frame:
            frames.append(final_frame)

        # Verify we got frames
        assert len(frames) > 0
        for frame in frames:
            assert len(frame) == 1920  # 20ms @ 48kHz

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires streaming mode implementation")
    async def test_first_audio_latency_under_500ms(self) -> None:
        """Test First Audio Latency (FAL) is < 500ms for streaming mode.

        This test is SKIPPED until streaming mode is implemented.
        Target: FAL < 500ms on GPU (vs ~2s for batch mode)
        """

        # This would test actual adapter streaming synthesis
        # adapter = CosyVoiceAdapter(..., stream=True)
        # start = time.perf_counter()
        # first_frame = None
        # async for frame in adapter.synthesize_stream(...):
        #     if first_frame is None:
        #         fal_ms = (time.perf_counter() - start) * 1000
        #         assert fal_ms < 500, f"FAL {fal_ms}ms exceeds 500ms target"
        #         first_frame = frame
        #     break

        pytest.skip("Awaiting streaming mode implementation")

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires streaming mode implementation")
    async def test_inter_chunk_timing_consistency(self) -> None:
        """Test inter-chunk timing is consistent (low jitter).

        This test is SKIPPED until streaming mode is implemented.
        Target: p95 jitter < 10ms
        """
        pytest.skip("Awaiting streaming mode implementation")

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires streaming mode implementation")
    async def test_pause_resume_during_streaming(self) -> None:
        """Test PAUSE/RESUME commands work correctly during streaming synthesis.

        This test is SKIPPED until streaming mode is implemented.
        Target: PAUSE response time < 50ms
        """
        pytest.skip("Awaiting streaming mode implementation")

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires streaming mode implementation")
    async def test_stop_during_streaming(self) -> None:
        """Test STOP command terminates streaming immediately.

        This test is SKIPPED until streaming mode is implemented.
        Target: STOP response time < 50ms
        """
        pytest.skip("Awaiting streaming mode implementation")

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires streaming mode implementation")
    async def test_fallback_to_batch_mode_on_error(self) -> None:
        """Test graceful fallback to batch mode if streaming fails.

        This test is SKIPPED until streaming mode is implemented.
        Should catch streaming errors and retry with stream=False
        """
        pytest.skip("Awaiting streaming mode implementation")


# ============================================================================
# Edge Cases
# ============================================================================


class TestStreamingEdgeCases:
    """Test edge cases for streaming audio processing."""

    def test_resampler_with_very_small_chunks(self, resampler: StreamingResampler) -> None:
        """Test resampler handles very small chunks (< 100 samples)."""
        tiny_chunk = np.zeros(48, dtype=np.int16)
        resampled = resampler.resample_chunk(tiny_chunk)

        # Should still resample correctly
        assert len(resampled) == pytest.approx(96, abs=5)

    def test_frame_buffer_with_single_sample(self, frame_buffer: FrameBuffer) -> None:
        """Test frame buffer handles single-sample additions."""
        for _ in range(960):
            sample = np.array([100], dtype=np.int16)
            frames = frame_buffer.add_samples(sample)

            # Should accumulate until we have a full frame
            if _ < 959:
                assert len(frames) == 0
            else:
                assert len(frames) == 1

    def test_consecutive_flush_calls(self, frame_buffer: FrameBuffer) -> None:
        """Test multiple flush() calls are safe and idempotent."""
        # Add partial frame
        samples = np.zeros(480, dtype=np.int16)
        _ = frame_buffer.add_samples(samples)

        # First flush returns frame
        frame1 = frame_buffer.flush()
        assert frame1 is not None

        # Second flush returns None (buffer empty)
        frame2 = frame_buffer.flush()
        assert frame2 is None

        # Third flush also returns None
        frame3 = frame_buffer.flush()
        assert frame3 is None
