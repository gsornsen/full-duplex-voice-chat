"""Unit tests for streaming audio processing utilities."""

import pytest
import torch
from tts.audio.streaming import FrameBuffer, StreamingResampler


class TestStreamingResampler:
    """Test suite for StreamingResampler class."""

    def test_initialization(self) -> None:
        """Test resampler initialization with valid parameters."""
        resampler = StreamingResampler(
            input_rate=24000,
            output_rate=48000,
            overlap_samples=441,
            device="cpu",
        )
        assert resampler.input_rate == 24000
        assert resampler.output_rate == 48000
        assert resampler.overlap_samples == 441
        assert resampler.device == "cpu"
        assert resampler.ratio == 2.0
        assert resampler.overlap_buffer is None

    def test_initialization_invalid_rates(self) -> None:
        """Test initialization fails with invalid sample rates."""
        with pytest.raises(ValueError, match="input_rate must be positive"):
            StreamingResampler(input_rate=-1, output_rate=48000)

        with pytest.raises(ValueError, match="output_rate must be positive"):
            StreamingResampler(input_rate=24000, output_rate=0)

    def test_initialization_invalid_overlap(self) -> None:
        """Test initialization fails with negative overlap."""
        with pytest.raises(ValueError, match="overlap_samples must be non-negative"):
            StreamingResampler(
                input_rate=24000,
                output_rate=48000,
                overlap_samples=-1,
            )

    def test_process_simple_upsampling(self) -> None:
        """Test basic upsampling (24kHz -> 48kHz)."""
        resampler = StreamingResampler(
            input_rate=24000,
            output_rate=48000,
            overlap_samples=0,  # Disable overlap for simple test
            device="cpu",
        )

        # Input: 1 second at 24kHz
        input_chunk = torch.randn(24000)
        output = resampler.process(input_chunk)

        # Output should be ~2x longer (48kHz)
        expected_length = int(24000 * 2.0)
        assert output.shape[0] == pytest.approx(expected_length, abs=10)

    def test_process_simple_downsampling(self) -> None:
        """Test basic downsampling (48kHz -> 24kHz)."""
        resampler = StreamingResampler(
            input_rate=48000,
            output_rate=24000,
            overlap_samples=0,
            device="cpu",
        )

        # Input: 1 second at 48kHz
        input_chunk = torch.randn(48000)
        output = resampler.process(input_chunk)

        # Output should be ~0.5x shorter (24kHz)
        expected_length = int(48000 * 0.5)
        assert output.shape[0] == pytest.approx(expected_length, abs=10)

    def test_process_with_overlap_buffer(self) -> None:
        """Test overlap buffer maintains state across chunks."""
        resampler = StreamingResampler(
            input_rate=24000,
            output_rate=48000,
            overlap_samples=100,
            device="cpu",
        )

        # Process first chunk
        chunk1 = torch.randn(1000)
        _ = resampler.process(chunk1)

        # Overlap buffer should be populated
        assert resampler.overlap_buffer is not None
        assert resampler.overlap_buffer.shape[0] == 100

        # Process second chunk
        chunk2 = torch.randn(1000)
        output2 = resampler.process(chunk2)

        # Output should include contribution from overlap
        assert output2 is not None

    def test_process_2d_tensor(self) -> None:
        """Test processing 2D tensor (multi-channel audio)."""
        resampler = StreamingResampler(
            input_rate=24000,
            output_rate=48000,
            device="cpu",
        )

        # 2-channel audio: (channels, samples)
        input_chunk = torch.randn(2, 1000)
        output = resampler.process(input_chunk)

        # Should preserve channel dimension
        assert output.dim() == 2
        assert output.shape[0] == 2
        assert output.shape[1] == pytest.approx(2000, abs=10)

    def test_process_empty_chunk_raises(self) -> None:
        """Test processing empty chunk raises ValueError."""
        resampler = StreamingResampler(input_rate=24000, output_rate=48000)

        with pytest.raises(ValueError, match="Cannot process empty chunk"):
            resampler.process(torch.tensor([]))

    def test_process_invalid_dimensions_raises(self) -> None:
        """Test processing 3D tensor raises ValueError."""
        resampler = StreamingResampler(input_rate=24000, output_rate=48000)

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            resampler.process(torch.randn(2, 2, 100))

    def test_process_device_transfer(self) -> None:
        """Test automatic device transfer for input chunks."""
        resampler = StreamingResampler(
            input_rate=24000,
            output_rate=48000,
            device="cpu",
        )

        # Create chunk on CPU
        chunk = torch.randn(1000, device="cpu")
        output = resampler.process(chunk)

        # Output should be on CPU
        assert output.device.type == "cpu"

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_process_gpu_acceleration(self) -> None:
        """Test GPU-accelerated resampling.

        This test requires an actual CUDA-capable GPU to validate:
        - GPU device allocation
        - CUDA tensor operations
        - GPU memory management
        - Performance characteristics of GPU-based resampling
        """
        resampler = StreamingResampler(
            input_rate=24000,
            output_rate=48000,
            device="cuda",
        )

        # Create chunk on GPU
        chunk = torch.randn(10000, device="cuda")
        output = resampler.process(chunk)

        # Output should be on GPU
        assert output.device.type == "cuda"
        assert output.shape[0] == pytest.approx(20000, abs=10)

    def test_flush_with_residual(self) -> None:
        """Test flush emits residual samples."""
        resampler = StreamingResampler(
            input_rate=24000,
            output_rate=48000,
            overlap_samples=100,
        )

        # Process chunk to populate overlap buffer
        chunk = torch.randn(1000)
        _ = resampler.process(chunk)

        # Flush should return resampled overlap
        final = resampler.flush()
        assert final is not None
        assert final.numel() > 0

        # Second flush should return None
        assert resampler.flush() is None

    def test_flush_without_residual(self) -> None:
        """Test flush returns None when no residual."""
        resampler = StreamingResampler(input_rate=24000, output_rate=48000)

        # No processing, no residual
        assert resampler.flush() is None

    def test_reset_clears_overlap_buffer(self) -> None:
        """Test reset clears internal state."""
        resampler = StreamingResampler(
            input_rate=24000,
            output_rate=48000,
            overlap_samples=100,
        )

        # Process chunk to populate overlap buffer
        chunk = torch.randn(1000)
        _ = resampler.process(chunk)
        assert resampler.overlap_buffer is not None

        # Reset should clear buffer
        resampler.reset()
        assert resampler.overlap_buffer is None

    def test_process_small_chunk_saves_entire_chunk(self) -> None:
        """Test processing chunk smaller than overlap saves entire chunk."""
        resampler = StreamingResampler(
            input_rate=24000,
            output_rate=48000,
            overlap_samples=1000,  # Larger than chunk
        )

        # Process small chunk
        chunk = torch.randn(100)
        _ = resampler.process(chunk)

        # Entire chunk should be saved as overlap
        assert resampler.overlap_buffer is not None
        assert resampler.overlap_buffer.shape[0] == 100

    def test_multiple_chunks_continuity(self) -> None:
        """Test processing multiple chunks maintains continuity."""
        resampler = StreamingResampler(
            input_rate=24000,
            output_rate=48000,
            overlap_samples=100,
        )

        # Process three chunks
        outputs = []
        for _ in range(3):
            chunk = torch.randn(1000)
            output = resampler.process(chunk)
            outputs.append(output)

        # All outputs should have reasonable length (~2x upsampling with overlap)
        for output in outputs:
            assert output.shape[0] > 1800  # ~2x upsampling
            assert output.shape[0] <= 2200  # Allow for overlap contribution


class TestFrameBuffer:
    """Test suite for FrameBuffer class."""

    def test_initialization(self) -> None:
        """Test frame buffer initialization with valid parameters."""
        buffer = FrameBuffer(
            frame_size=960,
            sample_rate=48000,
            device="cpu",
        )
        assert buffer.frame_size == 960
        assert buffer.sample_rate == 48000
        assert buffer.device == "cpu"
        assert buffer.buffer is None
        assert buffer.buffered_samples == 0

    def test_initialization_invalid_frame_size(self) -> None:
        """Test initialization fails with invalid frame size."""
        with pytest.raises(ValueError, match="frame_size must be positive"):
            FrameBuffer(frame_size=0, sample_rate=48000)

        with pytest.raises(ValueError, match="frame_size must be positive"):
            FrameBuffer(frame_size=-1, sample_rate=48000)

    def test_initialization_invalid_sample_rate(self) -> None:
        """Test initialization fails with invalid sample rate."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            FrameBuffer(frame_size=960, sample_rate=0)

    def test_push_single_frame_exact(self) -> None:
        """Test pushing exactly one frame worth of samples."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        samples = torch.randn(960)
        frames = buffer.push(samples)

        # Should emit exactly one frame
        assert len(frames) == 1
        assert frames[0].shape[0] == 960
        assert buffer.buffered_samples == 0

    def test_push_multiple_frames(self) -> None:
        """Test pushing multiple frames worth of samples."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        # Push 2.5 frames worth
        samples = torch.randn(2400)  # 2 * 960 + 480
        frames = buffer.push(samples)

        # Should emit 2 complete frames
        assert len(frames) == 2
        assert frames[0].shape[0] == 960
        assert frames[1].shape[0] == 960

        # Should have 480 samples buffered
        assert buffer.buffered_samples == 480

    def test_push_partial_frame(self) -> None:
        """Test pushing less than one frame accumulates samples."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        # Push partial frame
        samples = torch.randn(500)
        frames = buffer.push(samples)

        # Should emit no frames
        assert len(frames) == 0
        assert buffer.buffered_samples == 500

    def test_push_accumulate_then_emit(self) -> None:
        """Test accumulating samples across pushes then emitting."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        # Push partial frames
        frames1 = buffer.push(torch.randn(400))
        frames2 = buffer.push(torch.randn(400))

        # No frames yet
        assert len(frames1) == 0
        assert len(frames2) == 0
        assert buffer.buffered_samples == 800

        # Push enough to complete frame
        frames3 = buffer.push(torch.randn(200))

        # Should emit one frame
        assert len(frames3) == 1
        assert frames3[0].shape[0] == 960
        assert buffer.buffered_samples == 40  # 400 + 400 + 200 - 960

    def test_push_2d_tensor(self) -> None:
        """Test pushing 2D tensor (multi-channel audio)."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        # 2-channel audio
        samples = torch.randn(2, 1920)  # 2 frames
        frames = buffer.push(samples)

        # Should emit 2 frames
        assert len(frames) == 2
        assert frames[0].shape == (2, 960)
        assert frames[1].shape == (2, 960)

    def test_push_empty_samples_raises(self) -> None:
        """Test pushing empty samples raises ValueError."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        with pytest.raises(ValueError, match="Cannot push empty samples"):
            buffer.push(torch.tensor([]))

    def test_push_invalid_dimensions_raises(self) -> None:
        """Test pushing 3D tensor raises ValueError."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        with pytest.raises(ValueError, match="must be 1D or 2D"):
            buffer.push(torch.randn(2, 2, 100))

    def test_push_device_transfer(self) -> None:
        """Test automatic device transfer for input samples."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000, device="cpu")

        samples = torch.randn(1000, device="cpu")
        _ = buffer.push(samples)

        # Buffered samples should be on CPU
        if buffer.buffer is not None:
            assert buffer.buffer.device.type == "cpu"

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_push_gpu_processing(self) -> None:
        """Test GPU-based frame buffering.

        This test requires an actual CUDA-capable GPU to validate:
        - GPU device allocation
        - CUDA tensor operations
        - GPU memory management
        - Frame buffering on GPU device
        """
        buffer = FrameBuffer(frame_size=960, sample_rate=48000, device="cuda")

        samples = torch.randn(1920, device="cuda")
        frames = buffer.push(samples)

        # Frames should be on GPU
        assert len(frames) == 2
        assert frames[0].device.type == "cuda"

    def test_flush_with_partial_frame(self) -> None:
        """Test flush emits partial frame."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        # Push partial frame
        buffer.push(torch.randn(500))

        # Flush should return partial frame
        final = buffer.flush()
        assert final is not None
        assert final.shape[0] == 500

        # Buffer should be empty
        assert buffer.buffered_samples == 0

    def test_flush_empty_buffer(self) -> None:
        """Test flush returns None when buffer is empty."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        # No samples pushed
        assert buffer.flush() is None

    def test_flush_after_complete_frames(self) -> None:
        """Test flush after emitting complete frames."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        # Push exactly 2 frames
        frames = buffer.push(torch.randn(1920))
        assert len(frames) == 2

        # Flush should return None (no partial frame)
        assert buffer.flush() is None

    def test_reset_clears_buffer(self) -> None:
        """Test reset clears internal buffer."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        # Push partial frame
        buffer.push(torch.randn(500))
        assert buffer.buffered_samples == 500

        # Reset should clear buffer
        buffer.reset()
        assert buffer.buffered_samples == 0
        assert buffer.buffer is None

    def test_buffered_duration_ms(self) -> None:
        """Test buffered duration calculation."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        # Push 480 samples (10 ms at 48kHz)
        buffer.push(torch.randn(480))

        assert buffer.buffered_duration_ms == pytest.approx(10.0, abs=0.1)

    def test_total_samples_emitted(self) -> None:
        """Test total samples emitted counter."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        # Push 2.5 frames
        _ = buffer.push(torch.randn(2400))

        # Should emit 2 frames = 1920 samples
        assert buffer.total_samples_emitted == 1920

        # Flush partial frame
        final = buffer.flush()
        assert final is not None

        # Total should include partial frame
        assert buffer.total_samples_emitted == 2400

    def test_total_samples_emitted_resets(self) -> None:
        """Test total samples counter resets."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        buffer.push(torch.randn(1920))
        assert buffer.total_samples_emitted == 1920

        buffer.reset()
        assert buffer.total_samples_emitted == 0

    def test_streaming_workflow(self) -> None:
        """Test realistic streaming workflow."""
        buffer = FrameBuffer(frame_size=960, sample_rate=48000)

        # Simulate streaming: push 5 chunks of varying sizes
        all_frames = []
        chunk_sizes = [400, 800, 1200, 600, 500]

        for size in chunk_sizes:
            frames = buffer.push(torch.randn(size))
            all_frames.extend(frames)

        # Get final partial frame
        final = buffer.flush()
        if final is not None:
            all_frames.append(final)

        # Total input: 400 + 800 + 1200 + 600 + 500 = 3500 samples
        # Expected: 3 complete frames (2880) + 1 partial (620)
        assert len(all_frames) == 4
        assert all_frames[0].shape[0] == 960
        assert all_frames[1].shape[0] == 960
        assert all_frames[2].shape[0] == 960
        assert all_frames[3].shape[0] == 620

    def test_zero_overlap_samples_allowed(self) -> None:
        """Test StreamingResampler allows zero overlap samples."""
        resampler = StreamingResampler(
            input_rate=24000,
            output_rate=48000,
            overlap_samples=0,
        )
        assert resampler.overlap_samples == 0

        # Should work without overlap buffer
        chunk = torch.randn(1000)
        output = resampler.process(chunk)
        assert output is not None
