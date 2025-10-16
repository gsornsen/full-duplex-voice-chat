"""Unit tests for AudioBuffer class.

Tests audio buffer functionality for accumulating speech utterances
from VAD events before sending to ASR.
"""

import asyncio

import pytest

from src.orchestrator.audio.buffer import (
    AudioBuffer,
    BufferOverflowError,
    RingAudioBuffer,
)


class TestAudioBufferInit:
    """Test AudioBuffer initialization."""

    def test_default_initialization(self) -> None:
        """Test buffer with default parameters."""
        buffer = AudioBuffer()
        assert buffer.sample_rate == 48000
        assert buffer.channels == 1
        assert buffer.max_duration_s == 30.0
        assert buffer.frame_count == 0
        assert buffer.byte_count == 0
        assert buffer.duration_ms() == 0

    def test_custom_initialization(self) -> None:
        """Test buffer with custom parameters."""
        buffer = AudioBuffer(sample_rate=16000, channels=2, max_duration_s=10.0)
        assert buffer.sample_rate == 16000
        assert buffer.channels == 2
        assert buffer.max_duration_s == 10.0

    def test_invalid_sample_rate(self) -> None:
        """Test that negative sample rate raises ValueError."""
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            AudioBuffer(sample_rate=-1)

    def test_zero_sample_rate(self) -> None:
        """Test that zero sample rate raises ValueError."""
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            AudioBuffer(sample_rate=0)

    def test_invalid_channels(self) -> None:
        """Test that invalid channel count raises ValueError."""
        with pytest.raises(ValueError, match="Channels must be 1 .* or 2"):
            AudioBuffer(channels=3)

    def test_zero_max_duration(self) -> None:
        """Test that zero max duration raises ValueError."""
        with pytest.raises(ValueError, match="Max duration must be positive"):
            AudioBuffer(max_duration_s=0.0)

    def test_negative_max_duration(self) -> None:
        """Test that negative max duration raises ValueError."""
        with pytest.raises(ValueError, match="Max duration must be positive"):
            AudioBuffer(max_duration_s=-1.0)


class TestAudioBufferAppend:
    """Test AudioBuffer append functionality."""

    @pytest.mark.asyncio
    async def test_append_single_frame(self) -> None:
        """Test appending a single audio frame."""
        buffer = AudioBuffer(sample_rate=48000)
        # 20ms frame at 48kHz = 960 samples = 1920 bytes (16-bit mono)
        frame = b"\x00\x01" * 960
        await buffer.append(frame)

        assert buffer.frame_count == 1
        assert buffer.byte_count == 1920
        assert buffer.duration_ms() == 20

    @pytest.mark.asyncio
    async def test_append_multiple_frames(self) -> None:
        """Test appending multiple frames."""
        buffer = AudioBuffer(sample_rate=48000)
        frame = b"\x00\x01" * 960  # 20ms frame

        await buffer.append(frame)
        await buffer.append(frame)
        await buffer.append(frame)

        assert buffer.frame_count == 3
        assert buffer.byte_count == 1920 * 3
        assert buffer.duration_ms() == 60

    @pytest.mark.asyncio
    async def test_append_empty_frame(self) -> None:
        """Test that appending empty frame raises ValueError."""
        buffer = AudioBuffer()
        with pytest.raises(ValueError, match="Cannot append empty frame"):
            await buffer.append(b"")

    @pytest.mark.asyncio
    async def test_append_overflow(self) -> None:
        """Test that buffer overflow is detected."""
        # Create buffer with 1 second max duration
        buffer = AudioBuffer(sample_rate=48000, max_duration_s=1.0)

        # 20ms frame
        frame = b"\x00\x01" * 960

        # Append 50 frames (1 second)
        for _ in range(50):
            await buffer.append(frame)

        # 51st frame should overflow
        with pytest.raises(BufferOverflowError, match="Buffer overflow"):
            await buffer.append(frame)

    @pytest.mark.asyncio
    async def test_append_different_sample_rates(self) -> None:
        """Test appending with different sample rates."""
        # 16kHz buffer
        buffer = AudioBuffer(sample_rate=16000)
        # 20ms frame at 16kHz = 320 samples = 640 bytes
        frame = b"\x00\x01" * 320

        await buffer.append(frame)
        assert buffer.duration_ms() == 20

    @pytest.mark.asyncio
    async def test_append_stereo(self) -> None:
        """Test appending stereo audio."""
        buffer = AudioBuffer(sample_rate=48000, channels=2)
        # 20ms stereo frame at 48kHz = 960 samples * 2 channels = 3840 bytes
        frame = b"\x00\x01" * 1920

        await buffer.append(frame)
        assert buffer.duration_ms() == 20


class TestAudioBufferGetAudio:
    """Test AudioBuffer get_audio functionality."""

    @pytest.mark.asyncio
    async def test_get_empty_buffer(self) -> None:
        """Test getting audio from empty buffer."""
        buffer = AudioBuffer()
        audio = await buffer.get_audio()
        assert audio == b""

    @pytest.mark.asyncio
    async def test_get_single_frame(self) -> None:
        """Test getting single frame."""
        buffer = AudioBuffer()
        frame = b"\x00\x01\x02\x03"
        await buffer.append(frame)

        audio = await buffer.get_audio()
        assert audio == frame

    @pytest.mark.asyncio
    async def test_get_multiple_frames(self) -> None:
        """Test getting concatenated frames."""
        buffer = AudioBuffer()
        frame1 = b"\x00\x01"
        frame2 = b"\x02\x03"
        frame3 = b"\x04\x05"

        await buffer.append(frame1)
        await buffer.append(frame2)
        await buffer.append(frame3)

        audio = await buffer.get_audio()
        assert audio == b"\x00\x01\x02\x03\x04\x05"

    @pytest.mark.asyncio
    async def test_get_audio_does_not_clear(self) -> None:
        """Test that get_audio doesn't clear the buffer."""
        buffer = AudioBuffer()
        frame = b"\x00\x01\x02\x03"
        await buffer.append(frame)

        audio1 = await buffer.get_audio()
        audio2 = await buffer.get_audio()

        assert audio1 == audio2
        assert buffer.frame_count == 1


class TestAudioBufferClear:
    """Test AudioBuffer clear functionality."""

    @pytest.mark.asyncio
    async def test_clear_empty_buffer(self) -> None:
        """Test clearing empty buffer."""
        buffer = AudioBuffer()
        await buffer.clear()
        assert await buffer.is_empty()

    @pytest.mark.asyncio
    async def test_clear_with_data(self) -> None:
        """Test clearing buffer with data."""
        buffer = AudioBuffer()
        await buffer.append(b"\x00\x01\x02\x03")
        assert buffer.frame_count == 1

        await buffer.clear()
        assert buffer.frame_count == 0
        assert buffer.byte_count == 0
        assert buffer.duration_ms() == 0
        assert await buffer.is_empty()

    @pytest.mark.asyncio
    async def test_clear_and_reuse(self) -> None:
        """Test that buffer can be reused after clear."""
        buffer = AudioBuffer()
        frame = b"\x00\x01\x02\x03"

        await buffer.append(frame)
        await buffer.clear()
        await buffer.append(frame)

        assert buffer.frame_count == 1
        audio = await buffer.get_audio()
        assert audio == frame


class TestAudioBufferIsEmpty:
    """Test AudioBuffer is_empty functionality."""

    @pytest.mark.asyncio
    async def test_empty_on_init(self) -> None:
        """Test buffer is empty on initialization."""
        buffer = AudioBuffer()
        assert await buffer.is_empty()

    @pytest.mark.asyncio
    async def test_not_empty_after_append(self) -> None:
        """Test buffer is not empty after append."""
        buffer = AudioBuffer()
        await buffer.append(b"\x00\x01")
        assert not await buffer.is_empty()

    @pytest.mark.asyncio
    async def test_empty_after_clear(self) -> None:
        """Test buffer is empty after clear."""
        buffer = AudioBuffer()
        await buffer.append(b"\x00\x01")
        await buffer.clear()
        assert await buffer.is_empty()


class TestAudioBufferDuration:
    """Test AudioBuffer duration calculation."""

    def test_duration_empty_buffer(self) -> None:
        """Test duration of empty buffer is 0."""
        buffer = AudioBuffer()
        assert buffer.duration_ms() == 0

    @pytest.mark.asyncio
    async def test_duration_single_frame(self) -> None:
        """Test duration calculation for single frame."""
        buffer = AudioBuffer(sample_rate=48000)
        # 20ms frame: 960 samples * 2 bytes
        frame = b"\x00\x01" * 960
        await buffer.append(frame)
        assert buffer.duration_ms() == 20

    @pytest.mark.asyncio
    async def test_duration_multiple_frames(self) -> None:
        """Test duration accumulation."""
        buffer = AudioBuffer(sample_rate=48000)
        frame = b"\x00\x01" * 960  # 20ms

        await buffer.append(frame)
        await buffer.append(frame)
        await buffer.append(frame)

        assert buffer.duration_ms() == 60

    @pytest.mark.asyncio
    async def test_duration_different_sample_rate(self) -> None:
        """Test duration with different sample rate."""
        buffer = AudioBuffer(sample_rate=16000)
        # 20ms frame at 16kHz: 320 samples * 2 bytes
        frame = b"\x00\x01" * 320
        await buffer.append(frame)
        assert buffer.duration_ms() == 20

    @pytest.mark.asyncio
    async def test_duration_stereo(self) -> None:
        """Test duration with stereo audio."""
        buffer = AudioBuffer(sample_rate=48000, channels=2)
        # 20ms stereo: 960 samples * 2 channels * 2 bytes
        frame = b"\x00\x01" * (960 * 2)
        await buffer.append(frame)
        assert buffer.duration_ms() == 20


class TestAudioBufferProperties:
    """Test AudioBuffer property accessors."""

    @pytest.mark.asyncio
    async def test_frame_count(self) -> None:
        """Test frame_count property."""
        buffer = AudioBuffer()
        assert buffer.frame_count == 0

        await buffer.append(b"\x00\x01")
        assert buffer.frame_count == 1

        await buffer.append(b"\x02\x03")
        assert buffer.frame_count == 2

    @pytest.mark.asyncio
    async def test_byte_count(self) -> None:
        """Test byte_count property."""
        buffer = AudioBuffer()
        assert buffer.byte_count == 0

        await buffer.append(b"\x00\x01\x02\x03")
        assert buffer.byte_count == 4

        await buffer.append(b"\x04\x05")
        assert buffer.byte_count == 6

    @pytest.mark.asyncio
    async def test_utilization_empty(self) -> None:
        """Test utilization of empty buffer."""
        buffer = AudioBuffer(sample_rate=48000, max_duration_s=1.0)
        assert buffer.utilization == 0.0

    @pytest.mark.asyncio
    async def test_utilization_partial(self) -> None:
        """Test utilization of partially filled buffer."""
        buffer = AudioBuffer(sample_rate=48000, max_duration_s=1.0)
        # 20ms frame
        frame = b"\x00\x01" * 960

        # Add 25 frames (0.5 seconds = 50% utilization)
        for _ in range(25):
            await buffer.append(frame)

        assert 0.49 < buffer.utilization < 0.51  # Allow small floating point error

    @pytest.mark.asyncio
    async def test_utilization_full(self) -> None:
        """Test utilization of full buffer."""
        buffer = AudioBuffer(sample_rate=48000, max_duration_s=1.0)
        frame = b"\x00\x01" * 960

        # Fill to capacity (50 frames = 1 second)
        for _ in range(50):
            await buffer.append(frame)

        assert buffer.utilization > 0.99  # Should be very close to 1.0


class TestAudioBufferRepr:
    """Test AudioBuffer string representation."""

    def test_repr_empty(self) -> None:
        """Test repr of empty buffer."""
        buffer = AudioBuffer(sample_rate=16000, channels=1, max_duration_s=5.0)
        repr_str = repr(buffer)

        assert "AudioBuffer" in repr_str
        assert "sample_rate=16000" in repr_str
        assert "channels=1" in repr_str
        assert "frames=0" in repr_str

    @pytest.mark.asyncio
    async def test_repr_with_data(self) -> None:
        """Test repr with data."""
        buffer = AudioBuffer()
        await buffer.append(b"\x00\x01" * 960)

        repr_str = repr(buffer)
        assert "frames=1" in repr_str
        assert "duration_ms=20" in repr_str


class TestAudioBufferThreadSafety:
    """Test AudioBuffer thread-safety with asyncio."""

    @pytest.mark.asyncio
    async def test_concurrent_appends(self) -> None:
        """Test concurrent append operations."""
        buffer = AudioBuffer(sample_rate=48000, max_duration_s=10.0)
        frame = b"\x00\x01" * 960

        # Append 100 frames concurrently
        tasks = [buffer.append(frame) for _ in range(100)]
        await asyncio.gather(*tasks)

        assert buffer.frame_count == 100

    @pytest.mark.asyncio
    async def test_concurrent_append_and_get(self) -> None:
        """Test concurrent append and get operations."""
        buffer = AudioBuffer()
        frame = b"\x00\x01" * 100

        async def append_task() -> None:
            for _ in range(10):
                await buffer.append(frame)
                await asyncio.sleep(0.001)

        async def get_task() -> None:
            for _ in range(10):
                await buffer.get_audio()
                await asyncio.sleep(0.001)

        # Run append and get concurrently
        await asyncio.gather(append_task(), get_task())

        # Should complete without errors
        assert buffer.frame_count == 10


class TestRingAudioBuffer:
    """Test RingAudioBuffer (circular buffer variant)."""

    @pytest.mark.asyncio
    async def test_ring_buffer_basic(self) -> None:
        """Test basic ring buffer operation."""
        ring = RingAudioBuffer(sample_rate=48000, max_duration_s=1.0)
        frame = b"\x00\x01" * 960

        await ring.append(frame)
        assert ring.frame_count == 1

    @pytest.mark.asyncio
    async def test_ring_buffer_no_overflow_error(self) -> None:
        """Test that ring buffer doesn't raise overflow error."""
        ring = RingAudioBuffer(sample_rate=48000, max_duration_s=1.0)
        frame = b"\x00\x01" * 960

        # Try to add more than capacity (51 frames > 50 frames max)
        for _ in range(60):
            await ring.append(frame)  # Should not raise

        # Buffer should contain at most 50 frames
        assert ring.frame_count <= 50

    @pytest.mark.asyncio
    async def test_ring_buffer_discards_oldest(self) -> None:
        """Test that ring buffer discards oldest frames."""
        ring = RingAudioBuffer(sample_rate=48000, max_duration_s=1.0)

        # Create distinguishable frames
        frame1 = b"\x00\x00" * 960
        frame2 = b"\xFF\xFF" * 960

        # Fill with frame1
        for _ in range(50):
            await ring.append(frame1)

        # Add one frame2 (should discard oldest frame1)
        await ring.append(frame2)

        # Get all audio
        audio = await ring.get_audio()

        # Should contain mostly frame1 but end with frame2
        assert audio.endswith(frame2)

    @pytest.mark.asyncio
    async def test_ring_buffer_empty_frame_error(self) -> None:
        """Test that ring buffer still rejects empty frames."""
        ring = RingAudioBuffer()
        with pytest.raises(ValueError, match="Cannot append empty frame"):
            await ring.append(b"")
