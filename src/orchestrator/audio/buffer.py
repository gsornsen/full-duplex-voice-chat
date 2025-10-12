"""Audio buffer for accumulating speech utterances from VAD events.

This module provides an audio buffer that accumulates audio frames detected
by VAD (Voice Activity Detection) until speech ends, then provides the
complete utterance for ASR transcription.

Key features:
- Accumulate audio frames from VAD speech detection
- Enforce maximum buffer size to prevent memory abuse
- Thread-safe with asyncio locks
- Track buffer duration and sample rate
- Support for clearing and resetting buffer state

Design:
    VAD detects speech → frames accumulated in buffer → VAD detects silence
    → buffer contents sent to ASR → buffer cleared → repeat
"""

import asyncio
import logging
from typing import Final

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SAMPLE_RATE: Final[int] = 48000  # Hz
DEFAULT_CHANNELS: Final[int] = 1  # Mono
DEFAULT_MAX_DURATION_S: Final[float] = 30.0  # 30 seconds max
BYTES_PER_SAMPLE: Final[int] = 2  # 16-bit PCM


class AudioBufferError(Exception):
    """Base exception for audio buffer errors."""

    pass


class BufferOverflowError(AudioBufferError):
    """Raised when buffer would exceed maximum duration."""

    pass


class AudioBuffer:
    """Thread-safe audio buffer for accumulating speech utterances.

    Accumulates audio frames until speech ends (VAD silence detection),
    then provides the complete audio data for ASR transcription.

    Thread-safety: All methods use asyncio.Lock for thread-safe access.

    Example:
        ```python
        # Create buffer for 48kHz mono audio
        buffer = AudioBuffer(sample_rate=48000, channels=1)

        # VAD speech detected - start accumulating frames
        async def on_vad_frame(frame: bytes):
            try:
                await buffer.append(frame)
            except BufferOverflowError:
                logger.warning("Buffer full, discarding frame")

        # VAD silence detected - get audio and transcribe
        async def on_vad_silence():
            audio_data = await buffer.get_audio()
            if not await buffer.is_empty():
                result = await asr.transcribe(audio_data, buffer.sample_rate)
                print(f"Transcription: {result.text}")
            await buffer.clear()
        ```
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        max_duration_s: float = DEFAULT_MAX_DURATION_S,
    ) -> None:
        """Initialize audio buffer.

        Args:
            sample_rate: Sample rate in Hz (e.g., 16000, 48000)
            channels: Number of audio channels (1=mono, 2=stereo)
            max_duration_s: Maximum buffer duration in seconds (prevents memory abuse)

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if channels not in (1, 2):
            raise ValueError(f"Channels must be 1 (mono) or 2 (stereo), got {channels}")
        if max_duration_s <= 0:
            raise ValueError(f"Max duration must be positive, got {max_duration_s}")

        self.sample_rate = sample_rate
        self.channels = channels
        self.max_duration_s = max_duration_s

        # Calculate maximum buffer size in bytes
        # bytes = samples * channels * bytes_per_sample
        # samples = sample_rate * duration_s
        max_samples = int(sample_rate * max_duration_s)
        self._max_bytes = max_samples * channels * BYTES_PER_SAMPLE

        # Buffer state
        self._frames: list[bytes] = []
        self._total_bytes = 0
        self._lock = asyncio.Lock()

        logger.debug(
            f"AudioBuffer initialized: sample_rate={sample_rate}Hz, "
            f"channels={channels}, max_duration={max_duration_s}s, "
            f"max_bytes={self._max_bytes}"
        )

    async def append(self, frame: bytes) -> None:
        """Append audio frame to buffer.

        Args:
            frame: Audio frame bytes (16-bit PCM)

        Raises:
            BufferOverflowError: If adding frame would exceed max_duration_s
            ValueError: If frame is empty

        Example:
            ```python
            # 20ms frame at 48kHz
            frame_size = int(48000 * 0.020) * 2  # 1920 bytes
            frame = b"\\x00\\x01" * (frame_size // 2)
            await buffer.append(frame)
            ```
        """
        if not frame:
            raise ValueError("Cannot append empty frame")

        async with self._lock:
            # Check if adding frame would exceed max buffer size
            if self._total_bytes + len(frame) > self._max_bytes:
                # Calculate current duration
                current_duration_s = self.duration_ms() / 1000.0
                raise BufferOverflowError(
                    f"Buffer overflow: would exceed {self.max_duration_s}s "
                    f"(current: {current_duration_s:.2f}s, "
                    f"bytes: {self._total_bytes}/{self._max_bytes})"
                )

            # Append frame
            self._frames.append(frame)
            self._total_bytes += len(frame)

            # Log every 1 second worth of audio
            if len(self._frames) % (self.sample_rate // 1000 * 20) == 0:
                duration_s = self.duration_ms() / 1000.0
                logger.debug(
                    f"Buffer accumulated {duration_s:.2f}s "
                    f"({self._total_bytes} bytes, {len(self._frames)} frames)"
                )

    async def get_audio(self) -> bytes:
        """Get all buffered audio as single bytes object.

        Does NOT clear the buffer. Call clear() separately if needed.

        Returns:
            Concatenated audio bytes (16-bit PCM)

        Example:
            ```python
            audio_data = await buffer.get_audio()
            result = await asr.transcribe(audio_data, buffer.sample_rate)
            await buffer.clear()
            ```
        """
        async with self._lock:
            if not self._frames:
                return b""
            return b"".join(self._frames)

    async def clear(self) -> None:
        """Clear buffer and reset state.

        Removes all accumulated audio frames and resets counters.

        Example:
            ```python
            await buffer.clear()
            assert await buffer.is_empty()
            assert buffer.duration_ms() == 0
            ```
        """
        async with self._lock:
            frames_cleared = len(self._frames)
            bytes_cleared = self._total_bytes
            self._frames.clear()
            self._total_bytes = 0

            if frames_cleared > 0:
                logger.debug(
                    f"Buffer cleared: {frames_cleared} frames, {bytes_cleared} bytes"
                )

    async def is_empty(self) -> bool:
        """Check if buffer is empty.

        Returns:
            True if buffer contains no audio frames, False otherwise

        Example:
            ```python
            if not await buffer.is_empty():
                audio = await buffer.get_audio()
                # Process audio...
            ```
        """
        async with self._lock:
            return len(self._frames) == 0

    def duration_ms(self) -> int:
        """Get duration of buffered audio in milliseconds.

        Does not require lock since it's a simple calculation from
        atomic _total_bytes attribute.

        Returns:
            Duration in milliseconds (0 if buffer is empty)

        Example:
            ```python
            duration = buffer.duration_ms()
            print(f"Buffered {duration}ms of audio")
            ```
        """
        # Calculate duration from total bytes
        # bytes = samples * channels * bytes_per_sample
        # samples = bytes / (channels * bytes_per_sample)
        # duration_ms = samples / sample_rate * 1000

        if self._total_bytes == 0:
            return 0

        bytes_per_sample_per_channel = BYTES_PER_SAMPLE * self.channels
        samples = self._total_bytes // bytes_per_sample_per_channel
        duration_ms = int((samples / self.sample_rate) * 1000)
        return duration_ms

    @property
    def frame_count(self) -> int:
        """Get number of frames currently in buffer.

        Returns:
            Number of audio frames buffered
        """
        return len(self._frames)

    @property
    def byte_count(self) -> int:
        """Get number of bytes currently in buffer.

        Returns:
            Total bytes of audio data buffered
        """
        return self._total_bytes

    @property
    def utilization(self) -> float:
        """Get buffer utilization as fraction of max capacity.

        Returns:
            Utilization ratio (0.0 = empty, 1.0 = full)

        Example:
            ```python
            if buffer.utilization > 0.9:
                logger.warning("Buffer 90% full!")
            ```
        """
        if self._max_bytes == 0:
            return 0.0
        return self._total_bytes / self._max_bytes

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"AudioBuffer(sample_rate={self.sample_rate}, "
            f"channels={self.channels}, "
            f"frames={len(self._frames)}, "
            f"bytes={self._total_bytes}/{self._max_bytes}, "
            f"duration_ms={self.duration_ms()}, "
            f"utilization={self.utilization:.1%})"
        )


class RingAudioBuffer(AudioBuffer):
    """Ring buffer variant that overwrites old data instead of raising overflow.

    Useful for continuous recording scenarios where you want to keep
    the most recent N seconds of audio.

    Example:
        ```python
        # Keep last 5 seconds of audio
        ring_buffer = RingAudioBuffer(sample_rate=48000, max_duration_s=5.0)

        # Append frames continuously - old data automatically discarded
        async for frame in audio_stream:
            await ring_buffer.append(frame)  # Never raises BufferOverflowError
        ```
    """

    async def append(self, frame: bytes) -> None:
        """Append frame to buffer, discarding oldest frames if necessary.

        Unlike base AudioBuffer, this never raises BufferOverflowError.
        Instead, it discards the oldest frames to make room.

        Args:
            frame: Audio frame bytes (16-bit PCM)

        Raises:
            ValueError: If frame is empty
        """
        if not frame:
            raise ValueError("Cannot append empty frame")

        async with self._lock:
            # Add new frame
            self._frames.append(frame)
            self._total_bytes += len(frame)

            # Discard oldest frames until under limit
            while self._total_bytes > self._max_bytes and self._frames:
                oldest_frame = self._frames.pop(0)
                self._total_bytes -= len(oldest_frame)

            # Note: We don't log discards at debug level to avoid spam
            # Production systems should monitor utilization metrics instead
