"""Audio frame handler utilities.

Helper utilities for processing audio frames, including validation,
file writing, format conversion, and statistics.

Demonstrates:
- Audio frame validation (size, format)
- Writing frames to WAV file
- RMS level calculation
- Frame statistics collection
- Buffer management

Usage:
    python examples/audio_frame_handler.py
"""

import struct
import wave
from dataclasses import dataclass
from pathlib import Path

from src.common.types import AudioFrame


@dataclass
class AudioFrameStats:
    """Statistics for audio frame processing.

    Attributes:
        total_frames: Total number of frames processed
        total_bytes: Total bytes of audio data
        total_duration_ms: Total duration in milliseconds
        min_rms: Minimum RMS level across all frames
        max_rms: Maximum RMS level across all frames
        avg_rms: Average RMS level
        invalid_frames: Number of frames with invalid size
    """

    total_frames: int = 0
    total_bytes: int = 0
    total_duration_ms: float = 0.0
    min_rms: float = float("inf")
    max_rms: float = 0.0
    avg_rms: float = 0.0
    invalid_frames: int = 0


class AudioFrameHandler:
    """Handler for processing audio frames.

    This class provides utilities for validating, analyzing, and writing
    audio frames to disk. It maintains statistics and can write frames
    to a WAV file.

    Example:
        >>> handler = AudioFrameHandler()
        >>> for frame in audio_frames:
        ...     if handler.validate_frame(frame):
        ...         handler.add_frame(frame)
        >>> handler.write_wav("output.wav")
        >>> print(handler.get_stats())
    """

    def __init__(self, sample_rate: int = 48000, frame_duration_ms: int = 20) -> None:
        """Initialize audio frame handler.

        Args:
            sample_rate: Sample rate in Hz (default: 48000)
            frame_duration_ms: Frame duration in milliseconds (default: 20)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frames: list[AudioFrame] = []
        self.stats = AudioFrameStats()

        # Calculate expected frame size
        # samples_per_frame = (sample_rate / 1000) * frame_duration_ms
        # bytes_per_frame = samples_per_frame * 2 (16-bit)
        self.expected_frame_size = int(
            (sample_rate / 1000) * frame_duration_ms * 2
        )

    def validate_frame(self, frame: AudioFrame) -> bool:
        """Validate audio frame format.

        Args:
            frame: Audio frame to validate

        Returns:
            True if frame is valid, False otherwise
        """
        if not isinstance(frame, bytes):
            print(f"✗ Invalid frame type: {type(frame)}")
            return False

        if len(frame) != self.expected_frame_size:
            print(
                f"✗ Invalid frame size: {len(frame)} bytes "
                f"(expected {self.expected_frame_size})"
            )
            self.stats.invalid_frames += 1
            return False

        return True

    def calculate_rms(self, frame: AudioFrame) -> float:
        """Calculate RMS (Root Mean Square) level of audio frame.

        RMS provides a measure of the average amplitude of the audio signal.
        Useful for detecting silence, normalizing levels, and quality checks.

        Args:
            frame: Audio frame (16-bit PCM)

        Returns:
            RMS level (0.0 to 1.0, where 1.0 is maximum)
        """
        # Unpack 16-bit samples
        num_samples = len(frame) // 2
        samples = struct.unpack(f"<{num_samples}h", frame)

        # Calculate RMS
        sum_squares = sum(sample * sample for sample in samples)
        rms = (sum_squares / num_samples) ** 0.5

        # Normalize to 0.0-1.0 range (max 16-bit value is 32767)
        return rms / 32767.0

    def add_frame(self, frame: AudioFrame) -> None:
        """Add frame to buffer and update statistics.

        Args:
            frame: Audio frame to add
        """
        if not self.validate_frame(frame):
            return

        self.frames.append(frame)

        # Update statistics
        self.stats.total_frames += 1
        self.stats.total_bytes += len(frame)
        self.stats.total_duration_ms += self.frame_duration_ms

        # Calculate and track RMS
        rms = self.calculate_rms(frame)
        self.stats.min_rms = min(self.stats.min_rms, rms)
        self.stats.max_rms = max(self.stats.max_rms, rms)

        # Update running average
        prev_total = self.stats.avg_rms * (self.stats.total_frames - 1)
        self.stats.avg_rms = (prev_total + rms) / self.stats.total_frames

    def write_wav(self, output_path: str | Path) -> None:
        """Write buffered frames to WAV file.

        Args:
            output_path: Path to output WAV file

        Raises:
            ValueError: If no frames have been added
        """
        if not self.frames:
            raise ValueError("No frames to write")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(output_path), "wb") as wav_file:
            # Configure WAV parameters
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)

            # Write all frames
            for frame in self.frames:
                wav_file.writeframes(frame)

        print(f"✓ Wrote {len(self.frames)} frames to {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024:.2f} KB")
        print(f"  Duration: {self.stats.total_duration_ms / 1000:.2f} seconds")

    def get_stats(self) -> AudioFrameStats:
        """Get current frame statistics.

        Returns:
            Audio frame statistics
        """
        return self.stats

    def clear(self) -> None:
        """Clear buffered frames and reset statistics."""
        self.frames.clear()
        self.stats = AudioFrameStats()


def demo() -> None:
    """Demonstrate audio frame handler usage."""
    print("=== Audio Frame Handler Demo ===\n")

    handler = AudioFrameHandler(sample_rate=48000, frame_duration_ms=20)

    # Generate test frames (silence)
    print("Generating test frames...")
    silence_frame: AudioFrame = b"\x00" * 1920  # 20ms silence at 48kHz

    # Add 100 frames (2 seconds of audio)
    for i in range(100):
        handler.add_frame(silence_frame)
        if (i + 1) % 25 == 0:
            print(f"  Added frame {i + 1}/100")

    print()

    # Display statistics
    stats = handler.get_stats()
    print("Statistics:")
    print(f"  Total frames: {stats.total_frames}")
    print(f"  Total bytes: {stats.total_bytes:,} ({stats.total_bytes / 1024:.2f} KB)")
    print(f"  Duration: {stats.total_duration_ms / 1000:.2f} seconds")
    print(f"  RMS range: {stats.min_rms:.6f} - {stats.max_rms:.6f}")
    print(f"  Average RMS: {stats.avg_rms:.6f}")
    print(f"  Invalid frames: {stats.invalid_frames}")
    print()

    # Write to WAV file
    output_path = Path("/tmp/test_audio.wav")
    handler.write_wav(output_path)
    print()

    # Test validation with invalid frame
    print("Testing validation with invalid frame...")
    invalid_frame: AudioFrame = b"\x00" * 100  # Wrong size
    if not handler.validate_frame(invalid_frame):
        print("✓ Correctly rejected invalid frame")
    print()

    # Clean up
    if output_path.exists():
        output_path.unlink()
        print(f"✓ Cleaned up test file: {output_path}")


if __name__ == "__main__":
    demo()
