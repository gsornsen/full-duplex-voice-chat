"""Audio synthesis utilities for generating test signals and warmup audio."""

import numpy as np
from numpy.typing import NDArray


def float32_to_int16_pcm(audio: NDArray[np.float32]) -> bytes:
    """Convert float32 audio array to int16 PCM bytes.

    Args:
        audio: Float32 audio array with values in range [-1.0, 1.0]

    Returns:
        PCM audio data as bytes (little-endian int16)

    Raises:
        ValueError: If audio array is empty
    """
    if audio.size == 0:
        raise ValueError("Audio array cannot be empty")

    # Clip to valid range and convert to int16
    clipped = np.clip(audio, -1.0, 1.0)
    int16_audio = (clipped * 32767.0).astype(np.int16)

    # Convert to bytes (little-endian by default on most systems)
    return int16_audio.tobytes()


def generate_sine_wave(frequency: int, duration_ms: int, sample_rate: int) -> bytes:
    """Generate a sine wave at the specified frequency.

    Args:
        frequency: Frequency in Hz (must be positive and <= sample_rate/2)
        duration_ms: Duration in milliseconds (must be non-negative)
        sample_rate: Sample rate in Hz (will be 48000 for our system)

    Returns:
        int16 PCM audio data as bytes (little-endian)

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> # Generate 100ms of 440Hz sine wave at 48kHz
        >>> audio_bytes = generate_sine_wave(frequency=440, duration_ms=100, sample_rate=48000)
        >>> len(audio_bytes)  # 100ms × 48kHz × 2 bytes/sample
        9600
    """
    # Validate inputs - check sample rate first before using it in comparisons
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    if frequency <= 0:
        raise ValueError(f"Frequency must be positive, got {frequency}")
    if frequency > sample_rate / 2:
        raise ValueError(
            f"Frequency {frequency} exceeds Nyquist limit {sample_rate / 2}"
        )
    if duration_ms < 0:
        raise ValueError(f"Duration must be non-negative, got {duration_ms}")

    # Handle zero duration edge case
    if duration_ms == 0:
        return b""

    # Calculate number of samples
    num_samples = int(sample_rate * duration_ms / 1000.0)

    # Generate time array
    t = np.arange(num_samples, dtype=np.float32)

    # Generate sine wave: sin(2π f t / sample_rate)
    sine_wave = np.sin(2.0 * np.pi * frequency * t / sample_rate).astype(np.float32)

    # Convert to int16 PCM bytes
    return float32_to_int16_pcm(sine_wave)


def generate_sine_wave_frames(
    frequency: int, duration_ms: int, sample_rate: int, frame_duration_ms: int = 20
) -> list[bytes]:
    """Generate sine wave audio framed into fixed-duration chunks.

    Args:
        frequency: Frequency in Hz
        duration_ms: Total duration in milliseconds
        sample_rate: Sample rate in Hz (48000 for our system)
        frame_duration_ms: Frame duration in milliseconds (default 20ms)

    Returns:
        List of PCM audio frames, each containing frame_duration_ms of audio

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> # Generate 100ms of 440Hz sine wave framed into 20ms chunks at 48kHz
        >>> frames = generate_sine_wave_frames(
        ...     frequency=440, duration_ms=100, sample_rate=48000, frame_duration_ms=20
        ... )
        >>> len(frames)  # 100ms / 20ms = 5 frames
        5
        >>> len(frames[0])  # 20ms × 48kHz × 2 bytes/sample
        1920
    """
    if frame_duration_ms <= 0:
        raise ValueError(f"Frame duration must be positive, got {frame_duration_ms}")

    # Validate other inputs (will be checked by generate_sine_wave)
    if duration_ms == 0:
        return []

    # Generate complete audio
    audio_bytes = generate_sine_wave(frequency, duration_ms, sample_rate)

    # Calculate frame size in bytes
    samples_per_frame = int(sample_rate * frame_duration_ms / 1000.0)
    bytes_per_frame = samples_per_frame * 2  # int16 = 2 bytes per sample

    # Split into frames
    frames: list[bytes] = []
    for i in range(0, len(audio_bytes), bytes_per_frame):
        frame = audio_bytes[i : i + bytes_per_frame]
        # Only include complete frames
        if len(frame) == bytes_per_frame:
            frames.append(frame)

    return frames


def generate_silence(duration_ms: int, sample_rate: int) -> bytes:
    """Generate silence (zeros) for the specified duration.

    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz

    Returns:
        int16 PCM audio data as bytes (all zeros)

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> # Generate 20ms of silence at 48kHz
        >>> silence = generate_silence(duration_ms=20, sample_rate=48000)
        >>> len(silence)  # 20ms × 48kHz × 2 bytes/sample
        1920
    """
    if duration_ms < 0:
        raise ValueError(f"Duration must be non-negative, got {duration_ms}")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    if duration_ms == 0:
        return b""

    # Calculate number of samples
    num_samples = int(sample_rate * duration_ms / 1000.0)

    # Generate zeros
    silence_array = np.zeros(num_samples, dtype=np.int16)

    return silence_array.tobytes()


def calculate_frame_count(duration_ms: int, frame_duration_ms: int = 20) -> int:
    """Calculate the number of complete frames for a given duration.

    Args:
        duration_ms: Total duration in milliseconds
        frame_duration_ms: Frame duration in milliseconds (default 20ms)

    Returns:
        Number of complete frames

    Examples:
        >>> calculate_frame_count(duration_ms=100, frame_duration_ms=20)
        5
        >>> calculate_frame_count(duration_ms=105, frame_duration_ms=20)
        5
    """
    if duration_ms < 0:
        return 0
    if frame_duration_ms <= 0:
        raise ValueError(f"Frame duration must be positive, got {frame_duration_ms}")

    return duration_ms // frame_duration_ms


def calculate_pcm_byte_size(duration_ms: int, sample_rate: int, channels: int = 1) -> int:
    """Calculate the size in bytes of PCM audio for given parameters.

    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz
        channels: Number of channels (default 1 for mono)

    Returns:
        Size in bytes (int16 PCM)

    Examples:
        >>> # 20ms at 48kHz mono = 960 samples × 2 bytes = 1920 bytes
        >>> calculate_pcm_byte_size(duration_ms=20, sample_rate=48000, channels=1)
        1920
        >>> # 20ms at 48kHz stereo = 960 samples × 2 channels × 2 bytes = 3840 bytes
        >>> calculate_pcm_byte_size(duration_ms=20, sample_rate=48000, channels=2)
        3840
    """
    if duration_ms < 0:
        raise ValueError(f"Duration must be non-negative, got {duration_ms}")
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    if channels <= 0:
        raise ValueError(f"Channels must be positive, got {channels}")

    num_samples = int(sample_rate * duration_ms / 1000.0)
    return num_samples * channels * 2  # int16 = 2 bytes per sample
