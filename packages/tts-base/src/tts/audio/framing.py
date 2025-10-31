"""Audio framing utilities for 20ms frames at 48kHz.

This module provides frame repacketization for streaming TTS adapters.
Converts variable-size audio chunks into fixed 20ms frames required by
the WebRTC transport layer.
"""

import numpy as np
from numpy.typing import NDArray


def repacketize_to_20ms(
    audio: NDArray[np.int16],
    sample_rate: int = 48000,
    frame_duration_ms: int = 20,
) -> list[bytes]:
    """Repacketize audio into fixed-duration frames.

    Splits audio into 20ms frames at 48kHz (960 samples per frame).
    Last frame is zero-padded if needed to maintain fixed size.

    Args:
        audio: Audio samples at sample_rate (int16)
        sample_rate: Sample rate in Hz (default: 48000)
        frame_duration_ms: Frame duration in milliseconds (default: 20)

    Returns:
        List of PCM frames as bytes (int16 little-endian)

    Example:
        >>> audio = np.array([1, 2, 3, ..., 960], dtype=np.int16)
        >>> frames = repacketize_to_20ms(audio, sample_rate=48000)
        >>> len(frames)
        1
        >>> len(frames[0])
        1920  # 960 samples * 2 bytes/sample
    """
    frames: list[bytes] = []

    # Calculate samples per frame based on sample rate and duration
    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)

    # Split audio into fixed-duration frames
    for i in range(0, len(audio), samples_per_frame):
        frame_samples = audio[i : i + samples_per_frame]

        # Pad last frame if needed to maintain fixed size
        if len(frame_samples) < samples_per_frame:
            frame_samples = np.pad(
                frame_samples,
                (0, samples_per_frame - len(frame_samples)),
                mode="constant",
            )

        # Convert to bytes (int16 little-endian)
        frame_bytes = frame_samples.tobytes()
        frames.append(frame_bytes)

    return frames
