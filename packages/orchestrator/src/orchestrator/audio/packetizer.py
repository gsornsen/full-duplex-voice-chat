"""Audio frame packetization utilities.

Handles encoding/decoding and validation of 20ms PCM audio frames at 48kHz.
Audio frames are base64-encoded for JSON transport over WebSocket.

Frame specification:
    - Duration: 20ms
    - Sample rate: 48kHz
    - Channels: mono
    - Bit depth: 16-bit signed integer (little endian)
    - Frame size: 960 samples * 2 bytes = 1920 bytes
"""

import base64

# Audio constants
SAMPLE_RATE_HZ: int = 48000
FRAME_DURATION_MS: int = 20
CHANNELS: int = 1
BYTES_PER_SAMPLE: int = 2

# Calculated frame size: (48000 Hz * 20ms / 1000) * 1 channel * 2 bytes = 1920 bytes
EXPECTED_FRAME_SIZE_BYTES: int = (
    SAMPLE_RATE_HZ * FRAME_DURATION_MS // 1000 * CHANNELS * BYTES_PER_SAMPLE
)


def validate_frame_size(frame: bytes) -> None:
    """Validate that a PCM frame is exactly 20ms at 48kHz.

    Args:
        frame: Raw PCM audio bytes

    Raises:
        ValueError: If frame size is incorrect
    """
    if len(frame) != EXPECTED_FRAME_SIZE_BYTES:
        raise ValueError(
            f"Invalid frame size: expected {EXPECTED_FRAME_SIZE_BYTES} bytes "
            f"(20ms @ 48kHz mono), got {len(frame)} bytes"
        )


def encode_pcm_frame(frame: bytes) -> str:
    """Encode PCM frame to base64 string for JSON transport.

    Args:
        frame: Raw PCM audio bytes (1920 bytes for 20ms @ 48kHz)

    Returns:
        Base64-encoded string

    Raises:
        ValueError: If frame size is incorrect
    """
    validate_frame_size(frame)
    return base64.b64encode(frame).decode("ascii")


def decode_pcm_frame(encoded: str) -> bytes:
    """Decode base64 string to PCM frame.

    Args:
        encoded: Base64-encoded PCM frame

    Returns:
        Raw PCM audio bytes

    Raises:
        ValueError: If decoded frame size is incorrect or decoding fails
    """
    try:
        frame = base64.b64decode(encoded)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 audio frame: {e}") from e

    validate_frame_size(frame)
    return frame


class AudioFramePacketizer:
    """Manages audio frame sequence numbering and validation.

    Tracks sequence numbers for audio frames to enable proper ordering
    and loss detection on the client side.
    """

    def __init__(self) -> None:
        """Initialize packetizer with sequence number counter."""
        self._sequence_number: int = 0

    def create_frame_metadata(self, frame: bytes) -> dict[str, int | str]:
        """Create frame metadata with sequence number.

        Args:
            frame: Raw PCM audio bytes

        Returns:
            Dictionary with frame metadata including base64-encoded audio

        Raises:
            ValueError: If frame size is incorrect
        """
        validate_frame_size(frame)

        self._sequence_number += 1

        return {
            "pcm": encode_pcm_frame(frame),
            "sample_rate": SAMPLE_RATE_HZ,
            "frame_ms": FRAME_DURATION_MS,
            "sequence": self._sequence_number,
        }

    def reset(self) -> None:
        """Reset sequence number counter."""
        self._sequence_number = 0

    @property
    def current_sequence(self) -> int:
        """Get current sequence number."""
        return self._sequence_number
