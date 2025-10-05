"""Audio utilities for frame packetization and encoding.

This module provides utilities for handling 20ms PCM audio frames at 48kHz,
including base64 encoding/decoding and frame validation.
"""

from .packetizer import (
    AudioFramePacketizer,
    decode_pcm_frame,
    encode_pcm_frame,
    validate_frame_size,
)

__all__ = [
    "AudioFramePacketizer",
    "encode_pcm_frame",
    "decode_pcm_frame",
    "validate_frame_size",
]
