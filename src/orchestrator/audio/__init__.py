"""Audio utilities for frame packetization, encoding, and resampling.

This module provides utilities for handling 20ms PCM audio frames at 48kHz,
including base64 encoding/decoding, frame validation, and sample rate conversion
for VAD preprocessing.
"""

from .packetizer import (
    AudioFramePacketizer,
    decode_pcm_frame,
    encode_pcm_frame,
    validate_frame_size,
)
from .resampler import AudioResampler, create_vad_resampler

__all__ = [
    "AudioFramePacketizer",
    "encode_pcm_frame",
    "decode_pcm_frame",
    "validate_frame_size",
    "AudioResampler",
    "create_vad_resampler",
]
