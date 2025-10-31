"""Audio utilities for frame packetization, encoding, resampling, and buffering.

This module provides utilities for handling 20ms PCM audio frames at 48kHz,
including base64 encoding/decoding, frame validation, sample rate conversion
for VAD preprocessing, and audio buffering for ASR.
"""

from .buffer import AudioBuffer, AudioBufferError, BufferOverflowError, RingAudioBuffer
from .packetizer import (
    AudioFramePacketizer,
    decode_pcm_frame,
    encode_pcm_frame,
    validate_frame_size,
)
from .resampler import AudioResampler, create_vad_resampler

__all__ = [
    # Packetization
    "AudioFramePacketizer",
    "encode_pcm_frame",
    "decode_pcm_frame",
    "validate_frame_size",
    # Resampling
    "AudioResampler",
    "create_vad_resampler",
    # Buffering (M10)
    "AudioBuffer",
    "RingAudioBuffer",
    "AudioBufferError",
    "BufferOverflowError",
]
