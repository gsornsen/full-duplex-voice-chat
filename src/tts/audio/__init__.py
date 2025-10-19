"""Audio processing utilities for framing, normalization, and synthesis."""

from .framing import repacketize_to_20ms
from .loudness import normalize_lufs, normalize_rms
from .resampling import resample_audio
from .synthesis import (
    calculate_frame_count,
    calculate_pcm_byte_size,
    float32_to_int16_pcm,
    generate_silence,
    generate_sine_wave,
    generate_sine_wave_frames,
)

__all__ = [
    # Framing utilities
    "repacketize_to_20ms",
    # Resampling utilities
    "resample_audio",
    # Loudness normalization
    "normalize_rms",
    "normalize_lufs",
    # Synthesis utilities
    "generate_sine_wave",
    "generate_sine_wave_frames",
    "generate_silence",
    "float32_to_int16_pcm",
    "calculate_frame_count",
    "calculate_pcm_byte_size",
]
