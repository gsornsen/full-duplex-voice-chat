"""Audio resampling utilities for TTS adapters.

This module provides high-quality audio resampling using scipy's Fourier method.
Shared across all TTS adapters (Piper, CosyVoice 2, XTTS-v2, etc.) to ensure
consistent audio processing quality.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import signal


def resample_audio(
    audio: NDArray[np.int16], source_rate: int, target_rate: int
) -> NDArray[np.int16]:
    """Resample audio to target sample rate using scipy.signal.resample.

    Uses high-quality Fourier method for resampling. Handles edge cases:
    - Same source/target rate: returns input unchanged
    - Empty audio: returns empty array
    - Type conversion: int16 → float32 → resample → int16

    Args:
        audio: Input audio samples (int16)
        source_rate: Source sample rate in Hz
        target_rate: Target sample rate in Hz

    Returns:
        Resampled audio samples (int16)

    Example:
        >>> audio_22k = np.array([100, 200, 300], dtype=np.int16)
        >>> audio_48k = resample_audio(audio_22k, 22050, 48000)
        >>> len(audio_48k)  # ~6.5 samples
        7
    """
    if source_rate == target_rate:
        return audio

    # Handle empty audio
    if len(audio) == 0:
        return audio

    # Convert to float for resampling
    audio_float = audio.astype(np.float32)

    # Calculate resampling ratio
    num_samples = int(len(audio_float) * target_rate / source_rate)

    # Resample using scipy
    resampled = signal.resample(audio_float, num_samples)

    # Convert back to int16
    return resampled.astype(np.int16)  # type: ignore[no-any-return]
