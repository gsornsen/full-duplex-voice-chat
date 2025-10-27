"""Audio resampling utilities for TTS adapters.

This module provides high-quality audio resampling using scipy's polyphase
anti-aliasing filter. Shared across all TTS adapters (Piper, CosyVoice 2,
XTTS-v2, etc.) to ensure consistent audio processing quality.

Key improvements over basic Fourier resampling:
- Polyphase FIR filter with anti-aliasing
- No ringing artifacts
- Optimized for realtime streaming
"""

import numpy as np
from numpy.typing import NDArray
from scipy import signal


def resample_audio(
    audio: NDArray[np.int16], source_rate: int, target_rate: int
) -> NDArray[np.int16]:
    """Resample audio to target sample rate using high-quality polyphase filter.

    Uses scipy.signal.resample_poly for superior quality compared to Fourier
    resampling. Benefits:
    - Anti-aliasing filter prevents frequency folding artifacts
    - Polyphase implementation is efficient for realtime use
    - No ringing artifacts (unlike Fourier method)
    - Optimized for integer rate conversions (e.g., 24kHz → 48kHz = 2x upsampling)

    Args:
        audio: Input audio samples (int16)
        source_rate: Source sample rate in Hz
        target_rate: Target sample rate in Hz

    Returns:
        Resampled audio samples (int16)

    Example:
        >>> audio_24k = np.array([100, 200, 300], dtype=np.int16)
        >>> audio_48k = resample_audio(audio_24k, 24000, 48000)
        >>> len(audio_48k)  # Exactly 2x upsampling
        6

    Notes:
        - For CosyVoice (24kHz → 48kHz): Exactly 2x upsampling, very efficient
        - For Piper (22050Hz → 48kHz): Rational ratio (160/147), still efficient
        - Processing time: ~5-10ms for 1 second of audio
        - Memory efficient: no large FFT buffers required
    """
    if source_rate == target_rate:
        return audio

    # Handle empty audio
    if len(audio) == 0:
        return audio

    # Convert to float for resampling
    audio_float = audio.astype(np.float32)

    # Use polyphase filter for high-quality resampling
    # This is superior to Fourier method (signal.resample) because:
    # 1. Built-in anti-aliasing filter prevents aliasing artifacts
    # 2. No ringing artifacts from Gibbs phenomenon
    # 3. Optimized for integer rate ratios (e.g., 2x upsampling)
    # 4. Lower memory usage (no large FFT buffers)
    try:
        resampled = signal.resample_poly(audio_float, target_rate, source_rate)
    except Exception:
        # Fallback to Fourier method if polyphase fails (rare)
        # This can happen with unusual rate ratios or very short audio
        num_samples = int(len(audio_float) * target_rate / source_rate)
        resampled = signal.resample(audio_float, num_samples)

    # Convert back to int16 with proper clipping
    # Use clip to prevent overflow from resampling overshoots
    resampled_clipped = np.clip(resampled, -32768, 32767)

    return resampled_clipped.astype(np.int16)  # type: ignore[no-any-return]
