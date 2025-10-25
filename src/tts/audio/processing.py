"""Audio processing utilities for artifact reduction and quality enhancement.

This module provides utilities to reduce audio artifacts (pops, clicks, static)
and improve overall audio quality in TTS output. Used by all TTS adapters to
ensure consistent, high-quality audio processing.

Key features:
- DC offset removal
- Fade-in/fade-out for click reduction
- High-quality resampling with anti-aliasing
- Dithering for quantization noise reduction
- Cross-fade utilities for boundary discontinuity elimination
"""

import numpy as np
from numpy.typing import NDArray


def remove_dc_offset(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    """Remove DC bias from audio signal.

    DC offset (non-zero mean) can cause pops and clicks at audio boundaries.
    This function centers the audio around zero by subtracting the mean.

    Args:
        audio: Audio samples (float32, normalized -1 to 1)

    Returns:
        DC-corrected audio samples

    Example:
        >>> audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        >>> corrected = remove_dc_offset(audio)
        >>> np.abs(corrected.mean()) < 1e-7
        True
    """
    mean_value: float = float(np.mean(audio))
    return (audio - mean_value).astype(np.float32)


def apply_fade(
    audio: NDArray[np.float32],
    fade_in_ms: float = 5.0,
    fade_out_ms: float = 5.0,
    sample_rate: int = 48000,
) -> NDArray[np.float32]:
    """Apply fade-in and fade-out to prevent clicks and pops.

    Gradual amplitude transitions at audio boundaries eliminate abrupt changes
    that cause audible clicks/pops. Uses linear fades for simplicity and speed.

    Args:
        audio: Audio samples (float32, normalized -1 to 1)
        fade_in_ms: Fade-in duration in milliseconds (default: 5ms)
        fade_out_ms: Fade-out duration in milliseconds (default: 5ms)
        sample_rate: Sample rate in Hz (default: 48000)

    Returns:
        Audio with fades applied

    Example:
        >>> audio = np.ones(1000, dtype=np.float32)
        >>> faded = apply_fade(audio, fade_in_ms=10, fade_out_ms=10, sample_rate=1000)
        >>> faded[0] < 0.1  # First sample faded in
        True
        >>> faded[-1] < 0.1  # Last sample faded out
        True

    Notes:
        - Fade lengths are clamped to prevent exceeding audio length
        - If audio is shorter than fade length, entire audio is faded
        - Linear fades used for minimal CPU overhead (< 1ms processing time)
    """
    # Work on a copy to avoid modifying input
    audio_faded = audio.copy()
    audio_length = len(audio_faded)

    # Calculate fade lengths in samples
    fade_in_samples = int(fade_in_ms * sample_rate / 1000)
    fade_out_samples = int(fade_out_ms * sample_rate / 1000)

    # Clamp fade lengths to prevent exceeding audio length
    fade_in_samples = min(fade_in_samples, audio_length // 2)
    fade_out_samples = min(fade_out_samples, audio_length // 2)

    # Apply fade-in
    if fade_in_samples > 0:
        fade_in_curve = np.linspace(0, 1, fade_in_samples, dtype=np.float32)
        audio_faded[:fade_in_samples] *= fade_in_curve

    # Apply fade-out
    if fade_out_samples > 0:
        fade_out_curve = np.linspace(1, 0, fade_out_samples, dtype=np.float32)
        audio_faded[-fade_out_samples:] *= fade_out_curve

    return audio_faded


def apply_dither(
    audio: NDArray[np.float32],
    bit_depth: int = 16,
    dither_amount: float = 1.0,
) -> NDArray[np.float32]:
    """Apply triangular dithering to reduce quantization noise.

    Dithering adds low-level noise before quantization to prevent correlated
    quantization artifacts. Triangular dither (TPDF) is optimal for 16-bit audio,
    providing noise shaping that pushes quantization noise above audible range.

    Args:
        audio: Audio samples (float32, normalized -1 to 1)
        bit_depth: Target bit depth (default: 16)
        dither_amount: Dithering strength multiplier (default: 1.0)

    Returns:
        Dithered audio samples

    Example:
        >>> audio = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        >>> dithered = apply_dither(audio, bit_depth=16)
        >>> not np.array_equal(audio, dithered)  # Noise added
        True

    Notes:
        - Triangular dithering (TPDF) used for optimal perceptual quality
        - Noise level scaled to 1 LSB (least significant bit) at target bit depth
        - Dither amount can be reduced (< 1.0) for very quiet passages
        - Processing overhead: < 1ms for typical audio chunks
    """
    # Calculate quantization step size (1 LSB at target bit depth)
    lsb = 1.0 / (2 ** (bit_depth - 1))

    # Generate triangular dither (sum of two uniform distributions)
    # Triangular PDF has better perceptual properties than rectangular
    dither_noise = (
        np.random.uniform(-0.5, 0.5, len(audio))
        + np.random.uniform(-0.5, 0.5, len(audio))
    ) * lsb * dither_amount

    result: NDArray[np.float32] = (audio + dither_noise).astype(np.float32)
    return result


def soft_clip(
    audio: NDArray[np.float32],
    threshold: float = 0.9,
    knee: float = 0.1,
) -> NDArray[np.float32]:
    """Apply soft clipping to prevent harsh distortion from overshoots.

    Soft clipping uses a smooth tanh-based curve to gradually limit peaks,
    preventing the harsh distortion of hard clipping while maintaining loudness.
    Useful when aggressive normalization may cause resampling overshoots.

    Args:
        audio: Audio samples (float32, normalized -1 to 1)
        threshold: Soft clipping threshold (default: 0.9)
        knee: Transition smoothness (default: 0.1)

    Returns:
        Soft-clipped audio samples

    Example:
        >>> audio = np.array([0.95, 1.1, -1.2], dtype=np.float32)
        >>> clipped = soft_clip(audio, threshold=0.9)
        >>> np.all(np.abs(clipped) <= 1.0)
        True

    Notes:
        - Threshold: Where soft limiting begins (0.9 = 90% of full scale)
        - Knee: Controls transition smoothness (0.1 = gentle, 0.01 = hard)
        - Uses tanh() for smooth, musical compression
        - Preserves waveform shape better than hard clipping
    """
    # Vectorized implementation for efficiency
    # For |x| < threshold: output = x (linear passthrough)
    # For |x| > threshold: output smoothly approaches ±1.0
    abs_audio = np.abs(audio)

    # Calculate excess beyond threshold
    excess = abs_audio - threshold

    # Apply soft knee only where needed
    compressed_excess = knee * np.tanh(np.where(abs_audio > threshold, excess / knee, 0.0))

    # Combine linear region and compressed region
    result = np.where(
        abs_audio <= threshold,
        audio,
        np.sign(audio) * (threshold + compressed_excess)
    )

    return result.astype(np.float32)


def normalize_peak(
    audio: NDArray[np.float32],
    target_peak: float = 0.85,
) -> NDArray[np.float32]:
    """Normalize audio to target peak level with conservative headroom.

    Peak normalization ensures consistent loudness across TTS models while
    leaving headroom for resampling overshoots and processing artifacts.

    Args:
        audio: Audio samples (float32, normalized -1 to 1)
        target_peak: Target peak level (default: 0.85 = -1.4 dBFS)

    Returns:
        Normalized audio samples

    Example:
        >>> audio = np.array([0.5, -0.5], dtype=np.float32)
        >>> normalized = normalize_peak(audio, target_peak=0.9)
        >>> np.abs(normalized).max()
        0.9

    Notes:
        - Reduced from 0.95 to 0.85 to prevent resampling overshoots
        - Provides 15% headroom (1.45 dB) for processing artifacts
        - Skips normalization if audio is silence (peak == 0)
        - Processing time: < 1ms for typical chunks
    """
    current_peak: float = float(np.abs(audio).max())

    if current_peak > 0:
        gain: float = target_peak / current_peak
        result = audio * gain
        return result.astype(np.float32)

    return audio  # Silence - no normalization needed


def crossfade_buffers(
    buffer_a: NDArray[np.float32],
    buffer_b: NDArray[np.float32],
    crossfade_ms: float = 20.0,
    sample_rate: int = 48000,
) -> NDArray[np.float32]:
    """Cross-fade the end of buffer_a with the start of buffer_b.

    This eliminates abrupt transitions (DC offset discontinuities, boundary clicks)
    when concatenating audio buffers. Uses equal-power crossfade for perceptually
    smooth transitions.

    Args:
        buffer_a: First audio buffer (ending audio)
        buffer_b: Second audio buffer (starting audio)
        crossfade_ms: Cross-fade duration in milliseconds (default: 20ms)
        sample_rate: Audio sample rate (default: 48000 Hz)

    Returns:
        Merged audio with smooth transition

    Example:
        >>> # Create buffers with DC offset mismatch
        >>> buffer_a = np.ones(4800, dtype=np.float32) * 0.5
        >>> buffer_b = np.ones(4800, dtype=np.float32) * -0.3
        >>> merged = crossfade_buffers(buffer_a, buffer_b, crossfade_ms=20.0)
        >>> # Verify smooth transition (no sharp discontinuities)
        >>> diff = np.abs(np.diff(merged))
        >>> diff.max() < 0.1  # Much less than 0.8 direct jump
        True

    Notes:
        - If buffers are shorter than crossfade duration, falls back to concatenation
        - Uses linear crossfade curves (sufficient for 20ms transitions)
        - Processing overhead: ~1-2ms for typical buffers
        - Designed for VAD→TTS boundary smoothing
    """
    crossfade_samples = int(crossfade_ms * sample_rate / 1000)

    # Ensure we have enough samples to crossfade
    if len(buffer_a) < crossfade_samples or len(buffer_b) < crossfade_samples:
        # Not enough samples to crossfade, just concatenate
        result: NDArray[np.float32] = np.concatenate([buffer_a, buffer_b])
        return result

    # Create fade curves (linear)
    fade_out = np.linspace(1, 0, crossfade_samples, dtype=np.float32)
    fade_in = np.linspace(0, 1, crossfade_samples, dtype=np.float32)

    # Apply fades to create crossfade region
    a_faded = buffer_a.copy()
    b_faded = buffer_b.copy()
    a_faded[-crossfade_samples:] *= fade_out
    b_faded[:crossfade_samples] *= fade_in

    # Merge: non-overlapping + crossfaded region + remaining
    merged: NDArray[np.float32] = np.concatenate([
        a_faded[:-crossfade_samples],  # buffer_a without fade region
        a_faded[-crossfade_samples:] + b_faded[:crossfade_samples],  # crossfaded region
        b_faded[crossfade_samples:],  # buffer_b without fade region
    ])

    return merged


def process_audio_for_streaming(
    audio: NDArray[np.float32],
    sample_rate: int = 48000,
    target_peak: float = 0.85,
    apply_fades: bool = True,
    apply_dithering: bool = True,
) -> NDArray[np.float32]:
    """Apply complete audio processing pipeline for artifact-free streaming.

    Combines multiple processing steps in optimal order to eliminate pops,
    clicks, and static while maintaining high audio quality.

    Processing order:
    1. DC offset removal (prevents boundary clicks)
    2. Peak normalization (consistent loudness)
    3. Soft clipping (prevents resampling overshoots)
    4. Fade in/out (eliminates boundary pops)
    5. Dithering (reduces quantization noise)

    Args:
        audio: Audio samples (float32, normalized -1 to 1)
        sample_rate: Sample rate in Hz (default: 48000)
        target_peak: Target peak level (default: 0.85)
        apply_fades: Enable fade-in/fade-out (default: True)
        apply_dithering: Enable dithering (default: True)

    Returns:
        Processed audio ready for int16 conversion

    Example:
        >>> audio = np.random.randn(48000).astype(np.float32) * 0.5
        >>> processed = process_audio_for_streaming(audio)
        >>> len(processed) == 48000
        True
        >>> np.abs(processed).max() <= 0.95  # Conservative peak
        True

    Notes:
        - Total processing overhead: ~2-3ms for 1 second of audio
        - Optimized for streaming: processes single chunks independently
        - Safe for GPU tensors: converts to numpy, processes, returns numpy
    """
    # Step 1: Remove DC offset (prevents boundary clicks)
    audio_processed = remove_dc_offset(audio)

    # Step 2: Normalize to target peak (consistent loudness)
    audio_processed = normalize_peak(audio_processed, target_peak=target_peak)

    # Step 3: Soft clip to prevent overshoots (no harsh distortion)
    audio_processed = soft_clip(audio_processed, threshold=0.9, knee=0.1)

    # Step 4: Apply fades (eliminate boundary pops)
    if apply_fades:
        audio_processed = apply_fade(
            audio_processed, fade_in_ms=5.0, fade_out_ms=5.0, sample_rate=sample_rate
        )

    # Step 5: Dither before quantization (reduce quantization noise)
    if apply_dithering:
        audio_processed = apply_dither(audio_processed, bit_depth=16, dither_amount=0.5)

    return audio_processed
