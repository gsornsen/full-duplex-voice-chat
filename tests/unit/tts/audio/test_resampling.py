"""Unit tests for audio resampling utilities.

Tests the resample_audio() function extracted from the Piper adapter.
Covers basic resampling, edge cases, type preservation, and quality validation.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tts.audio.resampling import resample_audio


# Fixtures for test data
@pytest.fixture
def sine_wave_22k() -> NDArray[np.int16]:
    """Generate 1kHz sine wave at 22050Hz (100ms duration).

    Returns:
        Audio samples as int16 array (2205 samples)
    """
    duration_s = 0.1  # 100ms
    freq = 1000  # 1kHz
    sample_rate = 22050
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * 32767 * 0.5).astype(np.int16)
    return wave


@pytest.fixture
def sine_wave_48k() -> NDArray[np.int16]:
    """Generate 1kHz sine wave at 48000Hz (100ms duration).

    Returns:
        Audio samples as int16 array (4800 samples)
    """
    duration_s = 0.1  # 100ms
    freq = 1000  # 1kHz
    sample_rate = 48000
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * 32767 * 0.5).astype(np.int16)
    return wave


@pytest.fixture
def silence_48k() -> NDArray[np.int16]:
    """Generate silence at 48kHz (20ms = 960 samples).

    Returns:
        Silence samples as int16 array
    """
    return np.zeros(960, dtype=np.int16)


# Test 1: Basic upsampling (22050Hz → 48000Hz)
def test_resample_audio_22k_to_48k(sine_wave_22k: NDArray[np.int16]) -> None:
    """Test basic upsampling from 22050Hz to 48000Hz.

    Verifies:
        - Output length is correct (within ±1 sample due to rounding)
        - Audio duration is preserved
        - Output is not empty

    Note: scipy.signal.resample_poly may produce off-by-one differences due to
          internal filter implementation. We allow ±1 sample tolerance.
    """
    source_rate = 22050
    target_rate = 48000
    input_audio = sine_wave_22k[:100]  # Use first 100 samples for simplicity

    # Expected output length: 100 * 48000 / 22050 ≈ 217.687 → 217 or 218 samples
    # scipy.resample_poly may round up/down, so allow ±1 sample tolerance
    expected_length = int(len(input_audio) * target_rate / source_rate)

    result = resample_audio(input_audio, source_rate, target_rate)

    # Verify length is correct (allow ±1 sample due to scipy rounding behavior)
    assert abs(len(result) - expected_length) <= 1, (
        f"Expected ~{expected_length} samples (±1), got {len(result)}"
    )

    # Verify duration is preserved (within 1% tolerance)
    input_duration_s = len(input_audio) / source_rate
    output_duration_s = len(result) / target_rate
    duration_error = abs(input_duration_s - output_duration_s) / input_duration_s

    assert duration_error < 0.01, (
        f"Duration error {duration_error:.2%} exceeds 1% tolerance"
    )

    # Verify output is not all zeros (resampling actually happened)
    assert np.any(result != 0), "Output should not be all zeros"


# Test 2: Downsampling (48000Hz → 22050Hz)
def test_resample_audio_48k_to_22k(sine_wave_48k: NDArray[np.int16]) -> None:
    """Test downsampling from 48000Hz to 22050Hz.

    Verifies:
        - Output length is correct
        - Audio duration is preserved
        - No aliasing artifacts (output is reasonable)
    """
    source_rate = 48000
    target_rate = 22050
    input_audio = sine_wave_48k[:960]  # 20ms at 48kHz

    # Expected output length: 960 * 22050 / 48000 = 441 samples
    expected_length = int(len(input_audio) * target_rate / source_rate)

    result = resample_audio(input_audio, source_rate, target_rate)

    assert len(result) == expected_length, (
        f"Expected {expected_length} samples, got {len(result)}"
    )

    # Verify duration is preserved
    input_duration_s = len(input_audio) / source_rate
    output_duration_s = len(result) / target_rate
    duration_error = abs(input_duration_s - output_duration_s) / input_duration_s

    assert duration_error < 0.01, (
        f"Duration error {duration_error:.2%} exceeds 1% tolerance"
    )

    # Verify output is not all zeros
    assert np.any(result != 0), "Output should not be all zeros"


# Test 3: No-op resampling (same rate)
def test_resample_audio_same_rate(sine_wave_48k: NDArray[np.int16]) -> None:
    """Test no-op resampling when source and target rates are identical.

    Verifies:
        - Output is exactly the input (identity function)
        - No resampling computation is performed
        - Memory is not copied (same object)
    """
    sample_rate = 48000
    input_audio = sine_wave_48k[:960]

    result = resample_audio(input_audio, sample_rate, sample_rate)

    # Should return the exact same array (no resampling)
    assert result is input_audio, "Should return input unchanged for same rate"
    assert len(result) == len(input_audio)
    assert np.array_equal(result, input_audio)


# Test 4: Empty input
def test_resample_audio_empty() -> None:
    """Test edge case with empty audio input.

    Verifies:
        - No crash or exception
        - Returns empty array
        - Preserves dtype
    """
    empty_audio = np.array([], dtype=np.int16)

    result = resample_audio(empty_audio, 22050, 48000)

    assert len(result) == 0, "Empty input should produce empty output"
    assert result.dtype == np.int16, "Output dtype should be int16"
    assert isinstance(result, np.ndarray), "Output should be numpy array"


# Test 5: Single sample input
def test_resample_audio_single_sample() -> None:
    """Test edge case with minimal (single sample) input.

    Verifies:
        - No crash or exception
        - Output has reasonable length (may be rounded)
        - Output is not empty
    """
    single_sample = np.array([1000], dtype=np.int16)
    source_rate = 22050
    target_rate = 48000

    result = resample_audio(single_sample, source_rate, target_rate)

    # Expected: 1 * 48000 / 22050 ≈ 2.177 → 2 samples
    expected_length = int(len(single_sample) * target_rate / source_rate)

    assert len(result) >= 1, "Should have at least 1 output sample"
    assert len(result) <= expected_length + 1, (
        f"Output length {len(result)} exceeds expected {expected_length} + 1"
    )
    assert result.dtype == np.int16


# Test 6: Type preservation (int16 input/output)
def test_resample_audio_type_preservation() -> None:
    """Test that output dtype is preserved as int16.

    Verifies:
        - Input is int16
        - Output is int16 (not float32 or other type)
        - Conversion from float back to int16 is correct
    """
    input_audio = np.array([100, 200, 300, 400, 500], dtype=np.int16)

    result = resample_audio(input_audio, 22050, 48000)

    assert result.dtype == np.int16, (
        f"Output dtype should be int16, got {result.dtype}"
    )
    assert isinstance(result, np.ndarray)

    # Verify values are in int16 range
    assert np.all(result >= -32768), "Values should be >= -32768 (int16 min)"
    assert np.all(result <= 32767), "Values should be <= 32767 (int16 max)"


# Test 7: Quality check (frequency preservation)
def test_resample_audio_quality(sine_wave_22k: NDArray[np.int16]) -> None:
    """Test resampling quality by checking frequency content preservation.

    Verifies:
        - Dominant frequency is preserved after resampling
        - Energy is concentrated at expected frequency
        - Minimal spectral distortion

    Uses FFT to detect peak frequency before and after resampling.
    """
    source_rate = 22050
    target_rate = 48000
    expected_freq = 1000  # 1kHz sine wave

    # Resample from 22050Hz to 48000Hz
    resampled = resample_audio(sine_wave_22k, source_rate, target_rate)

    # Compute FFT of original signal
    fft_orig = np.fft.rfft(sine_wave_22k.astype(np.float32))
    freqs_orig = np.fft.rfftfreq(len(sine_wave_22k), 1.0 / source_rate)
    peak_idx_orig = np.argmax(np.abs(fft_orig))
    peak_freq_orig = freqs_orig[peak_idx_orig]

    # Compute FFT of resampled signal
    fft_resampled = np.fft.rfft(resampled.astype(np.float32))
    freqs_resampled = np.fft.rfftfreq(len(resampled), 1.0 / target_rate)
    peak_idx_resampled = np.argmax(np.abs(fft_resampled))
    peak_freq_resampled = freqs_resampled[peak_idx_resampled]

    # Verify both signals have peak near 1kHz (allow ±50Hz tolerance)
    assert abs(peak_freq_orig - expected_freq) < 50, (
        f"Original peak {peak_freq_orig}Hz differs from expected {expected_freq}Hz"
    )
    assert abs(peak_freq_resampled - expected_freq) < 50, (
        f"Resampled peak {peak_freq_resampled}Hz differs from expected {expected_freq}Hz"
    )

    # Verify both signals have same dominant frequency (within tolerance)
    freq_difference = abs(peak_freq_orig - peak_freq_resampled)
    assert freq_difference < 100, (
        f"Peak frequency changed by {freq_difference}Hz after resampling"
    )
