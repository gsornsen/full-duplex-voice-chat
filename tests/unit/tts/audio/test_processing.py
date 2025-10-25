"""Unit tests for audio processing utilities.

Tests DC offset removal, fades, dithering, normalization, crossfading,
and the complete processing pipeline.
"""

import numpy as np
import pytest

from src.tts.audio.processing import (
    apply_dither,
    apply_fade,
    crossfade_buffers,
    normalize_peak,
    process_audio_for_streaming,
    remove_dc_offset,
    soft_clip,
)


class TestRemoveDCOffset:
    """Tests for DC offset removal."""

    def test_removes_positive_offset(self) -> None:
        """Test that positive DC offset is removed."""
        audio = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        result = remove_dc_offset(audio)

        # Mean should be very close to zero
        assert abs(result.mean()) < 1e-6

        # Relative values should be preserved
        assert result[1] - result[0] == pytest.approx(0.1, abs=1e-6)
        assert result[2] - result[1] == pytest.approx(0.1, abs=1e-6)

    def test_removes_negative_offset(self) -> None:
        """Test that negative DC offset is removed."""
        audio = np.array([-0.5, -0.4, -0.3], dtype=np.float32)
        result = remove_dc_offset(audio)

        assert abs(result.mean()) < 1e-6

    def test_preserves_zero_mean(self) -> None:
        """Test that zero-mean audio is unchanged."""
        audio = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
        result = remove_dc_offset(audio)

        # Should be very close to original
        np.testing.assert_allclose(result, audio, atol=1e-6)


class TestApplyFade:
    """Tests for fade in/out."""

    def test_fade_in_starts_at_zero(self) -> None:
        """Test that fade-in starts at zero amplitude."""
        audio = np.ones(1000, dtype=np.float32)
        result = apply_fade(audio, fade_in_ms=10, fade_out_ms=0, sample_rate=1000)

        # First sample should be zero
        assert result[0] == pytest.approx(0.0, abs=1e-6)

        # Last sample should be unchanged (no fade-out)
        assert result[-1] == pytest.approx(1.0, abs=1e-6)

    def test_fade_out_ends_at_zero(self) -> None:
        """Test that fade-out ends at zero amplitude."""
        audio = np.ones(1000, dtype=np.float32)
        result = apply_fade(audio, fade_in_ms=0, fade_out_ms=10, sample_rate=1000)

        # First sample should be unchanged (no fade-in)
        assert result[0] == pytest.approx(1.0, abs=1e-6)

        # Last sample should be zero
        assert result[-1] == pytest.approx(0.0, abs=1e-6)

    def test_symmetric_fade(self) -> None:
        """Test that symmetric fades work correctly."""
        audio = np.ones(1000, dtype=np.float32)
        result = apply_fade(audio, fade_in_ms=10, fade_out_ms=10, sample_rate=1000)

        # Start and end should be faded
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[-1] == pytest.approx(0.0, abs=1e-6)

        # Middle should be unchanged
        assert result[500] == pytest.approx(1.0, abs=1e-3)

    def test_short_audio_clamping(self) -> None:
        """Test that fades are clamped for short audio."""
        audio = np.ones(10, dtype=np.float32)

        # Request 20ms fade on 10 samples (should be clamped to 5 samples each)
        result = apply_fade(audio, fade_in_ms=20, fade_out_ms=20, sample_rate=1000)

        # Should not raise an error, and should produce valid output
        assert len(result) == 10
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[-1] == pytest.approx(0.0, abs=1e-6)


class TestApplyDither:
    """Tests for dithering."""

    def test_adds_noise(self) -> None:
        """Test that dithering adds noise to constant signal."""
        audio = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        result = apply_dither(audio, bit_depth=16)

        # Result should not be identical (noise added)
        assert not np.array_equal(audio, result)

    def test_noise_scale(self) -> None:
        """Test that dither noise is appropriately scaled."""
        audio = np.zeros(10000, dtype=np.float32)
        result = apply_dither(audio, bit_depth=16, dither_amount=1.0)

        # Noise should be small (scaled to 1 LSB at 16-bit)
        lsb = 1.0 / (2 ** 15)
        noise = result - audio

        # RMS of triangular dither should be ~lsb/sqrt(6)
        rms = np.sqrt(np.mean(noise ** 2))
        assert rms < lsb * 2  # Conservative check


class TestSoftClip:
    """Tests for soft clipping."""

    def test_passthrough_below_threshold(self) -> None:
        """Test that audio below threshold is unchanged."""
        audio = np.array([0.5, 0.7, 0.8], dtype=np.float32)
        result = soft_clip(audio, threshold=0.9)

        # Should be identical
        np.testing.assert_allclose(result, audio, atol=1e-6)

    def test_limits_above_threshold(self) -> None:
        """Test that audio above threshold is limited."""
        audio = np.array([0.95, 1.1, -1.2], dtype=np.float32)
        result = soft_clip(audio, threshold=0.9)

        # All values should be within [-1, 1]
        assert np.all(np.abs(result) <= 1.0)

        # Relative order should be preserved
        assert result[0] < result[1]  # 0.95 < 1.1
        assert result[2] < result[0]  # -1.2 < 0.95


class TestNormalizePeak:
    """Tests for peak normalization."""

    def test_normalizes_to_target(self) -> None:
        """Test that peak is normalized to target level."""
        audio = np.array([0.5, -0.5], dtype=np.float32)
        result = normalize_peak(audio, target_peak=0.9)

        # Peak should be 0.9
        assert np.abs(result).max() == pytest.approx(0.9, abs=1e-6)

    def test_preserves_waveform_shape(self) -> None:
        """Test that normalization preserves waveform shape."""
        audio = np.array([0.25, 0.5, -0.5, 0.0], dtype=np.float32)
        result = normalize_peak(audio, target_peak=1.0)

        # Ratios should be preserved
        assert result[0] / result[1] == pytest.approx(0.5, abs=1e-6)

    def test_skips_silence(self) -> None:
        """Test that silence is not normalized."""
        audio = np.zeros(100, dtype=np.float32)
        result = normalize_peak(audio, target_peak=0.9)

        # Should still be zero
        np.testing.assert_array_equal(result, audio)


class TestCrossfadeBuffers:
    """Tests for buffer crossfading."""

    def test_eliminates_discontinuity(self) -> None:
        """Test that crossfade eliminates abrupt transitions."""
        # Create buffers with DC offset mismatch (simulates VAD→TTS boundary)
        buffer_a = np.ones(4800, dtype=np.float32) * 0.5  # 100ms @ 48kHz
        buffer_b = np.ones(4800, dtype=np.float32) * -0.3  # Different DC offset

        # Crossfade
        merged = crossfade_buffers(buffer_a, buffer_b, crossfade_ms=20.0)

        # Check for smooth transition (no sharp discontinuities)
        diff = np.abs(np.diff(merged))
        max_discontinuity = diff.max()

        # Should have gradual transition, not abrupt jump (0.8 direct jump)
        assert max_discontinuity < 0.1

    def test_preserves_total_length(self) -> None:
        """Test that crossfade preserves expected output length."""
        buffer_a = np.ones(4800, dtype=np.float32) * 0.5
        buffer_b = np.ones(4800, dtype=np.float32) * 0.3

        merged = crossfade_buffers(buffer_a, buffer_b, crossfade_ms=20.0)

        # Length should be sum minus crossfade region
        crossfade_samples = int(20.0 * 48000 / 1000)  # 960 samples
        expected_length = len(buffer_a) + len(buffer_b) - crossfade_samples

        assert len(merged) == expected_length

    def test_fallback_for_short_buffers(self) -> None:
        """Test that short buffers fall back to concatenation."""
        buffer_a = np.ones(100, dtype=np.float32) * 0.5
        buffer_b = np.ones(100, dtype=np.float32) * 0.3

        # Request 20ms crossfade (960 samples @ 48kHz), but buffers are only 100 samples
        merged = crossfade_buffers(buffer_a, buffer_b, crossfade_ms=20.0)

        # Should concatenate instead of crossfade
        assert len(merged) == len(buffer_a) + len(buffer_b)

    def test_smooth_transition_region(self) -> None:
        """Test that crossfade region has smooth amplitude transition."""
        buffer_a = np.ones(4800, dtype=np.float32) * 1.0
        buffer_b = np.ones(4800, dtype=np.float32) * 0.0

        merged = crossfade_buffers(buffer_a, buffer_b, crossfade_ms=20.0)

        crossfade_samples = int(20.0 * 48000 / 1000)  # 960 samples
        transition_start = len(buffer_a) - crossfade_samples
        transition_end = len(buffer_a)

        # Extract transition region
        transition = merged[transition_start:transition_end]

        # Should be monotonically decreasing (1.0 → 0.0)
        assert transition[0] == pytest.approx(1.0, abs=1e-3)
        assert transition[-1] == pytest.approx(0.0, abs=1e-3)

        # Check monotonic decrease
        for i in range(len(transition) - 1):
            assert transition[i] >= transition[i + 1]


class TestProcessAudioForStreaming:
    """Tests for complete processing pipeline."""

    def test_full_pipeline_runs(self) -> None:
        """Test that full pipeline processes audio without errors."""
        audio = np.random.randn(48000).astype(np.float32) * 0.5
        result = process_audio_for_streaming(audio)

        # Should produce same-length output
        assert len(result) == len(audio)

    def test_conservative_peak_limiting(self) -> None:
        """Test that processing doesn't exceed safe peak level."""
        audio = np.random.randn(48000).astype(np.float32)
        result = process_audio_for_streaming(audio, target_peak=0.85)

        # Peak should not exceed target + headroom
        assert np.abs(result).max() <= 0.95

    def test_removes_dc_offset_in_pipeline(self) -> None:
        """Test that DC offset is removed in pipeline."""
        audio = np.random.randn(48000).astype(np.float32) * 0.5 + 0.2  # Add DC offset
        result = process_audio_for_streaming(audio)

        # Mean should be close to zero (DC removed)
        assert abs(result.mean()) < 0.05

    def test_optional_fades(self) -> None:
        """Test that fades can be disabled."""
        # Use signal that doesn't start/end at zero
        audio = np.sin(np.linspace(np.pi / 4, 2 * np.pi + np.pi / 4, 1000)).astype(np.float32) * 0.5

        # With fades
        with_fades = process_audio_for_streaming(audio, apply_fades=True, apply_dithering=False)

        # Without fades
        without_fades = process_audio_for_streaming(audio, apply_fades=False, apply_dithering=False)

        # Fades should reduce amplitude at boundaries
        # With fades: edges are attenuated, without fades: edges preserve amplitude
        assert abs(with_fades[0]) < abs(without_fades[0])
        assert abs(with_fades[-1]) < abs(without_fades[-1])

    def test_optional_dithering(self) -> None:
        """Test that dithering can be disabled."""
        audio = np.ones(1000, dtype=np.float32) * 0.5

        # With dithering (should add noise)
        with_dither = process_audio_for_streaming(audio, apply_fades=False, apply_dithering=True)

        # Without dithering (deterministic)
        without_dither = process_audio_for_streaming(
            audio, apply_fades=False, apply_dithering=False
        )

        # Results should differ due to dither noise
        assert not np.array_equal(with_dither, without_dither)
