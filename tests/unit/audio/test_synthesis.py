"""Unit tests for audio synthesis utilities."""

import numpy as np
import pytest
from tts.audio.synthesis import (
    calculate_frame_count,
    calculate_pcm_byte_size,
    float32_to_int16_pcm,
    generate_silence,
    generate_sine_wave,
    generate_sine_wave_frames,
)


class TestFloat32ToInt16PCM:
    """Test float32 to int16 PCM conversion."""

    def test_converts_zero_amplitude(self) -> None:
        """Test conversion of zero amplitude audio."""
        audio = np.zeros(100, dtype=np.float32)
        result = float32_to_int16_pcm(audio)
        assert len(result) == 200  # 100 samples × 2 bytes
        assert result == b"\x00" * 200

    def test_converts_max_amplitude(self) -> None:
        """Test conversion of maximum positive amplitude."""
        audio = np.ones(10, dtype=np.float32)
        result = float32_to_int16_pcm(audio)
        # Convert back to verify
        int16_array = np.frombuffer(result, dtype=np.int16)
        assert np.all(int16_array == 32767)

    def test_converts_min_amplitude(self) -> None:
        """Test conversion of maximum negative amplitude."""
        audio = np.full(10, -1.0, dtype=np.float32)
        result = float32_to_int16_pcm(audio)
        int16_array = np.frombuffer(result, dtype=np.int16)
        assert np.all(int16_array == -32767)

    def test_clips_out_of_range_values(self) -> None:
        """Test that values outside [-1.0, 1.0] are clipped."""
        audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)
        result = float32_to_int16_pcm(audio)
        int16_array = np.frombuffer(result, dtype=np.int16)
        assert int16_array[0] == 32767  # Clipped to 1.0
        assert int16_array[1] == -32767  # Clipped to -1.0
        assert abs(int16_array[2] - 16383) <= 1  # 0.5 × 32767

    def test_raises_on_empty_array(self) -> None:
        """Test that empty array raises ValueError."""
        audio = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="Audio array cannot be empty"):
            float32_to_int16_pcm(audio)

    def test_preserves_little_endian_byte_order(self) -> None:
        """Test that output uses little-endian byte order."""
        audio = np.array([0.5], dtype=np.float32)
        result = float32_to_int16_pcm(audio)
        # Manually verify little-endian encoding
        int16_val = int(0.5 * 32767)
        expected = int16_val.to_bytes(2, byteorder="little", signed=True)
        assert result == expected


class TestGenerateSineWave:
    """Test sine wave generation."""

    def test_generates_correct_byte_size(self) -> None:
        """Test that generated audio has correct byte size."""
        # 100ms at 48kHz = 4800 samples × 2 bytes = 9600 bytes
        result = generate_sine_wave(frequency=440, duration_ms=100, sample_rate=48000)
        assert len(result) == 9600

    def test_generates_20ms_frame(self) -> None:
        """Test generation of standard 20ms frame."""
        # 20ms at 48kHz = 960 samples × 2 bytes = 1920 bytes
        result = generate_sine_wave(frequency=440, duration_ms=20, sample_rate=48000)
        assert len(result) == 1920

    def test_generates_valid_sine_wave_shape(self) -> None:
        """Test that generated audio is a valid sine wave."""
        # Generate one period of 100Hz sine wave at 4800Hz sample rate
        # Period = 1/100 = 10ms
        result = generate_sine_wave(frequency=100, duration_ms=10, sample_rate=4800)

        # Convert back to float for analysis
        int16_array = np.frombuffer(result, dtype=np.int16)
        float_audio = int16_array.astype(np.float32) / 32767.0

        # Check that values oscillate around zero
        assert float_audio.max() > 0.9
        assert float_audio.min() < -0.9
        assert abs(float_audio.mean()) < 0.1

    def test_zero_duration_returns_empty(self) -> None:
        """Test that zero duration returns empty bytes."""
        result = generate_sine_wave(frequency=440, duration_ms=0, sample_rate=48000)
        assert result == b""

    def test_raises_on_negative_frequency(self) -> None:
        """Test that negative frequency raises ValueError."""
        with pytest.raises(ValueError, match="Frequency must be positive"):
            generate_sine_wave(frequency=-100, duration_ms=100, sample_rate=48000)

    def test_raises_on_zero_frequency(self) -> None:
        """Test that zero frequency raises ValueError."""
        with pytest.raises(ValueError, match="Frequency must be positive"):
            generate_sine_wave(frequency=0, duration_ms=100, sample_rate=48000)

    def test_raises_on_frequency_above_nyquist(self) -> None:
        """Test that frequency above Nyquist limit raises ValueError."""
        with pytest.raises(ValueError, match="exceeds Nyquist limit"):
            generate_sine_wave(frequency=25000, duration_ms=100, sample_rate=48000)

    def test_raises_on_negative_duration(self) -> None:
        """Test that negative duration raises ValueError."""
        with pytest.raises(ValueError, match="Duration must be non-negative"):
            generate_sine_wave(frequency=440, duration_ms=-10, sample_rate=48000)

    def test_raises_on_invalid_sample_rate(self) -> None:
        """Test that invalid sample rate raises ValueError."""
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            generate_sine_wave(frequency=440, duration_ms=100, sample_rate=-48000)

    def test_different_frequencies_produce_different_output(self) -> None:
        """Test that different frequencies produce different waveforms."""
        audio_440 = generate_sine_wave(frequency=440, duration_ms=50, sample_rate=48000)
        audio_880 = generate_sine_wave(frequency=880, duration_ms=50, sample_rate=48000)
        assert audio_440 != audio_880


class TestGenerateSineWaveFrames:
    """Test framed sine wave generation."""

    def test_generates_correct_frame_count(self) -> None:
        """Test that correct number of frames is generated."""
        # 100ms / 20ms = 5 frames
        frames = generate_sine_wave_frames(
            frequency=440, duration_ms=100, sample_rate=48000, frame_duration_ms=20
        )
        assert len(frames) == 5

    def test_each_frame_has_correct_size(self) -> None:
        """Test that each frame has the correct byte size."""
        frames = generate_sine_wave_frames(
            frequency=440, duration_ms=100, sample_rate=48000, frame_duration_ms=20
        )
        # 20ms at 48kHz = 960 samples × 2 bytes = 1920 bytes
        for frame in frames:
            assert len(frame) == 1920

    def test_discards_incomplete_final_frame(self) -> None:
        """Test that incomplete final frame is discarded."""
        # 105ms / 20ms = 5 complete frames + 5ms partial
        frames = generate_sine_wave_frames(
            frequency=440, duration_ms=105, sample_rate=48000, frame_duration_ms=20
        )
        assert len(frames) == 5  # Partial frame discarded

    def test_zero_duration_returns_empty_list(self) -> None:
        """Test that zero duration returns empty list."""
        frames = generate_sine_wave_frames(
            frequency=440, duration_ms=0, sample_rate=48000, frame_duration_ms=20
        )
        assert frames == []

    def test_custom_frame_duration(self) -> None:
        """Test generation with custom frame duration."""
        # 100ms / 10ms = 10 frames
        frames = generate_sine_wave_frames(
            frequency=440, duration_ms=100, sample_rate=48000, frame_duration_ms=10
        )
        assert len(frames) == 10
        # 10ms at 48kHz = 480 samples × 2 bytes = 960 bytes
        for frame in frames:
            assert len(frame) == 960

    def test_raises_on_invalid_frame_duration(self) -> None:
        """Test that invalid frame duration raises ValueError."""
        with pytest.raises(ValueError, match="Frame duration must be positive"):
            generate_sine_wave_frames(
                frequency=440, duration_ms=100, sample_rate=48000, frame_duration_ms=-10
            )

    def test_frames_are_continuous(self) -> None:
        """Test that concatenated frames match non-framed generation."""
        # Generate framed and non-framed versions
        frames = generate_sine_wave_frames(
            frequency=440, duration_ms=100, sample_rate=48000, frame_duration_ms=20
        )
        continuous = generate_sine_wave(frequency=440, duration_ms=100, sample_rate=48000)

        # Concatenate frames
        concatenated = b"".join(frames)

        # Should match (up to any discarded partial frame)
        assert concatenated == continuous[: len(concatenated)]


class TestGenerateSilence:
    """Test silence generation."""

    def test_generates_correct_byte_size(self) -> None:
        """Test that silence has correct byte size."""
        # 20ms at 48kHz = 960 samples × 2 bytes = 1920 bytes
        result = generate_silence(duration_ms=20, sample_rate=48000)
        assert len(result) == 1920

    def test_all_bytes_are_zero(self) -> None:
        """Test that all bytes are zero."""
        result = generate_silence(duration_ms=20, sample_rate=48000)
        assert result == b"\x00" * 1920

    def test_zero_duration_returns_empty(self) -> None:
        """Test that zero duration returns empty bytes."""
        result = generate_silence(duration_ms=0, sample_rate=48000)
        assert result == b""

    def test_raises_on_negative_duration(self) -> None:
        """Test that negative duration raises ValueError."""
        with pytest.raises(ValueError, match="Duration must be non-negative"):
            generate_silence(duration_ms=-10, sample_rate=48000)

    def test_raises_on_invalid_sample_rate(self) -> None:
        """Test that invalid sample rate raises ValueError."""
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            generate_silence(duration_ms=20, sample_rate=-48000)


class TestCalculateFrameCount:
    """Test frame count calculation."""

    def test_calculates_exact_divisions(self) -> None:
        """Test calculation when duration divides evenly."""
        assert calculate_frame_count(duration_ms=100, frame_duration_ms=20) == 5
        assert calculate_frame_count(duration_ms=200, frame_duration_ms=20) == 10

    def test_rounds_down_partial_frames(self) -> None:
        """Test that partial frames are not counted."""
        assert calculate_frame_count(duration_ms=105, frame_duration_ms=20) == 5
        assert calculate_frame_count(duration_ms=199, frame_duration_ms=20) == 9

    def test_zero_duration_returns_zero(self) -> None:
        """Test that zero duration returns zero frames."""
        assert calculate_frame_count(duration_ms=0, frame_duration_ms=20) == 0

    def test_negative_duration_returns_zero(self) -> None:
        """Test that negative duration returns zero frames."""
        assert calculate_frame_count(duration_ms=-10, frame_duration_ms=20) == 0

    def test_raises_on_invalid_frame_duration(self) -> None:
        """Test that invalid frame duration raises ValueError."""
        with pytest.raises(ValueError, match="Frame duration must be positive"):
            calculate_frame_count(duration_ms=100, frame_duration_ms=0)

    def test_custom_frame_duration(self) -> None:
        """Test calculation with custom frame duration."""
        assert calculate_frame_count(duration_ms=100, frame_duration_ms=10) == 10
        assert calculate_frame_count(duration_ms=100, frame_duration_ms=25) == 4


class TestCalculatePCMByteSize:
    """Test PCM byte size calculation."""

    def test_calculates_mono_size(self) -> None:
        """Test calculation for mono audio."""
        # 20ms at 48kHz mono = 960 samples × 2 bytes = 1920 bytes
        assert calculate_pcm_byte_size(duration_ms=20, sample_rate=48000, channels=1) == 1920

    def test_calculates_stereo_size(self) -> None:
        """Test calculation for stereo audio."""
        # 20ms at 48kHz stereo = 960 samples × 2 channels × 2 bytes = 3840 bytes
        assert calculate_pcm_byte_size(duration_ms=20, sample_rate=48000, channels=2) == 3840

    def test_calculates_multichannel_size(self) -> None:
        """Test calculation for multichannel audio."""
        # 20ms at 48kHz 5.1 surround = 960 samples × 6 channels × 2 bytes = 11520 bytes
        assert calculate_pcm_byte_size(duration_ms=20, sample_rate=48000, channels=6) == 11520

    def test_zero_duration_returns_zero(self) -> None:
        """Test that zero duration returns zero bytes."""
        assert calculate_pcm_byte_size(duration_ms=0, sample_rate=48000, channels=1) == 0

    def test_raises_on_negative_duration(self) -> None:
        """Test that negative duration raises ValueError."""
        with pytest.raises(ValueError, match="Duration must be non-negative"):
            calculate_pcm_byte_size(duration_ms=-10, sample_rate=48000, channels=1)

    def test_raises_on_invalid_sample_rate(self) -> None:
        """Test that invalid sample rate raises ValueError."""
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            calculate_pcm_byte_size(duration_ms=20, sample_rate=-48000, channels=1)

    def test_raises_on_invalid_channels(self) -> None:
        """Test that invalid channel count raises ValueError."""
        with pytest.raises(ValueError, match="Channels must be positive"):
            calculate_pcm_byte_size(duration_ms=20, sample_rate=48000, channels=0)

    def test_different_sample_rates(self) -> None:
        """Test calculation with different sample rates."""
        # 20ms at 16kHz mono = 320 samples × 2 bytes = 640 bytes
        assert calculate_pcm_byte_size(duration_ms=20, sample_rate=16000, channels=1) == 640
        # 20ms at 96kHz mono = 1920 samples × 2 bytes = 3840 bytes
        assert calculate_pcm_byte_size(duration_ms=20, sample_rate=96000, channels=1) == 3840
