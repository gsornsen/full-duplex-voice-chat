"""Unit tests for audio framing utilities.

Tests the repacketize_to_20ms() function for converting variable-size audio
chunks into fixed 20ms frames required by WebRTC transport.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from tts.audio.framing import repacketize_to_20ms


# Fixtures for test data
@pytest.fixture
def audio_exact_frames() -> NDArray[np.int16]:
    """Generate audio that fits exactly into 2 frames (1920 samples).

    Returns:
        Audio samples as int16 array (1920 samples = 2 * 20ms @ 48kHz)
    """
    samples_per_frame = 960  # 20ms @ 48kHz
    audio = np.arange(1, samples_per_frame * 2 + 1, dtype=np.int16)
    return audio


@pytest.fixture
def audio_partial_frame() -> NDArray[np.int16]:
    """Generate audio that requires padding (1000 samples).

    Returns:
        Audio samples as int16 array (1 full frame + 40 partial samples)
    """
    audio = np.arange(1, 1001, dtype=np.int16)
    return audio


@pytest.fixture
def audio_single_sample() -> NDArray[np.int16]:
    """Generate minimal audio input (1 sample).

    Returns:
        Single audio sample
    """
    return np.array([100], dtype=np.int16)


@pytest.fixture
def silence_48k() -> NDArray[np.int16]:
    """Generate silence at 48kHz (960 samples = 20ms).

    Returns:
        Silence samples as int16 array
    """
    return np.zeros(960, dtype=np.int16)


# Test 1: Exact frames (no padding needed)
def test_repacketize_exact_frames(audio_exact_frames: NDArray[np.int16]) -> None:
    """Test repacketization with audio that fits exactly into frames.

    Verifies:
        - Correct number of frames (2)
        - Each frame is exactly 1920 bytes (960 samples * 2 bytes)
        - No padding needed
        - Frame content is correct
    """
    frames = repacketize_to_20ms(audio_exact_frames, sample_rate=48000)

    # Should produce exactly 2 frames
    assert len(frames) == 2, f"Expected 2 frames, got {len(frames)}"

    # Each frame should be 1920 bytes (960 samples * 2 bytes)
    expected_frame_size = 960 * 2
    for i, frame in enumerate(frames):
        assert len(frame) == expected_frame_size, (
            f"Frame {i} has size {len(frame)}, expected {expected_frame_size}"
        )

    # Verify frame content (decode and compare)
    frame0_samples = np.frombuffer(frames[0], dtype=np.int16)
    frame1_samples = np.frombuffer(frames[1], dtype=np.int16)

    assert np.array_equal(frame0_samples, audio_exact_frames[:960]), (
        "First frame content mismatch"
    )
    assert np.array_equal(frame1_samples, audio_exact_frames[960:]), (
        "Second frame content mismatch"
    )


# Test 2: Partial frame (requires padding)
def test_repacketize_partial_frame(audio_partial_frame: NDArray[np.int16]) -> None:
    """Test repacketization with audio that requires padding.

    Input: 1000 samples (1 full frame + 40 partial samples)
    Expected: 2 frames, second frame has 920 zeros appended

    Verifies:
        - Correct number of frames (2)
        - Each frame is exactly 1920 bytes
        - Second frame is zero-padded
        - Non-zero samples are preserved
    """
    frames = repacketize_to_20ms(audio_partial_frame, sample_rate=48000)

    # Should produce 2 frames (1 full + 1 partial with padding)
    assert len(frames) == 2, f"Expected 2 frames, got {len(frames)}"

    # Each frame should be 1920 bytes
    expected_frame_size = 960 * 2
    for i, frame in enumerate(frames):
        assert len(frame) == expected_frame_size, (
            f"Frame {i} has size {len(frame)}, expected {expected_frame_size}"
        )

    # Decode frames
    frame0_samples = np.frombuffer(frames[0], dtype=np.int16)
    frame1_samples = np.frombuffer(frames[1], dtype=np.int16)

    # First frame should match first 960 samples
    assert np.array_equal(frame0_samples, audio_partial_frame[:960]), (
        "First frame content mismatch"
    )

    # Second frame should have 40 non-zero samples followed by 920 zeros
    assert np.array_equal(frame1_samples[:40], audio_partial_frame[960:1000]), (
        "Second frame non-zero content mismatch"
    )
    assert np.all(frame1_samples[40:] == 0), (
        "Second frame should have zeros for padding"
    )


# Test 3: Single sample (requires extensive padding)
def test_repacketize_single_sample(audio_single_sample: NDArray[np.int16]) -> None:
    """Test repacketization with minimal input (1 sample).

    Expected: 1 frame with 1 sample + 959 zeros

    Verifies:
        - Correct number of frames (1)
        - Frame is exactly 1920 bytes
        - First sample matches input
        - Remaining samples are zero-padded
    """
    frames = repacketize_to_20ms(audio_single_sample, sample_rate=48000)

    # Should produce 1 frame
    assert len(frames) == 1, f"Expected 1 frame, got {len(frames)}"

    # Frame should be 1920 bytes
    expected_frame_size = 960 * 2
    assert len(frames[0]) == expected_frame_size, (
        f"Frame has size {len(frames[0])}, expected {expected_frame_size}"
    )

    # Decode frame
    frame_samples = np.frombuffer(frames[0], dtype=np.int16)

    # First sample should match input
    assert frame_samples[0] == audio_single_sample[0], (
        f"First sample {frame_samples[0]} != input {audio_single_sample[0]}"
    )

    # Remaining samples should be zero-padded
    assert np.all(frame_samples[1:] == 0), (
        "Remaining samples should be zero-padded"
    )


# Test 4: Empty input
def test_repacketize_empty() -> None:
    """Test edge case with empty audio input.

    Verifies:
        - No crash or exception
        - Returns empty list (no frames)
    """
    empty_audio = np.array([], dtype=np.int16)

    frames = repacketize_to_20ms(empty_audio, sample_rate=48000)

    assert len(frames) == 0, "Empty input should produce no frames"
    assert isinstance(frames, list), "Output should be a list"


# Test 5: Frame size validation (various input sizes)
def test_repacketize_frame_size() -> None:
    """Test that all frames have exactly the correct size.

    Tests various input sizes to ensure all frames are 1920 bytes,
    including the last frame which may require padding.

    Verifies:
        - All frames are exactly 1920 bytes (960 samples * 2)
        - No short frames (except last, which is padded)
        - Frame count is correct for input size
    """
    test_sizes = [100, 960, 1920, 5000]  # Various sizes
    expected_frame_size = 960 * 2  # bytes

    for size in test_sizes:
        audio = np.arange(1, size + 1, dtype=np.int16)
        frames = repacketize_to_20ms(audio, sample_rate=48000)

        # Calculate expected frame count
        expected_frame_count = (size + 959) // 960  # Ceiling division

        assert len(frames) == expected_frame_count, (
            f"For {size} samples, expected {expected_frame_count} frames, "
            f"got {len(frames)}"
        )

        # Verify all frames have exact size
        for i, frame in enumerate(frames):
            assert len(frame) == expected_frame_size, (
                f"Size {size}, frame {i}: expected {expected_frame_size} bytes, "
                f"got {len(frame)}"
            )


# Test 6: Byte format validation (little-endian int16)
def test_repacketize_byte_format() -> None:
    """Test that output bytes are correctly encoded as little-endian int16.

    Verifies:
        - Byte encoding is correct
        - Values can be decoded back to int16
        - Decoded values match original input
    """
    # Create known values
    known_values = np.array([100, 200, 300, 400, 500], dtype=np.int16)

    frames = repacketize_to_20ms(known_values, sample_rate=48000)

    # Should produce 1 frame (5 samples + 955 padding)
    assert len(frames) == 1, f"Expected 1 frame, got {len(frames)}"

    # Decode frame
    decoded = np.frombuffer(frames[0], dtype=np.int16)

    # Verify known values are preserved
    assert np.array_equal(decoded[:5], known_values), (
        "Decoded values do not match input"
    )

    # Verify little-endian encoding manually for first value (100)
    # 100 in int16 little-endian: 0x64 0x00 (100, 0)
    assert frames[0][0] == 0x64, "First byte should be 0x64 (100)"
    assert frames[0][1] == 0x00, "Second byte should be 0x00"

    # Verify padding is zeros
    assert np.all(decoded[5:] == 0), "Padding should be zeros"


# Test 7: Different sample rates
def test_repacketize_different_sample_rates() -> None:
    """Test repacketization with different sample rates.

    Verifies:
        - Frame size calculation works for various sample rates
        - 20ms duration is correct for each rate
        - Frame count matches expected value

    Sample rates tested:
        - 22050Hz: 441 samples per 20ms frame
        - 16000Hz: 320 samples per 20ms frame
        - 48000Hz: 960 samples per 20ms frame
    """
    test_cases = [
        (22050, 441),  # 22050 * 0.020 = 441 samples
        (16000, 320),  # 16000 * 0.020 = 320 samples
        (48000, 960),  # 48000 * 0.020 = 960 samples
    ]

    for sample_rate, expected_samples_per_frame in test_cases:
        # Create audio with exactly 2 frames worth of samples
        audio = np.arange(
            1, expected_samples_per_frame * 2 + 1, dtype=np.int16
        )

        frames = repacketize_to_20ms(audio, sample_rate=sample_rate)

        # Should produce exactly 2 frames
        assert len(frames) == 2, (
            f"Rate {sample_rate}Hz: expected 2 frames, got {len(frames)}"
        )

        # Each frame should have correct byte size
        expected_frame_size = expected_samples_per_frame * 2  # bytes
        for i, frame in enumerate(frames):
            assert len(frame) == expected_frame_size, (
                f"Rate {sample_rate}Hz, frame {i}: expected {expected_frame_size} "
                f"bytes, got {len(frame)}"
            )

        # Verify duration is 20ms per frame
        samples_per_frame = len(frames[0]) // 2  # bytes to samples
        duration_ms = (samples_per_frame / sample_rate) * 1000
        assert abs(duration_ms - 20.0) < 0.001, (
            f"Rate {sample_rate}Hz: frame duration {duration_ms}ms != 20ms"
        )


# Test 8: Custom frame durations
def test_repacketize_frame_duration() -> None:
    """Test repacketization with custom frame durations.

    Verifies:
        - Frame duration parameter is respected
        - Samples per frame calculation is correct
        - Output frame sizes match expected values

    Frame durations tested:
        - 10ms: 480 samples @ 48kHz
        - 20ms: 960 samples @ 48kHz
        - 30ms: 1440 samples @ 48kHz
    """
    sample_rate = 48000
    test_durations = [10, 20, 30]  # milliseconds

    for duration_ms in test_durations:
        # Calculate expected samples per frame
        expected_samples = int(sample_rate * duration_ms / 1000)

        # Create audio with exactly 2 frames worth of samples
        audio = np.arange(1, expected_samples * 2 + 1, dtype=np.int16)

        frames = repacketize_to_20ms(
            audio, sample_rate=sample_rate, frame_duration_ms=duration_ms
        )

        # Should produce exactly 2 frames
        assert len(frames) == 2, (
            f"Duration {duration_ms}ms: expected 2 frames, got {len(frames)}"
        )

        # Each frame should have correct byte size
        expected_frame_size = expected_samples * 2  # bytes
        for i, frame in enumerate(frames):
            assert len(frame) == expected_frame_size, (
                f"Duration {duration_ms}ms, frame {i}: expected "
                f"{expected_frame_size} bytes, got {len(frame)}"
            )

        # Verify actual duration matches expected
        actual_samples = len(frames[0]) // 2  # bytes to samples
        actual_duration_ms = (actual_samples / sample_rate) * 1000
        assert abs(actual_duration_ms - duration_ms) < 0.001, (
            f"Expected {duration_ms}ms, got {actual_duration_ms}ms"
        )
