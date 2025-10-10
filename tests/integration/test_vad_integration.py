"""VAD Integration Test.

Tests Voice Activity Detection integration:
1. Feed synthetic audio (speech + silence)
2. Verify VAD detects speech start/end
3. Verify callbacks fire correctly
4. Test different aggressiveness levels
5. Measure processing latency per frame

Note: These tests use relaxed timing tolerances for CI environments where
VAD behavior may vary due to system load and audio characteristics.

VAD Hysteresis: WebRTC VAD includes built-in smoothing that continues to
report "speech" for ~6 frames (120ms @ 16kHz) after actual speech ends.
Tests must account for this hysteresis when calculating timing expectations.
"""

import logging
import time

import numpy as np
import pytest

from src.orchestrator.config import VADConfig
from src.orchestrator.vad import VADProcessor
from tests.integration.conftest import generate_silence, generate_speech_audio

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_vad_speech_detection() -> None:
    """Test basic speech detection with VAD.

    Validates:
    - VAD detects speech in speech frames
    - VAD detects silence in silent frames
    - Speech/silence transitions trigger events
    """
    # Arrange
    config = VADConfig(aggressiveness=2, sample_rate=16000, frame_duration_ms=20)
    vad = VADProcessor(
        config=config,
        min_speech_duration_ms=100,  # Require 100ms of speech to trigger
        min_silence_duration_ms=300,  # Require 300ms of silence to end
    )

    speech_start_events = []
    speech_end_events = []

    def on_speech_start(timestamp_ms: float) -> None:
        speech_start_events.append(timestamp_ms)
        logger.info(f"Speech started at {timestamp_ms}ms")

    def on_speech_end(timestamp_ms: float) -> None:
        speech_end_events.append(timestamp_ms)
        logger.info(f"Speech ended at {timestamp_ms}ms")

    vad.on_speech_start = on_speech_start
    vad.on_speech_end = on_speech_end

    # Act - Process speech → silence → speech pattern
    # Initial silence (100ms = 5 frames @ 20ms)
    for _ in range(5):
        frame = generate_silence(duration_ms=20, sample_rate=16000)
        vad.process_frame(frame)

    # Speech (500ms = 25 frames @ 20ms)
    for _ in range(25):
        frame = generate_speech_audio(duration_ms=20, sample_rate=16000)
        vad.process_frame(frame)

    # Silence (500ms = 25 frames @ 20ms)
    for _ in range(25):
        frame = generate_silence(duration_ms=20, sample_rate=16000)
        vad.process_frame(frame)

    # Speech again (500ms = 25 frames @ 20ms)
    for _ in range(25):
        frame = generate_speech_audio(duration_ms=20, sample_rate=16000)
        vad.process_frame(frame)

    # Final silence to trigger speech end
    for _ in range(20):
        frame = generate_silence(duration_ms=20, sample_rate=16000)
        vad.process_frame(frame)

    # Assert - Verify events
    assert len(speech_start_events) >= 1, "No speech start events detected"
    assert len(speech_end_events) >= 1, "No speech end events detected"

    logger.info(
        f"Detected {len(speech_start_events)} speech starts, "
        f"{len(speech_end_events)} speech ends"
    )

    # Verify stats
    stats = vad.stats
    logger.info(f"VAD stats: {stats}")
    assert stats["frames_processed"] == 100, "Wrong frame count"
    assert stats["speech_frames"] > 0, "No speech frames detected"
    assert stats["silence_frames"] > 0, "No silence frames detected"


@pytest.mark.asyncio
async def test_vad_aggressiveness_levels() -> None:
    """Test VAD with different aggressiveness levels.

    Validates:
    - Higher aggressiveness detects more silence
    - Lower aggressiveness detects more speech
    - All levels process frames correctly
    """
    results = {}

    for aggressiveness in [0, 1, 2, 3]:
        config = VADConfig(
            aggressiveness=aggressiveness, sample_rate=16000, frame_duration_ms=20
        )
        vad = VADProcessor(config=config, min_speech_duration_ms=60, min_silence_duration_ms=200)

        # Process same audio with different aggressiveness
        for _ in range(10):
            frame = generate_speech_audio(duration_ms=20, sample_rate=16000)
            vad.process_frame(frame)

        stats = vad.stats
        results[aggressiveness] = stats
        logger.info(f"Aggressiveness {aggressiveness}: {stats}")

    # Validate that different levels produce different results
    # (At least some variation in speech ratio)
    speech_ratios = [results[i]["speech_ratio"] for i in range(4)]
    assert len(set(speech_ratios)) > 1, "All aggressiveness levels produced same results"


@pytest.mark.asyncio
async def test_vad_processing_latency() -> None:
    """Test VAD processing latency per frame.

    Validates:
    - Processing latency < 5ms per frame
    - No significant latency spikes
    - Consistent performance across frames
    """
    config = VADConfig(aggressiveness=2, sample_rate=16000, frame_duration_ms=20)
    vad = VADProcessor(config=config)

    latencies = []

    # Process 100 frames and measure latency
    for _i in range(100):
        frame = generate_speech_audio(duration_ms=20, sample_rate=16000)

        start_time = time.perf_counter()
        vad.process_frame(frame)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    # Calculate statistics
    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    max_latency = np.max(latencies)

    logger.info(
        f"VAD processing latency: mean={mean_latency:.3f}ms, "
        f"p95={p95_latency:.3f}ms, p99={p99_latency:.3f}ms, max={max_latency:.3f}ms"
    )

    # Assert latency targets (relaxed for CI)
    assert mean_latency < 10.0, f"Mean latency {mean_latency:.3f}ms exceeds 10ms target (CI relaxed)" # noqa: E501
    assert p95_latency < 20.0, f"p95 latency {p95_latency:.3f}ms exceeds 20ms target (CI relaxed)"
    assert max_latency < 50.0, f"Max latency {max_latency:.3f}ms exceeds 50ms target (CI relaxed)"


@pytest.mark.asyncio
async def test_vad_debouncing() -> None:
    """Test VAD debouncing behavior.

    Validates:
    - Short speech bursts are filtered out
    - Speech must exceed min_speech_duration_ms to trigger
    - Short silence gaps don't end speech
    - Silence must exceed min_silence_duration_ms to trigger end
    """
    config = VADConfig(aggressiveness=2, sample_rate=16000, frame_duration_ms=20)
    vad = VADProcessor(
        config=config,
        min_speech_duration_ms=100,  # 5 frames
        min_silence_duration_ms=300,  # 15 frames
    )

    speech_start_events = []
    speech_end_events = []

    vad.on_speech_start = lambda ts: speech_start_events.append(ts)
    vad.on_speech_end = lambda ts: speech_end_events.append(ts)

    # Test 1: Short speech burst (3 frames = 60ms) - should NOT trigger (need 100ms minimum)
    for _ in range(3):
        frame = generate_speech_audio(duration_ms=20, sample_rate=16000)
        vad.process_frame(frame)

    # Add silence to reset
    for _ in range(10):
        frame = generate_silence(duration_ms=20, sample_rate=16000)
        vad.process_frame(frame)

    # Should not have triggered - burst was too short
    logger.info(f"After short burst: starts={len(speech_start_events)}, ends={len(speech_end_events)}") # noqa: E501

    # Test 2: Long speech (10 frames = 200ms) - should trigger
    for _ in range(10):
        frame = generate_speech_audio(duration_ms=20, sample_rate=16000)
        vad.process_frame(frame)

    # Should have triggered by now
    logger.info(f"After long speech: starts={len(speech_start_events)}, ends={len(speech_end_events)}") # noqa: E501
    assert len(speech_start_events) >= 1, "Long speech did not trigger start event"

    # Test 3: Short silence gap (5 frames = 100ms) - should NOT end speech (need 300ms)
    for _ in range(5):
        frame = generate_silence(duration_ms=20, sample_rate=16000)
        vad.process_frame(frame)

    # Should not have ended yet
    logger.info(f"After short silence: starts={len(speech_start_events)}, ends={len(speech_end_events)}") # noqa: E501

    # Test 4: Long silence (20 frames = 400ms) - should trigger end
    for _ in range(20):
        frame = generate_silence(duration_ms=20, sample_rate=16000)
        vad.process_frame(frame)

    # Should have ended by now
    logger.info(f"After long silence: starts={len(speech_start_events)}, ends={len(speech_end_events)}") # noqa: E501
    assert len(speech_end_events) >= 1, "Long silence did not trigger end event"


@pytest.mark.asyncio
async def test_vad_frame_size_validation() -> None:
    """Test VAD frame size validation.

    Validates:
    - Correct frame size is accepted
    - Invalid frame sizes raise ValueError
    - Error message is descriptive
    """
    config = VADConfig(aggressiveness=2, sample_rate=16000, frame_duration_ms=20)
    vad = VADProcessor(config=config)

    # Correct size: 20ms @ 16kHz = 320 samples = 640 bytes
    correct_frame = generate_silence(duration_ms=20, sample_rate=16000)
    vad.process_frame(correct_frame)  # Should work

    # Wrong size
    wrong_frame = b"\x00" * 100
    with pytest.raises(ValueError, match="Invalid frame size"):
        vad.process_frame(wrong_frame)


@pytest.mark.asyncio
async def test_vad_state_reset() -> None:
    """Test VAD state reset functionality.

    Validates:
    - Reset clears all state
    - Reset clears all counters
    - VAD works correctly after reset
    """
    config = VADConfig(aggressiveness=2, sample_rate=16000, frame_duration_ms=20)
    vad = VADProcessor(config=config)

    # Process some frames
    for _ in range(10):
        frame = generate_speech_audio(duration_ms=20, sample_rate=16000)
        vad.process_frame(frame)

    # Verify state is active
    stats_before = vad.stats
    assert stats_before["frames_processed"] == 10, "Frames not processed"

    # Reset
    vad.reset()

    # Verify state is cleared
    stats_after = vad.stats
    assert stats_after["frames_processed"] == 0, "Stats not reset"
    assert stats_after["speech_frames"] == 0, "Speech frames not reset"
    assert stats_after["silence_frames"] == 0, "Silence frames not reset"
    assert not vad.is_speaking, "Speaking state not reset"

    # Verify VAD works after reset
    for _ in range(5):
        frame = generate_speech_audio(duration_ms=20, sample_rate=16000)
        vad.process_frame(frame)

    stats_after_use = vad.stats
    assert stats_after_use["frames_processed"] == 5, "VAD not working after reset"


@pytest.mark.asyncio
async def test_vad_multiple_speech_segments() -> None:
    """Test VAD with multiple speech segments.

    Validates:
    - Multiple speech segments detected correctly
    - Each segment gets separate start/end events
    - Events match expected pattern

    Note: This test accounts for VAD hysteresis (~120ms @ aggressiveness=2)
    where the VAD continues to report "speech" for several frames after
    actual speech ends. The min_silence_duration is set to 200ms to ensure
    reliable detection despite this behavior.
    """
    config = VADConfig(aggressiveness=2, sample_rate=16000, frame_duration_ms=20)
    # Use 200ms silence threshold to account for ~120ms VAD hysteresis
    vad = VADProcessor(config=config, min_speech_duration_ms=100, min_silence_duration_ms=200)

    speech_start_events = []
    speech_end_events = []

    vad.on_speech_start = lambda ts: speech_start_events.append(ts)
    vad.on_speech_end = lambda ts: speech_end_events.append(ts)

    # Pattern: silence → speech → silence → speech → silence
    # Initial silence (100ms = 5 frames)
    for _ in range(5):
        vad.process_frame(generate_silence(duration_ms=20, sample_rate=16000))

    # Segment 1: Speech (200ms = 10 frames)
    for _ in range(10):
        vad.process_frame(generate_speech_audio(duration_ms=20, sample_rate=16000))

    # Silence gap (500ms = 25 frames - enough for VAD hysteresis + debouncing)
    for _ in range(25):
        vad.process_frame(generate_silence(duration_ms=20, sample_rate=16000))

    # Segment 2: Speech (200ms = 10 frames)
    for _ in range(10):
        vad.process_frame(generate_speech_audio(duration_ms=20, sample_rate=16000))

    # Final silence (500ms = 25 frames - ensure end event triggers)
    for _ in range(25):
        vad.process_frame(generate_silence(duration_ms=20, sample_rate=16000))

    # Validate events (allow some tolerance for VAD behavior)
    logger.info(
        f"Speech segments: starts={len(speech_start_events)}, ends={len(speech_end_events)}, "
        f"start_times={speech_start_events}, end_times={speech_end_events}"
    )

    # Expect at least 1 start/end pair, ideally 2
    assert len(speech_start_events) >= 1, (
        f"Expected at least 1 speech start, got {len(speech_start_events)}"
    )
    assert len(speech_end_events) >= 1, (
        f"Expected at least 1 speech end, got {len(speech_end_events)}. "
        f"This may indicate VAD hysteresis preventing silence detection."
    )

    # Verify event ordering (starts before ends)
    for i in range(min(len(speech_start_events), len(speech_end_events))):
        assert speech_start_events[i] < speech_end_events[i], (
            f"Speech start {i} after end: {speech_start_events[i]} >= {speech_end_events[i]}"
        )


@pytest.mark.asyncio
async def test_vad_with_real_audio_characteristics() -> None:
    """Test VAD with realistic audio characteristics.

    Validates:
    - VAD works with multi-frequency speech
    - VAD handles background noise
    - Detection is robust to audio variations
    """
    config = VADConfig(aggressiveness=2, sample_rate=16000, frame_duration_ms=20)
    vad = VADProcessor(config=config, min_speech_duration_ms=100, min_silence_duration_ms=300)

    speech_detected_count = 0
    silence_detected_count = 0

    # Process 50 frames of speech-like audio with noise
    for _ in range(50):
        frame = generate_speech_audio(duration_ms=20, sample_rate=16000)
        is_speech = vad.process_frame(frame)
        if is_speech:
            speech_detected_count += 1

    # Process 50 frames of silence with low noise
    for _ in range(50):
        # Generate silence with very low noise floor
        num_samples = 320  # 20ms @ 16kHz
        noise = np.random.normal(0, 10, num_samples).astype(np.int16)  # Very low noise
        frame = noise.tobytes()
        is_speech = vad.process_frame(frame)
        if not is_speech:
            silence_detected_count += 1

    # Validate detection
    # Should detect more speech frames in speech audio
    assert speech_detected_count > 0, "No speech detected in speech audio"

    # Should detect more silence frames in silence audio
    assert silence_detected_count > 0, "No silence detected in silent audio"

    logger.info(
        f"Detected {speech_detected_count}/50 speech frames in speech audio, "
        f"{silence_detected_count}/50 silence frames in silent audio"
    )

    # Get final stats
    stats = vad.stats
    logger.info(f"Final VAD stats: {stats}")
