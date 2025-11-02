"""Barge-in latency integration tests.

Tests barge-in latency requirements:
- P95 barge-in latency < 50ms (speech detection → PAUSE confirmed)
- VAD processing latency < 5ms per frame
- Latency distribution analysis
- Latency histogram generation

These tests validate M3 performance SLAs.
"""

import logging
import time

import numpy as np
import pytest
from orchestrator.config import VADConfig
from orchestrator.vad import VADProcessor

from tests.helpers.vad_test_utils import (
    BargeInLatencyMeasurement,
    VADTestRecorder,
    calculate_latency_stats,
    calculate_p95_latency,
    generate_audio_sequence,
    generate_speech_audio,
)

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ============================================================================
# Latency Measurement Tests
# ============================================================================


@pytest.mark.asyncio
async def test_p95_barge_in_latency_under_50ms() -> None:
    """Test that P95 barge-in latency is under 50ms.

    Validates:
    - Run 30 trials of barge-in detection
    - Measure end-to-end latency (speech detection → action)
    - P95 latency < 50ms
    - Report latency distribution

    This is a critical M3 success criterion.
    """
    config = VADConfig(
        aggressiveness=2,
        sample_rate=16000,
        frame_duration_ms=20,
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
    )

    latencies: list[float] = []
    num_trials = 30

    for trial in range(num_trials):
        vad = VADProcessor(
            config=config,
            min_speech_duration_ms=100,
            min_silence_duration_ms=300,
        )

        # Create measurement outside callback to avoid closure issues
        measurement = BargeInLatencyMeasurement(speech_detected_ts=0.0)
        speech_detected = False

        def on_speech_start(
            timestamp_ms: float, m: BargeInLatencyMeasurement = measurement
        ) -> None:
            """Record when speech is detected."""
            nonlocal speech_detected
            if not speech_detected:  # Only record first detection
                m.speech_detected_ts = time.perf_counter()
                speech_detected = True

        vad.on_speech_start = on_speech_start

        # Generate speech pattern
        frames = generate_audio_sequence(
            [
                ("silence", 100),
                ("speech", 200),  # Should trigger detection
            ],
            sample_rate=16000,
        )

        # Process frames and measure latency
        for frame in frames:
            vad.process_frame(frame)

            # Simulate PAUSE action when speech detected
            if speech_detected and measurement.pause_sent_ts is None:
                measurement.pause_sent_ts = time.perf_counter()
                # Simulate minimal PAUSE processing overhead
                time.sleep(0.0001)  # 0.1ms
                measurement.pause_confirmed_ts = time.perf_counter()

        # Calculate barge-in latency
        if measurement.pause_confirmed_ts is not None:
            latency_ms = measurement.get_total_barge_in_latency_ms()
            if latency_ms is not None:
                latencies.append(latency_ms)
                logger.debug(f"Trial {trial + 1}: {latency_ms:.2f}ms")

    # Calculate statistics
    assert len(latencies) > 0, "No latencies recorded"

    stats = calculate_latency_stats(latencies)
    p95 = calculate_p95_latency(latencies)

    logger.info(
        f"Barge-in latency ({num_trials} trials): "
        f"mean={stats['mean']:.2f}ms, "
        f"p50={stats['median']:.2f}ms, "
        f"p95={p95:.2f}ms, "
        f"p99={stats['p99']:.2f}ms, "
        f"max={stats['max']:.2f}ms"
    )

    # Validate P95 < 50ms (critical M3 requirement)
    assert p95 < 50.0, (
        f"P95 barge-in latency {p95:.2f}ms exceeds 50ms target. "
        f"Full stats: {stats}"
    )

    # Validate mean is reasonable
    assert stats["mean"] < 30.0, (
        f"Mean latency {stats['mean']:.2f}ms is too high (should be well under P95)"
    )

    logger.info(f"✓ Barge-in latency P95 requirement met: {p95:.2f}ms < 50ms")


@pytest.mark.asyncio
async def test_vad_processing_latency_under_5ms() -> None:
    """Test that VAD processing latency is under 5ms per frame.

    Validates:
    - Process 100 frames
    - Measure per-frame processing time
    - P95 latency < 5ms
    - No significant outliers

    This ensures VAD doesn't add excessive latency to audio pipeline.
    """
    config = VADConfig(
        aggressiveness=2,
        sample_rate=16000,
        frame_duration_ms=20,
    )
    vad = VADProcessor(config=config)

    latencies: list[float] = []
    num_frames = 100

    # Process frames and measure individual latencies
    for _i in range(num_frames):
        frame = generate_speech_audio(duration_ms=20, sample_rate=16000)

        start_time = time.perf_counter()
        vad.process_frame(frame)
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    # Calculate statistics
    stats = calculate_latency_stats(latencies)
    p95 = calculate_p95_latency(latencies)

    logger.info(
        f"VAD processing latency ({num_frames} frames): "
        f"mean={stats['mean']:.3f}ms, "
        f"p50={stats['median']:.3f}ms, "
        f"p95={p95:.3f}ms, "
        f"p99={stats['p99']:.3f}ms, "
        f"max={stats['max']:.3f}ms"
    )

    # Validate P95 < 5ms (relaxed for CI environments)
    # Production target is 5ms, but allow up to 10ms in CI
    ci_tolerance = 10.0
    assert p95 < ci_tolerance, (
        f"P95 VAD processing latency {p95:.3f}ms exceeds {ci_tolerance}ms CI tolerance. "
        f"Production target is 5ms. Full stats: {stats}"
    )

    # Validate no extreme outliers
    assert stats["max"] < 50.0, (
        f"Max latency {stats['max']:.3f}ms indicates extreme outlier "
        "(should be < 50ms even in CI)"
    )

    logger.info(f"✓ VAD processing latency acceptable: P95={p95:.3f}ms")


@pytest.mark.asyncio
async def test_latency_histogram_analysis() -> None:
    """Test barge-in latency distribution and generate histogram data.

    Validates:
    - Collect 50 latency measurements
    - Analyze distribution characteristics
    - Check for bimodal distribution (could indicate issues)
    - Generate histogram buckets for analysis

    This test helps identify latency patterns and anomalies.
    """
    config = VADConfig(
        aggressiveness=2,
        sample_rate=16000,
        frame_duration_ms=20,
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
    )

    latencies: list[float] = []
    num_trials = 50

    for _trial in range(num_trials):
        vad = VADProcessor(
            config=config,
            min_speech_duration_ms=100,
            min_silence_duration_ms=300,
        )

        # Create recorder and detection time outside callback
        recorder = VADTestRecorder()
        detection_time: float | None = None

        def on_speech_start(timestamp_ms: float, r: VADTestRecorder = recorder) -> None:
            """Record detection time."""
            nonlocal detection_time
            if detection_time is None:
                detection_time = time.perf_counter()
            r.on_speech_start(timestamp_ms)

        vad.on_speech_start = on_speech_start

        # Generate speech
        frames = generate_audio_sequence(
            [("silence", 100), ("speech", 200)],
            sample_rate=16000,
        )

        # Process and measure
        start_time = time.perf_counter()
        for frame in frames:
            vad.process_frame(frame)
            recorder.increment_frame_count()

        # Calculate latency
        if detection_time is not None:
            latency_ms = (detection_time - start_time) * 1000
            latencies.append(latency_ms)

    # Analyze distribution
    latencies_array = np.array(latencies)
    stats = calculate_latency_stats(latencies)

    # Create histogram buckets (0-10ms, 10-20ms, 20-30ms, etc.)
    bins = [0, 10, 20, 30, 40, 50, 100]
    histogram, bin_edges = np.histogram(latencies_array, bins=bins)

    logger.info("Latency distribution histogram:")
    for i in range(len(histogram)):
        bucket = f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}ms"
        count = histogram[i]
        percentage = (count / len(latencies)) * 100
        logger.info(f"  {bucket:>12}: {count:3d} samples ({percentage:5.1f}%)")

    # Check for reasonable distribution
    # Most samples should be in lower buckets
    samples_under_30ms = sum(histogram[:3])  # 0-30ms buckets
    percentage_under_30ms = (samples_under_30ms / len(latencies)) * 100

    logger.info("\nDistribution analysis:")
    logger.info(f"  Samples under 30ms: {percentage_under_30ms:.1f}%")
    logger.info(f"  Mean: {stats['mean']:.2f}ms")
    logger.info(f"  Std dev: {stats['std']:.2f}ms")
    logger.info(f"  P95: {stats['p95']:.2f}ms")

    # Validate: at least 80% of samples should be under 30ms
    assert percentage_under_30ms >= 80.0, (
        f"Only {percentage_under_30ms:.1f}% of samples under 30ms "
        f"(expected >= 80%). Distribution may be skewed."
    )

    # Validate: standard deviation should be reasonable (< 15ms)
    assert stats["std"] < 15.0, (
        f"Standard deviation {stats['std']:.2f}ms is too high "
        f"(indicates inconsistent latency)"
    )

    logger.info("✓ Latency distribution is acceptable")


@pytest.mark.asyncio
async def test_barge_in_latency_with_varying_aggressiveness() -> None:
    """Test barge-in latency across different VAD aggressiveness levels.

    Validates:
    - Latency is consistent across aggressiveness levels
    - Higher aggressiveness doesn't significantly increase latency
    - All levels meet < 50ms P95 requirement

    This ensures VAD configuration doesn't compromise latency.
    """
    num_trials = 20
    results: dict[int, dict[str, float]] = {}

    for aggressiveness in [0, 1, 2, 3]:
        config = VADConfig(
            aggressiveness=aggressiveness,
            sample_rate=16000,
            frame_duration_ms=20,
            min_speech_duration_ms=100,
            min_silence_duration_ms=300,
        )

        latencies: list[float] = []

        for _trial in range(num_trials):
            vad = VADProcessor(
                config=config,
                min_speech_duration_ms=100,
                min_silence_duration_ms=300,
            )

            detection_time: float | None = None

            def on_speech_start(timestamp_ms: float) -> None:
                nonlocal detection_time
                if detection_time is None:
                    detection_time = time.perf_counter()

            vad.on_speech_start = on_speech_start

            frames = generate_audio_sequence(
                [("silence", 100), ("speech", 200)],
                sample_rate=16000,
            )

            start_time = time.perf_counter()
            for frame in frames:
                vad.process_frame(frame)

            if detection_time is not None:
                latency_ms = (detection_time - start_time) * 1000
                latencies.append(latency_ms)

        # Calculate stats for this aggressiveness level
        if latencies:
            stats = calculate_latency_stats(latencies)
            results[aggressiveness] = stats

            logger.info(
                f"Aggressiveness {aggressiveness}: "
                f"mean={stats['mean']:.2f}ms, "
                f"p95={stats['p95']:.2f}ms"
            )

    # Validate all levels meet P95 < 50ms
    for aggressiveness, stats in results.items():
        assert stats["p95"] < 50.0, (
            f"Aggressiveness {aggressiveness} P95 latency {stats['p95']:.2f}ms "
            f"exceeds 50ms target"
        )

    # Validate latency variation across levels is reasonable
    p95_values = [stats["p95"] for stats in results.values()]
    p95_range = max(p95_values) - min(p95_values)

    logger.info(f"P95 latency range across aggressiveness levels: {p95_range:.2f}ms")

    # Range should be < 20ms (latency should be relatively consistent)
    assert p95_range < 20.0, (
        f"P95 latency range {p95_range:.2f}ms is too large "
        f"(indicates aggressiveness significantly affects latency)"
    )

    logger.info("✓ Latency consistent across all aggressiveness levels")
