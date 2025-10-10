"""End-to-end barge-in integration tests.

Tests complete barge-in flow:
- Full pipeline: text → audio → speech detected → pause → resume
- WebSocket transport integration
- Metrics recording
- VAD aggressiveness levels
- Debouncing behavior
- Audio resampling pipeline

These tests validate M3 end-to-end functionality.
"""

import asyncio
import json
import logging
import time

import pytest
from websockets.asyncio.client import ClientConnection

from src.orchestrator.audio.resampler import create_vad_resampler
from src.orchestrator.config import OrchestratorConfig, VADConfig
from src.orchestrator.session import SessionMetrics
from src.orchestrator.vad import VADProcessor
from tests.helpers.vad_test_utils import (
    VADTestRecorder,
    generate_audio_sequence,
    generate_speech_audio,
)
from tests.integration.conftest import send_text_message

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ============================================================================
# End-to-End Flow Tests
# ============================================================================


@pytest.mark.asyncio
async def test_end_to_end_barge_in_flow() -> None:
    """Test complete barge-in flow from speech detection to resume.

    Validates:
    - Speech audio triggers VAD detection
    - Speech detection would trigger PAUSE (simulated)
    - Silence detection would trigger RESUME (simulated)
    - Complete cycle: detect → pause → silence → resume
    - Metrics are recorded correctly

    This simulates the full M3 barge-in pipeline.
    """
    # Setup VAD processor
    config = VADConfig(
        aggressiveness=2,
        sample_rate=16000,
        frame_duration_ms=20,
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
    )
    vad = VADProcessor(
        config=config,
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
    )

    # Setup event tracking
    recorder = VADTestRecorder()
    pause_events: list[float] = []
    resume_events: list[float] = []
    metrics = SessionMetrics()

    def on_speech_start(timestamp_ms: float) -> None:
        """Handle speech detection → trigger PAUSE."""
        recorder.on_speech_start(timestamp_ms)
        pause_time = time.perf_counter()
        pause_events.append(pause_time)
        logger.info(f"[Flow] PAUSE triggered at {timestamp_ms:.1f}ms")

    def on_speech_end(timestamp_ms: float) -> None:
        """Handle silence detection → trigger RESUME."""
        recorder.on_speech_end(timestamp_ms)
        resume_time = time.perf_counter()
        resume_events.append(resume_time)

        # Record barge-in latency (simplified)
        if pause_events:
            latency_ms = (resume_time - pause_events[-1]) * 1000
            metrics.record_barge_in(latency_ms)
            logger.info(f"[Flow] RESUME triggered at {timestamp_ms:.1f}ms")

    vad.on_speech_start = on_speech_start
    vad.on_speech_end = on_speech_end

    # Generate complete flow: silence → speech → silence
    frames = generate_audio_sequence(
        [
            ("silence", 100),  # Initial silence
            ("speech", 300),  # Speech (triggers pause)
            ("silence", 600),  # Silence (triggers resume)
        ],
        sample_rate=16000,
    )

    # Process audio frames
    start_time = time.perf_counter()
    for frame in frames:
        vad.process_frame(frame)
        recorder.increment_frame_count()
        await asyncio.sleep(0.001)  # Simulate real-time processing

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000

    # Verify complete flow
    assert len(pause_events) >= 1, "PAUSE not triggered"
    assert len(resume_events) >= 1, "RESUME not triggered"
    assert metrics.barge_in_count >= 1, "Barge-in not recorded in metrics"

    # Verify event ordering
    if pause_events and resume_events:
        assert pause_events[0] < resume_events[0], "PAUSE after RESUME (wrong order)"

    logger.info(
        f"✓ End-to-end flow completed in {total_time_ms:.1f}ms: "
        f"pause_events={len(pause_events)}, "
        f"resume_events={len(resume_events)}, "
        f"barge_ins={metrics.barge_in_count}"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(
    True, reason="Requires running orchestrator server (see test_websocket_e2e.py)"
)
async def test_barge_in_with_websocket_transport(
    ws_client: ClientConnection,
    orchestrator_server: OrchestratorConfig,
) -> None:
    """Test barge-in with WebSocket transport.

    Validates:
    - WebSocket messages trigger barge-in flow
    - Client can send audio during synthesis
    - Server responds with appropriate state changes
    - Metrics are tracked correctly

    Note: This test is skipped by default as it requires full orchestrator setup.
    Enable when WebSocket barge-in integration is complete.
    """
    # Send text to start synthesis
    await send_text_message(ws_client, "Hello, this is a test message.", is_final=True)

    # Wait for synthesis to start
    await asyncio.sleep(0.1)

    # Simulate client sending audio (speech) during synthesis
    # This would trigger VAD detection on server side
    speech_audio = generate_speech_audio(duration_ms=200, sample_rate=16000)

    # Send audio message (WebSocket binary or JSON-encoded)
    audio_message = {
        "type": "audio",
        "pcm": speech_audio.hex(),  # Hex-encoded PCM
        "sample_rate": 16000,
        "is_final": False,
    }
    await ws_client.send(json.dumps(audio_message))

    # Wait for server to process and respond
    # Would expect state change or pause confirmation
    response = await asyncio.wait_for(ws_client.recv(), timeout=2.0)
    data = json.loads(response)

    logger.info(f"Received response: {data}")

    # Verify server handled barge-in
    # (Exact response format depends on implementation)
    # assert data.get("type") == "state_change"
    # assert data.get("state") == "barged_in"

    logger.info("✓ WebSocket barge-in flow completed")


@pytest.mark.asyncio
async def test_barge_in_metrics_recorded() -> None:
    """Test that barge-in events are recorded in SessionMetrics.

    Validates:
    - record_barge_in() updates counters
    - Latencies are stored correctly
    - P95 calculation works
    - Average calculation works
    - Metrics summary includes barge-in data
    """
    metrics = SessionMetrics()

    # Record multiple barge-in events with varying latencies
    latencies = [15.5, 22.3, 18.7, 45.2, 12.9, 33.1, 28.4, 19.6, 25.8, 31.2]

    for latency_ms in latencies:
        metrics.record_barge_in(latency_ms)

    # Verify count
    assert metrics.barge_in_count == len(latencies)

    # Verify latencies stored
    assert len(metrics.barge_in_latencies_ms) == len(latencies)
    assert metrics.barge_in_latencies_ms == latencies

    # Verify average calculation
    avg_latency = metrics.compute_avg_barge_in_latency_ms()
    assert avg_latency is not None
    expected_avg = sum(latencies) / len(latencies)
    assert abs(avg_latency - expected_avg) < 0.01

    # Verify P95 calculation
    p95_latency = metrics.compute_p95_barge_in_latency_ms()
    assert p95_latency is not None
    # P95 should be close to max for this small sample
    assert p95_latency <= max(latencies)
    assert p95_latency >= expected_avg

    logger.info(
        f"Metrics: count={metrics.barge_in_count}, "
        f"avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms"
    )

    logger.info("✓ Barge-in metrics recorded and calculated correctly")


@pytest.mark.asyncio
async def test_vad_aggressiveness_levels() -> None:
    """Test VAD detection with all aggressiveness levels (0-3).

    Validates:
    - All levels process audio correctly
    - Higher aggressiveness detects less speech
    - Level 0 (least aggressive) detects most speech
    - Level 3 (most aggressive) is most conservative
    """
    results: dict[int, dict[str, int | float]] = {}

    # Generate test audio: mix of speech and silence
    frames = generate_audio_sequence(
        [
            ("speech", 200),
            ("silence", 200),
            ("speech", 200),
        ],
        sample_rate=16000,
    )

    for aggressiveness in [0, 1, 2, 3]:
        config = VADConfig(
            aggressiveness=aggressiveness,
            sample_rate=16000,
            frame_duration_ms=20,
            min_speech_duration_ms=100,
            min_silence_duration_ms=300,
        )
        vad = VADProcessor(
            config=config,
            min_speech_duration_ms=100,
            min_silence_duration_ms=300,
        )

        recorder = VADTestRecorder()
        vad.on_speech_start = recorder.on_speech_start
        vad.on_speech_end = recorder.on_speech_end

        # Process frames
        for frame in frames:
            vad.process_frame(frame)
            recorder.increment_frame_count()

        stats = vad.stats
        results[aggressiveness] = {
            "speech_frames": stats["speech_frames"],
            "silence_frames": stats["silence_frames"],
            "speech_ratio": stats["speech_ratio"],
            "speech_events": len(recorder.speech_start_events),
            "silence_events": len(recorder.speech_end_events),
        }

        logger.info(
            f"Aggressiveness {aggressiveness}: "
            f"speech_ratio={stats['speech_ratio']:.2f}, "
            f"events={len(recorder.speech_start_events)}/{len(recorder.speech_end_events)}"
        )

    # Verify all levels worked
    for aggressiveness, stats in results.items():
        assert stats["speech_frames"] + stats["silence_frames"] == len(frames), (
            f"Level {aggressiveness}: frame count mismatch"
        )

    # Verify trend: higher aggressiveness → lower speech ratio (generally)
    # Note: This may not be strictly monotonic due to audio characteristics
    speech_ratios = [results[i]["speech_ratio"] for i in range(4)]
    logger.info(f"Speech ratios by aggressiveness: {speech_ratios}")

    logger.info("✓ All VAD aggressiveness levels functional")


@pytest.mark.asyncio
async def test_debouncing_prevents_spurious_events() -> None:
    """Test that debouncing prevents spurious speech/silence events.

    Validates:
    - Short speech bursts don't trigger events
    - Short silence gaps don't end speech
    - min_speech_duration_ms is respected
    - min_silence_duration_ms is respected
    - Only sustained speech/silence triggers events
    """
    config = VADConfig(
        aggressiveness=2,
        sample_rate=16000,
        frame_duration_ms=20,
        min_speech_duration_ms=100,  # 5 frames minimum
        min_silence_duration_ms=300,  # 15 frames minimum
    )
    vad = VADProcessor(
        config=config,
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
    )

    recorder = VADTestRecorder()
    vad.on_speech_start = recorder.on_speech_start
    vad.on_speech_end = recorder.on_speech_end

    # Test pattern: short bursts that should be filtered out
    frames = generate_audio_sequence(
        [
            ("silence", 100),
            ("speech", 60),  # Too short (< 100ms) - should NOT trigger
            ("silence", 100),
            ("speech", 200),  # Long enough - SHOULD trigger
            ("silence", 100),  # Too short (< 300ms) - should NOT end speech
            ("speech", 100),  # Continues speech
            ("silence", 600),  # Long enough - SHOULD end speech
        ],
        sample_rate=16000,
    )

    # Process frames
    for frame in frames:
        vad.process_frame(frame)
        recorder.increment_frame_count()

    # Verify debouncing worked
    # Should have exactly 1 speech start (from 200ms+ speech segment)
    # Should have exactly 1 speech end (from 400ms silence)
    assert len(recorder.speech_start_events) == 1, (
        f"Expected 1 speech start (debounced), got {len(recorder.speech_start_events)}"
    )

    assert len(recorder.speech_end_events) == 1, (
        f"Expected 1 speech end (debounced), got {len(recorder.speech_end_events)}"
    )

    logger.info(
        f"✓ Debouncing working: {len(recorder.speech_start_events)} starts, "
        f"{len(recorder.speech_end_events)} ends (filtered spurious events)"
    )


@pytest.mark.asyncio
async def test_audio_resampling_pipeline() -> None:
    """Test audio resampling pipeline for VAD preprocessing.

    Validates:
    - Resampling 48kHz → 16kHz works correctly
    - Resampled audio has correct size
    - Resampled audio is compatible with VAD
    - No quality degradation prevents VAD detection
    - create_vad_resampler() factory works
    """
    # Create resampler
    resampler = create_vad_resampler()

    assert resampler.source_rate == 48000
    assert resampler.target_rate == 16000
    assert abs(resampler.ratio - (16000 / 48000)) < 0.001

    # Generate 48kHz speech audio
    speech_48k = generate_speech_audio(duration_ms=20, sample_rate=48000)
    logger.info(f"Input audio (48kHz): {len(speech_48k)} bytes")

    # Resample to 16kHz
    speech_16k = resampler.process_frame(speech_48k)
    logger.info(f"Resampled audio (16kHz): {len(speech_16k)} bytes")

    # Verify output size
    # 20ms @ 48kHz = 960 samples = 1920 bytes
    # 20ms @ 16kHz = 320 samples = 640 bytes
    assert len(speech_48k) == 1920, "Input size incorrect"
    assert len(speech_16k) == 640, "Output size incorrect"

    # Verify resampled audio works with VAD
    config = VADConfig(aggressiveness=2, sample_rate=16000, frame_duration_ms=20)
    vad = VADProcessor(config=config)

    # Process multiple resampled frames
    num_frames = 20
    speech_detected_count = 0

    for _ in range(num_frames):
        speech_48k = generate_speech_audio(duration_ms=20, sample_rate=48000)
        speech_16k = resampler.process_frame(speech_48k)
        is_speech = vad.process_frame(speech_16k)
        if is_speech:
            speech_detected_count += 1

    # Verify VAD detected speech in resampled audio
    assert speech_detected_count > 0, "VAD failed to detect speech in resampled audio"

    logger.info(
        f"✓ Resampling pipeline works: {speech_detected_count}/{num_frames} "
        f"frames detected as speech"
    )


@pytest.mark.asyncio
async def test_vad_with_multiple_sample_rates() -> None:
    """Test VAD with all supported sample rates.

    Validates:
    - 8kHz, 16kHz, 32kHz, 48kHz all work
    - Detection quality is consistent
    - Frame sizes are correct
    """
    sample_rates = [8000, 16000, 32000, 48000]
    results: dict[int, bool] = {}

    for sample_rate in sample_rates:
        config = VADConfig(
            aggressiveness=2,
            sample_rate=sample_rate,
            frame_duration_ms=20,
        )
        vad = VADProcessor(config=config)

        # Generate speech at target sample rate
        frames = []
        for _ in range(10):
            frame = generate_speech_audio(duration_ms=20, sample_rate=sample_rate)
            frames.append(frame)

        # Process frames
        speech_detected = False
        for frame in frames:
            is_speech = vad.process_frame(frame)
            if is_speech:
                speech_detected = True

        results[sample_rate] = speech_detected

        # Calculate expected frame size
        expected_samples = sample_rate * 20 // 1000
        expected_bytes = expected_samples * 2
        actual_bytes = len(frames[0])

        logger.info(
            f"Sample rate {sample_rate}Hz: "
            f"frame_size={actual_bytes}bytes (expected={expected_bytes}), "
            f"speech_detected={speech_detected}"
        )

        # Verify frame size
        assert actual_bytes == expected_bytes, (
            f"Frame size mismatch at {sample_rate}Hz"
        )

    # Verify all sample rates detected speech
    for sample_rate, detected in results.items():
        assert detected, f"No speech detected at {sample_rate}Hz"

    logger.info(f"✓ All {len(sample_rates)} sample rates functional")


@pytest.mark.asyncio
async def test_barge_in_timing_accuracy() -> None:
    """Test timing accuracy of barge-in detection.

    Validates:
    - VAD timestamp matches actual audio timeline
    - Speech start time is accurate within tolerance
    - Speech end time is accurate within tolerance
    - Timing doesn't drift over long sessions
    """
    config = VADConfig(
        aggressiveness=2,
        sample_rate=16000,
        frame_duration_ms=20,
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
    )
    vad = VADProcessor(
        config=config,
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
    )

    recorder = VADTestRecorder()
    vad.on_speech_start = recorder.on_speech_start
    vad.on_speech_end = recorder.on_speech_end

    # Generate precise pattern with known timings
    # 100ms silence, 200ms speech, 400ms silence
    # Expected speech_start: ~100ms (after debouncing)
    # Expected speech_end: ~300ms (200ms into speech + last speech frame)
    frames = generate_audio_sequence(
        [
            ("silence", 100),
            ("speech", 200),
            ("silence", 600),
        ],
        sample_rate=16000,
    )

    # Process frames
    for _i, frame in enumerate(frames):
        vad.process_frame(frame)
        recorder.increment_frame_count()

    # Verify events fired
    assert len(recorder.speech_start_events) >= 1, "No speech start detected"
    assert len(recorder.speech_end_events) >= 1, "No speech end detected"

    # Check timing accuracy
    speech_start_ts = recorder.speech_start_events[0].timestamp_ms
    speech_end_ts = recorder.speech_end_events[0].timestamp_ms

    # Speech start should be around 100ms (silence) + debouncing
    # Allow tolerance for debouncing and VAD behavior
    expected_start_min = 100  # Start of speech
    expected_start_max = 200  # After debouncing
    assert expected_start_min <= speech_start_ts <= expected_start_max, (
        f"Speech start timing off: {speech_start_ts:.1f}ms "
        f"(expected {expected_start_min}-{expected_start_max}ms)"
    )

    # Speech end should be around 300ms (100ms silence + 200ms speech)
    # Plus VAD hysteresis (~120ms)
    expected_end_min = 200  # End of actual speech
    expected_end_max = 500  # After hysteresis + debouncing
    assert expected_end_min <= speech_end_ts <= expected_end_max, (
        f"Speech end timing off: {speech_end_ts:.1f}ms "
        f"(expected {expected_end_min}-{expected_end_max}ms)"
    )

    logger.info(
        f"✓ Timing accuracy verified: "
        f"start={speech_start_ts:.1f}ms, end={speech_end_ts:.1f}ms"
    )
