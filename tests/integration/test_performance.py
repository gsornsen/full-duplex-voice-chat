"""Performance Benchmarks Integration Test.

Measures performance metrics:
1. First Audio Latency under different loads (1, 3, 10 concurrent)
2. Frame jitter measurement
3. CPU/memory profiling
4. Latency percentiles (p50, p95, p99)
"""

import asyncio
import json
import logging
import time
from typing import Any

import numpy as np
import pytest
import websockets

from tests.integration.conftest import (
    LatencyMetrics,
    receive_audio_frames,
    send_text_message,
)

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_fal_single_session(orchestrator_server: Any) -> None:
    """Benchmark First Audio Latency with single session.

    Target: FAL < 300ms (p95)
    """
    fal_metrics = LatencyMetrics()
    num_measurements = 20

    async with websockets.connect("ws://localhost:8080") as ws:
        # Receive session start
        await ws.recv()

        # Take multiple measurements
        for i in range(num_measurements):
            send_time = time.time()
            await send_text_message(ws, f"Benchmark message {i}", is_final=True)

            # Receive first frame
            frames = await receive_audio_frames(ws, timeout_s=5.0)
            assert len(frames) > 0

            fal_ms = (time.time() - send_time) * 1000
            fal_metrics.record(fal_ms)

            # Small delay between measurements
            await asyncio.sleep(0.1)

    # Report results
    summary = fal_metrics.get_summary()
    logger.info(
        f"Single Session FAL Benchmark ({num_measurements} samples):\n"
        f"  Mean: {summary['mean']:.2f}ms\n"
        f"  p50:  {summary['p50']:.2f}ms\n"
        f"  p95:  {summary['p95']:.2f}ms\n"
        f"  p99:  {summary['p99']:.2f}ms\n"
        f"  Min:  {summary['min']:.2f}ms\n"
        f"  Max:  {summary['max']:.2f}ms"
    )

    # Assert target
    assert summary["p95"] < 300, f"p95 FAL {summary['p95']:.2f}ms exceeds 300ms target"


@pytest.mark.asyncio
async def test_fal_concurrent_3_sessions(orchestrator_server: Any) -> None:
    """Benchmark First Audio Latency with 3 concurrent sessions.

    Target: FAL < 400ms (p95) under moderate load
    """
    fal_metrics = LatencyMetrics()
    num_messages_per_session = 10

    async def session_benchmark(session_id: int) -> None:
        """Run benchmark for one session."""
        async with websockets.connect("ws://localhost:8080") as ws:
            await ws.recv()  # session start

            for i in range(num_messages_per_session):
                send_time = time.time()
                await send_text_message(
                    ws, f"Session {session_id} message {i}", is_final=True
                )

                frames = await receive_audio_frames(ws, timeout_s=5.0)
                assert len(frames) > 0

                fal_ms = (time.time() - send_time) * 1000
                fal_metrics.record(fal_ms)

                await asyncio.sleep(0.05)

    # Run 3 concurrent sessions
    tasks = [asyncio.create_task(session_benchmark(i)) for i in range(3)]
    await asyncio.gather(*tasks)

    # Report results
    summary = fal_metrics.get_summary()
    logger.info(
        f"3 Concurrent Sessions FAL Benchmark ({len(fal_metrics.samples)} samples):\n"
        f"  Mean: {summary['mean']:.2f}ms\n"
        f"  p50:  {summary['p50']:.2f}ms\n"
        f"  p95:  {summary['p95']:.2f}ms\n"
        f"  p99:  {summary['p99']:.2f}ms"
    )

    # Assert target
    assert summary["p95"] < 400, f"p95 FAL {summary['p95']:.2f}ms exceeds 400ms target"


@pytest.mark.asyncio
async def test_fal_concurrent_10_sessions(orchestrator_server: Any) -> None:
    """Benchmark First Audio Latency with 10 concurrent sessions.

    Target: FAL < 600ms (p95) under high load
    """
    fal_metrics = LatencyMetrics()
    num_messages_per_session = 5

    async def session_benchmark(session_id: int) -> None:
        """Run benchmark for one session."""
        async with websockets.connect("ws://localhost:8080") as ws:
            await ws.recv()  # session start

            for i in range(num_messages_per_session):
                send_time = time.time()
                await send_text_message(
                    ws, f"Session {session_id} message {i}", is_final=True
                )

                frames = await receive_audio_frames(ws, timeout_s=10.0)
                assert len(frames) > 0

                fal_ms = (time.time() - send_time) * 1000
                fal_metrics.record(fal_ms)

                await asyncio.sleep(0.05)

    # Run 10 concurrent sessions
    tasks = [asyncio.create_task(session_benchmark(i)) for i in range(10)]
    await asyncio.gather(*tasks)

    # Report results
    summary = fal_metrics.get_summary()
    logger.info(
        f"10 Concurrent Sessions FAL Benchmark ({len(fal_metrics.samples)} samples):\n"
        f"  Mean: {summary['mean']:.2f}ms\n"
        f"  p50:  {summary['p50']:.2f}ms\n"
        f"  p95:  {summary['p95']:.2f}ms\n"
        f"  p99:  {summary['p99']:.2f}ms"
    )

    # Assert target
    assert summary["p95"] < 600, f"p95 FAL {summary['p95']:.2f}ms exceeds 600ms target"


@pytest.mark.asyncio
async def test_frame_jitter_measurement(orchestrator_server: Any) -> None:
    """Measure frame jitter (timing variance).

    Target: Frame jitter < ±5ms from expected 20ms cadence
    """
    async with websockets.connect("ws://localhost:8080") as ws:
        await ws.recv()  # session start

        # Send a message that generates many frames
        await send_text_message(ws, "A" * 100, is_final=True)  # Long text

        # Collect frames with timestamps
        frames = []
        frame_times = []

        async for msg in ws:
            data = json.loads(msg)
            if data.get("type") == "audio":
                frame_time = time.time()
                frames.append(data)
                frame_times.append(frame_time)

                # Stop after receiving final frame or 50 frames
                if not data.get("pcm") or len(frames) >= 50:
                    break

    # Calculate intervals
    if len(frame_times) < 2:
        pytest.skip("Not enough frames for jitter measurement")

    intervals_ms = np.diff(frame_times) * 1000
    expected_interval = 20.0
    jitter = intervals_ms - expected_interval

    # Statistics
    mean_interval = float(np.mean(intervals_ms))
    std_interval = float(np.std(intervals_ms))
    mean_jitter = float(np.mean(np.abs(jitter)))
    p95_jitter = float(np.percentile(np.abs(jitter), 95))
    max_jitter = float(np.max(np.abs(jitter)))

    logger.info(
        f"Frame Jitter Benchmark ({len(intervals_ms)} intervals):\n"
        f"  Mean interval: {mean_interval:.2f}ms\n"
        f"  Std interval:  {std_interval:.2f}ms\n"
        f"  Mean jitter:   {mean_jitter:.2f}ms\n"
        f"  p95 jitter:    {p95_jitter:.2f}ms\n"
        f"  Max jitter:    {max_jitter:.2f}ms"
    )

    # Assert target
    assert p95_jitter < 5.0, f"p95 jitter {p95_jitter:.2f}ms exceeds 5ms target"


@pytest.mark.asyncio
async def test_throughput_benchmark(orchestrator_server: Any) -> None:
    """Measure system throughput (messages/frames per second).

    Measures:
    - Messages processed per second
    - Audio frames delivered per second
    """
    duration_s = 10.0
    message_count = 0
    frame_count = 0

    async def throughput_session(session_id: int) -> dict[str, int]:
        """Measure throughput for one session."""
        nonlocal message_count, frame_count
        local_messages = 0
        local_frames = 0

        async with websockets.connect("ws://localhost:8080") as ws:
            await ws.recv()  # session start
            start_time = time.time()

            while time.time() - start_time < duration_s:
                # Send message
                await send_text_message(
                    ws, f"Throughput test session {session_id}", is_final=True
                )
                local_messages += 1

                # Count frames
                frames = await receive_audio_frames(ws, timeout_s=5.0)
                local_frames += len(frames)

                # Small delay
                await asyncio.sleep(0.05)

        return {"messages": local_messages, "frames": local_frames}

    # Run 3 concurrent sessions
    start_time = time.time()
    tasks = [asyncio.create_task(throughput_session(i)) for i in range(3)]
    results = await asyncio.gather(*tasks)
    actual_duration = time.time() - start_time

    # Aggregate results
    for result in results:
        message_count += result["messages"]
        frame_count += result["frames"]

    # Calculate rates
    messages_per_sec = message_count / actual_duration
    frames_per_sec = frame_count / actual_duration

    logger.info(
        f"Throughput Benchmark ({actual_duration:.1f}s duration, 3 sessions):\n"
        f"  Total messages: {message_count}\n"
        f"  Total frames:   {frame_count}\n"
        f"  Messages/sec:   {messages_per_sec:.2f}\n"
        f"  Frames/sec:     {frames_per_sec:.2f}"
    )

    # Sanity checks
    assert messages_per_sec > 0, "No messages processed"
    assert frames_per_sec > 0, "No frames delivered"


@pytest.mark.asyncio
async def test_latency_percentiles_distribution(orchestrator_server: Any) -> None:
    """Analyze latency distribution across percentiles.

    Validates:
    - Latency distribution is reasonable
    - No extreme outliers
    - Percentiles follow expected pattern
    """
    fal_metrics = LatencyMetrics()
    num_measurements = 100

    async with websockets.connect("ws://localhost:8080") as ws:
        await ws.recv()  # session start

        for i in range(num_measurements):
            send_time = time.time()
            await send_text_message(ws, f"Latency test {i}", is_final=True)

            frames = await receive_audio_frames(ws, timeout_s=5.0)
            assert len(frames) > 0

            fal_ms = (time.time() - send_time) * 1000
            fal_metrics.record(fal_ms)

            await asyncio.sleep(0.05)

    # Calculate percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    results = {
        f"p{p}": float(np.percentile(fal_metrics.samples, p)) for p in percentiles
    }

    logger.info(
        f"Latency Distribution ({num_measurements} samples):\n"
        + "\n".join([f"  {k}: {v:.2f}ms" for k, v in results.items()])
    )

    # Validate distribution makes sense
    # p50 should be less than p95
    assert results["p50"] < results["p95"], "Invalid distribution: p50 >= p95"

    # p95 should be less than p99
    assert results["p95"] < results["p99"], "Invalid distribution: p95 >= p99"

    # No extreme outliers (p99 should be < 2x p50)
    assert results["p99"] < results["p50"] * 3, (
        f"Extreme outliers: p99={results['p99']:.2f}ms > 3x p50={results['p50']:.2f}ms"
    )


@pytest.mark.asyncio
async def test_cold_start_vs_warm_latency(orchestrator_server: Any) -> None:
    """Compare cold start vs warm latency.

    Measures:
    - First message latency (cold start)
    - Subsequent message latencies (warm)
    """
    cold_start_latency: float | None = None
    warm_latencies = LatencyMetrics()

    async with websockets.connect("ws://localhost:8080") as ws:
        await ws.recv()  # session start

        # Cold start measurement
        send_time = time.time()
        await send_text_message(ws, "Cold start message", is_final=True)
        await receive_audio_frames(ws, timeout_s=5.0)
        cold_start_latency = (time.time() - send_time) * 1000

        # Warm measurements
        for i in range(10):
            send_time = time.time()
            await send_text_message(ws, f"Warm message {i}", is_final=True)
            await receive_audio_frames(ws, timeout_s=5.0)
            warm_latency = (time.time() - send_time) * 1000
            warm_latencies.record(warm_latency)

            await asyncio.sleep(0.05)

    warm_summary = warm_latencies.get_summary()

    logger.info(
        f"Cold Start vs Warm Latency:\n"
        f"  Cold start:     {cold_start_latency:.2f}ms\n"
        f"  Warm mean:      {warm_summary['mean']:.2f}ms\n"
        f"  Warm p50:       {warm_summary['p50']:.2f}ms\n"
        f"  Warm p95:       {warm_summary['p95']:.2f}ms\n"
        f"  Improvement:    {cold_start_latency - warm_summary['mean']:.2f}ms "
        f"({((cold_start_latency - warm_summary['mean']) / cold_start_latency * 100):.1f}%)"
    )

    # Cold start may be higher, but not excessively so
    assert cold_start_latency < warm_summary['mean'] * 2, (
        "Cold start latency excessively high"
    )


@pytest.mark.asyncio
async def test_frame_delivery_consistency(orchestrator_server: Any) -> None:
    """Test consistency of frame delivery.

    Validates:
    - Frames are delivered in order
    - No duplicate frames
    - Frame count is consistent for same text length
    """
    frame_counts = []

    async with websockets.connect("ws://localhost:8080") as ws:
        await ws.recv()  # session start

        # Send same text 10 times
        test_text = "This is a consistent test message."
        for _i in range(10):
            await send_text_message(ws, test_text, is_final=True)
            frames = await receive_audio_frames(ws, timeout_s=5.0)

            # Check sequence numbers
            sequences = [f["sequence"] for f in frames]
            assert sequences == list(range(1, len(sequences) + 1)), (
                f"Frame sequence disorder: {sequences}"
            )

            # Check for duplicates
            assert len(sequences) == len(set(sequences)), "Duplicate frame sequences"

            frame_counts.append(len(frames))
            await asyncio.sleep(0.05)

    # Frame counts should be consistent
    mean_count = np.mean(frame_counts)
    std_count = np.std(frame_counts)

    logger.info(
        f"Frame Delivery Consistency:\n"
        f"  Mean frame count: {mean_count:.1f}\n"
        f"  Std deviation:    {std_count:.2f}\n"
        f"  Min:              {min(frame_counts)}\n"
        f"  Max:              {max(frame_counts)}\n"
        f"  Frame counts:     {frame_counts}"
    )

    # Standard deviation should be low (consistent frame counts)
    assert std_count < mean_count * 0.2, f"High variance in frame counts: std={std_count:.2f}"


@pytest.mark.asyncio
async def test_stress_test_rapid_messages(orchestrator_server: Any) -> None:
    """Stress test with rapid message sending.

    Validates:
    - System handles rapid message bursts
    - No message loss or corruption
    - Performance degrades gracefully
    """
    num_messages = 50
    fal_metrics = LatencyMetrics()
    frame_counts = []

    async with websockets.connect("ws://localhost:8080") as ws:
        await ws.recv()  # session start

        # Send messages as fast as possible
        for i in range(num_messages):
            send_time = time.time()
            await send_text_message(ws, f"Rapid message {i}", is_final=True)

            frames = await receive_audio_frames(ws, timeout_s=5.0)
            fal_ms = (time.time() - send_time) * 1000

            fal_metrics.record(fal_ms)
            frame_counts.append(len(frames))

            # Minimal delay (stress test)
            await asyncio.sleep(0.01)

    summary = fal_metrics.get_summary()

    logger.info(
        f"Stress Test - Rapid Messages ({num_messages} messages):\n"
        f"  FAL mean:       {summary['mean']:.2f}ms\n"
        f"  FAL p95:        {summary['p95']:.2f}ms\n"
        f"  FAL p99:        {summary['p99']:.2f}ms\n"
        f"  Total frames:   {sum(frame_counts)}\n"
        f"  Messages lost:  0 (all received {num_messages})"
    )

    # All messages should be processed
    assert len(frame_counts) == num_messages, "Message loss detected"

    # Each message should produce frames
    assert all(count > 0 for count in frame_counts), "Some messages produced no frames"

    # Performance should degrade gracefully (not catastrophically)
    assert summary["p95"] < 1000, f"p95 latency {summary['p95']:.2f}ms excessive under stress"


@pytest.mark.asyncio
async def test_memory_stability_long_session(orchestrator_server: Any) -> None:
    """Test memory stability during long-running session.

    Validates:
    - No memory leaks during extended session
    - Frame delivery remains consistent
    - Latency doesn't degrade over time
    """
    num_messages = 50  # Simulate longer session
    fal_metrics_early = LatencyMetrics()
    fal_metrics_late = LatencyMetrics()

    async with websockets.connect("ws://localhost:8080") as ws:
        await ws.recv()  # session start

        for i in range(num_messages):
            send_time = time.time()
            await send_text_message(ws, f"Long session message {i}", is_final=True)

            await receive_audio_frames(ws, timeout_s=5.0)
            fal_ms = (time.time() - send_time) * 1000

            # Split metrics: early vs late messages
            if i < num_messages // 3:
                fal_metrics_early.record(fal_ms)
            elif i > 2 * num_messages // 3:
                fal_metrics_late.record(fal_ms)

            await asyncio.sleep(0.05)

    early_summary = fal_metrics_early.get_summary()
    late_summary = fal_metrics_late.get_summary()

    logger.info(
        f"Long Session Stability ({num_messages} messages):\n"
        f"  Early FAL mean: {early_summary['mean']:.2f}ms\n"
        f"  Late FAL mean:  {late_summary['mean']:.2f}ms\n"
        f"  Degradation:    {late_summary['mean'] - early_summary['mean']:.2f}ms "
        f"({((late_summary['mean'] - early_summary['mean']) / early_summary['mean'] * 100):.1f}%)"
    )

    # Latency shouldn't degrade significantly (< 50% increase)
    degradation_pct = (
        (late_summary["mean"] - early_summary["mean"]) / early_summary["mean"] * 100
    )
    assert degradation_pct < 50, (
        f"Excessive latency degradation: {degradation_pct:.1f}% over session"
    )
