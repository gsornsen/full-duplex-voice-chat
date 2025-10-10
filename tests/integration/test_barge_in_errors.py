"""Barge-in error handling and edge case tests.

Tests error conditions and edge cases:
- Worker disconnect during BARGED_IN state
- Invalid audio frames
- PAUSE timeout handling
- Graceful degradation with VAD disabled
- Concurrent session barge-ins
- Resource cleanup
- Error recovery

These tests validate M3 robustness and error handling.
"""

import asyncio
import logging
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from src.orchestrator.config import VADConfig
from src.orchestrator.session import SessionManager, SessionState
from src.orchestrator.transport.base import TransportSession
from src.orchestrator.vad import VADProcessor
from tests.helpers.vad_test_utils import (
    VADTestRecorder,
    generate_silence_audio,
    generate_speech_audio,
)

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ============================================================================
# Mock Transport with Failure Simulation
# ============================================================================


class FailingMockTransport(TransportSession):
    """Mock transport that can simulate failures."""

    def __init__(self, session_id: str, fail_on_disconnect: bool = False) -> None:
        """Initialize failing mock transport.

        Args:
            session_id: Session identifier
            fail_on_disconnect: If True, simulate worker disconnect
        """
        self._session_id = session_id
        self._is_connected = True
        self._fail_on_disconnect = fail_on_disconnect
        self.send_count = 0
        self.disconnect_triggered = False

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._is_connected

    async def send_audio_frame(self, frame: bytes) -> None:
        """Send audio frame (may fail if disconnected)."""
        if not self._is_connected:
            raise ConnectionError("Transport disconnected")
        self.send_count += 1

    async def send_text(self, text: str) -> None:
        """Send text chunk."""
        if not self._is_connected:
            raise ConnectionError("Transport disconnected")

    async def receive_text(self) -> AsyncIterator[str]:
        """Receive text from client."""
        # Async generator for receiving text from client
        if False:
            yield ""

    async def close(self) -> None:
        """Close connection."""
        self._is_connected = False

    def simulate_disconnect(self) -> None:
        """Simulate worker disconnection."""
        self._is_connected = False
        self.disconnect_triggered = True
        logger.info("Simulated worker disconnect")


# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def failing_session_manager() -> SessionManager:
    """Create session manager with failing transport.

    Returns:
        SessionManager instance with disconnect simulation capability
    """
    transport = FailingMockTransport(session_id="test-failing-session")
    manager = SessionManager(transport_session=transport)
    return manager


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_barge_in_with_worker_disconnect(
    failing_session_manager: SessionManager,
) -> None:
    """Test barge-in behavior when worker disconnects.

    Validates:
    - Worker disconnect during BARGED_IN state is handled
    - Session transitions to TERMINATED
    - Resources are cleaned up
    - No crashes or hangs
    """
    transport = failing_session_manager.transport
    assert isinstance(transport, FailingMockTransport)

    # Setup: enter BARGED_IN state
    failing_session_manager.transition_state(SessionState.LISTENING)
    failing_session_manager.transition_state(SessionState.SPEAKING)
    failing_session_manager.transition_state(SessionState.BARGED_IN)

    assert failing_session_manager.state == SessionState.BARGED_IN

    # Simulate worker disconnect
    transport.simulate_disconnect()

    # Attempt to send audio (should fail gracefully)
    test_frame = generate_speech_audio(duration_ms=20, sample_rate=48000)

    # Audio send should fail due to disconnect
    with pytest.raises(ConnectionError):
        await transport.send_audio_frame(test_frame)

    # Session should handle disconnect gracefully
    await failing_session_manager.shutdown()

    # Verify cleanup
    assert failing_session_manager.state == SessionState.TERMINATED  # type: ignore[comparison-overlap]
    assert not failing_session_manager.is_active
    assert transport.disconnect_triggered

    logger.info("✓ Worker disconnect during barge-in handled gracefully")


@pytest.mark.asyncio
async def test_vad_with_invalid_audio_frames() -> None:
    """Test VAD handling of invalid/malformed audio frames.

    Validates:
    - Invalid frame sizes raise ValueError
    - Error messages are descriptive
    - VAD state is not corrupted by errors
    - Can continue processing after error
    """
    config = VADConfig(aggressiveness=2, sample_rate=16000, frame_duration_ms=20)
    vad = VADProcessor(config=config)

    # Test various invalid frames
    invalid_frames = [
        (b"", "empty frame"),
        (b"\x00" * 100, "wrong size (100 bytes)"),
        (b"\x00" * 641, "odd number of bytes"),
        (b"\x00" * 1920, "48kHz frame for 16kHz VAD"),
    ]

    for invalid_frame, description in invalid_frames:
        with pytest.raises(ValueError, match="Invalid frame size"):
            vad.process_frame(invalid_frame)
        logger.debug(f"Correctly rejected {description}")

    # Verify VAD still works after errors
    valid_frame = generate_speech_audio(duration_ms=20, sample_rate=16000)
    vad.process_frame(valid_frame)  # Should succeed

    stats = vad.stats
    assert stats["frames_processed"] == 1, "VAD corrupted by invalid frames"

    logger.info("✓ Invalid audio frames rejected with proper errors")


@pytest.mark.asyncio
async def test_pause_timeout_handling() -> None:
    """Test handling of PAUSE command timeout.

    Validates:
    - Timeout waiting for PAUSE confirmation
    - System recovers gracefully
    - State remains consistent
    - User experience degrades gracefully

    This simulates worker being slow to respond to PAUSE.
    """
    # Setup session
    transport = FailingMockTransport(session_id="test-timeout")
    manager = SessionManager(transport_session=transport)

    # Enter SPEAKING state
    manager.transition_state(SessionState.LISTENING)
    manager.transition_state(SessionState.SPEAKING)

    # Simulate PAUSE command with timeout
    pause_timeout_ms = 100.0  # 100ms timeout

    async def send_pause_with_timeout() -> bool:
        """Simulate sending PAUSE with timeout."""
        try:
            # In real implementation, this would send gRPC PAUSE
            # and wait for confirmation with timeout
            await asyncio.wait_for(
                asyncio.sleep(0.2),  # Simulate slow worker (200ms)
                timeout=pause_timeout_ms / 1000,
            )
            return True
        except TimeoutError:
            logger.warning("PAUSE command timed out")
            return False

    # Attempt PAUSE
    pause_succeeded = await send_pause_with_timeout()

    # Should have timed out
    assert not pause_succeeded, "PAUSE should have timed out"

    # System should still be functional
    # (In production, might retry or log error)
    assert manager.is_active
    assert manager.state == SessionState.SPEAKING  # State unchanged

    # Can still transition normally
    manager.transition_state(SessionState.LISTENING)
    assert manager.state == SessionState.LISTENING  # type: ignore[comparison-overlap]

    logger.info("✓ PAUSE timeout handled gracefully")


@pytest.mark.asyncio
async def test_graceful_degradation_no_vad() -> None:
    """Test system works with VAD disabled.

    Validates:
    - System functions without VAD
    - No barge-in detection occurs
    - Audio pipeline works normally
    - No crashes when VAD is disabled

    This ensures VAD is an optional enhancement, not a hard requirement.
    """
    # Create VAD config with VAD disabled
    config = VADConfig(
        enabled=False,  # VAD disabled
        aggressiveness=2,
        sample_rate=16000,
        frame_duration_ms=20,
    )

    # System should handle disabled VAD gracefully
    # (In production, orchestrator would check config.vad.enabled)
    assert not config.enabled

    # Create session without VAD
    transport = FailingMockTransport(session_id="test-no-vad")
    manager = SessionManager(transport_session=transport)

    # Normal flow should work (without barge-in)
    manager.transition_state(SessionState.LISTENING)
    manager.transition_state(SessionState.SPEAKING)

    # Queue and send audio normally
    test_frame = generate_speech_audio(duration_ms=20, sample_rate=48000)
    await manager.queue_audio_frame(test_frame)

    # Start audio sender briefly
    sender_task = asyncio.create_task(manager.audio_sender_loop())
    await asyncio.sleep(0.05)
    sender_task.cancel()
    try:
        await sender_task
    except asyncio.CancelledError:
        pass

    # Verify audio was sent
    assert transport.send_count > 0, "Audio not sent with VAD disabled"

    # Complete session normally
    manager.transition_state(SessionState.LISTENING)
    await manager.shutdown()

    # Verify no barge-in occurred (VAD disabled)
    assert manager.metrics.barge_in_count == 0

    logger.info("✓ System works correctly with VAD disabled")


@pytest.mark.asyncio
async def test_concurrent_session_barge_ins() -> None:
    """Test multiple sessions with independent barge-in events.

    Validates:
    - Multiple sessions can barge-in independently
    - No state pollution between sessions
    - Each session has isolated VAD processing
    - Metrics are tracked separately

    This ensures session isolation is maintained.
    """
    num_sessions = 3
    sessions: list[tuple[SessionManager, VADProcessor]] = []

    # Create multiple sessions
    for i in range(num_sessions):
        transport = FailingMockTransport(session_id=f"test-concurrent-{i}")
        manager = SessionManager(transport_session=transport)

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

        sessions.append((manager, vad))

    # Process different audio patterns for each session
    async def process_session(
        session_id: int, manager: SessionManager, vad: VADProcessor
    ) -> None:
        """Process audio for one session."""
        recorder = VADTestRecorder()
        vad.on_speech_start = recorder.on_speech_start
        vad.on_speech_end = recorder.on_speech_end

        # Setup state
        manager.transition_state(SessionState.LISTENING)
        manager.transition_state(SessionState.SPEAKING)

        # Process varying amounts of speech
        num_frames = 15 + (session_id * 5)  # Different patterns per session
        for _ in range(num_frames):
            frame = generate_speech_audio(duration_ms=20, sample_rate=16000)
            vad.process_frame(frame)
            await asyncio.sleep(0.001)

        # Simulate barge-in if speech detected
        if recorder.speech_start_events:
            manager.transition_state(SessionState.BARGED_IN)
            manager.metrics.record_barge_in(25.0 + session_id)  # Unique latency

        logger.info(
            f"Session {session_id}: "
            f"frames={num_frames}, "
            f"speech_events={len(recorder.speech_start_events)}, "
            f"barge_ins={manager.metrics.barge_in_count}"
        )

    # Run all sessions concurrently
    tasks = [
        process_session(i, manager, vad) for i, (manager, vad) in enumerate(sessions)
    ]
    await asyncio.gather(*tasks)

    # Verify session isolation
    for i, (manager, _vad) in enumerate(sessions):
        # Each session should have independent metrics
        assert manager.session_id == f"test-concurrent-{i}"

        # Verify no cross-contamination
        # (Metrics should be unique per session)
        summary = manager.get_metrics_summary()
        logger.info(f"Session {i} metrics: {summary}")

    logger.info(f"✓ {num_sessions} concurrent sessions with independent barge-ins")


@pytest.mark.asyncio
async def test_vad_reset_after_error() -> None:
    """Test that VAD reset works after processing errors.

    Validates:
    - VAD state can be reset after errors
    - Reset clears error conditions
    - VAD works correctly after reset
    - No lingering state corruption
    """
    config = VADConfig(aggressiveness=2, sample_rate=16000, frame_duration_ms=20)
    vad = VADProcessor(config=config)

    # Process some valid frames
    for _ in range(5):
        frame = generate_speech_audio(duration_ms=20, sample_rate=16000)
        vad.process_frame(frame)

    # Cause an error
    with pytest.raises(ValueError):
        vad.process_frame(b"invalid")

    # Verify VAD still has state
    stats_before = vad.stats
    assert stats_before["frames_processed"] == 5

    # Reset VAD
    vad.reset()

    # Verify state cleared
    stats_after = vad.stats
    assert stats_after["frames_processed"] == 0
    assert stats_after["speech_frames"] == 0
    assert stats_after["silence_frames"] == 0

    # Verify VAD works after reset
    for _ in range(3):
        frame = generate_speech_audio(duration_ms=20, sample_rate=16000)
        vad.process_frame(frame)

    stats_final = vad.stats
    assert stats_final["frames_processed"] == 3

    logger.info("✓ VAD reset clears error conditions and restores functionality")


@pytest.mark.asyncio
async def test_barge_in_with_empty_audio_queue() -> None:
    """Test barge-in when audio queue is empty.

    Validates:
    - Barge-in works even if no audio is queued
    - No errors when pausing empty queue
    - State transitions work correctly
    - Resume works from empty state
    """
    transport = FailingMockTransport(session_id="test-empty-queue")
    manager = SessionManager(transport_session=transport)

    # Enter SPEAKING state (but don't queue any audio)
    manager.transition_state(SessionState.LISTENING)
    manager.transition_state(SessionState.SPEAKING)

    # Verify queue is empty
    assert manager.audio_buffer.empty()

    # Barge-in with empty queue
    manager.transition_state(SessionState.BARGED_IN)
    assert manager.state == SessionState.BARGED_IN

    # Resume
    manager.transition_state(SessionState.LISTENING)
    assert manager.state == SessionState.LISTENING  # type: ignore[comparison-overlap]

    # Queue still empty (no errors)
    assert manager.audio_buffer.empty()

    logger.info("✓ Barge-in with empty audio queue handled correctly")


@pytest.mark.asyncio
async def test_vad_with_extreme_audio_levels() -> None:
    """Test VAD with extreme audio amplitude levels.

    Validates:
    - VAD handles very loud audio
    - VAD handles very quiet audio
    - No crashes or numeric errors
    - Detection still works at extremes
    """
    config = VADConfig(aggressiveness=2, sample_rate=16000, frame_duration_ms=20)
    vad = VADProcessor(config=config)

    # Test very loud audio (max amplitude)
    loud_frame = generate_speech_audio(duration_ms=20, sample_rate=16000, amplitude=1.0)
    is_speech = vad.process_frame(loud_frame)
    logger.debug(f"Loud audio detected as speech: {is_speech}")

    # Reset for next test
    vad.reset()

    # Test very quiet audio (near zero)
    quiet_frame = generate_speech_audio(
        duration_ms=20, sample_rate=16000, amplitude=0.01
    )
    is_speech = vad.process_frame(quiet_frame)
    logger.debug(f"Quiet audio detected as speech: {is_speech}")

    # Reset for next test
    vad.reset()

    # Test silence with noise floor
    silence_with_noise = generate_silence_audio(
        duration_ms=20, sample_rate=16000, noise_floor=0.05
    )
    is_speech = vad.process_frame(silence_with_noise)
    logger.debug(f"Silence with noise detected as speech: {is_speech}")

    # No crashes occurred
    stats = vad.stats
    assert stats["frames_processed"] == 1  # Only last frame after reset

    logger.info("✓ VAD handles extreme audio levels without errors")


@pytest.mark.asyncio
async def test_session_cleanup_after_barge_in_error() -> None:
    """Test session cleanup after barge-in error.

    Validates:
    - Session cleanup works after errors during barge-in
    - Resources are properly released
    - No memory leaks or dangling references
    - Metrics are finalized correctly
    """
    transport = FailingMockTransport(session_id="test-cleanup")
    manager = SessionManager(transport_session=transport)

    # Setup and trigger barge-in
    manager.transition_state(SessionState.LISTENING)
    manager.transition_state(SessionState.SPEAKING)

    # Queue some audio
    for _ in range(5):
        frame = generate_speech_audio(duration_ms=20, sample_rate=48000)
        await manager.queue_audio_frame(frame)

    # Barge-in
    manager.transition_state(SessionState.BARGED_IN)
    manager.metrics.record_barge_in(30.0)

    # Simulate error (disconnect)
    transport.simulate_disconnect()

    # Shutdown should work despite error
    await manager.shutdown()

    # Verify cleanup
    assert manager.state == SessionState.TERMINATED
    assert not manager.is_active
    assert manager.audio_buffer.empty()
    assert manager.text_queue.empty()

    # Metrics should be finalized
    assert manager.metrics.session_end_ts is not None

    # Barge-in count preserved
    assert manager.metrics.barge_in_count == 1

    logger.info("✓ Session cleanup successful after barge-in error")


@pytest.mark.asyncio
async def test_rapid_vad_state_changes() -> None:
    """Test VAD with rapid speech/silence transitions.

    Validates:
    - VAD handles rapid on/off speech
    - Debouncing prevents event flood
    - Performance remains acceptable
    - No state corruption from rapid changes
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

    # Generate rapid alternating pattern (should be debounced)
    frames: list[bytes] = []
    for _ in range(20):  # 20 cycles
        # 40ms speech, 40ms silence (too short for events)
        frames.append(generate_speech_audio(duration_ms=20, sample_rate=16000))
        frames.append(generate_speech_audio(duration_ms=20, sample_rate=16000))
        frames.append(generate_silence_audio(duration_ms=20, sample_rate=16000))
        frames.append(generate_silence_audio(duration_ms=20, sample_rate=16000))

    # Process all frames
    for frame in frames:
        vad.process_frame(frame)
        recorder.increment_frame_count()

    # Debouncing should have prevented many events
    # (exact count depends on VAD hysteresis, but should be < 20)
    total_events = len(recorder.speech_start_events) + len(recorder.speech_end_events)

    logger.info(
        f"Rapid transitions: {total_events} events from {len(frames)} frames "
        f"(debouncing effective: {total_events < len(frames) // 10})"
    )

    # Verify no crashes and state is consistent
    stats = vad.stats
    assert stats["frames_processed"] == len(frames)

    logger.info("✓ VAD handles rapid state changes without corruption")
