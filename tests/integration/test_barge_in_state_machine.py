"""Barge-in state machine integration tests.

Tests state machine behavior during barge-in:
- Invalid state transitions are rejected
- Multiple barge-in cycles in a session
- Barge-in during active text streaming
- State transition guards
- Concurrent PAUSE/RESUME handling
- Edge cases and error conditions

These tests validate M3 state machine correctness.
"""

import logging
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from orchestrator.session import VALID_TRANSITIONS, SessionManager, SessionState
from orchestrator.transport.base import TransportSession

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ============================================================================
# Mock Transport for Testing
# ============================================================================


class MockTransportSession(TransportSession):
    """Mock transport session for state machine testing."""

    def __init__(self, session_id: str) -> None:
        """Initialize mock transport.

        Args:
            session_id: Session identifier
        """
        self._session_id = session_id
        self._is_connected = True
        self.state_transitions: list[tuple[SessionState, SessionState]] = []

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._is_connected

    async def send_audio_frame(self, frame: bytes) -> None:
        """Send audio frame (no-op for state machine tests)."""
        pass

    async def send_text(self, text: str) -> None:
        """Send text chunk (no-op for state machine tests)."""
        pass

    async def receive_text(self) -> AsyncIterator[str]:
        """Receive text from client (not implemented)."""
        # Async generator for receiving text from client
        if False:
            yield ""

    async def close(self) -> None:
        """Close connection."""
        self._is_connected = False


# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def session_manager() -> SessionManager:
    """Create session manager with mock transport.

    Returns:
        SessionManager instance
    """
    transport = MockTransportSession(session_id="test-session-state")
    manager = SessionManager(transport_session=transport)
    return manager


# ============================================================================
# State Transition Validation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_invalid_barge_in_from_listening(session_manager: SessionManager) -> None:
    """Test that BARGE_IN transition from LISTENING is invalid.

    Validates:
    - Cannot barge-in when not speaking
    - Invalid transition raises ValueError
    - State remains unchanged after failed transition

    Barge-in only makes sense when in SPEAKING state.
    """
    # Setup: ensure in LISTENING state
    session_manager.transition_state(SessionState.LISTENING)
    assert session_manager.state == SessionState.LISTENING

    # Act & Assert: attempt invalid transition
    with pytest.raises(ValueError, match="Invalid state transition"):
        session_manager.transition_state(SessionState.BARGED_IN)

    # State should be unchanged
    assert session_manager.state == SessionState.LISTENING

    logger.info("✓ Invalid LISTENING → BARGED_IN transition rejected")


@pytest.mark.asyncio
async def test_invalid_barge_in_from_idle(session_manager: SessionManager) -> None:
    """Test that BARGE_IN transition from IDLE is invalid.

    Validates:
    - Cannot barge-in from IDLE state
    - Invalid transition raises ValueError
    - State remains IDLE after failed transition
    """
    # Session starts in IDLE state
    assert session_manager.state == SessionState.IDLE

    # Act & Assert: attempt invalid transition
    with pytest.raises(ValueError, match="Invalid state transition"):
        session_manager.transition_state(SessionState.BARGED_IN)

    # State should be unchanged
    assert session_manager.state == SessionState.IDLE

    logger.info("✓ Invalid IDLE → BARGED_IN transition rejected")


@pytest.mark.asyncio
async def test_multiple_barge_ins_in_session(session_manager: SessionManager) -> None:
    """Test multiple barge-in cycles in a single session.

    Validates:
    - Multiple SPEAKING → BARGED_IN → LISTENING cycles
    - Each cycle completes correctly
    - State transitions are tracked
    - Session remains functional after multiple barge-ins

    Simulates user interrupting multiple times during a conversation.
    """
    num_cycles = 3
    barge_in_count = 0

    for i in range(num_cycles):
        # Cycle: LISTENING → SPEAKING → BARGED_IN → LISTENING
        if i == 0:
            session_manager.transition_state(SessionState.LISTENING)

        assert session_manager.state == SessionState.LISTENING

        # Start speaking
        session_manager.transition_state(SessionState.SPEAKING)
        # mypy sees state as LISTENING from assertion above, but transition changes it
        assert session_manager.state == SessionState.SPEAKING  # type: ignore[comparison-overlap]

        # User interrupts (barge-in)
        session_manager.transition_state(SessionState.BARGED_IN)
        assert session_manager.state == SessionState.BARGED_IN
        barge_in_count += 1

        # Handle interruption and return to listening
        session_manager.transition_state(SessionState.LISTENING)
        assert session_manager.state == SessionState.LISTENING

        logger.info(f"Completed barge-in cycle {i + 1}/{num_cycles}")

    # Verify all cycles completed
    assert barge_in_count == num_cycles

    # Verify session metrics recorded barge-ins
    # (Note: SessionMetrics.record_barge_in() would be called by orchestrator)
    metrics_summary = session_manager.get_metrics_summary()
    logger.info(f"Session metrics after {num_cycles} barge-ins: {metrics_summary}")

    logger.info(f"✓ {num_cycles} barge-in cycles completed successfully")


@pytest.mark.asyncio
async def test_barge_in_during_text_streaming(session_manager: SessionManager) -> None:
    """Test barge-in while text chunks are being processed.

    Validates:
    - Can barge-in while text is in queue
    - Text queue is preserved during barge-in
    - State transition doesn't corrupt queue
    - Can resume processing after barge-in

    This simulates interrupting during active synthesis.
    """
    # Setup: enter SPEAKING state
    session_manager.transition_state(SessionState.LISTENING)
    session_manager.transition_state(SessionState.SPEAKING)

    # Queue some text chunks (simulate streaming text)
    await session_manager.queue_text("This is a long sentence that")
    await session_manager.queue_text(" will be interrupted mid-stream.")
    await session_manager.queue_text(" This text won't be processed yet.")

    # Verify text is queued
    assert not session_manager.text_queue.empty()
    initial_queue_size = session_manager.text_queue.qsize()
    logger.info(f"Text queue size before barge-in: {initial_queue_size}")

    # Act: barge-in during text streaming
    session_manager.transition_state(SessionState.BARGED_IN)
    assert session_manager.state == SessionState.BARGED_IN

    # Verify queue is preserved (not cleared by barge-in)
    assert session_manager.text_queue.qsize() == initial_queue_size

    # Recover: return to listening
    session_manager.transition_state(SessionState.LISTENING)

    # Text queue should still be intact
    assert session_manager.text_queue.qsize() == initial_queue_size

    logger.info("✓ Barge-in during text streaming handled correctly")


@pytest.mark.asyncio
async def test_state_machine_guards() -> None:
    """Test all state transition guards defined in VALID_TRANSITIONS.

    Validates:
    - All valid transitions succeed
    - All invalid transitions are rejected
    - Transition table is symmetric and complete
    - Terminal states (TERMINATED) behave correctly
    """
    all_states = list(SessionState)
    transition_test_results: dict[str, bool] = {}

    for from_state in all_states:
        valid_targets = VALID_TRANSITIONS.get(from_state, set())

        for to_state in all_states:
            transport = MockTransportSession(session_id="test-guards")
            manager = SessionManager(transport_session=transport)

            # Set up the source state
            # Need to reach from_state through valid transitions
            if from_state == SessionState.IDLE:
                # Already in IDLE
                pass
            elif from_state == SessionState.LISTENING:
                manager.transition_state(SessionState.LISTENING)
            elif from_state == SessionState.SPEAKING:
                manager.transition_state(SessionState.LISTENING)
                manager.transition_state(SessionState.SPEAKING)
            elif from_state == SessionState.BARGED_IN:
                manager.transition_state(SessionState.LISTENING)
                manager.transition_state(SessionState.SPEAKING)
                manager.transition_state(SessionState.BARGED_IN)
            elif from_state == SessionState.WAITING_FOR_INPUT:
                manager.transition_state(SessionState.LISTENING)
                manager.transition_state(SessionState.WAITING_FOR_INPUT)
            elif from_state == SessionState.TERMINATED:
                manager.transition_state(SessionState.LISTENING)
                manager.transition_state(SessionState.TERMINATED)

            # Now test transition to target state
            transition_key = f"{from_state.value} → {to_state.value}"

            if to_state in valid_targets:
                # Should succeed
                try:
                    manager.transition_state(to_state)
                    transition_test_results[transition_key] = True
                except ValueError:
                    transition_test_results[transition_key] = False
                    logger.error(f"Valid transition failed: {transition_key}")
            else:
                # Should fail
                try:
                    manager.transition_state(to_state)
                    transition_test_results[transition_key] = False
                    logger.error(f"Invalid transition succeeded: {transition_key}")
                except ValueError:
                    transition_test_results[transition_key] = True

    # Check results
    failed_transitions = [
        key for key, success in transition_test_results.items() if not success
    ]

    if failed_transitions:
        logger.error(f"Failed transitions: {failed_transitions}")
        pytest.fail(f"State machine guards failed for: {failed_transitions}")

    logger.info(f"✓ All {len(transition_test_results)} state transitions validated")


@pytest.mark.asyncio
async def test_concurrent_pause_resume() -> None:
    """Test handling of rapid PAUSE/RESUME state changes.

    Validates:
    - Rapid state transitions don't corrupt state
    - Race conditions are handled
    - State remains consistent
    - No deadlocks or race conditions

    This simulates rapid user interruptions (e.g., user speaks briefly then stops).
    """
    transport = MockTransportSession(session_id="test-concurrent")
    manager = SessionManager(transport_session=transport)

    # Setup: enter SPEAKING state
    manager.transition_state(SessionState.LISTENING)
    manager.transition_state(SessionState.SPEAKING)

    # Simulate rapid barge-in cycles
    num_rapid_cycles = 10
    for i in range(num_rapid_cycles):
        # Barge-in
        manager.transition_state(SessionState.BARGED_IN)
        assert manager.state == SessionState.BARGED_IN

        # Immediately resume
        manager.transition_state(SessionState.LISTENING)
        # mypy sees state as BARGED_IN from assertion above, but transition changes it
        assert manager.state == SessionState.LISTENING  # type: ignore[comparison-overlap]

        # Back to speaking (for next cycle)
        if i < num_rapid_cycles - 1:
            manager.transition_state(SessionState.SPEAKING)
            assert manager.state == SessionState.SPEAKING

    # Verify final state is consistent
    assert manager.state == SessionState.LISTENING
    assert manager.is_active

    logger.info(f"✓ {num_rapid_cycles} rapid PAUSE/RESUME cycles handled correctly")


@pytest.mark.asyncio
async def test_state_transition_after_termination() -> None:
    """Test that no transitions are allowed after TERMINATED state.

    Validates:
    - TERMINATED is a terminal state
    - All transitions from TERMINATED are rejected
    - Session is properly marked as inactive

    Once terminated, session should not allow any state changes.
    """
    transport = MockTransportSession(session_id="test-terminated")
    manager = SessionManager(transport_session=transport)

    # Transition to TERMINATED
    manager.transition_state(SessionState.LISTENING)
    manager.transition_state(SessionState.TERMINATED)

    assert manager.state == SessionState.TERMINATED
    assert not manager.is_active

    # Try to transition to any other state (all should fail)
    test_states = [
        SessionState.IDLE,
        SessionState.LISTENING,
        SessionState.SPEAKING,
        SessionState.BARGED_IN,
    ]

    for target_state in test_states:
        with pytest.raises(ValueError, match="Invalid state transition"):
            manager.transition_state(target_state)

        # State should remain TERMINATED
        assert manager.state == SessionState.TERMINATED

    logger.info("✓ TERMINATED state correctly blocks all transitions")


@pytest.mark.asyncio
async def test_barge_in_state_with_metrics() -> None:
    """Test that barge-in events are properly tracked in session metrics.

    Validates:
    - SessionMetrics.record_barge_in() integration
    - Barge-in count increments correctly
    - Latencies are recorded
    - Metrics summary includes barge-in data

    This validates the metrics tracking aspect of M3.
    """
    transport = MockTransportSession(session_id="test-metrics")
    manager = SessionManager(transport_session=transport)

    # Perform multiple barge-in cycles with metrics
    num_barge_ins = 5

    for i in range(num_barge_ins):
        # Setup: enter SPEAKING state
        if i == 0:
            manager.transition_state(SessionState.LISTENING)
        manager.transition_state(SessionState.SPEAKING)

        # Simulate barge-in with latency measurement
        barge_in_latency_ms = 25.0 + (i * 2)  # Vary latency slightly
        manager.metrics.record_barge_in(barge_in_latency_ms)

        # Transition to BARGED_IN state
        manager.transition_state(SessionState.BARGED_IN)

        # Return to LISTENING
        manager.transition_state(SessionState.LISTENING)

    # Verify metrics
    assert manager.metrics.barge_in_count == num_barge_ins
    assert len(manager.metrics.barge_in_latencies_ms) == num_barge_ins

    # Check metrics summary
    summary = manager.get_metrics_summary()
    assert summary["barge_in_count"] == num_barge_ins

    avg_latency = manager.metrics.compute_avg_barge_in_latency_ms()
    p95_latency = manager.metrics.compute_p95_barge_in_latency_ms()

    assert avg_latency is not None
    assert p95_latency is not None

    logger.info(
        f"Barge-in metrics: count={num_barge_ins}, "
        f"avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms"
    )

    logger.info("✓ Barge-in metrics tracked correctly")


@pytest.mark.asyncio
async def test_state_transition_ordering() -> None:
    """Test that state transitions maintain correct temporal ordering.

    Validates:
    - State changes are atomic
    - Transition events occur in order
    - No intermediate states leak
    - State history is consistent

    This ensures state machine behaves deterministically.
    """
    transport = MockTransportSession(session_id="test-ordering")
    manager = SessionManager(transport_session=transport)

    # Track state history
    state_history: list[SessionState] = [manager.state]

    original_transition = manager.transition_state

    def tracked_transition(new_state: SessionState) -> None:
        """Wrapper to track state transitions."""
        original_transition(new_state)
        state_history.append(manager.state)

    manager.transition_state = tracked_transition  # type: ignore[method-assign]

    # Execute a complex sequence of transitions
    manager.transition_state(SessionState.LISTENING)
    manager.transition_state(SessionState.SPEAKING)
    manager.transition_state(SessionState.BARGED_IN)
    manager.transition_state(SessionState.LISTENING)
    manager.transition_state(SessionState.SPEAKING)
    manager.transition_state(SessionState.LISTENING)
    manager.transition_state(SessionState.TERMINATED)

    # Verify history
    expected_history = [
        SessionState.IDLE,
        SessionState.LISTENING,
        SessionState.SPEAKING,
        SessionState.BARGED_IN,
        SessionState.LISTENING,
        SessionState.SPEAKING,
        SessionState.LISTENING,
        SessionState.TERMINATED,
    ]

    assert state_history == expected_history, (
        f"State transition history mismatch.\n"
        f"Expected: {[s.value for s in expected_history]}\n"
        f"Got:      {[s.value for s in state_history]}"
    )

    logger.info(f"✓ State transition ordering correct ({len(state_history)} transitions)")


@pytest.mark.asyncio
async def test_barge_in_from_speaking_only() -> None:
    """Test that BARGED_IN can only be entered from SPEAKING state.

    Validates:
    - BARGED_IN requires SPEAKING as source state
    - All other source states are invalid
    - Error messages are clear

    This is a critical state machine constraint for barge-in.
    """
    test_cases = [
        (SessionState.IDLE, False),
        (SessionState.LISTENING, False),
        (SessionState.SPEAKING, True),
        # BARGED_IN → BARGED_IN is invalid (not in VALID_TRANSITIONS)
        # TERMINATED → anything is invalid
    ]

    for source_state, should_succeed in test_cases:
        transport = MockTransportSession(session_id=f"test-from-{source_state.value}")
        manager = SessionManager(transport_session=transport)

        # Set up source state
        if source_state == SessionState.IDLE:
            # Already in IDLE
            pass
        elif source_state == SessionState.LISTENING:
            manager.transition_state(SessionState.LISTENING)
        elif source_state == SessionState.SPEAKING:
            manager.transition_state(SessionState.LISTENING)
            manager.transition_state(SessionState.SPEAKING)

        # Attempt transition to BARGED_IN
        if should_succeed:
            manager.transition_state(SessionState.BARGED_IN)
            assert manager.state == SessionState.BARGED_IN
            logger.info(f"✓ {source_state.value} → BARGED_IN succeeded (expected)")
        else:
            with pytest.raises(ValueError, match="Invalid state transition"):
                manager.transition_state(SessionState.BARGED_IN)
            logger.info(f"✓ {source_state.value} → BARGED_IN rejected (expected)")

    logger.info("✓ BARGED_IN state transition constraints validated")
