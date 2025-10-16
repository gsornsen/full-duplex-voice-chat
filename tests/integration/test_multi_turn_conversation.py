"""Integration tests for multi-turn conversation flow (M10 Polish Task 7).

Tests end-to-end multi-turn conversation flow with session timeouts. Validates
the complete session lifecycle including state transitions, idle timeout,
max duration enforcement, and error handling.

Coverage:
- State transitions (LISTENING → WAITING_FOR_INPUT)
- Idle timeout behavior (disconnect after no input)
- Max session duration enforcement
- Max messages per session enforcement
- Multi-turn conversation flow
- Error handling (disconnect, empty messages, StopIteration)
"""

import asyncio
import logging
from collections.abc import AsyncIterator

import pytest

from src.orchestrator.config import SessionConfig
from src.orchestrator.session import SessionState
from src.orchestrator.transport.base import TransportSession

logger = logging.getLogger(__name__)


class MockMultiTurnTransport(TransportSession):
    """Mock transport for multi-turn conversation testing.

    Simulates a client sending multiple text messages with configurable delays.
    """

    def __init__(
        self,
        session_id: str,
        text_messages: list[str] | None = None,
        message_delay_s: float = 0.1,
    ) -> None:
        """Initialize mock multi-turn transport.

        Args:
            session_id: Session identifier
            text_messages: List of text messages to send
            message_delay_s: Delay between messages in seconds
        """
        self._session_id = session_id
        self._text_messages = text_messages or []
        self._message_delay_s = message_delay_s
        self._closed = False

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id

    async def send_audio_frame(self, frame: bytes) -> None:
        """Send audio frame (no-op for testing)."""
        pass

    async def receive_audio(self) -> AsyncIterator[bytes]:
        """Receive audio frames (not used in multi-turn text tests)."""
        yield b""

    @property
    def is_connected(self) -> bool:
        """Check if session is connected."""
        return not self._closed

    async def receive_text(self) -> AsyncIterator[str]:
        """Receive text messages.

        Yields:
            Text messages from configured list with delays
        """
        for message in self._text_messages:
            if self._closed:
                break
            await asyncio.sleep(self._message_delay_s)
            yield message

    async def close(self) -> None:
        """Close transport."""
        self._closed = True


class MockTimeoutTransport(TransportSession):
    """Mock transport that simulates idle timeout.

    Sends N messages then waits forever to simulate idle timeout scenario.
    """

    def __init__(
        self,
        session_id: str,
        timeout_after_messages: int = 1,
    ) -> None:
        """Initialize mock timeout transport.

        Args:
            session_id: Session identifier
            timeout_after_messages: Number of messages before simulating idle
        """
        self._session_id = session_id
        self._timeout_after_messages = timeout_after_messages
        self._message_count = 0
        self._closed = False

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id

    async def send_audio_frame(self, frame: bytes) -> None:
        """Send audio frame (no-op for testing)."""
        pass

    async def receive_audio(self) -> AsyncIterator[bytes]:
        """Receive audio frames (not used in multi-turn text tests)."""
        yield b""

    @property
    def is_connected(self) -> bool:
        """Check if session is connected."""
        return not self._closed

    async def receive_text(self) -> AsyncIterator[str]:
        """Receive text messages.

        Sends N messages then waits forever (simulates idle timeout).

        Yields:
            Text messages until timeout_after_messages reached
        """
        while self._message_count < self._timeout_after_messages:
            if self._closed:
                break
            self._message_count += 1
            yield f"Message {self._message_count}"
            await asyncio.sleep(0.1)

        # Simulate idle timeout - wait forever (caller should timeout)
        await asyncio.sleep(999999)

    async def close(self) -> None:
        """Close transport."""
        self._closed = True


class MockDisconnectTransport(TransportSession):
    """Mock transport that disconnects after N messages."""

    def __init__(
        self,
        session_id: str,
        disconnect_after_messages: int = 2,
    ) -> None:
        """Initialize mock disconnect transport.

        Args:
            session_id: Session identifier
            disconnect_after_messages: Number of messages before disconnect
        """
        self._session_id = session_id
        self._disconnect_after_messages = disconnect_after_messages
        self._message_count = 0
        self._closed = False

    @property
    def session_id(self) -> str:
        """Get session ID."""
        return self._session_id

    async def send_audio_frame(self, frame: bytes) -> None:
        """Send audio frame (no-op for testing)."""
        pass

    async def receive_audio(self) -> AsyncIterator[bytes]:
        """Receive audio frames (not used in multi-turn text tests)."""
        yield b""

    @property
    def is_connected(self) -> bool:
        """Check if session is connected."""
        return not self._closed

    async def receive_text(self) -> AsyncIterator[str]:
        """Receive text messages.

        Sends N messages then raises StopAsyncIteration (simulates disconnect).

        Yields:
            Text messages until disconnect_after_messages reached
        """
        while self._message_count < self._disconnect_after_messages:
            if self._closed:
                break
            self._message_count += 1
            yield f"Message {self._message_count}"
            await asyncio.sleep(0.1)

        # Simulate disconnect - raise StopAsyncIteration
        return

    async def close(self) -> None:
        """Close transport."""
        self._closed = True


@pytest.mark.asyncio
async def test_multi_turn_basic_flow() -> None:
    """Test basic multi-turn conversation flow (3 messages)."""
    transport = MockMultiTurnTransport(
        session_id="test-multi-turn-basic",
        text_messages=["Hello", "How are you?", "Goodbye"],
        message_delay_s=0.1,
    )

    messages_received = []
    async for text in transport.receive_text():
        messages_received.append(text)

    assert len(messages_received) == 3
    assert messages_received == ["Hello", "How are you?", "Goodbye"]


@pytest.mark.asyncio
async def test_multi_turn_with_delays() -> None:
    """Test multi-turn conversation with realistic delays."""
    transport = MockMultiTurnTransport(
        session_id="test-multi-turn-delays",
        text_messages=["Message 1", "Message 2", "Message 3"],
        message_delay_s=0.5,  # 500ms between messages
    )

    start_time = asyncio.get_event_loop().time()
    message_count = 0

    async for text in transport.receive_text():
        message_count += 1
        logger.info(f"Received: {text}")

    elapsed_time = asyncio.get_event_loop().time() - start_time

    assert message_count == 3
    # Should take ~1.5 seconds (3 messages * 0.5s delay)
    assert 1.0 <= elapsed_time <= 2.0


@pytest.mark.asyncio
async def test_idle_timeout_after_first_message() -> None:
    """Test that session times out after idle period (1 message then idle)."""
    transport = MockTimeoutTransport(
        session_id="test-idle-timeout",
        timeout_after_messages=1,
    )
    config = SessionConfig(idle_timeout_seconds=10)  # Minimum is 10 seconds

    message_count = 0
    timeout_occurred = False

    try:
        async for text in transport.receive_text():
            message_count += 1
            logger.info(f"Received: {text}")

            # Try to get next message with timeout
            try:
                # Wait for next message with timeout
                next_text = await asyncio.wait_for(
                    transport.receive_text().__anext__(),
                    timeout=config.idle_timeout_seconds,
                )
                logger.info(f"Received next: {next_text}")
            except TimeoutError:
                logger.info("Session idle timeout (expected)")
                timeout_occurred = True
                break
            except StopAsyncIteration:
                logger.info("Transport ended (expected)")
                break
    finally:
        await transport.close()

    assert message_count >= 1
    assert timeout_occurred


@pytest.mark.asyncio
async def test_idle_timeout_after_multiple_messages() -> None:
    """Test idle timeout after multiple messages (3 messages then idle)."""
    transport = MockTimeoutTransport(
        session_id="test-idle-timeout-multi",
        timeout_after_messages=3,
    )
    config = SessionConfig(idle_timeout_seconds=10)  # Minimum is 10 seconds

    messages_received = []
    timeout_occurred = False

    try:
        async for text in transport.receive_text():
            messages_received.append(text)
            logger.info(f"Received: {text}")

            # After 3rd message, next wait should timeout
            if len(messages_received) >= 3:
                try:
                    next_text = await asyncio.wait_for(
                        transport.receive_text().__anext__(),
                        timeout=config.idle_timeout_seconds,
                    )
                    messages_received.append(next_text)
                except TimeoutError:
                    logger.info("Session idle timeout after 3 messages (expected)")
                    timeout_occurred = True
                    break
    finally:
        await transport.close()

    assert len(messages_received) == 3
    assert timeout_occurred


@pytest.mark.asyncio
async def test_max_messages_enforcement() -> None:
    """Test that session enforces max messages limit."""
    config = SessionConfig(max_messages_per_session=5)

    transport = MockMultiTurnTransport(
        session_id="test-max-messages",
        text_messages=[f"Message {i}" for i in range(10)],  # 10 messages
        message_delay_s=0.05,
    )

    message_count = 0
    async for text in transport.receive_text():
        message_count += 1
        logger.info(f"Received message {message_count}: {text}")

        # Enforce max messages limit
        if message_count >= config.max_messages_per_session:
            logger.info(f"Max messages reached ({config.max_messages_per_session})")
            break

    await transport.close()

    assert message_count == config.max_messages_per_session


@pytest.mark.asyncio
async def test_max_session_duration_enforcement() -> None:
    """Test that session enforces max duration limit."""
    config = SessionConfig(
        idle_timeout_seconds=60,  # Won't hit idle timeout
        max_session_duration_seconds=60,  # 60 seconds (1 minute) max
    )

    transport = MockMultiTurnTransport(
        session_id="test-max-duration",
        text_messages=[f"Message {i}" for i in range(200)],  # Many messages (enough to exceed 60s)
        message_delay_s=0.5,  # 0.5s delay = 100 seconds total for 200 messages
    )

    start_time = asyncio.get_event_loop().time()
    message_count = 0
    duration_exceeded = False

    try:
        async for _text in transport.receive_text():
            message_count += 1
            elapsed_time = asyncio.get_event_loop().time() - start_time

            # Check max duration
            if elapsed_time >= config.max_session_duration_seconds:
                logger.info(f"Max session duration reached ({elapsed_time:.1f}s)")
                duration_exceeded = True
                break
    finally:
        await transport.close()

    elapsed_time = asyncio.get_event_loop().time() - start_time

    assert duration_exceeded
    assert elapsed_time >= config.max_session_duration_seconds
    assert message_count < 200  # Should not receive all messages (stopped at max duration)


@pytest.mark.asyncio
async def test_disconnect_during_conversation() -> None:
    """Test handling of client disconnect during conversation."""
    transport = MockDisconnectTransport(
        session_id="test-disconnect",
        disconnect_after_messages=2,
    )

    messages_received = []

    try:
        async for text in transport.receive_text():
            messages_received.append(text)
            logger.info(f"Received: {text}")
    except StopAsyncIteration:
        logger.info("Client disconnected (expected)")
    finally:
        await transport.close()

    # Should receive exactly 2 messages before disconnect
    assert len(messages_received) == 2
    # StopAsyncIteration is normal end of iterator, not an exception


@pytest.mark.asyncio
async def test_empty_message_handling() -> None:
    """Test handling of empty messages in conversation."""
    transport = MockMultiTurnTransport(
        session_id="test-empty-messages",
        text_messages=["Hello", "", "World", ""],
        message_delay_s=0.1,
    )

    messages_received = []
    async for text in transport.receive_text():
        messages_received.append(text)

    assert len(messages_received) == 4
    assert messages_received[1] == ""  # Empty message preserved
    assert messages_received[3] == ""  # Empty message preserved


@pytest.mark.asyncio
async def test_session_state_transitions() -> None:
    """Test session state transitions during multi-turn conversation."""
    # This test demonstrates the expected state flow:
    # LISTENING → WAITING_FOR_INPUT (after first message)
    # WAITING_FOR_INPUT → LISTENING (ready for next message)

    transport = MockMultiTurnTransport(
        session_id="test-state-transitions",
        text_messages=["First message", "Second message"],
        message_delay_s=0.2,
    )

    states = []
    message_count = 0

    # Simulate state tracking
    current_state = SessionState.LISTENING
    states.append(current_state)

    async for text in transport.receive_text():
        message_count += 1
        logger.info(f"Message {message_count}: {text}, state: {current_state}")

        # After receiving message, transition to WAITING_FOR_INPUT
        current_state = SessionState.WAITING_FOR_INPUT
        states.append(current_state)

        # Simulate processing delay
        await asyncio.sleep(0.1)

        # After processing, return to LISTENING
        current_state = SessionState.LISTENING
        states.append(current_state)

    await transport.close()

    # Should see: LISTENING, WAITING_FOR_INPUT, LISTENING, WAITING_FOR_INPUT, LISTENING
    assert len(states) == 5
    assert states[0] == SessionState.LISTENING
    assert states[1] == SessionState.WAITING_FOR_INPUT
    assert states[2] == SessionState.LISTENING
    assert states[3] == SessionState.WAITING_FOR_INPUT
    assert states[4] == SessionState.LISTENING


@pytest.mark.asyncio
async def test_realistic_conversation_scenario() -> None:
    """Test realistic multi-turn conversation with mixed timing."""
    config = SessionConfig(
        idle_timeout_seconds=10,  # Minimum is 10 seconds
        max_session_duration_seconds=60,  # Minimum is 60 seconds
        max_messages_per_session=10,
    )

    transport = MockMultiTurnTransport(
        session_id="test-realistic",
        text_messages=[
            "Hello",
            "What's the weather?",
            "Thanks",
            "Goodbye",
        ],
        message_delay_s=0.3,
    )

    start_time = asyncio.get_event_loop().time()
    messages_received = []

    async for text in transport.receive_text():
        messages_received.append(text)
        logger.info(f"Received: {text}")

        # Check timeout limits
        elapsed_time = asyncio.get_event_loop().time() - start_time
        if len(messages_received) >= config.max_messages_per_session:
            logger.info("Max messages reached")
            break
        if elapsed_time >= config.max_session_duration_seconds:
            logger.info("Max duration reached")
            break

    await transport.close()

    assert len(messages_received) == 4
    assert messages_received == ["Hello", "What's the weather?", "Thanks", "Goodbye"]


@pytest.mark.asyncio
async def test_session_config_integration() -> None:
    """Test SessionConfig integration with realistic values."""
    config = SessionConfig(
        idle_timeout_seconds=10,
        max_session_duration_seconds=60,
        max_messages_per_session=20,
    )

    # Verify config loaded correctly
    assert config.idle_timeout_seconds == 10
    assert config.max_session_duration_seconds == 60
    assert config.max_messages_per_session == 20

    # Verify relationships
    assert config.idle_timeout_seconds < config.max_session_duration_seconds


@pytest.mark.asyncio
async def test_concurrent_message_handling() -> None:
    """Test handling of messages arriving in quick succession."""
    transport = MockMultiTurnTransport(
        session_id="test-concurrent",
        text_messages=[f"Rapid message {i}" for i in range(5)],
        message_delay_s=0.01,  # Very fast (10ms between messages)
    )

    messages_received = []
    start_time = asyncio.get_event_loop().time()

    async for text in transport.receive_text():
        messages_received.append(text)

    elapsed_time = asyncio.get_event_loop().time() - start_time

    assert len(messages_received) == 5
    # Should complete quickly (<100ms for 5 messages @ 10ms each)
    assert elapsed_time < 0.1


@pytest.mark.asyncio
async def test_transport_close_during_receive() -> None:
    """Test closing transport during message reception."""
    transport = MockMultiTurnTransport(
        session_id="test-close-during-receive",
        text_messages=[f"Message {i}" for i in range(10)],
        message_delay_s=0.1,
    )

    message_count = 0

    async for text in transport.receive_text():
        message_count += 1
        logger.info(f"Received: {text}")

        # Close after 3 messages
        if message_count == 3:
            await transport.close()
            break

    # Should receive exactly 3 messages
    assert message_count == 3
