"""Unit tests for transport interface compliance.

Tests the base transport abstraction and ensures implementations
conform to the interface contract.
"""

from collections.abc import AsyncIterator

import pytest

from orchestrator.transport.base import Transport, TransportSession


class MockTransportSession(TransportSession):
    """Mock transport session for testing interface compliance."""

    def __init__(self, session_id: str = "test-session-001") -> None:
        self._session_id = session_id
        self._connected = True
        self._frames_sent: list[bytes] = []

    async def send_audio_frame(self, frame: bytes) -> None:
        """Mock send audio frame."""
        if not self._connected:
            raise ConnectionError("Session disconnected")
        if len(frame) != 1920:  # 960 samples * 2 bytes
            raise ValueError(f"Invalid frame size: {len(frame)}, expected 1920")
        self._frames_sent.append(frame)

    async def receive_text(self) -> AsyncIterator[str]:
        """Mock receive text."""
        if self._connected:
            yield "Hello"
            yield "World"

    async def close(self) -> None:
        """Mock close."""
        self._connected = False

    @property
    def session_id(self) -> str:
        """Return session ID."""
        return self._session_id

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected


class MockTransport(Transport):
    """Mock transport for testing interface compliance."""

    def __init__(self) -> None:
        self._running = False
        self._sessions: list[MockTransportSession] = []

    async def start(self) -> None:
        """Mock start."""
        if self._running:
            raise RuntimeError("Transport already running")
        self._running = True

    async def stop(self) -> None:
        """Mock stop."""
        self._running = False
        for session in self._sessions:
            await session.close()
        self._sessions.clear()

    async def accept_session(self) -> TransportSession:
        """Mock accept session."""
        if not self._running:
            raise RuntimeError("Transport not running")
        session = MockTransportSession(f"session-{len(self._sessions)}")
        self._sessions.append(session)
        return session

    @property
    def transport_type(self) -> str:
        """Return transport type."""
        return "mock"

    @property
    def is_running(self) -> bool:
        """Check running status."""
        return self._running


@pytest.mark.asyncio
async def test_transport_lifecycle() -> None:
    """Test transport lifecycle (start/stop)."""
    transport = MockTransport()

    # Initial state
    assert not transport.is_running

    # Start transport
    await transport.start()
    assert transport.is_running

    # Starting again should fail
    with pytest.raises(RuntimeError, match="already running"):
        await transport.start()

    # Stop transport
    await transport.stop()
    assert not transport.is_running


@pytest.mark.asyncio
async def test_transport_accept_session_not_running() -> None:
    """Test that accepting session fails when transport not running."""
    transport = MockTransport()

    with pytest.raises(RuntimeError, match="not running"):
        await transport.accept_session()


@pytest.mark.asyncio
async def test_transport_accept_session() -> None:
    """Test accepting a new session."""
    transport = MockTransport()
    await transport.start()

    session = await transport.accept_session()
    assert isinstance(session, TransportSession)
    assert session.is_connected
    assert session.session_id == "session-0"

    await transport.stop()


@pytest.mark.asyncio
async def test_transport_session_send_audio() -> None:
    """Test sending audio frames through session."""
    session = MockTransportSession()

    # Valid frame (1920 bytes)
    frame = b"\x00" * 1920
    await session.send_audio_frame(frame)
    assert len(session._frames_sent) == 1

    # Invalid frame size
    with pytest.raises(ValueError, match="Invalid frame size"):
        await session.send_audio_frame(b"\x00" * 1000)


@pytest.mark.asyncio
async def test_transport_session_send_audio_disconnected() -> None:
    """Test that sending audio fails when disconnected."""
    session = MockTransportSession()
    await session.close()

    with pytest.raises(ConnectionError, match="disconnected"):
        await session.send_audio_frame(b"\x00" * 1920)


@pytest.mark.asyncio
async def test_transport_session_receive_text() -> None:
    """Test receiving text from session."""
    session = MockTransportSession()

    chunks = []
    async for chunk in session.receive_text():
        chunks.append(chunk)

    assert chunks == ["Hello", "World"]


@pytest.mark.asyncio
async def test_transport_session_properties() -> None:
    """Test session property accessors."""
    session = MockTransportSession("custom-id")

    assert session.session_id == "custom-id"
    assert session.is_connected is True

    await session.close()
    assert session.is_connected is False


@pytest.mark.asyncio
async def test_transport_type_identifier() -> None:
    """Test transport type identifier."""
    transport = MockTransport()
    assert transport.transport_type == "mock"


@pytest.mark.asyncio
async def test_transport_stop_closes_sessions() -> None:
    """Test that stopping transport closes all sessions."""
    transport = MockTransport()
    await transport.start()

    # Create multiple sessions
    session1 = await transport.accept_session()
    session2 = await transport.accept_session()

    assert session1.is_connected
    assert session2.is_connected

    # Stop transport
    await transport.stop()

    # Sessions should be closed
    assert not session1.is_connected
    assert not session2.is_connected
