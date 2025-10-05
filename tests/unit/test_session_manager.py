"""Unit tests for session manager.

Tests session state management, audio buffering, text queuing,
and metrics collection.
"""

import asyncio
from collections.abc import AsyncIterator

import pytest

from src.orchestrator.session import SessionManager, SessionMetrics, SessionState
from src.orchestrator.transport.base import TransportSession


class MockSession(TransportSession):
    """Mock transport session for testing."""

    def __init__(self, session_id: str = "test-001") -> None:
        self._session_id = session_id
        self._connected = True
        self._sent_frames: list[bytes] = []
        self._text_chunks: list[str] = []

    async def send_audio_frame(self, frame: bytes) -> None:
        """Mock send audio frame."""
        if not self._connected:
            raise ConnectionError("Not connected")
        self._sent_frames.append(frame)

    async def receive_text(self) -> AsyncIterator[str]:
        """Mock receive text."""
        for chunk in self._text_chunks:
            if not self._connected:
                break
            yield chunk

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

    def queue_text(self, text: str) -> None:
        """Add text to be yielded by receive_text."""
        self._text_chunks.append(text)


def test_session_metrics_initial_state() -> None:
    """Test initial state of session metrics."""
    metrics = SessionMetrics()

    assert metrics.first_audio_latency_ms is None
    assert metrics.frame_count == 0
    assert metrics.text_chunks_received == 0
    assert metrics.last_text_received_ts is None
    assert metrics.first_audio_sent_ts is None
    assert metrics.session_end_ts is None
    assert metrics.barge_in_count == 0


def test_session_metrics_record_text() -> None:
    """Test recording text reception."""
    metrics = SessionMetrics()

    metrics.record_text_received()
    assert metrics.text_chunks_received == 1
    assert metrics.last_text_received_ts is not None

    metrics.record_text_received()
    assert metrics.text_chunks_received == 2


def test_session_metrics_record_audio() -> None:
    """Test recording audio frame sent."""
    metrics = SessionMetrics()

    metrics.record_audio_sent()
    assert metrics.frame_count == 1
    assert metrics.first_audio_sent_ts is not None

    metrics.record_audio_sent()
    assert metrics.frame_count == 2


def test_session_metrics_first_audio_latency() -> None:
    """Test first audio latency calculation."""
    metrics = SessionMetrics()

    # Record text reception
    metrics.record_text_received()

    # Small delay
    import time

    time.sleep(0.01)

    # Record first audio frame
    metrics.record_audio_sent()

    # FAL should be calculated
    assert metrics.first_audio_latency_ms is not None
    assert metrics.first_audio_latency_ms > 0
    assert metrics.first_audio_latency_ms < 100  # Should be < 100ms in test


def test_session_metrics_frame_jitter_insufficient_data() -> None:
    """Test frame jitter with insufficient data."""
    metrics = SessionMetrics()

    # No frames
    assert metrics.compute_frame_jitter_ms() is None

    # One frame
    metrics.record_audio_sent()
    assert metrics.compute_frame_jitter_ms() is None


def test_session_metrics_frame_jitter_calculation() -> None:
    """Test frame jitter calculation with multiple frames."""
    metrics = SessionMetrics()

    # Record several frames
    for _ in range(10):
        metrics.record_audio_sent()
        import time

        time.sleep(0.001)  # Small delay

    jitter = metrics.compute_frame_jitter_ms()
    assert jitter is not None
    assert jitter >= 0


def test_session_metrics_finalize() -> None:
    """Test finalizing session metrics."""
    metrics = SessionMetrics()

    assert metrics.session_end_ts is None

    metrics.finalize()

    assert metrics.session_end_ts is not None
    assert metrics.session_end_ts > metrics.session_start_ts


@pytest.mark.asyncio
async def test_session_manager_initialization() -> None:
    """Test session manager initialization."""
    mock_transport = MockSession()
    manager = SessionManager(mock_transport)

    assert manager.session_id == "test-001"
    assert manager.state == SessionState.IDLE
    assert manager.is_active
    assert isinstance(manager.metrics, SessionMetrics)


@pytest.mark.asyncio
async def test_session_manager_queue_audio() -> None:
    """Test queueing audio frames."""
    mock_transport = MockSession()
    manager = SessionManager(mock_transport)

    frame = b"\x00" * 1920
    await manager.queue_audio_frame(frame)

    # Frame should be in buffer
    assert not manager.audio_buffer.empty()


@pytest.mark.asyncio
async def test_session_manager_queue_text() -> None:
    """Test queueing text chunks."""
    mock_transport = MockSession()
    manager = SessionManager(mock_transport)

    await manager.queue_text("Hello")

    # Text should be in queue
    assert not manager.text_queue.empty()
    assert manager.metrics.text_chunks_received == 1


@pytest.mark.asyncio
async def test_session_manager_state_transition() -> None:
    """Test session state transitions."""
    mock_transport = MockSession()
    manager = SessionManager(mock_transport)

    assert manager.state == SessionState.IDLE

    manager.transition_state(SessionState.LISTENING)
    assert manager.state == SessionState.LISTENING  # type: ignore[comparison-overlap]

    manager.transition_state(SessionState.SPEAKING)
    assert manager.state == SessionState.SPEAKING


@pytest.mark.asyncio
async def test_session_manager_shutdown() -> None:
    """Test session shutdown."""
    mock_transport = MockSession()
    manager = SessionManager(mock_transport)

    # Queue some data
    await manager.queue_audio_frame(b"\x00" * 1920)
    await manager.queue_text("test")

    assert manager.is_active

    # Shutdown
    await manager.shutdown()

    assert manager.state == SessionState.TERMINATED
    assert not manager.is_active
    assert not mock_transport.is_connected
    assert manager.audio_buffer.empty()
    assert manager.text_queue.empty()


@pytest.mark.asyncio
async def test_session_manager_audio_sender_loop() -> None:
    """Test audio sender loop."""
    mock_transport = MockSession()
    manager = SessionManager(mock_transport)

    # Queue some frames
    frames = [b"\x00" * 1920, b"\x01" * 1920, b"\x02" * 1920]
    for frame in frames:
        await manager.queue_audio_frame(frame)

    # Start sender loop
    sender_task = asyncio.create_task(manager.audio_sender_loop())

    # Give it time to process
    await asyncio.sleep(0.1)

    # Cancel the task
    sender_task.cancel()
    try:
        await sender_task
    except asyncio.CancelledError:
        pass

    # Frames should have been sent
    assert len(mock_transport._sent_frames) == 3
    assert manager.metrics.frame_count == 3


@pytest.mark.asyncio
async def test_session_manager_text_receiver_loop() -> None:
    """Test text receiver loop."""
    mock_transport = MockSession()
    mock_transport.queue_text("Hello")
    mock_transport.queue_text("World")

    manager = SessionManager(mock_transport)

    # Start receiver loop
    receiver_task = asyncio.create_task(manager.text_receiver_loop())

    # Give it time to process
    await asyncio.sleep(0.1)

    # Cancel the task
    receiver_task.cancel()
    try:
        await receiver_task
    except asyncio.CancelledError:
        pass

    # Text should have been queued
    assert manager.metrics.text_chunks_received == 2
    assert manager.text_queue.qsize() == 2


@pytest.mark.asyncio
async def test_session_manager_is_active_after_disconnect() -> None:
    """Test that session is not active after transport disconnects."""
    mock_transport = MockSession()
    manager = SessionManager(mock_transport)

    assert manager.is_active

    # Disconnect transport
    await mock_transport.close()

    assert not manager.is_active


@pytest.mark.asyncio
async def test_session_manager_get_metrics_summary() -> None:
    """Test getting metrics summary."""
    mock_transport = MockSession()
    manager = SessionManager(mock_transport)

    # Record some activity
    await manager.queue_text("test")
    await manager.queue_audio_frame(b"\x00" * 1920)

    summary = manager.get_metrics_summary()

    assert summary["session_id"] == "test-001"
    assert summary["state"] == "idle"
    assert summary["text_chunks"] == 1
    assert summary["frame_count"] == 0  # Not sent yet, just queued
    assert "session_duration_s" in summary
    assert isinstance(summary["session_duration_s"], float)


@pytest.mark.asyncio
async def test_session_manager_queue_audio_when_inactive() -> None:
    """Test that queuing audio when inactive is a no-op."""
    mock_transport = MockSession()
    manager = SessionManager(mock_transport)

    await manager.shutdown()

    # This should not raise, just return
    await manager.queue_audio_frame(b"\x00" * 1920)

    # Buffer should be empty
    assert manager.audio_buffer.empty()


@pytest.mark.asyncio
async def test_session_manager_queue_text_when_inactive() -> None:
    """Test that queuing text when inactive is a no-op."""
    mock_transport = MockSession()
    manager = SessionManager(mock_transport)

    await manager.shutdown()

    # This should not raise, just return
    await manager.queue_text("test")

    # Queue should be empty
    assert manager.text_queue.empty()
