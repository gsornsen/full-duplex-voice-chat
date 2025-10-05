"""Unit tests for LiveKit transport implementation.

Tests LiveKitSession and LiveKitTransport classes with mocked LiveKit SDK
objects to verify session lifecycle, audio frame handling, and text reception.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from livekit import rtc

from src.orchestrator.config import LiveKitConfig
from src.orchestrator.livekit.room_manager import LiveKitRoomManager
from src.orchestrator.transport.livekit_transport import (
    LiveKitSession,
    LiveKitTransport,
)


@pytest.fixture
def livekit_config() -> LiveKitConfig:
    """Create LiveKit configuration for testing."""
    return LiveKitConfig(
        enabled=True,
        url="http://localhost:7880",
        api_key="test-key",
        api_secret="test-secret",
        room_prefix="test-room",
    )


@pytest.fixture
def mock_room() -> Mock:
    """Create mock LiveKit Room."""
    room = Mock(spec=rtc.Room)
    room.connection_state = rtc.ConnectionState.CONN_CONNECTED
    room.local_participant = Mock()
    room.local_participant.publish_track = AsyncMock()
    room.local_participant.unpublish_track = AsyncMock()
    room.disconnect = AsyncMock()
    room.connect = AsyncMock()
    room.on = Mock()
    return room


@pytest.fixture
def mock_participant() -> Mock:
    """Create mock RemoteParticipant."""
    participant = Mock(spec=rtc.RemoteParticipant)
    participant.identity = "test-participant"
    participant.sid = "participant-123"
    return participant


@pytest.fixture
def livekit_session(mock_room: Mock, mock_participant: Mock) -> LiveKitSession:
    """Create LiveKitSession instance for testing."""
    return LiveKitSession(
        room=mock_room,
        participant=mock_participant,
        session_id="test-session-123",
        room_name="test-room-abc",
    )


class TestLiveKitSession:
    """Test suite for LiveKitSession."""

    def test_session_initialization(
        self, livekit_session: LiveKitSession, mock_room: Mock, mock_participant: Mock
    ) -> None:
        """Test session is properly initialized."""
        assert livekit_session.session_id == "test-session-123"
        assert livekit_session.is_connected
        assert livekit_session._room == mock_room
        assert livekit_session._participant == mock_participant
        assert livekit_session._room_name == "test-room-abc"

    def test_session_id_property(self, livekit_session: LiveKitSession) -> None:
        """Test session_id property returns correct value."""
        assert livekit_session.session_id == "test-session-123"

    def test_is_connected_when_room_connected(
        self, livekit_session: LiveKitSession, mock_room: Mock
    ) -> None:
        """Test is_connected returns True when room is connected."""
        mock_room.connection_state = rtc.ConnectionState.CONN_CONNECTED
        assert livekit_session.is_connected

    def test_is_connected_when_room_disconnected(
        self, livekit_session: LiveKitSession, mock_room: Mock
    ) -> None:
        """Test is_connected returns False when room is disconnected."""
        mock_room.connection_state = rtc.ConnectionState.CONN_DISCONNECTED
        assert not livekit_session.is_connected

    async def test_initialize_audio_track(self, livekit_session: LiveKitSession) -> None:
        """Test audio track initialization."""
        with (
            patch("src.orchestrator.transport.livekit_transport.rtc.AudioSource") as mock_source,
            patch(
                "src.orchestrator.transport.livekit_transport.rtc.LocalAudioTrack"
            ) as mock_track_class,
        ):
            mock_audio_source = Mock()
            mock_source.return_value = mock_audio_source

            mock_audio_track = Mock()
            mock_track_class.create_audio_track.return_value = mock_audio_track

            await livekit_session.initialize_audio_track()

            # Verify audio source created with correct parameters
            mock_source.assert_called_once_with(48000, num_channels=1)

            # Verify track created
            mock_track_class.create_audio_track.assert_called_once_with(
                "orchestrator-audio",
                mock_audio_source,
            )

            # Verify track published
            livekit_session._room.local_participant.publish_track.assert_called_once()  # type: ignore[attr-defined]

            # Verify internal state
            assert livekit_session._audio_source == mock_audio_source
            assert livekit_session._audio_track == mock_audio_track

    async def test_send_audio_frame_without_initialization(
        self, livekit_session: LiveKitSession
    ) -> None:
        """Test sending audio frame before initialization raises error."""
        # Create valid 20ms frame at 48kHz (1920 bytes)
        frame = b"\x00" * 1920

        with pytest.raises(RuntimeError, match="Audio track not initialized"):
            await livekit_session.send_audio_frame(frame)

    async def test_send_audio_frame_invalid_size(self, livekit_session: LiveKitSession) -> None:
        """Test sending audio frame with invalid size raises ValueError."""
        # Initialize audio track
        livekit_session._audio_source = Mock()

        # Create invalid frame (wrong size)
        invalid_frame = b"\x00" * 1000

        with pytest.raises(ValueError, match="Invalid frame size"):
            await livekit_session.send_audio_frame(invalid_frame)

    async def test_send_audio_frame_when_disconnected(
        self, livekit_session: LiveKitSession, mock_room: Mock
    ) -> None:
        """Test sending audio frame when disconnected raises ConnectionError."""
        # Disconnect room
        mock_room.connection_state = rtc.ConnectionState.CONN_DISCONNECTED

        frame = b"\x00" * 1920

        with pytest.raises(ConnectionError, match="LiveKit connection is closed"):
            await livekit_session.send_audio_frame(frame)

    async def test_send_audio_frame_success(self, livekit_session: LiveKitSession) -> None:
        """Test successful audio frame sending."""
        # Create valid frame
        frame = b"\x00" * 1920

        # Mock audio source
        mock_audio_source = Mock()
        mock_audio_source.capture_frame = AsyncMock()
        livekit_session._audio_source = mock_audio_source

        with (
            patch(
                "src.orchestrator.transport.livekit_transport.rtc.AudioFrame"
            ) as mock_frame_class,
            patch("src.orchestrator.transport.livekit_transport.np.frombuffer") as mock_frombuffer,
            patch("src.orchestrator.transport.livekit_transport.np.copyto"),
            patch("src.orchestrator.transport.livekit_transport.np.asarray") as mock_asarray,
        ):
            mock_audio_frame = Mock()
            mock_audio_frame.data = bytearray(1920)
            mock_frame_class.create.return_value = mock_audio_frame

            # Mock numpy operations
            mock_pcm_data = Mock()
            mock_frombuffer.return_value = mock_pcm_data
            mock_asarray.return_value = mock_audio_frame.data

            # Send frame
            await livekit_session.send_audio_frame(frame)

            # Verify frame creation
            mock_frame_class.create.assert_called_once_with(
                sample_rate=48000,
                num_channels=1,
                samples_per_channel=960,
            )

            # Verify frame captured
            mock_audio_source.capture_frame.assert_called_once()

            # Verify sequence number incremented
            assert livekit_session._sequence_number == 1

    async def test_receive_text_yields_messages(self, livekit_session: LiveKitSession) -> None:
        """Test receive_text yields messages from queue."""
        # Queue some text messages
        await livekit_session._text_queue.put("Hello")
        await livekit_session._text_queue.put("World")

        # Mark as disconnected after messages to stop iteration
        async def delayed_disconnect() -> None:
            await asyncio.sleep(0.1)
            livekit_session._connected = False

        asyncio.create_task(delayed_disconnect())

        # Collect messages
        messages = []
        async for text in livekit_session.receive_text():
            messages.append(text)
            if len(messages) >= 2:
                livekit_session._connected = False

        assert messages == ["Hello", "World"]

    async def test_receive_text_stops_when_disconnected(
        self, livekit_session: LiveKitSession, mock_room: Mock
    ) -> None:
        """Test receive_text stops when connection is closed."""
        # Disconnect immediately
        livekit_session._connected = False
        mock_room.connection_state = rtc.ConnectionState.CONN_DISCONNECTED

        # Should not yield anything
        messages = []
        async for text in livekit_session.receive_text():
            messages.append(text)

        assert messages == []

    def test_on_data_received(
        self, livekit_session: LiveKitSession, mock_participant: Mock
    ) -> None:
        """Test data channel message handling."""
        # Simulate receiving data
        data = b"Test message"

        livekit_session.on_data_received(data, mock_participant)

        # Verify message queued
        assert not livekit_session._text_queue.empty()
        assert livekit_session._text_queue.get_nowait() == "Test message"

    def test_on_data_received_invalid_utf8(
        self, livekit_session: LiveKitSession, mock_participant: Mock
    ) -> None:
        """Test handling of invalid UTF-8 data."""
        # Invalid UTF-8 sequence
        invalid_data = b"\xff\xfe"

        # Should not raise, but log error
        livekit_session.on_data_received(invalid_data, mock_participant)

        # Queue should be empty
        assert livekit_session._text_queue.empty()

    async def test_close_session(self, livekit_session: LiveKitSession, mock_room: Mock) -> None:
        """Test session close cleans up resources."""
        # Set up audio track
        mock_track = Mock()
        mock_track.sid = "track-123"
        livekit_session._audio_track = mock_track

        await livekit_session.close()

        # Verify track unpublished
        mock_room.local_participant.unpublish_track.assert_called_once_with("track-123")

        # Verify room disconnected
        mock_room.disconnect.assert_called_once()

        # Verify connection state updated
        assert not livekit_session._connected
        assert livekit_session._disconnect_event.is_set()

    async def test_close_session_idempotent(
        self, livekit_session: LiveKitSession, mock_room: Mock
    ) -> None:
        """Test closing session multiple times is safe."""
        await livekit_session.close()
        await livekit_session.close()

        # Should only disconnect once
        assert mock_room.disconnect.call_count == 1


class TestLiveKitTransport:
    """Test suite for LiveKitTransport."""

    def test_transport_initialization(self, livekit_config: LiveKitConfig) -> None:
        """Test transport is properly initialized."""
        transport = LiveKitTransport(livekit_config)

        assert transport.transport_type == "livekit"
        assert not transport.is_running
        assert transport._config == livekit_config

    async def test_start_transport_success(self, livekit_config: LiveKitConfig) -> None:
        """Test starting transport successfully."""
        with patch.object(LiveKitRoomManager, "list_rooms", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []

            transport = LiveKitTransport(livekit_config)

            await transport.start()

            assert transport.is_running
            mock_list.assert_called_once()

    async def test_start_transport_already_running(self, livekit_config: LiveKitConfig) -> None:
        """Test starting transport when already running raises error."""
        with patch.object(LiveKitRoomManager, "list_rooms", new_callable=AsyncMock):
            transport = LiveKitTransport(livekit_config)
            await transport.start()

            with pytest.raises(RuntimeError, match="already running"):
                await transport.start()

    async def test_start_transport_connection_failure(self, livekit_config: LiveKitConfig) -> None:
        """Test starting transport with connection failure."""
        with patch.object(
            LiveKitRoomManager,
            "list_rooms",
            new_callable=AsyncMock,
            side_effect=Exception("Connection failed"),
        ):
            transport = LiveKitTransport(livekit_config)

            with pytest.raises(RuntimeError, match="Failed to start LiveKit transport"):
                await transport.start()

            assert not transport.is_running

    async def test_stop_transport(self, livekit_config: LiveKitConfig) -> None:
        """Test stopping transport cleans up resources."""
        with (
            patch.object(LiveKitRoomManager, "list_rooms", new_callable=AsyncMock),
            patch.object(LiveKitRoomManager, "delete_room", new_callable=AsyncMock) as mock_delete,
            patch.object(LiveKitRoomManager, "close", new_callable=AsyncMock) as mock_close,
        ):
            transport = LiveKitTransport(livekit_config)
            await transport.start()

            # Add mock session
            mock_session = Mock()
            mock_session.session_id = "test-123"
            mock_session.close = AsyncMock()
            transport._active_sessions["test-123"] = mock_session

            # Add mock room
            transport._active_rooms["room-123"] = Mock()

            await transport.stop()

            assert not transport.is_running
            assert len(transport._active_sessions) == 0
            assert len(transport._active_rooms) == 0

            # Verify session closed
            mock_session.close.assert_called_once()

            # Verify room deleted
            mock_delete.assert_called_once_with("room-123")

            # Verify room manager closed
            mock_close.assert_called_once()

    async def test_stop_transport_idempotent(self, livekit_config: LiveKitConfig) -> None:
        """Test stopping transport multiple times is safe."""
        with (
            patch.object(LiveKitRoomManager, "list_rooms", new_callable=AsyncMock),
            patch.object(LiveKitRoomManager, "close", new_callable=AsyncMock),
        ):
            transport = LiveKitTransport(livekit_config)
            await transport.start()

            await transport.stop()
            await transport.stop()

            assert not transport.is_running

    async def test_accept_session_when_not_running(self, livekit_config: LiveKitConfig) -> None:
        """Test accept_session raises error when transport not running."""
        transport = LiveKitTransport(livekit_config)

        with pytest.raises(RuntimeError, match="not running"):
            await transport.accept_session()

    async def test_accept_session_returns_queued_session(
        self, livekit_config: LiveKitConfig
    ) -> None:
        """Test accept_session returns session from queue."""
        with patch.object(LiveKitRoomManager, "list_rooms", new_callable=AsyncMock):
            transport = LiveKitTransport(livekit_config)
            await transport.start()

            # Queue a mock session
            mock_session = Mock(spec=LiveKitSession)
            await transport._session_queue.put(mock_session)

            # Accept session
            session = await transport.accept_session()

            assert session == mock_session

    async def test_create_room_and_wait_for_participant_timeout(
        self, livekit_config: LiveKitConfig
    ) -> None:
        """Test timeout when waiting for participant."""
        with (
            patch.object(LiveKitRoomManager, "list_rooms", new_callable=AsyncMock),
            patch.object(LiveKitRoomManager, "create_room", new_callable=AsyncMock) as mock_create,
            patch.object(LiveKitRoomManager, "delete_room", new_callable=AsyncMock),
            patch("src.orchestrator.transport.livekit_transport.rtc.Room") as mock_room_class,
        ):
            mock_create.return_value = "test-room-123"

            mock_room = Mock()
            mock_room.connect = AsyncMock()
            mock_room.disconnect = AsyncMock()
            mock_room.on = Mock()
            mock_room_class.return_value = mock_room

            transport = LiveKitTransport(livekit_config)
            await transport.start()

            # Should timeout waiting for participant (wrapped in RuntimeError)
            with pytest.raises(RuntimeError, match="Failed to create session"):
                await transport.create_room_and_wait_for_participant(timeout_seconds=0.1)

    async def test_create_room_and_wait_for_participant_success(
        self, livekit_config: LiveKitConfig
    ) -> None:
        """Test successful participant connection."""
        with (
            patch.object(LiveKitRoomManager, "list_rooms", new_callable=AsyncMock),
            patch.object(LiveKitRoomManager, "create_room", new_callable=AsyncMock) as mock_create,
            patch.object(LiveKitRoomManager, "create_access_token") as mock_token,
            patch("src.orchestrator.transport.livekit_transport.rtc.Room") as mock_room_class,
            patch.object(LiveKitSession, "initialize_audio_track", new_callable=AsyncMock),
        ):
            mock_create.return_value = "test-room-123"
            mock_token.return_value = "test-token"

            # Mock room with participant event
            mock_room = Mock()
            mock_room.connect = AsyncMock()
            mock_room.disconnect = AsyncMock()
            mock_room.connection_state = rtc.ConnectionState.CONN_CONNECTED

            # Track event handlers
            event_handlers: dict[str, Any] = {}

            def on_handler(event: str, callback: Any) -> None:
                event_handlers[event] = callback

            mock_room.on = on_handler
            mock_room_class.return_value = mock_room

            transport = LiveKitTransport(livekit_config)
            await transport.start()

            # Create task to simulate participant joining
            async def simulate_participant_join() -> None:
                await asyncio.sleep(0.05)
                mock_participant = Mock(spec=rtc.RemoteParticipant)
                mock_participant.identity = "test-user"
                if "participant_connected" in event_handlers:
                    event_handlers["participant_connected"](mock_participant)

            asyncio.create_task(simulate_participant_join())

            # Wait for participant
            session = await transport.create_room_and_wait_for_participant(timeout_seconds=1.0)

            assert session is not None
            assert session.session_id.startswith("lk-")
            assert "test-room-123" in transport._active_rooms
            assert session.session_id in transport._active_sessions

    async def test_transport_type_property(self, livekit_config: LiveKitConfig) -> None:
        """Test transport_type property returns correct value."""
        transport = LiveKitTransport(livekit_config)
        assert transport.transport_type == "livekit"

    async def test_is_running_property(self, livekit_config: LiveKitConfig) -> None:
        """Test is_running property reflects state correctly."""
        with (
            patch.object(LiveKitRoomManager, "list_rooms", new_callable=AsyncMock),
            patch.object(LiveKitRoomManager, "close", new_callable=AsyncMock),
        ):
            transport = LiveKitTransport(livekit_config)

            assert not transport.is_running

            await transport.start()
            assert transport.is_running

            await transport.stop()
            assert not transport.is_running
