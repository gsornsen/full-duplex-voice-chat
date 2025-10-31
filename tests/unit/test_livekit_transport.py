"""Unit tests for LiveKit transport implementation.

Tests LiveKitSession and LiveKitTransport classes with mocked LiveKit SDK
objects to verify session lifecycle, audio frame handling, and text reception.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from livekit import rtc

from orchestrator.config import LiveKitConfig
from orchestrator.livekit_utils.room_manager import LiveKitRoomManager
from orchestrator.transport.livekit_transport import (
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
def mock_room_manager() -> Mock:
    """Create mock LiveKitRoomManager to avoid real aiohttp connections.

    This fixture prevents the creation of real aiohttp.ClientSession instances
    that would leak resources in unit tests.
    """
    manager = Mock(spec=LiveKitRoomManager)
    manager.list_rooms = AsyncMock(return_value=[])
    manager.create_room = AsyncMock()
    manager.delete_room = AsyncMock()
    manager.close = AsyncMock()
    manager.create_access_token = Mock(return_value="mock-token-123")
    manager.generate_room_name = Mock(return_value="test-room-abc")
    return manager


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
            patch("orchestrator.transport.livekit_transport.rtc.AudioSource") as mock_source,
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

    async def test_send_audio_frame_success(self, livekit_session: LiveKitSession) -> None:
        """Test sending valid audio frame."""
        with (
            patch("orchestrator.transport.livekit_transport.rtc.AudioFrame") as mock_frame_class, # noqa: E501
            patch("orchestrator.transport.livekit_transport.np.frombuffer") as mock_frombuffer,
            patch("orchestrator.transport.livekit_transport.np.copyto"),
            patch("orchestrator.transport.livekit_transport.np.asarray"),
        ):
            # Setup mocks
            mock_audio_source = AsyncMock()
            livekit_session._audio_source = mock_audio_source

            mock_frame = Mock()
            mock_frame.data = bytearray(1920)
            mock_frame_class.create.return_value = mock_frame

            mock_pcm = Mock()
            mock_frombuffer.return_value = mock_pcm

            # Create valid 20ms frame at 48kHz (1920 bytes)
            frame = b"\x00" * 1920

            await livekit_session.send_audio_frame(frame)

            # Verify frame created with correct parameters
            mock_frame_class.create.assert_called_once_with(
                sample_rate=48000,
                num_channels=1,
                samples_per_channel=960,  # 1920 bytes / 2 bytes per sample
            )

            # Verify frame captured
            mock_audio_source.capture_frame.assert_called_once_with(mock_frame)

    async def test_receive_text_not_implemented(self, livekit_session: LiveKitSession) -> None:
        """Test receive_text async generator (currently not implemented)."""
        # For now, just verify the method exists and can be called
        # Implementation depends on LiveKit data channels
        text_gen = livekit_session.receive_text()
        assert text_gen is not None

    async def test_close_session(self, livekit_session: LiveKitSession, mock_room: Mock) -> None:
        """Test closing session disconnects room."""
        await livekit_session.close()

        mock_room.disconnect.assert_called_once()


class TestLiveKitTransport:
    """Test suite for LiveKitTransport."""

    def test_transport_initialization(
        self, livekit_config: LiveKitConfig, mock_room_manager: Mock
    ) -> None:
        """Test transport is properly initialized."""
        with patch(
            "src.orchestrator.transport.livekit_transport.LiveKitRoomManager",
            return_value=mock_room_manager,
        ):
            transport = LiveKitTransport(livekit_config)

            assert transport.transport_type == "livekit"
            assert not transport.is_running
            assert transport._config == livekit_config

    async def test_start_transport_success(
        self, livekit_config: LiveKitConfig, mock_room_manager: Mock
    ) -> None:
        """Test starting transport successfully."""
        with patch(
            "src.orchestrator.transport.livekit_transport.LiveKitRoomManager",
            return_value=mock_room_manager,
        ):
            transport = LiveKitTransport(livekit_config)

            await transport.start()

            assert transport.is_running
            mock_room_manager.list_rooms.assert_called_once()

            # Clean up
            await transport.stop()

    async def test_start_transport_already_running(
        self, livekit_config: LiveKitConfig, mock_room_manager: Mock
    ) -> None:
        """Test starting transport when already running raises error."""
        with patch(
            "src.orchestrator.transport.livekit_transport.LiveKitRoomManager",
            return_value=mock_room_manager,
        ):
            transport = LiveKitTransport(livekit_config)
            await transport.start()

            with pytest.raises(RuntimeError, match="already running"):
                await transport.start()

            # Clean up
            await transport.stop()

    async def test_start_transport_connection_failure(
        self, livekit_config: LiveKitConfig, mock_room_manager: Mock
    ) -> None:
        """Test starting transport with connection failure."""
        mock_room_manager.list_rooms.side_effect = Exception("Connection failed")

        with patch(
            "src.orchestrator.transport.livekit_transport.LiveKitRoomManager",
            return_value=mock_room_manager,
        ):
            transport = LiveKitTransport(livekit_config)

            with pytest.raises(RuntimeError, match="Failed to start LiveKit transport"):
                await transport.start()

            assert not transport.is_running

    async def test_stop_transport(
        self, livekit_config: LiveKitConfig, mock_room_manager: Mock
    ) -> None:
        """Test stopping transport cleans up resources."""
        with patch(
            "src.orchestrator.transport.livekit_transport.LiveKitRoomManager",
            return_value=mock_room_manager,
        ):
            transport = LiveKitTransport(livekit_config)
            await transport.start()

            # Add mock session
            mock_session = Mock()
            mock_session.session_id = "test-123"
            mock_session.close = AsyncMock()
            transport._active_sessions["test-123"] = mock_session

            # Add mock room
            transport._active_rooms["test-room"] = Mock()

            await transport.stop()

            # Verify cleanup
            assert not transport.is_running
            mock_session.close.assert_called_once()
            mock_room_manager.delete_room.assert_called_once_with("test-room")
            mock_room_manager.close.assert_called_once()

    async def test_stop_transport_idempotent(
        self, livekit_config: LiveKitConfig, mock_room_manager: Mock
    ) -> None:
        """Test stopping transport multiple times is safe."""
        with patch(
            "src.orchestrator.transport.livekit_transport.LiveKitRoomManager",
            return_value=mock_room_manager,
        ):
            transport = LiveKitTransport(livekit_config)

            # Stop without starting
            await transport.stop()

            # Start and stop twice
            await transport.start()
            await transport.stop()
            await transport.stop()

            assert not transport.is_running

    async def test_accept_session_not_running(
        self, livekit_config: LiveKitConfig, mock_room_manager: Mock
    ) -> None:
        """Test accept_session raises error when transport not running."""
        with patch(
            "src.orchestrator.transport.livekit_transport.LiveKitRoomManager",
            return_value=mock_room_manager,
        ):
            transport = LiveKitTransport(livekit_config)

            with pytest.raises(RuntimeError, match="not running"):
                await asyncio.wait_for(transport.accept_session(), timeout=0.1)

