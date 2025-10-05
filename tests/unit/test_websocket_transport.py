"""Unit tests for WebSocket transport implementation.

Tests WebSocket message protocol, audio frame packetization, and transport
session lifecycle.
"""

import base64
import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from websockets.protocol import State

from src.orchestrator.audio.packetizer import (
    EXPECTED_FRAME_SIZE_BYTES,
    AudioFramePacketizer,
    decode_pcm_frame,
    encode_pcm_frame,
    validate_frame_size,
)
from src.orchestrator.transport.websocket_protocol import (
    AudioMessage,
    ErrorMessage,
    SessionEndMessage,
    SessionStartMessage,
    TextMessage,
)
from src.orchestrator.transport.websocket_transport import (
    WebSocketSession,
    WebSocketTransport,
)


class TestAudioFramePacketizer:
    """Test audio frame packetization utilities."""

    def test_validate_frame_size_valid(self) -> None:
        """Test frame size validation with valid 20ms frame."""
        # 20ms @ 48kHz mono = 960 samples * 2 bytes = 1920 bytes
        valid_frame = b"\x00" * EXPECTED_FRAME_SIZE_BYTES
        validate_frame_size(valid_frame)  # Should not raise

    def test_validate_frame_size_invalid(self) -> None:
        """Test frame size validation with invalid frame size."""
        invalid_frame = b"\x00" * 1000
        with pytest.raises(ValueError, match="Invalid frame size"):
            validate_frame_size(invalid_frame)

    def test_encode_pcm_frame(self) -> None:
        """Test base64 encoding of PCM frame."""
        frame = b"\x00" * EXPECTED_FRAME_SIZE_BYTES
        encoded = encode_pcm_frame(frame)

        # Should be valid base64
        assert isinstance(encoded, str)
        # Decode to verify
        decoded = base64.b64decode(encoded)
        assert decoded == frame

    def test_encode_pcm_frame_invalid_size(self) -> None:
        """Test encoding with invalid frame size."""
        invalid_frame = b"\x00" * 1000
        with pytest.raises(ValueError, match="Invalid frame size"):
            encode_pcm_frame(invalid_frame)

    def test_decode_pcm_frame(self) -> None:
        """Test base64 decoding of PCM frame."""
        frame = b"\x00" * EXPECTED_FRAME_SIZE_BYTES
        encoded = base64.b64encode(frame).decode("ascii")
        decoded = decode_pcm_frame(encoded)

        assert decoded == frame

    def test_decode_pcm_frame_invalid_base64(self) -> None:
        """Test decoding with invalid base64."""
        with pytest.raises(ValueError, match="Failed to decode"):
            decode_pcm_frame("not-valid-base64!!!")

    def test_decode_pcm_frame_invalid_size(self) -> None:
        """Test decoding with wrong frame size."""
        # Valid base64 but wrong size
        wrong_size = b"\x00" * 1000
        encoded = base64.b64encode(wrong_size).decode("ascii")

        with pytest.raises(ValueError, match="Invalid frame size"):
            decode_pcm_frame(encoded)

    def test_packetizer_sequence_numbering(self) -> None:
        """Test sequence number tracking."""
        packetizer = AudioFramePacketizer()
        frame = b"\x00" * EXPECTED_FRAME_SIZE_BYTES

        # First frame
        meta1 = packetizer.create_frame_metadata(frame)
        assert meta1["sequence"] == 1
        assert meta1["sample_rate"] == 48000
        assert meta1["frame_ms"] == 20

        # Second frame
        meta2 = packetizer.create_frame_metadata(frame)
        assert meta2["sequence"] == 2

        # Third frame
        meta3 = packetizer.create_frame_metadata(frame)
        assert meta3["sequence"] == 3

    def test_packetizer_reset(self) -> None:
        """Test sequence number reset."""
        packetizer = AudioFramePacketizer()
        frame = b"\x00" * EXPECTED_FRAME_SIZE_BYTES

        # Create a few frames
        packetizer.create_frame_metadata(frame)
        packetizer.create_frame_metadata(frame)
        assert packetizer.current_sequence == 2

        # Reset
        packetizer.reset()
        assert packetizer.current_sequence == 0

        # Next frame should be sequence 1
        meta = packetizer.create_frame_metadata(frame)
        assert meta["sequence"] == 1


class TestWebSocketProtocol:
    """Test WebSocket message protocol models."""

    def test_text_message_validation(self) -> None:
        """Test text message validation."""
        msg = TextMessage(text="Hello, world!", is_final=True)
        assert msg.type == "text"
        assert msg.text == "Hello, world!"
        assert msg.is_final is True

    def test_text_message_empty_text(self) -> None:
        """Test text message with empty text."""
        with pytest.raises(ValueError):
            TextMessage(text="", is_final=True)

    def test_audio_message_validation(self) -> None:
        """Test audio message validation."""
        frame = b"\x00" * EXPECTED_FRAME_SIZE_BYTES
        encoded = base64.b64encode(frame).decode("ascii")

        msg = AudioMessage(pcm=encoded, sequence=1)
        assert msg.type == "audio"
        assert msg.sample_rate == 48000
        assert msg.frame_ms == 20
        assert msg.sequence == 1

    def test_session_start_message(self) -> None:
        """Test session start message."""
        msg = SessionStartMessage(session_id="test-session")
        assert msg.type == "session_start"
        assert msg.session_id == "test-session"

    def test_session_end_message(self) -> None:
        """Test session end message."""
        msg = SessionEndMessage(session_id="test-session", reason="timeout")
        assert msg.type == "session_end"
        assert msg.session_id == "test-session"
        assert msg.reason == "timeout"

    def test_error_message(self) -> None:
        """Test error message."""
        msg = ErrorMessage(message="Test error", code="TEST_ERROR")
        assert msg.type == "error"
        assert msg.message == "Test error"
        assert msg.code == "TEST_ERROR"


class TestWebSocketSession:
    """Test WebSocket session implementation."""

    @pytest.fixture
    def mock_websocket(self) -> MagicMock:
        """Create mock WebSocket connection."""
        ws = MagicMock()
        ws.state = State.OPEN
        ws.remote_address = ("127.0.0.1", 12345)
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        return ws

    def test_session_initialization(self, mock_websocket: MagicMock) -> None:
        """Test session initialization."""
        session = WebSocketSession(mock_websocket, "test-session")

        assert session.session_id == "test-session"
        assert session.is_connected is True

    @pytest.mark.asyncio
    async def test_send_audio_frame(self, mock_websocket: MagicMock) -> None:
        """Test sending audio frame."""
        session = WebSocketSession(mock_websocket, "test-session")
        frame = b"\x00" * EXPECTED_FRAME_SIZE_BYTES

        await session.send_audio_frame(frame)

        # Verify WebSocket send was called
        mock_websocket.send.assert_called_once()

        # Verify message format
        sent_data = mock_websocket.send.call_args[0][0]
        msg = json.loads(sent_data)

        assert msg["type"] == "audio"
        assert msg["sample_rate"] == 48000
        assert msg["frame_ms"] == 20
        assert msg["sequence"] == 1
        assert "pcm" in msg

    @pytest.mark.asyncio
    async def test_send_audio_frame_invalid_size(self, mock_websocket: MagicMock) -> None:
        """Test sending audio frame with invalid size."""
        session = WebSocketSession(mock_websocket, "test-session")
        invalid_frame = b"\x00" * 1000

        with pytest.raises(ValueError, match="Invalid frame size"):
            await session.send_audio_frame(invalid_frame)

    @pytest.mark.asyncio
    async def test_send_audio_frame_disconnected(self, mock_websocket: MagicMock) -> None:
        """Test sending audio frame when disconnected."""
        mock_websocket.state = State.CLOSED
        session = WebSocketSession(mock_websocket, "test-session")
        frame = b"\x00" * EXPECTED_FRAME_SIZE_BYTES

        with pytest.raises(ConnectionError, match="connection is closed"):
            await session.send_audio_frame(frame)

    @pytest.mark.asyncio
    async def test_receive_text(self, mock_websocket: MagicMock) -> None:
        """Test receiving text messages."""
        # Mock incoming messages
        messages = [
            json.dumps({"type": "text", "text": "Hello", "is_final": True}),
            json.dumps({"type": "text", "text": "World", "is_final": True}),
        ]

        async def mock_iter() -> AsyncGenerator[str]:
            for msg in messages:
                yield msg

        mock_websocket.__aiter__ = lambda self: mock_iter()

        session = WebSocketSession(mock_websocket, "test-session")

        texts = []
        async for text in session.receive_text():
            texts.append(text)

        assert texts == ["Hello", "World"]

    @pytest.mark.asyncio
    async def test_send_session_start(self, mock_websocket: MagicMock) -> None:
        """Test sending session start notification."""
        session = WebSocketSession(mock_websocket, "test-session")

        await session.send_session_start()

        mock_websocket.send.assert_called_once()
        sent_data = mock_websocket.send.call_args[0][0]
        msg = json.loads(sent_data)

        assert msg["type"] == "session_start"
        assert msg["session_id"] == "test-session"

    @pytest.mark.asyncio
    async def test_close_session(self, mock_websocket: MagicMock) -> None:
        """Test session close."""
        session = WebSocketSession(mock_websocket, "test-session")

        await session.close()

        assert session.is_connected is False
        mock_websocket.close.assert_called_once()


class TestWebSocketTransport:
    """Test WebSocket transport server."""

    def test_transport_initialization(self) -> None:
        """Test transport initialization."""
        transport = WebSocketTransport(host="0.0.0.0", port=8080, max_connections=100)  # noqa: S104

        assert transport.transport_type == "websocket"
        assert transport.is_running is False

    @pytest.mark.asyncio
    async def test_transport_start_stop(self) -> None:
        """Test transport start and stop."""
        transport = WebSocketTransport(host="127.0.0.1", port=8081, max_connections=10)

        # Start transport
        await transport.start()
        assert transport.is_running is True

        # Stop transport
        await transport.stop()
        assert transport.is_running is False

    @pytest.mark.asyncio
    async def test_transport_double_start(self) -> None:
        """Test starting transport twice."""
        transport = WebSocketTransport(host="127.0.0.1", port=8082, max_connections=10)

        await transport.start()

        with pytest.raises(RuntimeError, match="already running"):
            await transport.start()

        await transport.stop()

    @pytest.mark.asyncio
    async def test_accept_session_not_running(self) -> None:
        """Test accepting session when transport not running."""
        transport = WebSocketTransport(host="127.0.0.1", port=8083, max_connections=10)

        with pytest.raises(RuntimeError, match="not running"):
            await transport.accept_session()
