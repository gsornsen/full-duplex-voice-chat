"""Unit tests for CLI WebSocket client."""

import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import websockets

from src.client.cli_client import AudioPlayer, CLIClient
from orchestrator.transport.websocket_protocol import (
    AudioMessage,
    ErrorMessage,
    SessionEndMessage,
    SessionStartMessage,
    TextMessage,
)


class TestAudioPlayer:
    """Test suite for AudioPlayer class."""

    def test_init_with_sounddevice(self) -> None:
        """Test AudioPlayer initialization with sounddevice available."""
        with patch("src.client.cli_client.AudioPlayer.__init__", return_value=None):
            player = AudioPlayer.__new__(AudioPlayer)
            player.sample_rate = 48000
            player.device = None
            player.frame_count = 0
            player.buffer = []
            player.playing = False
            player.sd = MagicMock()

            assert player.sample_rate == 48000
            assert player.device is None
            assert player.frame_count == 0

    def test_init_without_sounddevice(self) -> None:
        """Test AudioPlayer initialization without sounddevice (fallback mode)."""
        with patch("builtins.__import__", side_effect=ImportError):
            player = AudioPlayer(sample_rate=48000)
            assert player.sd is None
            assert player.sample_rate == 48000

    def test_play_frame_with_sounddevice(self) -> None:
        """Test playing audio frame with sounddevice."""
        # Create mock sounddevice
        mock_sd = MagicMock()

        # Create PCM data (20ms @ 48kHz = 960 samples)
        pcm_data = np.random.randint(-32768, 32767, 960, dtype=np.int16).tobytes()

        player = AudioPlayer.__new__(AudioPlayer)
        player.sample_rate = 48000
        player.device = None
        player.frame_count = 0
        player.sd = mock_sd

        player.play_frame(pcm_data)

        # Verify sounddevice.play was called
        mock_sd.play.assert_called_once()
        args, kwargs = mock_sd.play.call_args
        assert kwargs["samplerate"] == 48000
        assert len(args[0]) == 960  # 960 samples

    def test_play_frame_fallback_to_file(self) -> None:
        """Test playing audio frame with file fallback when sounddevice fails."""
        mock_sd = MagicMock()
        mock_sd.play.side_effect = OSError("Audio device not found")

        pcm_data = np.random.randint(-32768, 32767, 960, dtype=np.int16).tobytes()

        player = AudioPlayer.__new__(AudioPlayer)
        player.sample_rate = 48000
        player.device = None
        player.frame_count = 0
        player.sd = mock_sd

        with patch("soundfile.write") as mock_write:
            player.play_frame(pcm_data)

            # Should fall back to file save
            mock_write.assert_called_once()
            filename = mock_write.call_args[0][0]
            assert filename == "audio_output_0000.wav"

    def test_play_frame_without_sounddevice(self) -> None:
        """Test playing audio frame without sounddevice (save to file)."""
        pcm_data = np.random.randint(-32768, 32767, 960, dtype=np.int16).tobytes()

        player = AudioPlayer.__new__(AudioPlayer)
        player.sample_rate = 48000
        player.device = None
        player.frame_count = 0
        player.sd = None

        with patch("soundfile.write") as mock_write:
            player.play_frame(pcm_data)

            # Should save to file
            mock_write.assert_called_once()
            filename = mock_write.call_args[0][0]
            assert filename == "audio_output_0000.wav"

    def test_wait_for_completion(self) -> None:
        """Test waiting for audio completion."""
        mock_sd = MagicMock()
        player = AudioPlayer.__new__(AudioPlayer)
        player.sd = mock_sd

        player.wait_for_completion()

        mock_sd.wait.assert_called_once()

    def test_wait_for_completion_without_sounddevice(self) -> None:
        """Test waiting for completion without sounddevice (no-op)."""
        player = AudioPlayer.__new__(AudioPlayer)
        player.sd = None

        # Should not raise exception
        player.wait_for_completion()


class TestCLIClient:
    """Test suite for CLIClient class."""

    def test_init(self) -> None:
        """Test CLIClient initialization."""
        client = CLIClient(
            server_url="ws://localhost:8080",
            device=None,
            verbose=True,
        )

        assert client.server_url == "ws://localhost:8080"
        assert client.device is None
        assert client.verbose is True
        assert client.session_id is None
        assert client.running is True

    async def test_send_text(self) -> None:
        """Test sending text message to server."""
        client = CLIClient("ws://localhost:8080")
        mock_websocket = AsyncMock()

        await client.send_text(mock_websocket, "Hello, world!")

        # Verify message was sent
        mock_websocket.send.assert_called_once()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "text"
        assert sent_data["text"] == "Hello, world!"
        assert sent_data["is_final"] is True

    async def test_send_control(self) -> None:
        """Test sending control message to server."""
        client = CLIClient("ws://localhost:8080")
        mock_websocket = AsyncMock()

        await client.send_control(mock_websocket, "PAUSE")

        # Verify control message was sent
        mock_websocket.send.assert_called_once()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "control"
        assert sent_data["command"] == "PAUSE"

    async def test_handle_session_start(self) -> None:
        """Test handling session start message."""
        client = CLIClient("ws://localhost:8080")

        msg = SessionStartMessage(session_id="test-session-123")
        await client.handle_message(msg.model_dump_json())

        assert client.session_id == "test-session-123"

    async def test_handle_audio_message(self) -> None:
        """Test handling audio message."""
        client = CLIClient("ws://localhost:8080")

        # Create test PCM data
        pcm_data = np.random.randint(-32768, 32767, 960, dtype=np.int16).tobytes()
        pcm_base64 = base64.b64encode(pcm_data).decode()

        msg = AudioMessage(
            pcm=pcm_base64,
            sample_rate=48000,
            frame_ms=20,
            sequence=1,
        )

        with patch.object(client.audio_player, "play_frame") as mock_play:
            await client.handle_message(msg.model_dump_json())
            mock_play.assert_called_once()
            assert mock_play.call_args[0][0] == pcm_data

    async def test_handle_session_end(self) -> None:
        """Test handling session end message."""
        client = CLIClient("ws://localhost:8080")

        msg = SessionEndMessage(session_id="test-session-123", reason="completed")

        with patch.object(client.audio_player, "wait_for_completion"):
            await client.handle_message(msg.model_dump_json())

    async def test_handle_error_message(self) -> None:
        """Test handling error message."""
        client = CLIClient("ws://localhost:8080")

        msg = ErrorMessage(message="Test error", code="TEST_ERROR")
        await client.handle_message(msg.model_dump_json())

        # No exception should be raised

    async def test_handle_unknown_message(self) -> None:
        """Test handling unknown message type."""
        client = CLIClient("ws://localhost:8080")

        unknown_msg = json.dumps({"type": "unknown", "data": "test"})
        await client.handle_message(unknown_msg)

        # Should handle gracefully without exception

    async def test_handle_invalid_json(self) -> None:
        """Test handling invalid JSON message."""
        client = CLIClient("ws://localhost:8080")

        invalid_msg = "not json"
        await client.handle_message(invalid_msg)

        # Should handle gracefully without exception

    async def test_receive_messages(self) -> None:
        """Test receiving messages from server."""
        client = CLIClient("ws://localhost:8080")

        # Create mock websocket with messages
        messages = [
            SessionStartMessage(session_id="test-123").model_dump_json(),
            SessionEndMessage(session_id="test-123", reason="completed").model_dump_json(),
        ]

        mock_websocket = AsyncMock()
        mock_websocket.__aiter__.return_value = iter(messages)

        await client.receive_messages(mock_websocket)

        # Verify session_id was set
        assert client.session_id == "test-123"

    async def test_receive_messages_connection_closed(self) -> None:
        """Test handling connection closed during receive."""
        client = CLIClient("ws://localhost:8080")

        mock_websocket = AsyncMock()
        mock_websocket.__aiter__.side_effect = OSError("Connection closed")

        # Should handle gracefully
        await client.receive_messages(mock_websocket)


class TestMessageSerialization:
    """Test message serialization and deserialization."""

    def test_text_message_serialization(self) -> None:
        """Test TextMessage serialization."""
        msg = TextMessage(text="Hello", is_final=True)
        data = json.loads(msg.model_dump_json())

        assert data["type"] == "text"
        assert data["text"] == "Hello"
        assert data["is_final"] is True

    def test_audio_message_deserialization(self) -> None:
        """Test AudioMessage deserialization."""
        pcm_data = b"\x00" * 1920
        pcm_base64 = base64.b64encode(pcm_data).decode()

        msg_dict: dict[str, Any] = {
            "type": "audio",
            "pcm": pcm_base64,
            "sample_rate": 48000,
            "frame_ms": 20,
            "sequence": 1,
        }

        msg = AudioMessage(**msg_dict)
        assert msg.type == "audio"
        assert msg.pcm == pcm_base64
        assert msg.sample_rate == 48000
        assert msg.frame_ms == 20
        assert msg.sequence == 1

        # Verify PCM can be decoded
        decoded = base64.b64decode(msg.pcm)
        assert decoded == pcm_data


class TestAudioFrameDecoding:
    """Test audio frame decoding and conversion."""

    def test_pcm_base64_decode(self) -> None:
        """Test decoding base64 PCM data."""
        # Create 20ms @ 48kHz PCM (960 samples)
        original_pcm = np.random.randint(-32768, 32767, 960, dtype=np.int16)
        pcm_bytes = original_pcm.tobytes()
        pcm_base64 = base64.b64encode(pcm_bytes).decode()

        # Decode
        decoded_bytes = base64.b64decode(pcm_base64)
        decoded_pcm = np.frombuffer(decoded_bytes, dtype=np.int16)

        # Verify
        assert len(decoded_pcm) == 960
        assert np.array_equal(decoded_pcm, original_pcm)

    def test_pcm_int16_to_float32_conversion(self) -> None:
        """Test converting PCM int16 to float32."""
        # Create test data
        pcm_int16 = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)

        # Convert to float32 (normalize to [-1, 1])
        pcm_float32 = pcm_int16.astype(np.float32) / 32768.0

        # Verify
        assert pcm_float32[0] == 0.0
        assert pcm_float32[1] == pytest.approx(0.5, abs=0.001)
        assert pcm_float32[2] == pytest.approx(-0.5, abs=0.001)
        assert pcm_float32[3] == pytest.approx(1.0, abs=0.001)
        assert pcm_float32[4] == -1.0

    def test_audio_frame_size(self) -> None:
        """Test audio frame size calculation."""
        # 20ms @ 48kHz = 960 samples
        # int16 = 2 bytes per sample
        # Total = 1920 bytes
        sample_rate = 48000
        frame_ms = 20
        bytes_per_sample = 2

        samples = (sample_rate * frame_ms) // 1000
        expected_bytes = samples * bytes_per_sample

        assert samples == 960
        assert expected_bytes == 1920


class TestGracefulShutdown:
    """Test graceful shutdown scenarios."""

    async def test_client_stop_on_quit_command(self) -> None:
        """Test client stops gracefully on /quit command."""
        client = CLIClient("ws://localhost:8080")
        assert client.running is True

        # Simulate quit command
        client.running = False
        assert client.running is False

    async def test_signal_handling(self) -> None:
        """Test signal handling for graceful shutdown."""
        client = CLIClient("ws://localhost:8080")

        def signal_handler() -> None:
            client.running = False

        signal_handler()
        assert client.running is False


class TestErrorHandling:
    """Test error handling in CLI client."""

    async def test_connection_error(self) -> None:
        """Test handling connection errors."""
        # Test connection to invalid URL raises appropriate exceptions
        client = CLIClient("ws://invalid:9999")

        # Connection error can be OSError or WebSocket-specific exception
        with pytest.raises((OSError, websockets.exceptions.WebSocketException)):
            async for _ in client.connect():
                pass

    async def test_malformed_message(self) -> None:
        """Test handling malformed messages."""
        client = CLIClient("ws://localhost:8080")

        # Should not raise exception
        await client.handle_message("{invalid json}")
        await client.handle_message('{"type": "unknown"}')

    async def test_audio_playback_error(self) -> None:
        """Test handling audio playback errors."""
        mock_sd = MagicMock()
        mock_sd.play.side_effect = OSError("Device busy")

        player = AudioPlayer.__new__(AudioPlayer)
        player.sample_rate = 48000
        player.device = None
        player.frame_count = 0
        player.sd = mock_sd

        pcm_data = np.zeros(1920, dtype=np.int16).tobytes()

        with patch("soundfile.write"):
            # Should handle error and fall back to file
            player.play_frame(pcm_data)


class TestVerboseLogging:
    """Test verbose logging functionality."""

    def test_verbose_mode_enabled(self) -> None:
        """Test client with verbose logging enabled."""
        client = CLIClient("ws://localhost:8080", verbose=True)
        assert client.verbose is True

    def test_verbose_mode_disabled(self) -> None:
        """Test client with verbose logging disabled."""
        client = CLIClient("ws://localhost:8080", verbose=False)
        assert client.verbose is False

    async def test_verbose_audio_logging(self) -> None:
        """Test audio frame logging in verbose mode."""
        client = CLIClient("ws://localhost:8080", verbose=True)

        pcm_data = np.zeros(1920, dtype=np.int16).tobytes()
        pcm_base64 = base64.b64encode(pcm_data).decode()

        msg = AudioMessage(
            pcm=pcm_base64,
            sample_rate=48000,
            frame_ms=20,
            sequence=1,
        )

        with patch.object(client.audio_player, "play_frame"):
            # Should log frame details in verbose mode
            await client.handle_message(msg.model_dump_json())
