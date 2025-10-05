"""WebSocket CLI client for testing the orchestrator.

Provides a command-line interface for connecting to the WebSocket orchestrator,
sending text input, and receiving/playing audio responses.
"""

import argparse
import asyncio
import base64
import json
import logging
import signal
import sys
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import websockets
from websockets.asyncio.client import ClientConnection

from src.orchestrator.transport.websocket_protocol import (
    AudioMessage,
    ControlMessage,
    ErrorMessage,
    SessionEndMessage,
    SessionStartMessage,
    TextMessage,
)

# Configure logging
logger = logging.getLogger(__name__)


class AudioPlayer:
    """Manages audio playback with proper buffering.

    Uses soundfile for cross-platform audio output. Falls back to file output
    if no audio device is available.
    """

    def __init__(self, sample_rate: int = 48000, device: str | None = None) -> None:
        """Initialize audio player.

        Args:
            sample_rate: Audio sample rate in Hz
            device: Optional audio device name/index
        """
        self.sample_rate = sample_rate
        self.device = device
        self.frame_count = 0
        self.buffer: list[np.ndarray] = []
        self.playing = False

        # Try to import sounddevice for playback
        self.sd = None
        try:
            import sounddevice as sd
            self.sd = sd
            logger.info(f"Audio output initialized (device: {device or 'default'})")
        except ImportError:
            logger.warning("sounddevice not available, audio will be saved to file")

    def play_frame(self, pcm_data: bytes) -> None:
        """Play a single PCM audio frame.

        Args:
            pcm_data: Raw PCM bytes (16-bit signed integers)
        """
        # Convert bytes to numpy array (int16 → float32)
        audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        audio_array = audio_array / 32768.0  # Normalize to [-1, 1]

        if self.sd is not None:
            # Play audio through sounddevice
            try:
                self.sd.play(audio_array, samplerate=self.sample_rate, device=self.device)
            except Exception as e:
                logger.error(f"Audio playback failed: {e}")
                self._save_to_file(audio_array)
        else:
            # Fallback: save to file
            self._save_to_file(audio_array)

        self.frame_count += 1

    def _save_to_file(self, audio_array: np.ndarray) -> None:
        """Save audio to file as fallback.

        Args:
            audio_array: Audio data as float32 array
        """
        import soundfile as sf

        filename = f"audio_output_{self.frame_count:04d}.wav"
        sf.write(filename, audio_array, self.sample_rate)
        logger.debug(f"Saved audio to {filename}")

    def wait_for_completion(self) -> None:
        """Wait for all audio playback to complete."""
        if self.sd is not None:
            try:
                self.sd.wait()
            except Exception as e:  # noqa: S110
                logger.debug(f"Audio wait failed (non-critical): {e}")


class CLIClient:
    """WebSocket CLI client for orchestrator communication."""

    def __init__(
        self,
        server_url: str,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize CLI client.

        Args:
            server_url: WebSocket server URL (e.g., ws://localhost:8080)
            device: Optional audio output device
            verbose: Enable verbose logging
        """
        self.server_url = server_url
        self.device = device
        self.verbose = verbose
        self.session_id: str | None = None
        self.running = True
        self.audio_player = AudioPlayer(sample_rate=48000, device=device)

        # Setup logging
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )

    async def connect(self) -> AsyncIterator[ClientConnection]:
        """Connect to WebSocket server.

        Yields:
            WebSocket connection
        """
        try:
            async with websockets.connect(self.server_url) as websocket:
                logger.info(f"Connected to {self.server_url}")
                yield websocket
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    async def send_text(self, websocket: ClientConnection, text: str) -> None:
        """Send text message to server.

        Args:
            websocket: WebSocket connection
            text: Text to send for synthesis
        """
        message = TextMessage(text=text, is_final=True)
        await websocket.send(message.model_dump_json())
        logger.debug(f"Sent: {text}")

    async def send_control(
        self, websocket: ClientConnection, command: str
    ) -> None:
        """Send control message to server.

        Args:
            websocket: WebSocket connection
            command: Control command (PAUSE/RESUME/STOP)
        """
        message = ControlMessage(command=command)  # type: ignore[arg-type]
        await websocket.send(message.model_dump_json())
        logger.info(f"Sent control: {command}")

    async def handle_message(self, message_data: str) -> None:
        """Handle incoming message from server.

        Args:
            message_data: Raw JSON message from server
        """
        try:
            data = json.loads(message_data)
            msg_type = data.get("type")

            if msg_type == "session_start":
                msg = SessionStartMessage(**data)
                self.session_id = msg.session_id
                logger.info(f"Session started: {msg.session_id}")
                print(f"\n🔗 Session started: {msg.session_id}")

            elif msg_type == "audio":
                audio_msg = AudioMessage(**data)
                # Decode base64 PCM and play
                pcm_data = base64.b64decode(audio_msg.pcm)
                self.audio_player.play_frame(pcm_data)

                if self.verbose:
                    logger.debug(
                        f"Received audio frame {audio_msg.sequence} "
                        f"({len(pcm_data)} bytes, {audio_msg.frame_ms}ms)"
                    )
                else:
                    # Show progress indicator
                    print(".", end="", flush=True)

            elif msg_type == "session_end":
                end_msg = SessionEndMessage(**data)
                logger.info(f"Session ended: {end_msg.reason}")
                print(f"\n✓ Session ended: {end_msg.reason}")
                # Wait for audio completion
                self.audio_player.wait_for_completion()

            elif msg_type == "error":
                error_msg = ErrorMessage(**data)
                logger.error(f"Server error [{error_msg.code}]: {error_msg.message}")
                print(f"\n❌ Error: {error_msg.message}")

            else:
                logger.warning(f"Unknown message type: {msg_type}")

        except Exception as e:
            logger.error(f"Failed to handle message: {e}")

    async def receive_messages(self, websocket: ClientConnection) -> None:
        """Receive and handle messages from server.

        Args:
            websocket: WebSocket connection
        """
        try:
            async for message in websocket:
                # WebSocket messages can be str or bytes, convert bytes to str
                if isinstance(message, bytes):
                    message_str = message.decode("utf-8")
                else:
                    message_str = message
                await self.handle_message(message_str)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by server")
        except Exception as e:
            logger.error(f"Error receiving messages: {e}")

    async def input_loop(self, websocket: ClientConnection) -> None:
        """Handle user input from stdin.

        Args:
            websocket: WebSocket connection
        """
        print("\n" + "=" * 60)
        print("WebSocket CLI Client")
        print("=" * 60)
        print("\nCommands:")
        print("  /pause  - Pause audio playback")
        print("  /resume - Resume audio playback")
        print("  /stop   - Stop current synthesis")
        print("  /quit   - Exit client")
        print("  /help   - Show this help")
        print("\nEnter text to synthesize, or a command (starting with /):\n")

        loop = asyncio.get_event_loop()

        while self.running:
            try:
                # Read input asynchronously
                text = await loop.run_in_executor(None, input, "You: ")
                text = text.strip()

                if not text:
                    continue

                # Handle commands
                if text.startswith("/"):
                    command = text[1:].lower()

                    if command == "quit":
                        self.running = False
                        print("\nGoodbye!")
                        break

                    elif command == "help":
                        print("\nCommands:")
                        print("  /pause  - Pause audio playback")
                        print("  /resume - Resume audio playback")
                        print("  /stop   - Stop current synthesis")
                        print("  /quit   - Exit client")
                        print("  /help   - Show this help")
                        print()

                    elif command in ("pause", "resume", "stop"):
                        await self.send_control(websocket, command.upper())

                    else:
                        print(f"Unknown command: {command}")
                        print("Type /help for available commands")

                else:
                    # Send text for synthesis
                    await self.send_text(websocket, text)

            except EOFError:
                # Handle Ctrl+D
                self.running = False
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C
                self.running = False
                print("\n\nInterrupted!")
                break
            except Exception as e:
                logger.error(f"Input error: {e}")

    async def run(self) -> None:
        """Run the CLI client."""
        try:
            async with websockets.connect(self.server_url) as websocket:
                logger.info(f"Connected to {self.server_url}")

                # Setup signal handlers
                def signal_handler() -> None:
                    self.running = False

                loop = asyncio.get_event_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, signal_handler)

                # Run input and receive loops concurrently
                try:
                    await asyncio.gather(
                        self.input_loop(websocket),
                        self.receive_messages(websocket),
                    )
                finally:
                    # Cleanup signal handlers
                    for sig in (signal.SIGINT, signal.SIGTERM):
                        loop.remove_signal_handler(sig)

                    # Wait for final audio completion
                    self.audio_player.wait_for_completion()

        except Exception as e:
            logger.error(f"Client error: {e}")
            sys.exit(1)


async def run_client(server_url: str, config: dict[str, Any] | None = None) -> None:
    """Run CLI client connecting to orchestrator via WebSocket.

    Args:
        server_url: WebSocket server URL
        config: Optional client configuration
    """
    config = config or {}
    client = CLIClient(
        server_url=server_url,
        device=config.get("device"),
        verbose=config.get("verbose", False),
    )
    await client.run()


def main() -> None:
    """Main entry point for CLI client."""
    parser = argparse.ArgumentParser(
        description="WebSocket CLI client for realtime duplex voice chat"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="ws://localhost:8080",
        help="WebSocket server URL (default: ws://localhost:8080)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Audio output device name or index",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    config = {
        "device": args.device,
        "verbose": args.verbose,
    }

    try:
        asyncio.run(run_client(args.host, config))
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
