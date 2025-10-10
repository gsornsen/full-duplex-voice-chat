"""Basic WebSocket client example.

Demonstrates:
- WebSocket connection to orchestrator
- Session start
- Text chunk streaming
- Audio frame reception
- Session end
- Proper error handling

Usage:
    python examples/basic_websocket_client.py
    python examples/basic_websocket_client.py --url ws://localhost:8080
    python examples/basic_websocket_client.py --text "Custom text to synthesize"
"""

import argparse
import asyncio
import base64
import json
import sys
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection

from src.common.types import AudioFrame, SessionID


async def send_session_start(
    ws: ClientConnection,
    model_id: str = "mock-440hz",
    sample_rate: int = 48000,
    language: str = "en",
) -> SessionID:
    """Send SessionStart message and wait for SessionStarted response.

    Args:
        ws: WebSocket connection
        model_id: TTS model to use
        sample_rate: Audio sample rate (48000 Hz)
        language: Target language code

    Returns:
        Session ID from server response

    Raises:
        RuntimeError: If session start fails
    """
    session_start = {
        "type": "SessionStart",
        "model_id": model_id,
        "sample_rate": sample_rate,
        "language": language,
    }
    await ws.send(json.dumps(session_start))
    print(f"→ Sent SessionStart (model={model_id}, language={language})")

    # Receive SessionStarted
    response_str = await ws.recv()
    response: dict[str, Any] = json.loads(response_str)

    if response["type"] != "SessionStarted":
        raise RuntimeError(f"Expected SessionStarted, got {response['type']}")

    session_id: SessionID = response["session_id"]
    print(f"← Received SessionStarted (session_id={session_id})")
    return session_id


async def send_text_chunk(
    ws: ClientConnection, text: str, is_final: bool = True
) -> None:
    """Send text chunk for synthesis.

    Args:
        ws: WebSocket connection
        text: Text to synthesize
        is_final: Whether this is the final chunk
    """
    text_chunk = {
        "type": "TextChunk",
        "text": text,
        "is_final": is_final,
    }
    await ws.send(json.dumps(text_chunk))
    print(f"→ Sent TextChunk (text='{text[:50]}...', is_final={is_final})")


async def receive_audio_frames(ws: ClientConnection, session_id: SessionID) -> int:
    """Receive and process audio frames until session ends.

    Args:
        ws: WebSocket connection
        session_id: Current session ID

    Returns:
        Total number of audio frames received

    Raises:
        RuntimeError: If unexpected message type received
    """
    frame_count = 0
    total_bytes = 0

    while True:
        response_str = await ws.recv()
        response: dict[str, Any] = json.loads(response_str)

        if response["type"] == "AudioFrame":
            frame_count += 1

            # Decode base64 audio data
            audio_data: AudioFrame = base64.b64decode(response["data"])
            total_bytes += len(audio_data)

            # Validate frame format (should be 1920 bytes for 20ms at 48kHz)
            expected_size = 1920  # 960 samples * 2 bytes/sample
            if len(audio_data) != expected_size:
                print(
                    f"⚠  Frame {frame_count}: unexpected size {len(audio_data)} "
                    f"(expected {expected_size})"
                )

            # Print progress every 10 frames (200ms)
            if frame_count % 10 == 0:
                print(f"← Received frame {frame_count} ({len(audio_data)} bytes)")

        elif response["type"] == "SessionEnded":
            reason = response.get("reason", "unknown")
            print(f"← Session ended: {reason}")
            break

        elif response["type"] == "Error":
            error_msg = response.get("message", "Unknown error")
            print(f"✗ Error: {error_msg}")
            raise RuntimeError(f"Server error: {error_msg}")

        else:
            print(f"⚠  Unexpected message type: {response['type']}")

    print(
        f"\nSummary: {frame_count} frames, {total_bytes} bytes "
        f"({total_bytes / 1024:.2f} KB)"
    )
    return frame_count


async def run_client(
    url: str = "ws://localhost:8080",
    text: str = "Hello, this is a test of the text-to-speech system.",
    model_id: str = "mock-440hz",
) -> None:
    """Connect to M2 orchestrator and synthesize text.

    Args:
        url: WebSocket URL of orchestrator
        text: Text to synthesize
        model_id: TTS model to use
    """
    print(f"Connecting to {url}...")

    async with websockets.connect(url) as ws:
        print(f"✓ Connected to {url}\n")

        # Start session
        session_id = await send_session_start(ws, model_id=model_id)

        # Send text for synthesis
        await send_text_chunk(ws, text, is_final=True)

        # Receive audio frames
        frame_count = await receive_audio_frames(ws, session_id)

        print(f"\n✓ Synthesis complete ({frame_count} frames)")


def main() -> None:
    """Parse arguments and run client."""
    parser = argparse.ArgumentParser(description="Basic WebSocket client for M2")
    parser.add_argument(
        "--url",
        default="ws://localhost:8080",
        help="WebSocket URL of orchestrator (default: ws://localhost:8080)",
    )
    parser.add_argument(
        "--text",
        default="Hello, this is a test of the text-to-speech system.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--model",
        default="mock-440hz",
        help="TTS model ID (default: mock-440hz)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_client(url=args.url, text=args.text, model_id=args.model))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except ConnectionRefusedError:
        print(
            "\n✗ Connection refused. Make sure the orchestrator is running:\n"
            "  just run-orch"
        )
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
