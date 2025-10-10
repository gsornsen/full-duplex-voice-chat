"""gRPC worker client example.

Demonstrates direct gRPC communication with a TTS worker, bypassing the
orchestrator. Useful for testing worker functionality in isolation.

Demonstrates:
- Direct gRPC connection to worker
- Bidirectional streaming synthesis
- Session management
- Control commands (PAUSE/RESUME/STOP)
- Capability querying
- Model listing

Usage:
    python examples/grpc_worker_client.py
    python examples/grpc_worker_client.py --addr localhost:7001
    python examples/grpc_worker_client.py --text "Custom text"
"""

import argparse
import asyncio
import sys
import uuid
from collections.abc import AsyncIterator
from typing import Any

import grpc

from src.common.types import AudioFrame, SessionID
from src.rpc.generated import tts_pb2, tts_pb2_grpc


async def text_chunk_generator(text: str, session_id: SessionID) -> AsyncIterator[tts_pb2.TextChunk]:
    """Generate text chunks for streaming synthesis.

    Args:
        text: Text to synthesize
        session_id: Session identifier

    Yields:
        TextChunk protobuf messages
    """
    # Send text as single chunk (could be split for streaming)
    chunk = tts_pb2.TextChunk(
        session_id=session_id,
        text=text,
        is_final=True,
    )
    yield chunk


async def query_capabilities(stub: tts_pb2_grpc.TTSServiceStub) -> dict[str, Any]:
    """Query worker capabilities.

    Args:
        stub: gRPC service stub

    Returns:
        Capabilities dictionary
    """
    print("Querying worker capabilities...")

    request = tts_pb2.GetCapabilitiesRequest()
    response = await stub.GetCapabilities(request)

    capabilities = {
        "streaming": response.capabilities.streaming,
        "zero_shot": response.capabilities.zero_shot,
        "lora": response.capabilities.lora,
        "cpu_ok": response.capabilities.cpu_ok,
        "languages": list(response.capabilities.languages),
        "emotive_zero_prompt": response.capabilities.emotive_zero_prompt,
        "max_concurrent_sessions": response.capabilities.max_concurrent_sessions,
    }

    print(f"✓ Capabilities: {capabilities}")
    print(f"  Resident models: {list(response.resident_models)}")
    print(f"  Metrics: {dict(response.metrics)}\n")

    return capabilities


async def list_models(stub: tts_pb2_grpc.TTSServiceStub) -> list[str]:
    """List available models.

    Args:
        stub: gRPC service stub

    Returns:
        List of model IDs
    """
    print("Listing available models...")

    request = tts_pb2.ListModelsRequest()
    response = await stub.ListModels(request)

    model_ids = []
    for model in response.models:
        model_ids.append(model.model_id)
        print(
            f"  - {model.model_id} (family={model.family}, "
            f"loaded={model.is_loaded}, languages={list(model.languages)})"
        )

    print()
    return model_ids


async def synthesize_text(
    stub: tts_pb2_grpc.TTSServiceStub,
    text: str,
    model_id: str = "mock-440hz",
    session_id: SessionID | None = None,
) -> int:
    """Synthesize text using streaming gRPC.

    Args:
        stub: gRPC service stub
        text: Text to synthesize
        model_id: TTS model to use
        session_id: Optional session ID (generated if not provided)

    Returns:
        Number of audio frames received
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    print(f"Starting synthesis session: {session_id}")

    # Start session
    start_request = tts_pb2.StartSessionRequest(
        session_id=session_id,
        model_id=model_id,
    )
    start_response = await stub.StartSession(start_request)

    if not start_response.success:
        print(f"✗ Failed to start session: {start_response.message}")
        return 0

    print(f"✓ Session started: {start_response.message}\n")

    try:
        # Stream text and receive audio
        print(f"Synthesizing: '{text[:50]}...'\n")

        frame_count = 0
        total_bytes = 0

        async for audio_frame in stub.Synthesize(text_chunk_generator(text, session_id)):
            frame_count += 1
            frame_data: AudioFrame = audio_frame.audio_data
            total_bytes += len(frame_data)

            # Print progress every 10 frames
            if frame_count % 10 == 0:
                print(
                    f"← Frame {frame_count}: {len(frame_data)} bytes "
                    f"(seq={audio_frame.sequence_number}, "
                    f"final={audio_frame.is_final})"
                )

            # Check for final frame
            if audio_frame.is_final:
                print(f"← Final frame received (seq={audio_frame.sequence_number})")
                break

        print(
            f"\nSummary: {frame_count} frames, {total_bytes} bytes "
            f"({total_bytes / 1024:.2f} KB)"
        )

        return frame_count

    finally:
        # End session
        end_request = tts_pb2.EndSessionRequest(session_id=session_id)
        end_response = await stub.EndSession(end_request)

        if end_response.success:
            print(f"✓ Session ended successfully\n")
        else:
            print(f"⚠  Session end failed\n")


async def test_control_commands(
    stub: tts_pb2_grpc.TTSServiceStub, session_id: SessionID
) -> None:
    """Test control commands (PAUSE/RESUME/STOP).

    Args:
        stub: gRPC service stub
        session_id: Session to control
    """
    print("Testing control commands...")

    commands = [
        (tts_pb2.PAUSE, "PAUSE"),
        (tts_pb2.RESUME, "RESUME"),
        (tts_pb2.STOP, "STOP"),
    ]

    for command_enum, command_name in commands:
        request = tts_pb2.ControlRequest(
            session_id=session_id,
            command=command_enum,
        )
        response = await stub.Control(request)

        status = "✓" if response.success else "✗"
        print(
            f"  {status} {command_name}: {response.message} "
            f"(ts={response.timestamp_ms})"
        )

    print()


async def run_client(addr: str, text: str, model_id: str) -> None:
    """Run gRPC client demo.

    Args:
        addr: Worker gRPC address (host:port)
        text: Text to synthesize
        model_id: TTS model to use
    """
    print(f"Connecting to worker at {addr}...\n")

    async with grpc.aio.insecure_channel(addr) as channel:
        stub = tts_pb2_grpc.TTSServiceStub(channel)

        # Query capabilities
        await query_capabilities(stub)

        # List models
        await list_models(stub)

        # Synthesize text
        session_id: SessionID = str(uuid.uuid4())
        frame_count = await synthesize_text(stub, text, model_id, session_id)

        # Test control commands (on a new session)
        test_session_id: SessionID = str(uuid.uuid4())
        start_request = tts_pb2.StartSessionRequest(
            session_id=test_session_id,
            model_id=model_id,
        )
        await stub.StartSession(start_request)
        await test_control_commands(stub, test_session_id)

        # Clean up test session
        end_request = tts_pb2.EndSessionRequest(session_id=test_session_id)
        await stub.EndSession(end_request)

        print(f"✓ Demo complete ({frame_count} frames synthesized)")


def main() -> None:
    """Parse arguments and run client."""
    parser = argparse.ArgumentParser(description="gRPC worker client for M2")
    parser.add_argument(
        "--addr",
        default="localhost:7001",
        help="Worker gRPC address (default: localhost:7001)",
    )
    parser.add_argument(
        "--text",
        default="This is a test of direct worker communication.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--model",
        default="mock-440hz",
        help="TTS model ID (default: mock-440hz)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_client(addr=args.addr, text=args.text, model_id=args.model))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except grpc.RpcError as e:
        print(
            f"\n✗ gRPC error: {e.code()}: {e.details()}\n"
            "Make sure the worker is running:\n"
            "  just run-tts-mock"
        )
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
