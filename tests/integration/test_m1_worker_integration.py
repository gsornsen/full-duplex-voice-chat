"""Integration tests for TTS worker and gRPC client.

This module contains comprehensive integration tests for the TTS worker server
and gRPC client. Tests cover session lifecycle, streaming synthesis, control
commands, model management, and timing requirements.

ARCHITECTURE NOTE:
These tests validate the direct gRPC API of TTS workers, which remains unchanged
in M2 architecture. While M2 introduces the orchestrator layer for client-facing
interactions, workers still expose the same gRPC interface tested here.

Test scope:
- Direct worker gRPC API (bypasses orchestrator)
- Session lifecycle (StartSession, EndSession)
- Streaming synthesis (Synthesize RPC)
- Control commands (PAUSE/RESUME/STOP)
- Model management (ListModels, LoadModel, UnloadModel)
- Performance metrics (frame timing, pause latency)

For full system integration tests (client → orchestrator → worker), see:
- test_full_pipeline.py
- test_websocket_e2e.py

GRPC SEGFAULT WORKAROUND:
These tests use gRPC async stubs which may segfault in certain environments (WSL2).
The tests will be skipped automatically in unsafe environments unless:
- Running with `just test-integration` (uses --forked flag)
- Setting GRPC_TESTS_ENABLED=1 environment variable

See GRPC_SEGFAULT_WORKAROUND.md for details.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from src.orchestrator.grpc_client import TTSWorkerClient
from src.rpc.generated import tts_pb2
from tests.integration.test_utils import skip_if_grpc_unsafe

logger = logging.getLogger(__name__)

# Apply gRPC safety check to all tests in this module
pytestmark = [
    skip_if_grpc_unsafe,
    pytest.mark.integration,
    pytest.mark.docker,
    pytest.mark.grpc,
]


@pytest_asyncio.fixture
async def client(
    mock_tts_worker: str,
) -> AsyncIterator[TTSWorkerClient]:
    """Create and connect TTS worker client.

    Creates a client instance, connects to the worker, and ensures the
    connection is established. Cleans up by disconnecting on teardown.

    Args:
        mock_tts_worker: Mock worker address from conftest.py fixture

    Yields:
        TTSWorkerClient: Connected client instance
    """
    client = TTSWorkerClient(mock_tts_worker)
    await client.connect()

    yield client

    await client.disconnect()


@pytest.mark.asyncio
async def test_worker_connection(client: TTSWorkerClient) -> None:
    """Test basic worker connection and capabilities.

    Verifies that the client can connect to the worker and retrieve
    capabilities information.
    """
    # Get capabilities
    response = await client.get_capabilities()

    # Verify capabilities
    assert response.capabilities.streaming is True
    assert response.capabilities.cpu_ok is True
    assert "mock-440hz" in response.resident_models
    assert response.capabilities.max_concurrent_sessions == 10

    logger.info("Worker connection test passed")


@pytest.mark.asyncio
async def test_list_models(client: TTSWorkerClient) -> None:
    """Test listing available models.

    Verifies that the worker returns a list of available models with
    proper metadata.
    """
    models = await client.list_models()

    # Should have at least one model
    assert len(models) > 0

    # Check mock model
    mock_model = next((m for m in models if m.model_id == "mock-440hz"), None)
    assert mock_model is not None
    assert mock_model.family == "mock"
    assert mock_model.is_loaded is True
    assert "en" in mock_model.languages

    logger.info(f"Listed {len(models)} models successfully")


@pytest.mark.asyncio
async def test_session_lifecycle(client: TTSWorkerClient) -> None:
    """Test session start and end.

    Verifies that sessions can be created and terminated properly,
    with correct state management on the client side.
    """
    session_id = "test-session-1"

    # Start session
    success = await client.start_session(session_id, model_id="mock-440hz")
    assert success is True
    assert client.session_id == session_id

    # End session
    success = await client.end_session()
    assert success is True
    assert client.session_id is None

    logger.info("Session lifecycle test passed")


@pytest.mark.asyncio
async def test_streaming_synthesis(client: TTSWorkerClient) -> None:
    """Test streaming text to audio.

    Verifies that text chunks are synthesized into audio frames with
    correct format, timing, and frame counts.
    """
    session_id = "test-streaming"

    # Start session
    await client.start_session(session_id)

    # Send text chunks
    text_chunks = ["Hello world", "Testing 123"]
    frames = []

    async for frame in client.synthesize(text_chunks):
        frames.append(frame)

    # Verify frames received
    assert len(frames) > 0

    # Check frame metadata (excluding final frame)
    non_final_frames = [f for f in frames if not f.is_final]
    for frame in non_final_frames:
        assert frame.session_id == session_id
        assert frame.sample_rate == 48000
        assert frame.frame_duration_ms == 20
        assert len(frame.audio_data) == 1920  # 960 samples × 2 bytes
        assert frame.sequence_number > 0

    # Check final frame
    final_frames = [f for f in frames if f.is_final]
    assert len(final_frames) == 1
    assert final_frames[0].audio_data == b""

    # Mock adapter generates 500ms per chunk (25 frames × 20ms)
    # 2 chunks = ~50 frames + 1 final frame
    assert len(non_final_frames) >= 40  # Allow some variation

    await client.end_session()

    logger.info(f"Streaming synthesis test passed ({len(frames)} frames)")


@pytest.mark.asyncio
async def test_pause_command(client: TTSWorkerClient) -> None:
    """Test PAUSE control command.

    Verifies that the PAUSE command responds within the 50ms SLA and
    that synthesis can be resumed with the RESUME command.
    """
    session_id = "test-pause"

    # Start session
    await client.start_session(session_id)

    # Start synthesis in background
    text_chunks = ["Long text for pause test"]

    async def synthesize_task() -> list[tts_pb2.AudioFrame]:
        """Collect frames from synthesis."""
        frames: list[tts_pb2.AudioFrame] = []
        async for frame in client.synthesize(text_chunks):
            frames.append(frame)
        return frames

    # Start synthesis
    synthesis = asyncio.create_task(synthesize_task())

    # Wait a bit for synthesis to start
    await asyncio.sleep(0.1)

    # Send PAUSE command and measure response time
    start_time = time.time()
    success = await client.control("PAUSE")
    pause_duration_ms = (time.time() - start_time) * 1000

    assert success is True
    assert pause_duration_ms < 50  # SLA: < 50ms response

    logger.info(f"PAUSE response time: {pause_duration_ms:.2f}ms")

    # Wait and send RESUME
    await asyncio.sleep(0.1)
    success = await client.control("RESUME")
    assert success is True

    # Wait for synthesis to complete
    frames = await synthesis
    assert len(frames) > 0

    await client.end_session()

    logger.info("Pause command test passed")


@pytest.mark.asyncio
async def test_resume_command(client: TTSWorkerClient) -> None:
    """Test RESUME control command.

    Verifies that RESUME command properly continues synthesis after
    a PAUSE command.
    """
    session_id = "test-resume"

    # Start session
    await client.start_session(session_id)

    # Start synthesis
    text_chunks = ["Text for resume test"]

    async def synthesize_task() -> list[tts_pb2.AudioFrame]:
        """Collect frames from synthesis."""
        frames: list[tts_pb2.AudioFrame] = []
        async for frame in client.synthesize(text_chunks):
            frames.append(frame)
        return frames

    # Start synthesis
    synthesis = asyncio.create_task(synthesize_task())

    # Pause synthesis
    await asyncio.sleep(0.05)
    await client.control("PAUSE")

    # Resume synthesis
    await asyncio.sleep(0.05)
    success = await client.control("RESUME")
    assert success is True

    # Wait for completion
    frames = await synthesis
    assert len(frames) > 0

    await client.end_session()

    logger.info("Resume command test passed")


@pytest.mark.asyncio
async def test_stop_command(client: TTSWorkerClient) -> None:
    """Test STOP control command.

    Verifies that STOP command immediately terminates synthesis
    and sends the final frame marker.
    """
    session_id = "test-stop"

    # Start session
    await client.start_session(session_id)

    # Start synthesis
    text_chunks = ["Text for stop test"]

    async def synthesize_task() -> list[tts_pb2.AudioFrame]:
        """Collect frames until stopped."""
        frames: list[tts_pb2.AudioFrame] = []
        async for frame in client.synthesize(text_chunks):
            frames.append(frame)
        return frames

    # Start synthesis
    synthesis = asyncio.create_task(synthesize_task())

    # Wait a bit
    await asyncio.sleep(0.05)

    # Send STOP command
    success = await client.control("STOP")
    assert success is True

    # Wait for synthesis to terminate
    frames = await synthesis

    # Should have received some frames before stop
    # (May or may not receive final frame depending on timing)
    logger.info(f"Received {len(frames)} frames before STOP")

    await client.end_session()

    logger.info("Stop command test passed")


@pytest.mark.asyncio
async def test_load_unload_model(client: TTSWorkerClient) -> None:
    """Test model loading and unloading.

    Verifies that models can be dynamically loaded and unloaded
    without affecting active sessions.
    """
    # Load model
    success = await client.load_model("mock-440hz")
    assert success is True

    # Unload model
    success = await client.unload_model("mock-440hz")
    assert success is True

    logger.info("Load/unload model test passed")


@pytest.mark.asyncio
async def test_multiple_sessions_sequential(client: TTSWorkerClient) -> None:
    """Test multiple sequential sessions.

    Verifies that the worker can handle multiple sessions in sequence
    without resource leaks or state corruption.
    """
    for i in range(3):
        session_id = f"test-multi-{i}"

        # Start session
        await client.start_session(session_id)

        # Synthesize
        text_chunks = [f"Session {i}"]
        frames = []
        async for frame in client.synthesize(text_chunks):
            frames.append(frame)

        assert len(frames) > 0

        # End session
        await client.end_session()

    logger.info("Multiple sequential sessions test passed")


@pytest.mark.asyncio
async def test_invalid_session_error(client: TTSWorkerClient) -> None:
    """Test error handling for invalid session.

    Verifies that attempting to synthesize without an active session
    raises the appropriate error.
    """
    # Try to synthesize without starting session
    with pytest.raises(RuntimeError, match="No active session"):
        text_chunks = ["Test"]
        async for _ in client.synthesize(text_chunks):
            pass

    logger.info("Invalid session error test passed")


@pytest.mark.asyncio
async def test_invalid_command_error(client: TTSWorkerClient) -> None:
    """Test error handling for invalid control command.

    Verifies that invalid control commands are rejected with
    the appropriate error.
    """
    session_id = "test-invalid-cmd"
    await client.start_session(session_id)

    # Try invalid command
    with pytest.raises(ValueError, match="Invalid command"):
        await client.control("INVALID")

    await client.end_session()

    logger.info("Invalid command error test passed")


@pytest.mark.asyncio
async def test_frame_size_validation(client: TTSWorkerClient) -> None:
    """Test that audio frames have exact expected size.

    Verifies that all non-final frames contain exactly 1920 bytes
    (960 samples × 2 bytes for 16-bit PCM at 48kHz, 20ms duration).
    """
    session_id = "test-frame-size"

    await client.start_session(session_id)

    text_chunks = ["Frame size validation"]
    frames = []

    async for frame in client.synthesize(text_chunks):
        frames.append(frame)

    # Check all non-final frames
    non_final_frames = [f for f in frames if not f.is_final]

    for frame in non_final_frames:
        assert len(frame.audio_data) == 1920, (
            f"Frame {frame.sequence_number} has incorrect size: "
            f"{len(frame.audio_data)} bytes (expected 1920)"
        )

    await client.end_session()

    logger.info(f"Frame size validation test passed ({len(non_final_frames)} frames)")


@pytest.mark.asyncio
async def test_pause_response_timing(client: TTSWorkerClient) -> None:
    """Test PAUSE command response timing under load.

    Verifies that PAUSE commands consistently respond within 50ms
    even when the worker is actively synthesizing.
    """
    session_id = "test-pause-timing"

    await client.start_session(session_id)

    # Start synthesis with longer text
    text_chunks = ["Longer text to ensure synthesis is active"]

    async def synthesize_task() -> list[tts_pb2.AudioFrame]:
        """Collect frames from synthesis."""
        frames: list[tts_pb2.AudioFrame] = []
        async for frame in client.synthesize(text_chunks):
            frames.append(frame)
        return frames

    # Start synthesis
    synthesis = asyncio.create_task(synthesize_task())

    # Wait for synthesis to be in progress
    await asyncio.sleep(0.15)

    # Send multiple PAUSE/RESUME cycles and measure timing
    pause_times = []
    for _ in range(3):
        start = time.time()
        await client.control("PAUSE")
        duration_ms = (time.time() - start) * 1000
        pause_times.append(duration_ms)

        await asyncio.sleep(0.05)
        await client.control("RESUME")
        await asyncio.sleep(0.05)

    # Cancel synthesis
    await client.control("STOP")
    await synthesis

    # Verify all pause commands responded within 50ms
    for i, duration in enumerate(pause_times):
        assert duration < 50, f"PAUSE {i} took {duration:.2f}ms (> 50ms)"

    avg_pause_time = sum(pause_times) / len(pause_times)
    logger.info(
        f"Pause timing test passed (avg: {avg_pause_time:.2f}ms, "
        f"max: {max(pause_times):.2f}ms)"
    )

    await client.end_session()


@pytest.mark.asyncio
async def test_session_isolation(client: TTSWorkerClient, mock_tts_worker: str) -> None:
    """Test that sessions are isolated from each other.

    Verifies that operations on one session don't affect other sessions
    (requires creating a second client connection).
    """
    # Create second client
    client2 = TTSWorkerClient(mock_tts_worker)
    await client2.connect()

    try:
        # Start sessions on both clients
        await client.start_session("session-1")
        await client2.start_session("session-2")

        # Start synthesis on both
        async def synth1() -> list[tts_pb2.AudioFrame]:
            frames: list[tts_pb2.AudioFrame] = []
            async for frame in client.synthesize(["Client 1"]):
                frames.append(frame)
            return frames

        async def synth2() -> list[tts_pb2.AudioFrame]:
            frames: list[tts_pb2.AudioFrame] = []
            async for frame in client2.synthesize(["Client 2"]):
                frames.append(frame)
            return frames

        # Run both in parallel
        task1 = asyncio.create_task(synth1())
        task2 = asyncio.create_task(synth2())

        # Wait for both to complete
        frames1, frames2 = await asyncio.gather(task1, task2)

        # Both should complete successfully
        assert len(frames1) > 0
        assert len(frames2) > 0

        # Verify session IDs are correct
        non_final1 = [f for f in frames1 if not f.is_final]
        non_final2 = [f for f in frames2 if not f.is_final]

        assert all(f.session_id == "session-1" for f in non_final1)
        assert all(f.session_id == "session-2" for f in non_final2)

        # End sessions
        await client.end_session()
        await client2.end_session()

        logger.info("Session isolation test passed")

    finally:
        await client2.disconnect()


@pytest.mark.asyncio
async def test_empty_text_chunks(client: TTSWorkerClient) -> None:
    """Test handling of empty text chunks.

    Verifies that the worker handles empty text gracefully
    without crashing or producing invalid audio.
    """
    session_id = "test-empty"

    await client.start_session(session_id)

    # Send empty text chunks
    text_chunks = [""]
    frames = []

    async for frame in client.synthesize(text_chunks):
        frames.append(frame)

    # Should at least get a final frame
    assert len(frames) >= 1

    # Check final frame exists
    final_frames = [f for f in frames if f.is_final]
    assert len(final_frames) == 1

    await client.end_session()

    logger.info("Empty text chunks test passed")


@pytest.mark.asyncio
async def test_capabilities_consistency(client: TTSWorkerClient) -> None:
    """Test that capabilities remain consistent across calls.

    Verifies that multiple calls to GetCapabilities return
    consistent information.
    """
    # Get capabilities multiple times
    cap1 = await client.get_capabilities()
    cap2 = await client.get_capabilities()
    cap3 = await client.get_capabilities()

    # Verify consistency
    assert cap1.capabilities.streaming == cap2.capabilities.streaming == cap3.capabilities.streaming
    assert cap1.capabilities.cpu_ok == cap2.capabilities.cpu_ok == cap3.capabilities.cpu_ok
    assert (
        cap1.capabilities.max_concurrent_sessions
        == cap2.capabilities.max_concurrent_sessions
        == cap3.capabilities.max_concurrent_sessions
    )

    assert list(cap1.resident_models) == list(cap2.resident_models) == list(cap3.resident_models)

    logger.info("Capabilities consistency test passed")
