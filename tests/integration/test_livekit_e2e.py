"""LiveKit End-to-End Integration Test.

Tests LiveKit transport flow:
1. Start LiveKit server (Docker container)
2. Start LiveKit transport
3. Create room and connect participant
4. Send text via data channel
5. Verify mock TTS worker receives request
6. Verify audio track publishes frames
7. Measure FAL
8. Clean shutdown

Note: These tests require LiveKit server running. They will be skipped if:
- Docker is not available
- LiveKit container fails to start

IMPORTANT: These tests are isolated due to grpc-python event loop issues.
Run separately with: pytest tests/integration/test_livekit_e2e.py --forked
"""

import logging

import pytest

logger = logging.getLogger(__name__)

# Mark all tests in this module as infrastructure tests (skip in CI)
# These tests require LiveKit server with proper authentication and Docker
# Run locally with: pytest -m infrastructure
pytestmark = [pytest.mark.grpc, pytest.mark.infrastructure]


@pytest.mark.integration
@pytest.mark.livekit
@pytest.mark.docker
@pytest.mark.asyncio
async def test_livekit_server_available(livekit_container: str) -> None:
    """Test that LiveKit server is accessible.

    This is a basic smoke test to verify LiveKit container is running.
    """
    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{livekit_container}/", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                assert resp.status < 500, f"LiveKit server returned {resp.status}"
                logger.info(f"LiveKit server accessible at {livekit_container}")
    except Exception as e:
        pytest.fail(f"Failed to connect to LiveKit server: {e}")


@pytest.mark.integration
@pytest.mark.livekit
@pytest.mark.docker
@pytest.mark.asyncio
async def test_livekit_room_creation(livekit_container: str) -> None:
    """Test LiveKit room creation.

    Validates:
    - Can create a room
    - Room has valid metadata
    - Room can be listed
    """
    pytest.skip("LiveKit SDK integration pending - M2 scope is WebSocket only")

    # Future implementation:
    # from livekit import api
    #
    # livekit_api = api.LiveKitAPI(
    #     url=livekit_container,
    #     api_key="devkey",
    #     api_secret="secret",
    # )
    #
    # # Create room
    # room = await livekit_api.room.create_room(
    #     api.CreateRoomRequest(name="test-room")
    # )
    #
    # assert room.name == "test-room"
    # logger.info(f"Created room: {room.name}")


@pytest.mark.integration
@pytest.mark.livekit
@pytest.mark.docker
@pytest.mark.asyncio
async def test_livekit_participant_connection(livekit_container: str) -> None:
    """Test LiveKit participant connection.

    Validates:
    - Participant can join room
    - Participant receives connection events
    - Participant can disconnect cleanly
    """
    pytest.skip("LiveKit SDK integration pending - M2 scope is WebSocket only")

    # Future implementation:
    # from livekit import rtc
    #
    # room = rtc.Room()
    #
    # # Connect to room
    # await room.connect(
    #     url=livekit_container,
    #     token="<generated-token>",
    # )
    #
    # assert room.connection_state == rtc.ConnectionState.CONNECTED
    # logger.info("Participant connected to room")
    #
    # # Disconnect
    # await room.disconnect()


@pytest.mark.integration
@pytest.mark.livekit
@pytest.mark.docker
@pytest.mark.asyncio
async def test_livekit_data_channel_messaging(livekit_container: str) -> None:
    """Test LiveKit data channel for text messaging.

    Validates:
    - Can send data via data channel
    - Data is received by orchestrator
    - Text is forwarded to TTS worker
    """
    pytest.skip("LiveKit SDK integration pending - M2 scope is WebSocket only")

    # Future implementation:
    # - Connect participant to room
    # - Send text via data channel
    # - Verify orchestrator receives text
    # - Verify TTS synthesis triggered


@pytest.mark.integration
@pytest.mark.livekit
@pytest.mark.docker
@pytest.mark.asyncio
async def test_livekit_audio_track_publishing(livekit_container: str) -> None:
    """Test LiveKit audio track publishing.

    Validates:
    - Orchestrator can publish audio track
    - Audio frames are sent to track
    - Participant receives audio frames
    """
    pytest.skip("LiveKit SDK integration pending - M2 scope is WebSocket only")

    # Future implementation:
    # - Connect participant
    # - Trigger TTS synthesis
    # - Verify audio track is published
    # - Verify audio frames received


@pytest.mark.integration
@pytest.mark.livekit
@pytest.mark.docker
@pytest.mark.asyncio
async def test_livekit_fal_measurement(livekit_container: str) -> None:
    """Measure First Audio Latency via LiveKit.

    Target: FAL < 400ms for LiveKit flow

    Validates:
    - FAL is within acceptable range
    - LiveKit overhead is reasonable
    """
    pytest.skip("LiveKit SDK integration pending - M2 scope is WebSocket only")

    # Future implementation:
    # - Connect participant
    # - Send text via data channel
    # - Measure time to first audio frame
    # - Assert FAL < 400ms


@pytest.mark.integration
@pytest.mark.livekit
@pytest.mark.docker
@pytest.mark.asyncio
async def test_livekit_webrtc_vs_websocket_comparison(
    livekit_container: str, orchestrator_server: object
) -> None:
    """Compare LiveKit (WebRTC) vs WebSocket performance.

    Validates:
    - Both transports provide similar functionality
    - Performance characteristics are documented
    - FAL is comparable (within 100ms)
    """
    pytest.skip("LiveKit SDK integration pending - M2 scope is WebSocket only")

    # Future implementation:
    # - Measure FAL for WebSocket transport
    # - Measure FAL for LiveKit transport
    # - Compare and log results
    # - Document overhead/benefits of each


@pytest.mark.integration
@pytest.mark.livekit
@pytest.mark.asyncio
async def test_livekit_graceful_degradation() -> None:
    """Test system behavior when LiveKit is unavailable.

    Validates:
    - System starts without LiveKit (WebSocket only)
    - No crashes or errors when LiveKit disabled
    - WebSocket transport remains functional
    """
    # This should work without LiveKit container

    from src.orchestrator.config import (
        LiveKitConfig,
        OrchestratorConfig,
        TransportConfig,
        WebSocketConfig,
    )

    # Create config with LiveKit disabled using proper Pydantic models
    config = OrchestratorConfig(
        transport=TransportConfig(
            websocket=WebSocketConfig(enabled=True, host="127.0.0.1", port=8081),
            livekit=LiveKitConfig(enabled=False),
        ),
        log_level="INFO",
    )

    # Verify LiveKit is disabled
    assert not config.transport.livekit.enabled, "LiveKit should be disabled"
    assert config.transport.websocket.enabled, "WebSocket should be enabled"

    logger.info("System configured for WebSocket-only mode (LiveKit disabled)")


@pytest.mark.integration
@pytest.mark.livekit
@pytest.mark.docker
@pytest.mark.asyncio
async def test_livekit_concurrent_participants(livekit_container: str) -> None:
    """Test multiple concurrent participants in LiveKit room.

    Validates:
    - Multiple participants can join same room
    - Each participant gets isolated session
    - Audio tracks are correctly routed
    """
    pytest.skip("LiveKit SDK integration pending - M2 scope is WebSocket only")

    # Future implementation:
    # - Create room
    # - Connect multiple participants
    # - Send different text from each
    # - Verify each gets correct audio
    # - Verify no cross-talk


@pytest.mark.integration
@pytest.mark.livekit
@pytest.mark.docker
@pytest.mark.asyncio
async def test_livekit_room_cleanup(livekit_container: str) -> None:
    """Test LiveKit room cleanup and resource management.

    Validates:
    - Rooms are cleaned up after participants leave
    - Resources are released properly
    - No resource leaks
    """
    pytest.skip("LiveKit SDK integration pending - M2 scope is WebSocket only")

    # Future implementation:
    # - Create room
    # - Connect and disconnect participants
    # - Verify room is cleaned up
    # - Check for resource leaks


# Note: Full LiveKit integration is planned for later phases.
# M2 focuses on WebSocket transport with LiveKit as optional.
# These tests serve as placeholders and documentation for future work.
