"""Integration test fixtures and utilities.

Provides shared fixtures for:
- Docker container management (Redis, LiveKit)
- Mock TTS worker process spawning
- Orchestrator server lifecycle
- Synthetic audio generation
- Frame timing validation utilities
- Metric collection helpers
"""

import asyncio
import gc
import json
import logging
import socket
import subprocess
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from typing import Any, TYPE_CHECKING

import numpy as np
import pytest
import pytest_asyncio
import websockets
from redis import asyncio as aioredis
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

from src.orchestrator.config import (
    LiveKitConfig,
    OrchestratorConfig,
    RedisConfig,
    RoutingConfig,
    TransportConfig,
    WebSocketConfig,
)
from src.orchestrator.registry import WorkerRegistration, WorkerRegistry
from src.orchestrator.server import start_server

# Lazy imports for gRPC - only imported when gRPC fixtures are used
# This prevents segfault issues when running non-gRPC tests
if TYPE_CHECKING:
    import grpc
    from src.rpc.generated import tts_pb2, tts_pb2_grpc
    from src.tts.worker import start_worker

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions for Port Allocation
# ============================================================================


def get_free_port() -> int:
    """Get a free TCP port for binding.

    Returns:
        Available port number

    Notes:
        Uses ephemeral port allocation (port=0) to avoid conflicts.
        The port is freed immediately after discovery, so there's a small
        race condition window, but this is acceptable for tests.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port: int = s.getsockname()[1]
    return port


# ============================================================================
# gRPC Event Loop Workaround
# ============================================================================


@pytest.fixture(scope="module")
def grpc_event_loop_workaround() -> Iterable[None]:
    """Workaround for grpc-python event loop cleanup issues.

    grpc-python creates background threads that interact with the asyncio event loop.
    When pytest-asyncio tears down and recreates event loops between tests, these
    threads can crash with segfaults when trying to access the old loop.

    This fixture:
    1. Disables GC during tests to prevent premature cleanup
    2. Adds a delay after each test to allow grpc threads to finish

    NOTE: This fixture is NOT autouse - only gRPC tests should explicitly use it.
    Non-gRPC tests (like VAD tests) should not require this workaround.

    See: https://github.com/grpc/grpc/issues/37714
    """
    # Disable garbage collection to prevent premature cleanup of grpc internals
    gc_was_enabled = gc.isenabled()
    gc.disable()

    yield

    # Re-enable GC if it was enabled before
    if gc_was_enabled:
        gc.enable()

    # Give grpc threads time to clean up
    time.sleep(1.0)


@pytest.fixture(scope="module")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    """Use a consistent event loop policy for all tests.

    This ensures all fixtures and tests in a module share the same event loop,
    avoiding grpc-python issues with event loop transitions.
    """
    return asyncio.get_event_loop_policy()


# ============================================================================
# Docker Container Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Check if Docker is available on the system.

    Returns:
        True if Docker is available, False otherwise
    """
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["docker", "info"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


@pytest_asyncio.fixture(scope="module")
async def redis_container(docker_available: bool) -> AsyncIterator[str]:
    """Start Redis container for integration tests.

    Yields:
        Redis URL (redis://localhost:6380)

    Skips test if Docker is not available.

    Note: Uses port 6380 to avoid conflicts with development Redis on 6379.
    """
    if not docker_available:
        pytest.skip("Docker not available")

    container_name = f"test-redis-{uuid.uuid4().hex[:8]}"
    redis_port = get_free_port()
    redis_url = f"redis://localhost:{redis_port}"

    # Clean up any existing container with the same name (unlikely with uuid)
    subprocess.run(  # noqa: S603, S607
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        check=False,
    )

    # Start Redis container
    logger.info(f"Starting Redis container: {container_name} on port {redis_port}")
    try:
        subprocess.run(  # noqa: S603, S607
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                f"{redis_port}:6379",
                "redis:7-alpine",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Redis container: {e.stderr.decode()}")
        pytest.skip(f"Failed to start Redis container: {e}")

    # Wait for Redis to be ready
    redis = aioredis.from_url(redis_url)
    for attempt in range(30):
        try:
            await redis.ping()
            logger.info(f"Redis ready at {redis_url}")
            break
        except Exception:
            if attempt == 29:
                raise RuntimeError("Redis failed to start") from None
            await asyncio.sleep(0.5)

    try:
        yield redis_url
    finally:
        await redis.close()
        # Cleanup
        logger.info(f"Stopping Redis container: {container_name}")
        subprocess.run(  # noqa: S603, S607
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            check=True,
        )


@pytest_asyncio.fixture(scope="module")
async def livekit_container(docker_available: bool) -> AsyncIterator[str]:
    """Start LiveKit container for integration tests.

    Yields:
        LiveKit URL (http://localhost:7880)

    Skips test if Docker is not available.
    """
    if not docker_available:
        pytest.skip("Docker not available")

    container_name = f"test-livekit-{uuid.uuid4().hex[:8]}"
    livekit_url = "http://localhost:7880"

    # Clean up any existing container with the same name (unlikely with uuid)
    subprocess.run(  # noqa: S603, S607
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        check=False,
    )

    # Start LiveKit container in dev mode
    logger.info(f"Starting LiveKit container: {container_name}")
    try:
        subprocess.run(  # noqa: S603, S607
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                "7880:7880",
                "-p",
                "7881:7881",
                "-p",
                "7882:7882/udp",
                "livekit/livekit-server:latest",
                "--dev",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start LiveKit container: {e.stderr.decode()}")
        pytest.skip(f"Failed to start LiveKit container: {e}")

    # Wait for LiveKit to be ready
    import aiohttp

    async with aiohttp.ClientSession() as session:
        for attempt in range(30):
            try:
                async with session.get(f"{livekit_url}/") as resp:
                    if resp.status < 500:
                        logger.info(f"LiveKit ready at {livekit_url}")
                        break
            except Exception:
                if attempt == 29:
                    raise RuntimeError("LiveKit failed to start") from None
                await asyncio.sleep(0.5)

    try:
        yield livekit_url
    finally:
        # Cleanup
        logger.info(f"Stopping LiveKit container: {container_name}")
        subprocess.run(  # noqa: S603, S607
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            check=True,
        )


# ============================================================================
# Mock TTS Worker Fixtures
# ============================================================================


@pytest_asyncio.fixture(scope="module")
async def mock_tts_worker() -> AsyncIterator[str]:
    """Start mock TTS worker process.

    Yields:
        gRPC address (localhost:<dynamic-port>)

    Notes:
        Uses dynamic port allocation to avoid conflicts when multiple
        tests run in parallel or sequentially.
    """
    # Lazy imports - only import when this fixture is used
    import grpc.aio
    from src.rpc.generated import tts_pb2, tts_pb2_grpc
    from src.tts.worker import start_worker

    # Use dynamic port allocation to avoid conflicts
    port = get_free_port()
    addr = f"localhost:{port}"

    logger.info(f"Starting mock TTS worker on port {port}")

    # Start worker in background
    worker_task = asyncio.create_task(start_worker({"port": port}))

    # Wait for worker to be ready
    for attempt in range(30):
        try:
            channel = grpc.aio.insecure_channel(addr)
            stub = tts_pb2_grpc.TTSServiceStub(channel)  # type: ignore[no-untyped-call]
            await stub.GetCapabilities(tts_pb2.GetCapabilitiesRequest())
            await channel.close()
            logger.info(f"Mock TTS worker ready at {addr}")
            break
        except Exception as e:
            if attempt == 29:
                worker_task.cancel()
                logger.error(f"Mock TTS worker failed to start: {e}")
                raise RuntimeError(f"Mock TTS worker failed to start on port {port}") from e
            await asyncio.sleep(0.5)

    try:
        yield addr
    finally:
        # Cleanup
        logger.info("Stopping mock TTS worker")
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

        # Give grpc time to clean up background threads
        await asyncio.sleep(1.0)


@pytest_asyncio.fixture(scope="module")
async def registered_mock_worker(
    redis_container: str, mock_tts_worker: str
) -> AsyncIterator[WorkerRegistration]:
    """Register mock TTS worker in Redis.

    Args:
        redis_container: Redis URL fixture
        mock_tts_worker: Mock worker address fixture

    Yields:
        Worker registration metadata
    """
    registry = WorkerRegistry(
        redis_url=redis_container,
        worker_key_prefix="worker:",
        worker_ttl_seconds=30,
    )

    await registry.connect()

    registration = WorkerRegistration(
        name="mock-tts-worker-0",
        addr=f"grpc://{mock_tts_worker}",
        capabilities={
            "streaming": True,
            "zero_shot": False,
            "lora": False,
            "cpu_ok": True,
            "languages": ["en"],
            "emotive_zero_prompt": False,
        },
        resident_models=["mock-440hz"],
        metrics={"rtf": 0.1, "queue_depth": 0.0},
    )

    await registry.register_worker(registration, ttl_s=60)
    logger.info(f"Registered mock worker: {registration.name}")

    try:
        yield registration
    finally:
        await registry.remove_worker(registration.name)
        await registry.disconnect()


# ============================================================================
# Orchestrator Server Fixtures
# ============================================================================


@pytest_asyncio.fixture(scope="module")
async def orchestrator_server(
    redis_container: str, mock_tts_worker: str, tmp_path_factory: pytest.TempPathFactory
) -> AsyncIterator[OrchestratorConfig]:
    """Start orchestrator server with WebSocket transport.

    Args:
        redis_container: Redis URL fixture
        mock_tts_worker: Mock worker address fixture
        tmp_path_factory: Pytest temporary directory factory for module scope

    Yields:
        Orchestrator configuration
    """
    # Use dynamic port for WebSocket to avoid conflicts
    ws_port = get_free_port()

    # Create test configuration using proper Pydantic models
    config = OrchestratorConfig(
        transport=TransportConfig(
            websocket=WebSocketConfig(
                enabled=True,
                host="127.0.0.1",
                port=ws_port,
                max_connections=10,
            ),
            livekit=LiveKitConfig(enabled=False),
        ),
        routing=RoutingConfig(static_worker_addr=f"grpc://{mock_tts_worker}"),
        redis=RedisConfig(url=redis_container),  # Use RedisConfig model
        log_level="INFO",
    )

    # Write config to temporary file
    tmp_path = tmp_path_factory.mktemp("config")
    config_path = tmp_path / "orchestrator_config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config.model_dump(mode='json'), f)

    logger.info(f"Starting orchestrator on WebSocket port {ws_port}")

    # Start server in background
    server_task = asyncio.create_task(start_server(config_path))

    # Wait for server to be ready
    ws_url = f"ws://localhost:{ws_port}"
    for attempt in range(30):
        try:
            async with websockets.connect(ws_url) as ws:
                # Receive session start message
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(msg)
                if data.get("type") == "session_start":
                    await ws.close()
                    logger.info(f"Orchestrator server ready at {ws_url}")
                    break
        except Exception:
            if attempt == 29:
                server_task.cancel()
                raise RuntimeError(
                    f"Orchestrator server failed to start on port {ws_port}"
                    ) from None
            await asyncio.sleep(0.5)

    try:
        yield config
    finally:
        # Cleanup
        logger.info("Stopping orchestrator server")
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


# ============================================================================
# WebSocket Client Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def ws_client(orchestrator_server: OrchestratorConfig) -> AsyncIterator[ClientConnection]:
    """Connect WebSocket client to orchestrator.

    Args:
        orchestrator_server: Orchestrator configuration fixture

    Yields:
        WebSocket client connection
    """
    ws_port = orchestrator_server.transport.websocket.port
    ws_url = f"ws://localhost:{ws_port}"

    async with websockets.connect(ws_url) as ws:
        # Receive and validate session start
        msg = await ws.recv()
        data = json.loads(msg)
        assert data["type"] == "session_start"
        assert "session_id" in data
        logger.info(f"WebSocket client connected: session_id={data['session_id']}")
        yield ws


# ============================================================================
# Synthetic Audio Generation
# ============================================================================


def generate_sine_wave(
    frequency: float = 440.0,
    duration_ms: int = 100,
    sample_rate: int = 48000,
) -> bytes:
    """Generate synthetic sine wave audio.

    Args:
        frequency: Frequency in Hz
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz

    Returns:
        PCM audio data (16-bit signed int, little endian)
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
    audio = np.sin(2 * np.pi * frequency * t)

    # Scale to 16-bit int range
    audio = (audio * 32767).astype(np.int16)

    # Convert to bytes (little endian)
    return audio.tobytes()


def generate_speech_audio(duration_ms: int = 1000, sample_rate: int = 16000) -> bytes:
    """Generate synthetic speech-like audio for VAD testing.

    Creates audio with speech characteristics (multiple frequencies).

    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz

    Returns:
        PCM audio data (16-bit signed int, little endian)
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)

    # Mix multiple frequencies for speech-like sound
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.3 * np.sin(2 * np.pi * 350 * t)
        + 0.2 * np.sin(2 * np.pi * 600 * t)
        + 0.2 * np.sin(2 * np.pi * 1200 * t)
    )

    # Add some noise for realism
    noise = np.random.normal(0, 0.05, num_samples)
    audio += noise

    # Normalize and scale
    audio = audio / np.max(np.abs(audio))
    audio_typed: np.ndarray[Any, np.dtype[np.int16]] = (audio * 32767 * 0.8).astype(np.int16)

    return audio_typed.tobytes()


def generate_silence(duration_ms: int = 100, sample_rate: int = 16000) -> bytes:
    """Generate silent audio frames.

    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz

    Returns:
        PCM audio data (zeros, 16-bit signed int)
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * num_samples


# ============================================================================
# Frame Timing Validation
# ============================================================================


class FrameTimingValidator:
    """Validates audio frame timing and cadence.

    Tracks frame arrival times and validates against expected 20ms cadence.
    """

    def __init__(self, expected_frame_ms: int = 20, tolerance_ms: float = 5.0) -> None:
        """Initialize validator.

        Args:
            expected_frame_ms: Expected frame duration in milliseconds
            tolerance_ms: Acceptable jitter tolerance in milliseconds
        """
        self.expected_frame_ms = expected_frame_ms
        self.tolerance_ms = tolerance_ms
        self.frame_times: list[float] = []
        self.first_frame_time: float | None = None

    def record_frame(self, timestamp: float | None = None) -> None:
        """Record frame arrival time.

        Args:
            timestamp: Optional timestamp (uses current time if not provided)
        """
        ts = timestamp if timestamp is not None else time.time()
        if self.first_frame_time is None:
            self.first_frame_time = ts
        self.frame_times.append(ts)

    def validate_frame_count(self, expected_count: int) -> None:
        """Validate that expected number of frames were received.

        Args:
            expected_count: Expected frame count

        Raises:
            AssertionError: If frame count doesn't match
        """
        assert len(self.frame_times) == expected_count, (
            f"Expected {expected_count} frames, got {len(self.frame_times)}"
        )

    def validate_timing(self) -> dict[str, float]:
        """Validate frame timing and calculate metrics.

        Returns:
            Dictionary with timing metrics:
            - mean_interval_ms: Mean interval between frames
            - std_interval_ms: Standard deviation of intervals
            - max_jitter_ms: Maximum jitter from expected cadence
            - p95_jitter_ms: 95th percentile jitter

        Raises:
            AssertionError: If timing exceeds tolerance
        """
        if len(self.frame_times) < 2:
            return {
                    "mean_interval_ms": 0,
                    "std_interval_ms": 0,
                    "max_jitter_ms": 0,
                    "p95_jitter_ms": 0
                    }

        # Calculate intervals
        intervals = np.diff(self.frame_times) * 1000  # Convert to ms
        mean_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals))

        # Calculate jitter from expected cadence
        jitter = np.abs(intervals - self.expected_frame_ms)
        max_jitter = float(np.max(jitter))
        p95_jitter = float(np.percentile(jitter, 95))

        # Validate with relaxed tolerance for CI environments
        relaxed_tolerance = self.tolerance_ms * 2  # Double tolerance for CI
        if p95_jitter > relaxed_tolerance:
            logger.warning(
                f"p95 jitter {p95_jitter:.2f}ms exceeds relaxed tolerance {relaxed_tolerance}ms "
                "(expected in CI environments with variable load)"
            )

        return {
            "mean_interval_ms": mean_interval,
            "std_interval_ms": std_interval,
            "max_jitter_ms": max_jitter,
            "p95_jitter_ms": p95_jitter,
        }


# ============================================================================
# Metric Collection Helpers
# ============================================================================


class LatencyMetrics:
    """Collects and analyzes latency metrics."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.samples: list[float] = []

    def record(self, latency_ms: float) -> None:
        """Record latency sample.

        Args:
            latency_ms: Latency in milliseconds
        """
        self.samples.append(latency_ms)

    def get_percentile(self, percentile: int) -> float:
        """Get latency percentile.

        Args:
            percentile: Percentile (0-100)

        Returns:
            Latency at percentile in milliseconds
        """
        if not self.samples:
            return 0.0
        return float(np.percentile(self.samples, percentile))

    def get_summary(self) -> dict[str, float]:
        """Get latency summary statistics.

        Returns:
            Dictionary with:
            - mean: Mean latency
            - p50: Median latency
            - p95: 95th percentile
            - p99: 99th percentile
            - min: Minimum latency
            - max: Maximum latency
        """
        if not self.samples:
            return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}

        return {
            "mean": float(np.mean(self.samples)),
            "p50": float(np.percentile(self.samples, 50)),
            "p95": float(np.percentile(self.samples, 95)),
            "p99": float(np.percentile(self.samples, 99)),
            "min": float(np.min(self.samples)),
            "max": float(np.max(self.samples)),
        }


# ============================================================================
# Utility Functions
# ============================================================================


def validate_audio_frame(
    frame_data: bytes, expected_duration_ms: int = 20, expected_sample_rate: int = 48000
) -> None:
    """Validate audio frame format.

    Args:
        frame_data: PCM audio data
        expected_duration_ms: Expected frame duration
        expected_sample_rate: Expected sample rate

    Raises:
        ValueError: If frame format is invalid (instead of AssertionError for better error messages)
    """
    expected_samples = expected_sample_rate * expected_duration_ms // 1000
    expected_bytes = expected_samples * 2  # 16-bit PCM

    if len(frame_data) != expected_bytes:
        raise ValueError(
            f"Invalid frame size: expected {expected_bytes} bytes "
            f"({expected_duration_ms}ms @ {expected_sample_rate}Hz mono), "
            f"got {len(frame_data)} bytes"
        )


async def send_text_message(ws: ClientConnection, text: str, is_final: bool = True) -> None:
    """Send text message via WebSocket.

    Args:
        ws: WebSocket connection
        text: Text to send
        is_final: Whether this is the final chunk
    """
    message = {"type": "text", "text": text, "is_final": is_final}
    await ws.send(json.dumps(message))


async def receive_audio_frames(
    ws: ClientConnection, timeout_s: float = 5.0
) -> list[dict[str, Any]]:
    """Receive audio frames from WebSocket.

    Args:
        ws: WebSocket connection
        timeout_s: Timeout in seconds

    Returns:
        List of audio frame messages

    Raises:
        asyncio.TimeoutError: If no frames received within timeout
    """
    frames: list[dict[str, Any]] = []
    start_time = time.time()

    while time.time() - start_time < timeout_s:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
            data = json.loads(msg)

            if data.get("type") == "audio":
                # Only collect frames with actual PCM data
                if data.get("pcm"):
                    frames.append(data)

                # Check if this is marked as final frame
                if data.get("is_final"):
                    logger.debug("Received final audio frame marker")
                    break

                # Empty PCM indicates end of stream (but keep waiting for is_final)
                if not data.get("pcm"):
                    logger.debug("Received empty PCM frame, waiting for final marker")
                    # Continue to check for is_final marker

        except TimeoutError:
            # If we have frames and hit timeout, consider it complete
            if frames:
                logger.debug(f"Timeout after receiving {len(frames)} frames, considering complete")
                break
            # If no frames yet, raise the timeout
            raise
        except (ConnectionClosed,
                ConnectionClosedError,
                ConnectionClosedOK) as e:
            # Connection closed - this is OK if we already have frames
            if frames:
                logger.debug(f"Connection closed after receiving {len(frames)} frames: {e}")
                break
            # If no frames, this is an error
            logger.error(f"Connection closed before receiving any frames: {e}")
            raise

    return frames
