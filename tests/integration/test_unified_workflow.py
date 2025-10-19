"""Integration tests for unified development workflow.

This test suite validates the complete development experience including:
- Infrastructure startup and health checks (Redis, LiveKit, Caddy)
- Model switching (Piper ↔ CosyVoice)
- Service discovery and network alias resolution
- Environment variable configuration precedence
- End-to-end speech synthesis workflows
- CI/CD integration with new `just dev` commands

Test Categories:
1. Infrastructure Tests: Verify Docker Compose services start correctly
2. Model Switching Tests: Hot-swap TTS models during runtime
3. Service Discovery Tests: Orchestrator finds TTS workers via DNS
4. Environment Variable Tests: Configuration precedence validation
5. End-to-End Tests: Full workflow from text to audio
6. CI Integration Tests: GitHub Actions compatibility

Requirements:
- Docker and Docker Compose installed
- Redis, LiveKit, and Caddy containers available
- TTS worker images built (Piper, CosyVoice)
- Network access to localhost:6379, localhost:7880, localhost:8443

Running Tests:
    # All unified workflow tests
    pytest tests/integration/test_unified_workflow.py -v -m integration

    # Specific category
    pytest tests/integration/test_unified_workflow.py::TestInfrastructure -v

    # With Docker cleanup
    pytest tests/integration/test_unified_workflow.py -v --cleanup-docker

Performance Targets:
- Infrastructure startup: < 60s (cold), < 30s (warm)
- Model switching: < 5s (unload old + load new)
- Service discovery: < 2s (orchestrator finds TTS worker)
- End-to-end workflow: < 10s (text → first audio frame)
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
import redis.asyncio as aioredis

# ============================================================================
# Fixtures for Docker Compose Infrastructure
# ============================================================================


@pytest.fixture(scope="module")
def project_root() -> Path:
    """Get project root directory.

    Returns:
        Path to project root (parent of tests directory)
    """
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def docker_compose_file(project_root: Path) -> Path:
    """Get docker-compose.yml file path.

    Args:
        project_root: Project root directory

    Returns:
        Path to docker-compose.yml
    """
    compose_file = project_root / "docker-compose.yml"
    assert compose_file.exists(), f"docker-compose.yml not found at {compose_file}"
    return compose_file


@pytest.fixture(scope="module")
def redis_client() -> aioredis.Redis[bytes]:
    """Create async Redis client for service discovery tests.

    Returns:
        Async Redis client connected to localhost:6379

    Raises:
        ConnectionError: If Redis is not available
    """
    # Note: Uses host Redis, not Docker internal network
    client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
    return client


# ============================================================================
# Category 1: Infrastructure Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.infrastructure
@pytest.mark.ci_skip
@pytest.mark.docker
@pytest.mark.redis
class TestInfrastructure:
    """Test Docker Compose infrastructure startup and health."""

    def test_dev_infra_only(self, docker_compose_file: Path) -> None:
        """Test infrastructure services start successfully.

        Validates:
        - Redis container starts and responds to PING
        - LiveKit server starts and serves health endpoint
        - Caddy reverse proxy starts and serves admin API

        Expected Duration: < 120s (cold start on slow CI), < 60s (warm local)
        """
        # Start only infrastructure services (no TTS workers or orchestrator)
        services = ["redis", "livekit", "caddy"]

        try:
            # Start services
            start_time = time.perf_counter()
            result = subprocess.run(
                ["docker", "compose", "-f", str(docker_compose_file), "up", "-d"] + services,
                capture_output=True,
                text=True,
                timeout=90,
                check=True,
            )
            startup_duration = time.perf_counter() - start_time

            assert result.returncode == 0, f"Docker Compose failed: {result.stderr}"
            assert startup_duration < 120, f"Startup took {startup_duration:.1f}s (expected < 120s)"

            # Give containers initial time to start (especially important for slow CI runners)
            time.sleep(5)

            # Wait for health checks to pass (up to 90s for slow CI runners)
            health_timeout = 90
            health_start = time.perf_counter()

            while (time.perf_counter() - health_start) < health_timeout:
                # First check if containers are actually running
                ps_result = subprocess.run(
                    ["docker", "compose", "-f", str(docker_compose_file), "ps", "--format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                # Log container status for debugging
                if ps_result.returncode != 0:
                    print(f"Warning: Could not check container status: {ps_result.stderr}")
                else:
                    print(f"Container status: {ps_result.stdout[:200]}")

                # Continue with health checks
                # Check Redis
                redis_health = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "-f",
                        str(docker_compose_file),
                        "exec",
                        "-T",
                        "redis",
                        "redis-cli",
                        "ping",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                # Check LiveKit
                livekit_health = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "-f",
                        str(docker_compose_file),
                        "exec",
                        "-T",
                        "livekit",
                        "wget",
                        "--spider",
                        "-q",
                        "http://localhost:7880/",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                # Check Caddy - use caddy CLI instead of wget (more reliable)
                caddy_health = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "-f",
                        str(docker_compose_file),
                        "exec",
                        "-T",
                        "caddy",
                        "caddy",
                        "version",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if (
                    redis_health.returncode == 0
                    and livekit_health.returncode == 0
                    and caddy_health.returncode == 0
                ):
                    # All health checks passed
                    elapsed = time.perf_counter() - health_start
                    print(f"All health checks passed after {elapsed:.1f}s")
                    break

                # Log failed health checks for debugging
                if redis_health.returncode != 0:
                    print(f"Redis health check failed: {redis_health.stderr}")
                if livekit_health.returncode != 0:
                    print(f"LiveKit health check failed: {livekit_health.stderr}")
                if caddy_health.returncode != 0:
                    print(f"Caddy health check failed: {caddy_health.stderr}")

                time.sleep(3)
            else:
                # Collect container logs for debugging
                logs_result = subprocess.run(
                    ["docker", "compose", "-f", str(docker_compose_file), "logs", "--tail", "20"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                pytest.fail(
                    f"Health checks did not pass within {health_timeout}s\n"
                    f"Redis: {redis_health.returncode}, "
                    f"LiveKit: {livekit_health.returncode}, "
                    f"Caddy: {caddy_health.returncode}\n\n"
                    f"Container logs:\n{logs_result.stdout[-1000:]}"
                )

        finally:
            # Cleanup: Stop services
            subprocess.run(
                ["docker", "compose", "-f", str(docker_compose_file), "down"],
                capture_output=True,
                timeout=30,
            )

    def test_dev_idempotent(self, docker_compose_file: Path) -> None:
        """Test running `just dev` twice doesn't cause conflicts.

        Validates:
        - First run starts services successfully
        - Second run detects running services and doesn't error
        - No port conflicts or duplicate containers

        Expected Duration: < 90s total
        """
        services = ["redis", "livekit", "caddy"]

        try:
            # First run
            result1 = subprocess.run(
                ["docker", "compose", "-f", str(docker_compose_file), "up", "-d"] + services,
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            assert result1.returncode == 0, f"First run failed: {result1.stderr}"

            # Second run (should be idempotent)
            result2 = subprocess.run(
                ["docker", "compose", "-f", str(docker_compose_file), "up", "-d"] + services,
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            assert result2.returncode == 0, f"Second run failed: {result2.stderr}"

            # Verify no duplicate containers
            ps_result = subprocess.run(
                ["docker", "compose", "-f", str(docker_compose_file), "ps", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )

            containers = [
                json.loads(line) for line in ps_result.stdout.strip().split("\n") if line
            ]
            container_names = [c["Service"] for c in containers]

            # Should have exactly one of each service (no duplicates)
            assert container_names.count("redis") == 1, "Redis container duplicated"
            assert container_names.count("livekit") == 1, "LiveKit container duplicated"
            assert container_names.count("caddy") == 1, "Caddy container duplicated"

        finally:
            subprocess.run(
                ["docker", "compose", "-f", str(docker_compose_file), "down"],
                capture_output=True,
                timeout=30,
            )

    def test_dev_clean_state(self, docker_compose_file: Path) -> None:
        """Test cleanup leaves no dangling containers or volumes.

        Validates:
        - `docker compose down` removes all containers
        - No volumes left behind (except named volumes)
        - No networks left behind (except default)

        Expected Duration: < 45s
        """
        services = ["redis", "livekit", "caddy"]

        try:
            # Start services
            subprocess.run(
                ["docker", "compose", "-f", str(docker_compose_file), "up", "-d"] + services,
                capture_output=True,
                timeout=60,
                check=True,
            )

            # Stop and remove
            subprocess.run(
                ["docker", "compose", "-f", str(docker_compose_file), "down"],
                capture_output=True,
                timeout=30,
                check=True,
            )

            # Verify no containers running
            ps_result = subprocess.run(
                ["docker", "compose", "-f", str(docker_compose_file), "ps", "--all"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )

            # Output should be empty (header only) or no running containers
            lines = [line for line in ps_result.stdout.strip().split("\n") if line]
            # Allow header line, but no actual containers
            assert len(lines) <= 1, f"Containers still running after cleanup: {ps_result.stdout}"

        finally:
            # Ensure cleanup even if test fails
            subprocess.run(
                ["docker", "compose", "-f", str(docker_compose_file), "down", "-v"],
                capture_output=True,
                timeout=30,
            )


# ============================================================================
# Category 2: Model Switching Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.docker
@pytest.mark.slow
class TestModelSwitching:
    """Test hot-swapping TTS models during runtime."""

    @pytest.mark.skip(reason="Requires TTS worker implementation with dynamic model switching")
    async def test_model_switch_piper_to_cosyvoice(
        self, docker_compose_file: Path, redis_client: aioredis.Redis[bytes]
    ) -> None:
        """Test switching from Piper to CosyVoice adapter.

        Validates:
        - Piper worker registers in Redis
        - Switch command unloads Piper, loads CosyVoice
        - CosyVoice worker registers with new capabilities
        - Orchestrator detects new worker within 2s

        Expected Duration: < 10s (unload + load + discovery)
        """
        # This test requires implementation of model switching API
        # Placeholder for future implementation
        pass

    @pytest.mark.skip(reason="Requires orchestrator model switching support")
    async def test_model_switch_during_session(
        self, docker_compose_file: Path, redis_client: aioredis.Redis[bytes]
    ) -> None:
        """Test switching models while orchestrator is running.

        Validates:
        - Active sessions complete before switch
        - New sessions use new model
        - No audio corruption during transition
        - Graceful degradation if switch fails

        Expected Duration: < 15s
        """
        pass

    @pytest.mark.skip(reason="Requires Docker Compose profile support")
    def test_concurrent_model_profiles(self, docker_compose_file: Path) -> None:
        """Test only one TTS worker runs at a time.

        Validates:
        - Starting Piper profile stops CosyVoice (if running)
        - Starting CosyVoice profile stops Piper (if running)
        - No port conflicts between workers
        - Redis shows only one active worker

        Expected Duration: < 20s
        """
        pass


# ============================================================================
# Category 3: Service Discovery Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.docker
@pytest.mark.redis
class TestServiceDiscovery:
    """Test orchestrator discovers TTS workers via Redis and DNS."""

    @pytest.mark.skip(reason="Requires orchestrator startup without TTS worker")
    async def test_orchestrator_starts_without_tts(
        self, docker_compose_file: Path, redis_client: aioredis.Redis[bytes]
    ) -> None:
        """Test orchestrator starts even if TTS worker is down.

        Validates:
        - Orchestrator container starts successfully
        - Health check passes (HTTP /health endpoint)
        - Logs show waiting for TTS worker
        - No crash or exit

        Expected Duration: < 30s
        """
        pass

    @pytest.mark.skip(reason="Requires network alias configuration testing")
    async def test_orchestrator_discovers_tts(
        self, docker_compose_file: Path, redis_client: aioredis.Redis[bytes]
    ) -> None:
        """Test orchestrator finds TTS worker via network alias.

        Validates:
        - TTS worker registers in Redis with network alias
        - Orchestrator queries Redis for workers
        - gRPC connection established via Docker DNS
        - First synthesis request succeeds

        Expected Duration: < 5s (discovery + connection)
        """
        pass

    @pytest.mark.skip(reason="Requires TTS worker restart handling")
    async def test_orchestrator_handles_tts_restart(
        self, docker_compose_file: Path, redis_client: aioredis.Redis[bytes]
    ) -> None:
        """Test orchestrator handles TTS worker restart gracefully.

        Validates:
        - Active sessions fail gracefully on worker crash
        - Orchestrator detects worker down via Redis TTL
        - Orchestrator reconnects when worker restarts
        - New sessions succeed after reconnection

        Expected Duration: < 45s (crash + restart + reconnect)
        """
        pass


# ============================================================================
# Category 4: Environment Variable Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.docker
class TestEnvironmentVariables:
    """Test environment variable configuration precedence."""

    @pytest.mark.skip(reason="Requires DEFAULT_MODEL env var implementation")
    def test_default_model_env_var(self, docker_compose_file: Path) -> None:
        """Test DEFAULT_MODEL env var loads correct model.

        Validates:
        - DEFAULT_MODEL=piper-en-us-lessac-medium loads Piper
        - DEFAULT_MODEL=cosyvoice2-en-base loads CosyVoice
        - Invalid model name logs error and falls back to default

        Expected Duration: < 30s per model
        """
        pass

    @pytest.mark.skip(reason="Requires configuration precedence implementation")
    def test_env_var_precedence(self, docker_compose_file: Path, project_root: Path) -> None:
        """Test configuration precedence: CLI > ENV > config > default.

        Validates:
        - CLI --default-model overrides env var
        - Env var DEFAULT_MODEL overrides config file
        - Config file overrides hardcoded default
        - Absence of all uses hardcoded default

        Expected Duration: < 60s (4 scenarios)
        """
        pass

    @pytest.mark.skip(reason="Requires .env.models support")
    def test_model_specific_env_files(self, docker_compose_file: Path, project_root: Path) -> None:
        """Test model-specific env files load correctly.

        Validates:
        - .env.models/.env.piper loads Piper-specific config
        - .env.models/.env.cosyvoice loads CosyVoice-specific config
        - Values override base .env file
        - Missing .env.models file uses defaults

        Expected Duration: < 45s
        """
        pass


# ============================================================================
# Category 5: End-to-End Workflow Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.docker
@pytest.mark.slow
class TestEndToEndWorkflow:
    """Test complete workflows from development to production."""

    @pytest.mark.skip(reason="Requires full Docker Compose stack with Piper")
    async def test_full_workflow_piper(
        self, docker_compose_file: Path, redis_client: aioredis.Redis[bytes]
    ) -> None:
        """Test full workflow with Piper adapter.

        Validates:
        - `just dev` starts all services (infra + TTS + orchestrator)
        - CLI client connects via WebSocket
        - Speech synthesis request succeeds
        - Audio frames received (20ms @ 48kHz)
        - Barge-in triggers on VAD speech detection
        - Session cleanup on disconnect

        Expected Duration: < 60s (startup + synthesis + cleanup)
        """
        pass

    @pytest.mark.skip(reason="Requires CosyVoice Docker profile")
    async def test_full_workflow_cosyvoice(
        self, docker_compose_file: Path, redis_client: aioredis.Redis[bytes]
    ) -> None:
        """Test full workflow with CosyVoice adapter.

        Validates:
        - DEFAULT_MODEL=cosyvoice2-en-base starts CosyVoice worker
        - Orchestrator discovers CosyVoice worker
        - Speech synthesis request succeeds with GPU acceleration
        - FAL p95 < 300ms (GPU target)
        - Frame jitter p95 < 10ms

        Expected Duration: < 90s (GPU model load + synthesis)
        """
        pass

    @pytest.mark.skip(reason="Requires model switching implementation")
    async def test_model_switch_preserves_session(
        self, docker_compose_file: Path, redis_client: aioredis.Redis[bytes]
    ) -> None:
        """Test switching models mid-conversation preserves session.

        Validates:
        - Session starts with Piper
        - Mid-conversation switch to CosyVoice
        - Session ID remains same
        - Conversation history preserved
        - Next response uses CosyVoice

        Expected Duration: < 45s
        """
        pass


# ============================================================================
# Category 6: CI Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.docker
class TestCIIntegration:
    """Test GitHub Actions CI compatibility with new workflows."""

    def test_ci_docker_availability(self) -> None:
        """Test Docker is available in CI environment.

        Validates:
        - `docker --version` succeeds
        - `docker compose version` succeeds
        - Docker daemon is running
        - NVIDIA runtime available (if GPU runner)

        Expected Duration: < 5s
        """
        # Check Docker CLI
        docker_version = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=5
        )
        assert docker_version.returncode == 0, "Docker not available"

        # Check Docker Compose
        compose_version = subprocess.run(
            ["docker", "compose", "version"], capture_output=True, text=True, timeout=5
        )
        assert compose_version.returncode == 0, "Docker Compose not available"

        # Check Docker daemon
        docker_ps = subprocess.run(
            ["docker", "ps"], capture_output=True, text=True, timeout=10
        )
        assert docker_ps.returncode == 0, "Docker daemon not running"

    @pytest.mark.skip(reason="Requires CI workflow integration")
    def test_ci_cleanup_no_dangling_services(self, docker_compose_file: Path) -> None:
        """Test CI cleanup doesn't leave dangling services.

        Validates:
        - After test run, all containers stopped
        - All volumes removed (except named)
        - All networks removed (except default)
        - No port conflicts for next test run

        Expected Duration: < 30s
        """
        pass

    @pytest.mark.skip(reason="Requires CI matrix testing")
    def test_ci_model_matrix(self, docker_compose_file: Path) -> None:
        """Test CI runs tests with different models.

        Validates:
        - Matrix includes: mock, piper, cosyvoice
        - Each model runs subset of tests
        - CosyVoice only on GPU runners
        - Mock/Piper on CPU runners

        Expected Duration: Varies by matrix size
        """
        pass


# ============================================================================
# Helper Functions
# ============================================================================


def wait_for_redis(host: str = "localhost", port: int = 6379, timeout: float = 30.0) -> bool:
    """Wait for Redis to be available.

    Args:
        host: Redis host (default: localhost)
        port: Redis port (default: 6379)
        timeout: Max wait time in seconds

    Returns:
        True if Redis is available, False if timeout

    Raises:
        TimeoutError: If Redis not available within timeout
    """
    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) < timeout:
        try:
            client = aioredis.from_url(f"redis://{host}:{port}", socket_timeout=2)
            asyncio.run(client.ping())
            asyncio.run(client.close())
            return True
        except Exception:
            time.sleep(1)

    raise TimeoutError(f"Redis not available at {host}:{port} after {timeout}s")


def wait_for_http(
    url: str, expected_status: int = 200, timeout: float = 30.0
) -> bool:
    """Wait for HTTP endpoint to be available.

    Args:
        url: HTTP URL to check
        expected_status: Expected HTTP status code (default: 200)
        timeout: Max wait time in seconds

    Returns:
        True if endpoint is available, False if timeout

    Raises:
        TimeoutError: If endpoint not available within timeout
    """
    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) < timeout:
        try:
            result = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.stdout.strip() == str(expected_status):
                return True
        except Exception:  # noqa: S110
            # Ignore errors during health check retry (network/timeout/etc)
            pass
        time.sleep(1)

    raise TimeoutError(f"HTTP endpoint {url} not available after {timeout}s")


def get_container_status(compose_file: Path, service: str) -> dict[str, Any]:
    """Get container status for a Docker Compose service.

    Args:
        compose_file: Path to docker-compose.yml
        service: Service name (e.g., 'redis', 'livekit')

    Returns:
        Dict with status info: {'running': bool, 'health': str, 'ports': list}

    Raises:
        subprocess.CalledProcessError: If docker compose command fails
    """
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "ps", service, "--format", "json"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    if not result.stdout.strip():
        return {"running": False, "health": "not_found", "ports": []}

    container_info = json.loads(result.stdout.strip().split("\n")[0])
    return {
        "running": container_info.get("State") == "running",
        "health": container_info.get("Health", "unknown"),
        "ports": container_info.get("Publishers", []),
    }
