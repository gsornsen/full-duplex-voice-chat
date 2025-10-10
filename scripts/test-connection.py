#!/usr/bin/env python3
"""Test connections to all M2 services.

This tool validates connectivity to:
- Redis (service discovery)
- TTS Worker (gRPC)
- Orchestrator (WebSocket and HTTP health)

Exit codes:
- 0: All services connected successfully
- 1: One or more services failed

Usage:
    python scripts/test-connection.py
    # Or with custom URLs:
    REDIS_URL=redis://localhost:6379 \\
    WORKER_ADDR=localhost:7001 \\
    ORCH_WS=ws://localhost:8080 \\
    ORCH_HEALTH=http://localhost:8081/health \\
    python scripts/test-connection.py
"""

import asyncio
import os
import sys
from typing import Any

try:
    import aiohttp
    import grpc
    from redis import asyncio as aioredis
except ImportError as e:
    print("ERROR: Required dependencies not installed")
    print(f"Missing: {e.name}")
    print("Resolution: uv sync")
    sys.exit(1)


async def test_redis(url: str) -> bool:
    """Test Redis connectivity.

    Args:
        url: Redis connection URL

    Returns:
        True if connected successfully, False otherwise
    """
    print(f"Testing Redis ({url})... ", end="", flush=True)
    try:
        redis = aioredis.from_url(url, decode_responses=True)
        result = await redis.ping()
        await redis.close()

        if result:
            print("OK")
            return True
        else:
            print("FAILED")
            print(f"   Cause: Redis ping returned {result}")
            print("   Resolution: Check Redis server status")
            return False

    except ConnectionRefusedError:
        print("FAILED")
        print("   Cause: Connection refused - Redis not running")
        print("   Resolution: just redis (or docker run redis)")
        print("   See: docs/runbooks/REDIS.md")
        return False
    except Exception as e:
        print("FAILED")
        print(f"   Cause: {type(e).__name__}: {e}")
        print("   Resolution: Check Redis URL and server status")
        print("   See: docs/runbooks/REDIS.md")
        return False


async def test_grpc_worker(addr: str) -> bool:
    """Test gRPC worker connectivity.

    Args:
        addr: Worker gRPC address (host:port)

    Returns:
        True if connected successfully, False otherwise
    """
    print(f"Testing gRPC Worker ({addr})... ", end="", flush=True)
    try:
        channel = grpc.aio.insecure_channel(addr)

        # Try to connect with timeout
        try:
            await asyncio.wait_for(channel.channel_ready(), timeout=5.0)
        except asyncio.TimeoutError:
            print("FAILED")
            print("   Cause: Connection timeout - worker not responding")
            print("   Resolution: Check worker is running and accessible")
            print("   See: docs/runbooks/GRPC_WORKER.md")
            await channel.close()
            return False

        print("OK")
        await channel.close()
        return True

    except Exception as e:
        print("FAILED")
        print(f"   Cause: {type(e).__name__}: {e}")
        print("   Resolution: Ensure TTS worker is running")
        print("   Commands: docker ps | grep tts-worker")
        print("   Or: just run-tts-sesame")
        print("   See: docs/runbooks/GRPC_WORKER.md")
        return False


async def test_orchestrator_ws(url: str) -> bool:
    """Test WebSocket endpoint.

    Args:
        url: WebSocket URL

    Returns:
        True if connected successfully, False otherwise
    """
    print(f"Testing Orchestrator WebSocket ({url})... ", end="", flush=True)
    try:
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(url) as ws:
                await ws.close()

        print("OK")
        return True

    except aiohttp.ClientConnectorError:
        print("FAILED")
        print("   Cause: Connection refused - orchestrator not running")
        print("   Resolution: just run-orch")
        print("   Or: docker compose up orchestrator")
        print("   See: docs/runbooks/WEBSOCKET.md")
        return False
    except asyncio.TimeoutError:
        print("FAILED")
        print("   Cause: Connection timeout")
        print("   Resolution: Check orchestrator logs and network")
        print("   See: docs/runbooks/WEBSOCKET.md")
        return False
    except Exception as e:
        print("FAILED")
        print(f"   Cause: {type(e).__name__}: {e}")
        print("   Resolution: Check orchestrator is running on correct port")
        print("   See: docs/runbooks/WEBSOCKET.md")
        return False


async def test_orchestrator_health(url: str) -> bool:
    """Test HTTP health endpoint.

    Args:
        url: Health check URL

    Returns:
        True if healthy, False otherwise
    """
    print(f"Testing Orchestrator Health ({url})... ", end="", flush=True)
    try:
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                data = await resp.json()

                if resp.status == 200:
                    status = data.get("status", "unknown")
                    uptime = data.get("uptime_seconds", 0)
                    print(f"OK (status={status}, uptime={uptime:.1f}s)")
                    return True
                else:
                    print(f"UNHEALTHY (HTTP {resp.status})")
                    print(f"   Health data: {data}")
                    print("   Resolution: Check orchestrator logs for errors")
                    return False

    except aiohttp.ClientConnectorError:
        print("FAILED")
        print("   Cause: Connection refused - health endpoint not available")
        print("   Resolution: Ensure orchestrator is running")
        print("   Note: Health endpoint should be on port 8081")
        return False
    except asyncio.TimeoutError:
        print("FAILED")
        print("   Cause: Health check timeout")
        print("   Resolution: Orchestrator may be overloaded or unresponsive")
        return False
    except Exception as e:
        print("FAILED")
        print(f"   Cause: {type(e).__name__}: {e}")
        print("   Resolution: Check orchestrator configuration")
        return False


async def main() -> int:
    """Run all connection tests.

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    print("=" * 60)
    print("M2 Connection Test")
    print("=" * 60)
    print()

    # Get URLs from environment or use defaults
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    worker_addr = os.getenv("WORKER_ADDR", "localhost:7001")
    orch_ws = os.getenv("ORCH_WS", "ws://localhost:8080")
    orch_health = os.getenv("ORCH_HEALTH", "http://localhost:8081/health")

    # Run all tests
    results = await asyncio.gather(
        test_redis(redis_url),
        test_grpc_worker(worker_addr),
        test_orchestrator_ws(orch_ws),
        test_orchestrator_health(orch_health),
        return_exceptions=False,
    )

    print()
    print("=" * 60)

    all_ok = all(results)
    if all_ok:
        print("Status: ALL SERVICES CONNECTED")
        print()
        print("Next steps:")
        print("  - Connect client: just cli")
        print("  - Send test message: echo 'Hello, world!' | just cli")
        print("  - View logs: docker compose logs -f")
        return 0
    else:
        failed_count = sum(1 for r in results if not r)
        print(f"Status: {failed_count} SERVICE(S) FAILED")
        print()
        print("Troubleshooting:")
        print("  1. Run pre-flight check: ./scripts/preflight-check.sh")
        print("  2. Check all services running: docker ps")
        print("  3. View logs: docker compose logs")
        print("  4. Consult runbooks: docs/runbooks/")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
