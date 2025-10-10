"""Worker registration and discovery example.

Demonstrates:
- Worker registration in Redis
- Capability advertisement
- Heartbeat mechanism
- Worker discovery queries
- TTL and expiration handling

Usage:
    python examples/worker_registration.py
    python examples/worker_registration.py --redis-url redis://localhost:6379
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Any

import redis.asyncio as redis

from src.common.types import WorkerAddress, WorkerCapabilities, WorkerMetrics


class WorkerRegistrar:
    """Worker registration manager.

    Handles worker registration, heartbeat, and discovery using Redis.
    Workers announce their presence and capabilities to enable dynamic
    routing by the orchestrator.

    Example:
        >>> registrar = WorkerRegistrar("redis://localhost:6379")
        >>> await registrar.connect()
        >>> await registrar.register_worker(
        ...     name="tts-worker-0",
        ...     addr="grpc://localhost:7001",
        ...     capabilities={"streaming": True, "languages": ["en"]},
        ... )
    """

    def __init__(self, redis_url: str, key_prefix: str = "worker:") -> None:
        """Initialize worker registrar.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for worker keys in Redis
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.client: redis.Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        self.client = redis.from_url(self.redis_url, decode_responses=True)
        await self.client.ping()
        print(f"✓ Connected to Redis at {self.redis_url}\n")

    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.aclose()
            print("\n✓ Disconnected from Redis")

    async def register_worker(
        self,
        name: str,
        addr: WorkerAddress,
        capabilities: WorkerCapabilities,
        resident_models: list[str] | None = None,
        metrics: WorkerMetrics | None = None,
        ttl_seconds: int = 30,
    ) -> None:
        """Register worker in Redis with TTL.

        Args:
            name: Worker name (unique identifier)
            addr: Worker gRPC address
            capabilities: Worker capability flags
            resident_models: List of loaded models (default: empty)
            metrics: Current performance metrics (default: zeros)
            ttl_seconds: Registration TTL in seconds

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self.client:
            raise RuntimeError("Not connected to Redis")

        # Build registration data
        registration = {
            "name": name,
            "addr": addr,
            "capabilities": capabilities,
            "resident_models": resident_models or [],
            "metrics": metrics
            or {"rtf": 0.0, "queue_depth": 0.0, "active_sessions": 0, "models_loaded": 0},
            "registered_at": int(time.time()),
        }

        # Store in Redis with TTL
        key = f"{self.key_prefix}{name}"
        await self.client.setex(
            key,
            ttl_seconds,
            json.dumps(registration),
        )

        print(f"→ Registered worker '{name}' at {addr}")
        print(f"  Capabilities: {capabilities}")
        print(f"  Resident models: {resident_models or []}")
        print(f"  TTL: {ttl_seconds} seconds")

    async def heartbeat(self, name: str, ttl_seconds: int = 30) -> bool:
        """Update worker TTL (heartbeat).

        Args:
            name: Worker name
            ttl_seconds: New TTL in seconds

        Returns:
            True if worker exists and TTL updated, False otherwise

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self.client:
            raise RuntimeError("Not connected to Redis")

        key = f"{self.key_prefix}{name}"

        # Check if worker exists
        if not await self.client.exists(key):
            print(f"⚠  Worker '{name}' not found (expired?)")
            return False

        # Update TTL
        await self.client.expire(key, ttl_seconds)
        print(f"→ Heartbeat for '{name}' (TTL reset to {ttl_seconds}s)")
        return True

    async def get_workers(self) -> list[dict[str, Any]]:
        """Get all registered workers.

        Returns:
            List of worker registration dictionaries

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self.client:
            raise RuntimeError("Not connected to Redis")

        # Scan for all worker keys
        keys = []
        async for key in self.client.scan_iter(match=f"{self.key_prefix}*"):
            keys.append(key)

        # Fetch all worker data
        workers = []
        for key in keys:
            data = await self.client.get(key)
            if data:
                worker = json.loads(data)
                workers.append(worker)

        return workers

    async def unregister_worker(self, name: str) -> bool:
        """Unregister worker from Redis.

        Args:
            name: Worker name

        Returns:
            True if worker was removed, False if not found

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self.client:
            raise RuntimeError("Not connected to Redis")

        key = f"{self.key_prefix}{name}"
        deleted = await self.client.delete(key)

        if deleted:
            print(f"→ Unregistered worker '{name}'")
            return True
        else:
            print(f"⚠  Worker '{name}' not found")
            return False


async def demo_registration() -> None:
    """Demonstrate worker registration and discovery."""
    print("=== Worker Registration Demo ===\n")

    registrar = WorkerRegistrar("redis://localhost:6379")

    try:
        await registrar.connect()

        # Register multiple workers
        workers = [
            {
                "name": "tts-worker-0",
                "addr": "grpc://localhost:7001",
                "capabilities": {
                    "streaming": True,
                    "zero_shot": False,
                    "lora": False,
                    "cpu_ok": True,
                    "languages": ["en"],
                    "emotive_zero_prompt": False,
                },
                "resident_models": ["mock-440hz"],
                "metrics": {
                    "rtf": 0.1,
                    "queue_depth": 0.0,
                    "active_sessions": 0,
                    "models_loaded": 1,
                },
            },
            {
                "name": "tts-worker-1",
                "addr": "grpc://localhost:7002",
                "capabilities": {
                    "streaming": True,
                    "zero_shot": True,
                    "lora": False,
                    "cpu_ok": False,
                    "languages": ["en", "zh"],
                    "emotive_zero_prompt": True,
                },
                "resident_models": ["cosyvoice2-en-base"],
                "metrics": {
                    "rtf": 0.3,
                    "queue_depth": 1.0,
                    "active_sessions": 1,
                    "models_loaded": 1,
                },
            },
        ]

        print("Registering workers...\n")
        for worker in workers:
            await registrar.register_worker(
                name=worker["name"],
                addr=worker["addr"],
                capabilities=worker["capabilities"],  # type: ignore[arg-type]
                resident_models=worker["resident_models"],
                metrics=worker["metrics"],  # type: ignore[arg-type]
                ttl_seconds=30,
            )
            print()

        # Query all workers
        print("Querying all workers...\n")
        all_workers = await registrar.get_workers()
        print(f"Found {len(all_workers)} workers:")
        for worker in all_workers:
            print(f"  - {worker['name']} at {worker['addr']}")
            print(f"    Languages: {worker['capabilities']['languages']}")
            print(f"    RTF: {worker['metrics']['rtf']}")
        print()

        # Simulate heartbeat
        print("Simulating heartbeat for 5 seconds...")
        for i in range(5):
            await asyncio.sleep(1)
            for worker in workers:
                await registrar.heartbeat(worker["name"], ttl_seconds=30)
        print()

        # Test TTL expiration (wait without heartbeat)
        print("Waiting 10 seconds to test TTL expiration...")
        print("(Workers will expire after 30s without heartbeat)")
        await asyncio.sleep(10)
        print()

        # Clean up - unregister workers
        print("Cleaning up...")
        for worker in workers:
            await registrar.unregister_worker(worker["name"])
        print()

        # Verify removal
        remaining_workers = await registrar.get_workers()
        print(f"Remaining workers: {len(remaining_workers)}")

    finally:
        await registrar.close()


def main() -> None:
    """Parse arguments and run demo."""
    parser = argparse.ArgumentParser(description="Worker registration demo")
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379",
        help="Redis connection URL (default: redis://localhost:6379)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(demo_registration())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except redis.ConnectionError:
        print(
            "\n✗ Redis connection failed. Make sure Redis is running:\n"
            "  just redis"
        )
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
