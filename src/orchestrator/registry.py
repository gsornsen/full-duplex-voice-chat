"""Redis-based worker discovery and registration.

This module provides worker service discovery using Redis with TTL-based
heartbeat registration. Designed for M2 scope with static routing, extensible
for M9+ dynamic routing.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from redis import asyncio as aioredis
from redis.asyncio import ConnectionPool

logger = logging.getLogger(__name__)


@dataclass
class WorkerRegistration:
    """Worker registration metadata.

    Attributes:
        name: Unique worker identifier (e.g., "tts-cosyvoice2@0")
        addr: Worker gRPC address (e.g., "grpc://tts-cosy:7002")
        capabilities: Worker capabilities dictionary
            - streaming: bool (supports streaming synthesis)
            - zero_shot: bool (supports zero-shot voice cloning)
            - lora: bool (supports LoRA adapters)
            - cpu_ok: bool (can run on CPU)
            - languages: list[str] (supported language codes)
            - emotive_zero_prompt: bool (supports emotion without prompt)
        resident_models: List of currently loaded model IDs
        metrics: Performance metrics dictionary
            - rtf: float (real-time factor)
            - queue_depth: int (current queue depth)
        last_heartbeat_ts: Unix timestamp of last heartbeat (auto-updated)
    """

    name: str
    addr: str
    capabilities: dict[str, Any]
    resident_models: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    last_heartbeat_ts: float = 0.0

    def to_json(self) -> str:
        """Serialize to JSON string for Redis storage.

        Returns:
            JSON-encoded registration data
        """
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "WorkerRegistration":
        """Deserialize from JSON string.

        Args:
            data: JSON-encoded registration data

        Returns:
            Deserialized WorkerRegistration

        Raises:
            ValueError: If JSON is invalid or missing required fields
        """
        try:
            obj = json.loads(data)
            return cls(**obj)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid worker registration JSON: {e}") from e


class WorkerRegistry:
    """Worker service discovery via Redis.

    Provides TTL-based worker registration with automatic expiration.
    Workers must periodically send heartbeats to maintain registration.

    For M2: Single worker registration and discovery.
    For M9+: Multi-worker load balancing and capability matching.
    """

    def __init__(
        self,
        redis_url: str,
        db: int = 0,
        worker_key_prefix: str = "worker:",
        worker_ttl_seconds: int = 30,
        connection_pool_size: int = 10,
    ) -> None:
        """Initialize registry with Redis connection.

        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379")
            db: Redis database number (0-15)
            worker_key_prefix: Key prefix for worker registrations
            worker_ttl_seconds: Worker registration TTL (heartbeat interval)
            connection_pool_size: Redis connection pool size
        """
        self.redis_url = redis_url
        self.db = db
        self.worker_key_prefix = worker_key_prefix
        self.worker_ttl_seconds = worker_ttl_seconds
        self.connection_pool_size = connection_pool_size

        # Connection pool (lazy initialization)
        self._pool: Any = None
        self._redis: Any = None
        self._connected = False

    async def connect(self) -> None:
        """Establish Redis connection pool.

        This method is idempotent - safe to call multiple times.

        Raises:
            ConnectionError: If Redis connection fails
        """
        if self._connected:
            return

        try:
            self._pool = ConnectionPool.from_url(
                self.redis_url,
                db=self.db,
                max_connections=self.connection_pool_size,
                decode_responses=True,
            )
            self._redis = aioredis.Redis(connection_pool=self._pool)

            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info(
                f"Connected to Redis at {self.redis_url} (db={self.db}, "
                f"pool_size={self.connection_pool_size})"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Redis connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Close Redis connection pool gracefully.

        This method is idempotent - safe to call multiple times.
        """
        if not self._connected:
            return

        try:
            if self._redis:
                await self._redis.close()
            if self._pool:
                await self._pool.disconnect()
            self._connected = False
            logger.info("Disconnected from Redis")
        except Exception as e:
            logger.warning(f"Error during Redis disconnect: {e}")

    async def health_check(self) -> bool:
        """Check if Redis connection is healthy.

        Returns:
            True if Redis is reachable and responsive, False otherwise
        """
        if not self._connected or not self._redis:
            return False

        try:
            await self._redis.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False

    async def register_worker(
        self, registration: WorkerRegistration, ttl_s: int | None = None
    ) -> None:
        """Register worker with TTL-based expiration.

        Workers must re-register before TTL expires to maintain presence.

        Args:
            registration: Worker registration metadata
            ttl_s: Optional TTL override (defaults to worker_ttl_seconds)

        Raises:
            ConnectionError: If Redis is not connected
            ValueError: If registration data is invalid
        """
        if not self._connected or not self._redis:
            raise ConnectionError("Redis not connected. Call connect() first.")

        ttl = ttl_s if ttl_s is not None else self.worker_ttl_seconds
        key = f"{self.worker_key_prefix}{registration.name}"

        try:
            # Update heartbeat timestamp
            import time

            registration.last_heartbeat_ts = time.time()

            # Store with TTL
            await self._redis.setex(key, ttl, registration.to_json())
            logger.debug(
                f"Registered worker '{registration.name}' at {registration.addr} (ttl={ttl}s)"
            )
        except Exception as e:
            logger.error(f"Failed to register worker '{registration.name}': {e}")
            raise ValueError(f"Worker registration failed: {e}") from e

    async def get_workers(self) -> list[WorkerRegistration]:
        """Retrieve all registered workers.

        Only returns workers with valid (non-expired) registrations.

        Returns:
            List of registered workers (may be empty)

        Raises:
            ConnectionError: If Redis is not connected
        """
        if not self._connected or not self._redis:
            raise ConnectionError("Redis not connected. Call connect() first.")

        try:
            # Find all worker keys
            pattern = f"{self.worker_key_prefix}*"
            keys = await self._redis.keys(pattern)

            if not keys:
                logger.debug("No workers currently registered")
                return []

            # Fetch all worker registrations
            workers: list[WorkerRegistration] = []
            for key in keys:
                data = await self._redis.get(key)
                if data:
                    try:
                        worker = WorkerRegistration.from_json(data)
                        workers.append(worker)
                    except ValueError as e:
                        logger.warning(f"Skipping invalid worker data at {key}: {e}")

            logger.debug(f"Retrieved {len(workers)} registered workers")
            return workers

        except Exception as e:
            logger.error(f"Failed to retrieve workers: {e}")
            raise ConnectionError(f"Worker retrieval failed: {e}") from e

    async def get_worker_by_name(self, name: str) -> WorkerRegistration | None:
        """Retrieve specific worker by name.

        Args:
            name: Worker name identifier

        Returns:
            Worker registration if found and valid, None otherwise

        Raises:
            ConnectionError: If Redis is not connected
        """
        if not self._connected or not self._redis:
            raise ConnectionError("Redis not connected. Call connect() first.")

        try:
            key = f"{self.worker_key_prefix}{name}"
            data = await self._redis.get(key)

            if not data:
                logger.debug(f"Worker '{name}' not found")
                return None

            worker = WorkerRegistration.from_json(data)
            logger.debug(f"Retrieved worker '{name}' at {worker.addr}")
            return worker

        except ValueError as e:
            logger.warning(f"Invalid worker data for '{name}': {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve worker '{name}': {e}")
            raise ConnectionError(f"Worker retrieval failed: {e}") from e

    async def remove_worker(self, name: str) -> None:
        """Remove worker registration manually.

        Typically not needed - workers expire automatically via TTL.
        Useful for graceful shutdown.

        Args:
            name: Worker name identifier

        Raises:
            ConnectionError: If Redis is not connected
        """
        if not self._connected or not self._redis:
            raise ConnectionError("Redis not connected. Call connect() first.")

        try:
            key = f"{self.worker_key_prefix}{name}"
            deleted = await self._redis.delete(key)

            if deleted:
                logger.info(f"Removed worker '{name}'")
            else:
                logger.debug(f"Worker '{name}' not found (already expired?)")

        except Exception as e:
            logger.error(f"Failed to remove worker '{name}': {e}")
            raise ConnectionError(f"Worker removal failed: {e}") from e
