"""Unit tests for Redis-based worker registry.

Tests worker registration, discovery, TTL expiration, and graceful degradation.
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestrator.registry import WorkerRegistration, WorkerRegistry


# Test data fixtures
@pytest.fixture
def sample_worker_registration() -> WorkerRegistration:
    """Create sample worker registration for testing."""
    return WorkerRegistration(
        name="tts-mock@0",
        addr="grpc://localhost:7001",
        capabilities={
            "streaming": True,
            "zero_shot": False,
            "lora": False,
            "cpu_ok": True,
            "languages": ["en"],
            "emotive_zero_prompt": False,
        },
        resident_models=["mock-model"],
        metrics={"rtf": 0.5, "queue_depth": 0},
        last_heartbeat_ts=time.time(),
    )


@pytest.fixture
def sample_gpu_worker_registration() -> WorkerRegistration:
    """Create sample GPU worker registration for testing."""
    return WorkerRegistration(
        name="tts-cosyvoice2@0",
        addr="grpc://localhost:7002",
        capabilities={
            "streaming": True,
            "zero_shot": True,
            "lora": False,
            "cpu_ok": False,
            "languages": ["en", "zh"],
            "emotive_zero_prompt": True,
        },
        resident_models=["cosyvoice2-en-base"],
        metrics={"rtf": 0.2, "queue_depth": 1},
        last_heartbeat_ts=time.time(),
    )


class TestWorkerRegistration:
    """Test WorkerRegistration dataclass and serialization."""

    def test_worker_registration_creation(
        self, sample_worker_registration: WorkerRegistration
    ) -> None:
        """Test creating worker registration."""
        assert sample_worker_registration.name == "tts-mock@0"
        assert sample_worker_registration.addr == "grpc://localhost:7001"
        assert sample_worker_registration.capabilities["streaming"] is True
        assert "en" in sample_worker_registration.capabilities["languages"]
        assert "mock-model" in sample_worker_registration.resident_models
        assert sample_worker_registration.metrics["rtf"] == 0.5

    def test_worker_registration_defaults(self) -> None:
        """Test worker registration with default values."""
        worker = WorkerRegistration(
            name="test-worker",
            addr="grpc://test:7001",
            capabilities={"streaming": True},
            resident_models=[],
        )
        assert worker.metrics == {}
        assert worker.last_heartbeat_ts == 0.0

    def test_worker_registration_to_json(
        self, sample_worker_registration: WorkerRegistration
    ) -> None:
        """Test serialization to JSON."""
        json_str = sample_worker_registration.to_json()
        assert isinstance(json_str, str)

        # Parse and validate
        data = json.loads(json_str)
        assert data["name"] == "tts-mock@0"
        assert data["addr"] == "grpc://localhost:7001"
        assert data["capabilities"]["streaming"] is True
        assert "mock-model" in data["resident_models"]

    def test_worker_registration_from_json(
        self, sample_worker_registration: WorkerRegistration
    ) -> None:
        """Test deserialization from JSON."""
        json_str = sample_worker_registration.to_json()
        restored = WorkerRegistration.from_json(json_str)

        assert restored.name == sample_worker_registration.name
        assert restored.addr == sample_worker_registration.addr
        assert restored.capabilities == sample_worker_registration.capabilities
        assert restored.resident_models == sample_worker_registration.resident_models
        assert restored.metrics == sample_worker_registration.metrics

    def test_worker_registration_from_json_invalid(self) -> None:
        """Test deserialization from invalid JSON."""
        # Invalid JSON
        with pytest.raises(ValueError, match="Invalid worker registration JSON"):
            WorkerRegistration.from_json("not valid json")

        # Missing required fields
        with pytest.raises(ValueError, match="Invalid worker registration JSON"):
            WorkerRegistration.from_json('{"name": "test"}')

    def test_worker_registration_roundtrip(
        self, sample_worker_registration: WorkerRegistration
    ) -> None:
        """Test JSON roundtrip (serialize + deserialize)."""
        json_str = sample_worker_registration.to_json()
        restored = WorkerRegistration.from_json(json_str)

        # Verify complete equality
        assert restored.name == sample_worker_registration.name
        assert restored.addr == sample_worker_registration.addr
        assert restored.capabilities == sample_worker_registration.capabilities
        assert restored.resident_models == sample_worker_registration.resident_models
        assert restored.metrics == sample_worker_registration.metrics
        assert restored.last_heartbeat_ts == sample_worker_registration.last_heartbeat_ts


class TestWorkerRegistry:
    """Test WorkerRegistry Redis operations."""

    @pytest.fixture
    def mock_redis(self) -> AsyncMock:
        """Create mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.ping = AsyncMock(return_value=True)
        redis_mock.setex = AsyncMock()
        redis_mock.get = AsyncMock()
        redis_mock.keys = AsyncMock(return_value=[])
        redis_mock.delete = AsyncMock(return_value=1)
        redis_mock.close = AsyncMock()
        return redis_mock

    @pytest.fixture
    def mock_pool(self) -> AsyncMock:
        """Create mock connection pool."""
        pool_mock = AsyncMock()
        pool_mock.disconnect = AsyncMock()
        return pool_mock

    @pytest.fixture
    def registry(self) -> WorkerRegistry:
        """Create registry instance for testing."""
        return WorkerRegistry(
            redis_url="redis://localhost:6379",
            db=0,
            worker_key_prefix="worker:",
            worker_ttl_seconds=30,
            connection_pool_size=10,
        )

    async def test_registry_initialization(self, registry: WorkerRegistry) -> None:
        """Test registry initialization."""
        assert registry.redis_url == "redis://localhost:6379"
        assert registry.db == 0
        assert registry.worker_key_prefix == "worker:"
        assert registry.worker_ttl_seconds == 30
        assert registry.connection_pool_size == 10
        assert registry._connected is False

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_registry_connect(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
    ) -> None:
        """Test Redis connection establishment."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis

        await registry.connect()

        assert registry._connected is True
        mock_redis.ping.assert_awaited_once()

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_registry_connect_idempotent(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
    ) -> None:
        """Test that connect() is idempotent."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis

        # Connect twice
        await registry.connect()
        await registry.connect()

        # Should only ping once
        assert mock_redis.ping.await_count == 1

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_registry_connect_failure(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
    ) -> None:
        """Test Redis connection failure."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.ping.side_effect = ConnectionError("Connection refused")

        with pytest.raises(ConnectionError, match="Redis connection failed"):
            await registry.connect()

        assert registry._connected is False

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_registry_disconnect(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
    ) -> None:
        """Test Redis disconnection."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis

        await registry.connect()
        await registry.disconnect()

        assert registry._connected is False
        mock_redis.close.assert_awaited_once()
        mock_pool.disconnect.assert_awaited_once()

    async def test_registry_disconnect_idempotent(
        self, registry: WorkerRegistry
    ) -> None:
        """Test that disconnect() is idempotent."""
        # Disconnect without connecting should not raise
        await registry.disconnect()
        await registry.disconnect()

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_health_check_healthy(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
    ) -> None:
        """Test health check when Redis is healthy."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis

        await registry.connect()
        is_healthy = await registry.health_check()

        assert is_healthy is True
        assert mock_redis.ping.await_count >= 2  # connect + health_check

    async def test_health_check_not_connected(self, registry: WorkerRegistry) -> None:
        """Test health check when not connected."""
        is_healthy = await registry.health_check()
        assert is_healthy is False

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_health_check_failed(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
    ) -> None:
        """Test health check when Redis ping fails."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis

        await registry.connect()

        # Make ping fail for health check
        mock_redis.ping.reset_mock()
        mock_redis.ping.side_effect = Exception("Connection lost")

        is_healthy = await registry.health_check()
        assert is_healthy is False

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    @patch("time.time")
    async def test_register_worker(
        self,
        mock_time: MagicMock,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
        sample_worker_registration: WorkerRegistration,
    ) -> None:
        """Test registering a worker."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_time.return_value = 12345.0

        await registry.connect()
        await registry.register_worker(sample_worker_registration)

        # Verify Redis setex called with correct parameters
        mock_redis.setex.assert_awaited_once()
        args = mock_redis.setex.call_args[0]
        assert args[0] == "worker:tts-mock@0"  # key
        assert args[1] == 30  # ttl
        assert isinstance(args[2], str)  # JSON data

        # Verify heartbeat timestamp was updated
        assert sample_worker_registration.last_heartbeat_ts == 12345.0

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_register_worker_custom_ttl(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
        sample_worker_registration: WorkerRegistration,
    ) -> None:
        """Test registering a worker with custom TTL."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis

        await registry.connect()
        await registry.register_worker(sample_worker_registration, ttl_s=60)

        # Verify custom TTL used
        args = mock_redis.setex.call_args[0]
        assert args[1] == 60  # ttl

    async def test_register_worker_not_connected(
        self,
        registry: WorkerRegistry,
        sample_worker_registration: WorkerRegistration,
    ) -> None:
        """Test registering worker when not connected."""
        with pytest.raises(ConnectionError, match="Redis not connected"):
            await registry.register_worker(sample_worker_registration)

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_get_workers_empty(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
    ) -> None:
        """Test retrieving workers when none registered."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.keys.return_value = []

        await registry.connect()
        workers = await registry.get_workers()

        assert workers == []
        mock_redis.keys.assert_awaited_once_with("worker:*")

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_get_workers_single(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
        sample_worker_registration: WorkerRegistration,
    ) -> None:
        """Test retrieving single worker."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.keys.return_value = ["worker:tts-mock@0"]
        mock_redis.get.return_value = sample_worker_registration.to_json()

        await registry.connect()
        workers = await registry.get_workers()

        assert len(workers) == 1
        assert workers[0].name == "tts-mock@0"
        assert workers[0].addr == "grpc://localhost:7001"

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_get_workers_multiple(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
        sample_worker_registration: WorkerRegistration,
        sample_gpu_worker_registration: WorkerRegistration,
    ) -> None:
        """Test retrieving multiple workers."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.keys.return_value = ["worker:tts-mock@0", "worker:tts-cosyvoice2@0"]

        # Return different data for each key
        async def mock_get(key: str) -> str:
            if key == "worker:tts-mock@0":
                return sample_worker_registration.to_json()
            else:
                return sample_gpu_worker_registration.to_json()

        mock_redis.get.side_effect = mock_get

        await registry.connect()
        workers = await registry.get_workers()

        assert len(workers) == 2
        worker_names = {w.name for w in workers}
        assert "tts-mock@0" in worker_names
        assert "tts-cosyvoice2@0" in worker_names

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_get_workers_invalid_data(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
    ) -> None:
        """Test retrieving workers with invalid data (should skip)."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.keys.return_value = ["worker:invalid"]
        mock_redis.get.return_value = "invalid json"

        await registry.connect()
        workers = await registry.get_workers()

        # Invalid worker should be skipped
        assert workers == []

    async def test_get_workers_not_connected(self, registry: WorkerRegistry) -> None:
        """Test retrieving workers when not connected."""
        with pytest.raises(ConnectionError, match="Redis not connected"):
            await registry.get_workers()

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_get_worker_by_name_found(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
        sample_worker_registration: WorkerRegistration,
    ) -> None:
        """Test retrieving worker by name when found."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.get.return_value = sample_worker_registration.to_json()

        await registry.connect()
        worker = await registry.get_worker_by_name("tts-mock@0")

        assert worker is not None
        assert worker.name == "tts-mock@0"
        assert worker.addr == "grpc://localhost:7001"
        mock_redis.get.assert_awaited_once_with("worker:tts-mock@0")

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_get_worker_by_name_not_found(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
    ) -> None:
        """Test retrieving worker by name when not found."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.get.return_value = None

        await registry.connect()
        worker = await registry.get_worker_by_name("nonexistent")

        assert worker is None

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_get_worker_by_name_invalid_data(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
    ) -> None:
        """Test retrieving worker by name with invalid data."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.get.return_value = "invalid json"

        await registry.connect()
        worker = await registry.get_worker_by_name("invalid")

        # Should return None for invalid data
        assert worker is None

    async def test_get_worker_by_name_not_connected(
        self, registry: WorkerRegistry
    ) -> None:
        """Test retrieving worker by name when not connected."""
        with pytest.raises(ConnectionError, match="Redis not connected"):
            await registry.get_worker_by_name("test")

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_remove_worker_found(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
    ) -> None:
        """Test removing worker when found."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.delete.return_value = 1  # 1 key deleted

        await registry.connect()
        await registry.remove_worker("tts-mock@0")

        mock_redis.delete.assert_awaited_once_with("worker:tts-mock@0")

    @patch("src.orchestrator.registry.ConnectionPool")
    @patch("src.orchestrator.registry.aioredis.Redis")
    async def test_remove_worker_not_found(
        self,
        mock_redis_class: MagicMock,
        mock_pool_class: MagicMock,
        registry: WorkerRegistry,
        mock_redis: AsyncMock,
        mock_pool: AsyncMock,
    ) -> None:
        """Test removing worker when not found (already expired)."""
        mock_pool_class.from_url.return_value = mock_pool
        mock_redis_class.return_value = mock_redis
        mock_redis.delete.return_value = 0  # 0 keys deleted

        await registry.connect()
        await registry.remove_worker("nonexistent")

        mock_redis.delete.assert_awaited_once_with("worker:nonexistent")

    async def test_remove_worker_not_connected(self, registry: WorkerRegistry) -> None:
        """Test removing worker when not connected."""
        with pytest.raises(ConnectionError, match="Redis not connected"):
            await registry.remove_worker("test")
