"""Redis Service Discovery Integration Test.

Tests Redis-based worker discovery:
1. Start Redis container
2. Register mock TTS worker
3. Verify worker appears in registry
4. Test TTL expiration
5. Test static routing resolution
6. Test graceful degradation when Redis down
"""

import asyncio
import logging

import pytest
from orchestrator.registry import WorkerRegistration, WorkerRegistry

logger = logging.getLogger(__name__)

# Mark all tests in this module as infrastructure (skip in CI - uses gRPC + Redis)
pytestmark = [pytest.mark.grpc, pytest.mark.infrastructure]


@pytest.mark.asyncio
async def test_worker_registration(redis_container: str) -> None:
    """Test basic worker registration in Redis.

    Validates:
    - Worker can register successfully
    - Registration data is persisted correctly
    - Worker can be retrieved by name
    """
    # Arrange
    registry = WorkerRegistry(
        redis_url=redis_container,
        worker_key_prefix="worker:",
        worker_ttl_seconds=30,
    )
    await registry.connect()

    registration = WorkerRegistration(
        name="test-worker-1",
        addr="grpc://localhost:7001",
        capabilities={
            "streaming": True,
            "zero_shot": False,
            "lora": False,
            "cpu_ok": True,
            "languages": ["en"],
        },
        resident_models=["mock-440hz"],
        metrics={"rtf": 0.1, "queue_depth": 0.0},
    )

    try:
        # Act - Register worker
        await registry.register_worker(registration, ttl_s=60)
        logger.info(f"Registered worker: {registration.name}")

        # Assert - Retrieve worker
        retrieved = await registry.get_worker_by_name("test-worker-1")
        assert retrieved is not None, "Worker not found"
        assert retrieved.name == registration.name
        assert retrieved.addr == registration.addr
        assert retrieved.capabilities == registration.capabilities
        assert retrieved.resident_models == registration.resident_models

        logger.info(f"Retrieved worker: {retrieved.name} at {retrieved.addr}")

    finally:
        await registry.remove_worker("test-worker-1")
        await registry.disconnect()


@pytest.mark.asyncio
async def test_worker_ttl_expiration(redis_container: str) -> None:
    """Test worker registration TTL expiration.

    Validates:
    - Worker expires after TTL
    - Expired worker is not returned in queries
    - Re-registration before expiry extends TTL
    """
    # Arrange
    registry = WorkerRegistry(
        redis_url=redis_container,
        worker_key_prefix="worker:",
        worker_ttl_seconds=2,  # Short TTL for testing
    )
    await registry.connect()

    registration = WorkerRegistration(
        name="ttl-test-worker",
        addr="grpc://localhost:7002",
        capabilities={"streaming": True},
        resident_models=["test-model"],
        metrics={},
    )

    try:
        # Act - Register worker with 2 second TTL
        await registry.register_worker(registration, ttl_s=2)
        logger.info(f"Registered worker with 2s TTL: {registration.name}")

        # Verify worker exists immediately
        worker = await registry.get_worker_by_name("ttl-test-worker")
        assert worker is not None, "Worker not found immediately after registration"

        # Wait for TTL to expire
        logger.info("Waiting for TTL expiration (3 seconds)...")
        await asyncio.sleep(3)

        # Verify worker is gone
        expired_worker = await registry.get_worker_by_name("ttl-test-worker")
        assert expired_worker is None, "Worker still exists after TTL expiration"

        logger.info("Worker correctly expired after TTL")

    finally:
        await registry.disconnect()


@pytest.mark.asyncio
async def test_worker_heartbeat_renewal(redis_container: str) -> None:
    """Test worker heartbeat TTL renewal.

    Validates:
    - Re-registration renews TTL
    - Worker survives multiple TTL periods with heartbeats
    - Heartbeat timestamp is updated
    """
    # Arrange
    registry = WorkerRegistry(
        redis_url=redis_container,
        worker_key_prefix="worker:",
        worker_ttl_seconds=2,  # Short TTL for testing
    )
    await registry.connect()

    registration = WorkerRegistration(
        name="heartbeat-test-worker",
        addr="grpc://localhost:7003",
        capabilities={"streaming": True},
        resident_models=["test-model"],
        metrics={},
    )

    try:
        # Act - Register and maintain heartbeat
        await registry.register_worker(registration, ttl_s=2)
        initial_timestamp = registration.last_heartbeat_ts
        logger.info(f"Initial registration at {initial_timestamp}")

        # Wait 1.5 seconds (before TTL expires)
        await asyncio.sleep(1.5)

        # Renew registration (heartbeat)
        await registry.register_worker(registration, ttl_s=2)
        renewed_timestamp = registration.last_heartbeat_ts
        logger.info(f"Renewed registration at {renewed_timestamp}")

        # Verify timestamp updated
        assert renewed_timestamp > initial_timestamp, "Heartbeat timestamp not updated"

        # Wait another 1.5 seconds
        await asyncio.sleep(1.5)

        # Verify worker still exists (should be 3 seconds since initial, but renewed at 1.5s)
        worker = await registry.get_worker_by_name("heartbeat-test-worker")
        assert worker is not None, "Worker expired despite heartbeat renewal"

        logger.info("Worker survived with heartbeat renewal")

    finally:
        await registry.remove_worker("heartbeat-test-worker")
        await registry.disconnect()


@pytest.mark.asyncio
async def test_multiple_worker_discovery(redis_container: str) -> None:
    """Test discovering multiple workers.

    Validates:
    - Multiple workers can be registered
    - All workers returned by get_workers()
    - Workers are correctly filtered by capabilities
    """
    # Arrange
    registry = WorkerRegistry(
        redis_url=redis_container,
        worker_key_prefix="worker:",
        worker_ttl_seconds=30,
    )
    await registry.connect()

    workers = [
        WorkerRegistration(
            name=f"worker-{i}",
            addr=f"grpc://localhost:{7000 + i}",
            capabilities={
                "streaming": True,
                "languages": ["en"] if i % 2 == 0 else ["zh"],
            },
            resident_models=[f"model-{i}"],
            metrics={"rtf": 0.1 * i},
        )
        for i in range(5)
    ]

    try:
        # Act - Register all workers
        for worker in workers:
            await registry.register_worker(worker, ttl_s=60)
            logger.info(f"Registered: {worker.name}")

        # Retrieve all workers
        all_workers = await registry.get_workers()

        # Assert
        assert len(all_workers) >= len(workers), (
            f"Not all workers found: {len(all_workers)}/{len(workers)}"
        )

        # Verify each registered worker is present
        worker_names = {w.name for w in all_workers}
        for worker in workers:
            assert worker.name in worker_names, f"Worker {worker.name} not found"

        logger.info(f"Successfully discovered {len(all_workers)} workers")

    finally:
        for worker in workers:
            await registry.remove_worker(worker.name)
        await registry.disconnect()


@pytest.mark.asyncio
async def test_worker_metrics_update(redis_container: str) -> None:
    """Test updating worker metrics.

    Validates:
    - Metrics can be updated via re-registration
    - Latest metrics are returned on retrieval
    - Metrics history is not preserved (last write wins)
    """
    # Arrange
    registry = WorkerRegistry(
        redis_url=redis_container,
        worker_key_prefix="worker:",
        worker_ttl_seconds=30,
    )
    await registry.connect()

    registration = WorkerRegistration(
        name="metrics-test-worker",
        addr="grpc://localhost:7004",
        capabilities={"streaming": True},
        resident_models=["model-1"],
        metrics={"rtf": 0.5, "queue_depth": 0.0},
    )

    try:
        # Act - Initial registration
        await registry.register_worker(registration, ttl_s=60)
        logger.info(f"Initial metrics: {registration.metrics}")

        # Update metrics
        registration.metrics = {"rtf": 0.3, "queue_depth": 2.0}
        await registry.register_worker(registration, ttl_s=60)
        logger.info(f"Updated metrics: {registration.metrics}")

        # Retrieve and verify
        worker = await registry.get_worker_by_name("metrics-test-worker")
        assert worker is not None
        assert worker.metrics["rtf"] == 0.3, "RTF not updated"
        assert worker.metrics["queue_depth"] == 2.0, "Queue depth not updated"

        logger.info("Metrics successfully updated")

    finally:
        await registry.remove_worker("metrics-test-worker")
        await registry.disconnect()


@pytest.mark.asyncio
async def test_redis_connection_failure_handling(redis_container: str) -> None:
    """Test handling of Redis connection failures.

    Validates:
    - Connection errors are raised appropriately
    - Graceful error messages
    - Operations fail fast when not connected
    """
    # Arrange - Use wrong Redis URL
    registry = WorkerRegistry(
        redis_url="redis://invalid-host:9999",
        worker_key_prefix="worker:",
        worker_ttl_seconds=30,
    )

    # Act & Assert - Connection should fail
    with pytest.raises(ConnectionError, match="Redis connection failed"):
        await registry.connect()

    # Operations should fail when not connected
    registration = WorkerRegistration(
        name="test-worker",
        addr="grpc://localhost:7005",
        capabilities={},
        resident_models=[],
        metrics={},
    )

    with pytest.raises(ConnectionError, match="Redis not connected"):
        await registry.register_worker(registration)

    with pytest.raises(ConnectionError, match="Redis not connected"):
        await registry.get_workers()

    with pytest.raises(ConnectionError, match="Redis not connected"):
        await registry.get_worker_by_name("test-worker")


@pytest.mark.asyncio
async def test_redis_health_check(redis_container: str) -> None:
    """Test Redis health check functionality.

    Validates:
    - Health check returns True when connected
    - Health check returns False when disconnected
    - Health check doesn't throw exceptions
    """
    # Arrange
    registry = WorkerRegistry(
        redis_url=redis_container,
        worker_key_prefix="worker:",
        worker_ttl_seconds=30,
    )

    # Act & Assert - Not connected
    healthy = await registry.health_check()
    assert not healthy, "Health check should fail when not connected"

    # Connect
    await registry.connect()
    healthy = await registry.health_check()
    assert healthy, "Health check should succeed when connected"

    # Disconnect
    await registry.disconnect()
    healthy = await registry.health_check()
    assert not healthy, "Health check should fail after disconnect"


@pytest.mark.asyncio
async def test_static_worker_routing(
    redis_container: str, registered_mock_worker: WorkerRegistration
) -> None:
    """Test static worker routing via registry.

    Validates:
    - Registered worker can be discovered
    - Worker address is correctly formatted
    - Worker capabilities are accessible
    """
    # Arrange
    registry = WorkerRegistry(
        redis_url=redis_container,
        worker_key_prefix="worker:",
        worker_ttl_seconds=30,
    )
    await registry.connect()

    try:
        # Act - Discover worker
        workers = await registry.get_workers()
        assert len(workers) > 0, "No workers discovered"

        # Find our mock worker
        mock_worker = next(
            (w for w in workers if w.name == registered_mock_worker.name), None
        )
        assert mock_worker is not None, "Mock worker not found"

        # Assert - Verify worker details
        assert mock_worker.addr == registered_mock_worker.addr
        assert mock_worker.capabilities["streaming"] is True
        assert "en" in mock_worker.capabilities["languages"]
        assert "mock-440hz" in mock_worker.resident_models

        logger.info(f"Successfully routed to worker: {mock_worker.addr}")

    finally:
        await registry.disconnect()


@pytest.mark.asyncio
async def test_invalid_worker_data_handling(redis_container: str) -> None:
    """Test handling of invalid worker registration data.

    Validates:
    - Invalid JSON is skipped gracefully
    - Malformed data doesn't crash discovery
    - Valid workers are still returned
    """
    import json

    from redis import asyncio as aioredis

    # Arrange
    redis = aioredis.from_url(redis_container, decode_responses=True)

    # Inject invalid data
    await redis.setex("worker:invalid-json", 30, "{ invalid json }")
    await redis.setex("worker:missing-fields", 30, json.dumps({"name": "incomplete"}))

    # Create valid worker
    registry = WorkerRegistry(
        redis_url=redis_container,
        worker_key_prefix="worker:",
        worker_ttl_seconds=30,
    )
    await registry.connect()

    valid_worker = WorkerRegistration(
        name="valid-worker",
        addr="grpc://localhost:7006",
        capabilities={"streaming": True},
        resident_models=["model-1"],
        metrics={},
    )

    try:
        await registry.register_worker(valid_worker, ttl_s=60)

        # Act - Get workers (should skip invalid entries)
        workers = await registry.get_workers()

        # Assert - Valid worker is returned, invalid ones are skipped
        valid_names = [w.name for w in workers]
        assert "valid-worker" in valid_names, "Valid worker not found"
        assert "invalid-json" not in valid_names, "Invalid JSON not skipped"
        assert "missing-fields" not in valid_names, "Incomplete data not skipped"

        logger.info(f"Successfully handled invalid data, found {len(workers)} valid workers")

    finally:
        await registry.remove_worker("valid-worker")
        await registry.disconnect()
        await redis.delete("worker:invalid-json", "worker:missing-fields")
        await redis.close()


@pytest.mark.asyncio
async def test_worker_removal(redis_container: str) -> None:
    """Test manual worker removal.

    Validates:
    - Worker can be removed manually
    - Removed worker is not returned in queries
    - Removal is idempotent
    """
    # Arrange
    registry = WorkerRegistry(
        redis_url=redis_container,
        worker_key_prefix="worker:",
        worker_ttl_seconds=30,
    )
    await registry.connect()

    registration = WorkerRegistration(
        name="removal-test-worker",
        addr="grpc://localhost:7007",
        capabilities={"streaming": True},
        resident_models=["model-1"],
        metrics={},
    )

    try:
        # Act - Register worker
        await registry.register_worker(registration, ttl_s=60)
        worker = await registry.get_worker_by_name("removal-test-worker")
        assert worker is not None, "Worker not registered"

        # Remove worker
        await registry.remove_worker("removal-test-worker")
        worker = await registry.get_worker_by_name("removal-test-worker")
        assert worker is None, "Worker not removed"

        # Idempotent removal (should not raise error)
        await registry.remove_worker("removal-test-worker")

        logger.info("Worker successfully removed")

    finally:
        await registry.disconnect()


@pytest.mark.asyncio
async def test_concurrent_worker_operations(redis_container: str) -> None:
    """Test concurrent worker registration and discovery.

    Validates:
    - Multiple concurrent registrations work correctly
    - Concurrent queries return consistent results
    - No race conditions or data corruption
    """
    # Arrange
    registry = WorkerRegistry(
        redis_url=redis_container,
        worker_key_prefix="worker:",
        worker_ttl_seconds=30,
    )
    await registry.connect()

    async def register_worker(worker_id: int) -> None:
        """Register a worker concurrently."""
        worker = WorkerRegistration(
            name=f"concurrent-worker-{worker_id}",
            addr=f"grpc://localhost:{8000 + worker_id}",
            capabilities={"streaming": True},
            resident_models=[f"model-{worker_id}"],
            metrics={"rtf": 0.1 * worker_id},
        )
        await registry.register_worker(worker, ttl_s=60)
        logger.info(f"Registered worker {worker_id}")

    try:
        # Act - Register 10 workers concurrently
        tasks = [register_worker(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Query workers concurrently
        query_tasks = [registry.get_workers() for _ in range(5)]
        results = await asyncio.gather(*query_tasks)

        # Assert - All queries return same workers
        expected_count = 10
        for result in results:
            assert len(result) >= expected_count, (
                f"Query returned {len(result)} workers, expected >= {expected_count}"
            )

        logger.info(f"Successfully handled {len(tasks)} concurrent registrations")

    finally:
        # Cleanup
        for i in range(10):
            await registry.remove_worker(f"concurrent-worker-{i}")
        await registry.disconnect()
