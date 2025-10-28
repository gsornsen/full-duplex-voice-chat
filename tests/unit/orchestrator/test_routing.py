"""Unit tests for M9 routing module.

Tests routing logic, capability matching, load balancing, and session affinity.
"""

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestrator.registry import WorkerRegistration, WorkerRegistry
from src.orchestrator.routing import Router, RoutingStrategy


@pytest.fixture
def mock_registry() -> Any:
    """Create mock worker registry."""
    registry = MagicMock(spec=WorkerRegistry)
    registry.health_check = AsyncMock(return_value=True)
    registry.get_workers = AsyncMock(return_value=[])
    registry.get_worker_by_name = AsyncMock(return_value=None)
    return registry


@pytest.fixture
def sample_workers() -> Any:
    """Create sample worker registrations for testing."""
    current_time = time.time()

    return [
        WorkerRegistration(
            name="worker-0",
            addr="grpc://localhost:7001",
            capabilities={
                "streaming": True,
                "zero_shot": False,
                "languages": ["en"],
            },
            resident_models=["piper-en-us-lessac-medium"],
            metrics={"queue_depth": 0, "rtf": 0.3},
            last_heartbeat_ts=current_time,
        ),
        WorkerRegistration(
            name="worker-1",
            addr="grpc://localhost:7002",
            capabilities={
                "streaming": True,
                "zero_shot": True,
                "languages": ["en", "zh"],
            },
            resident_models=["cosyvoice2-en-base"],
            metrics={"queue_depth": 2, "rtf": 0.4},
            last_heartbeat_ts=current_time,
        ),
        WorkerRegistration(
            name="worker-2",
            addr="grpc://localhost:7003",
            capabilities={
                "streaming": True,
                "zero_shot": False,
                "languages": ["en"],
            },
            resident_models=["piper-en-us-amy-medium"],
            metrics={"queue_depth": 1, "rtf": 0.35},
            last_heartbeat_ts=current_time,
        ),
    ]


@pytest.fixture
async def router_with_redis(mock_registry: Any) -> Any:
    """Create router with Redis support."""
    # Mock Redis connection
    with patch("src.orchestrator.routing.aioredis") as mock_redis:
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping = AsyncMock()
        mock_redis_instance.get = AsyncMock(return_value=None)
        mock_redis_instance.setex = AsyncMock()
        mock_redis_instance.close = AsyncMock()
        mock_redis.from_url.return_value = mock_redis_instance

        router = Router(
            registry=mock_registry,
            redis_url="redis://localhost:6379",
            affinity_enabled=True,
        )

        await router.initialize()
        yield router
        await router.shutdown()


# =============================================================================
# Static Routing Tests (M2 compatibility)
# =============================================================================


@pytest.mark.asyncio
async def test_static_routing(mock_registry: Any) -> None:
    """Test static routing bypasses discovery."""
    router = Router(
        registry=mock_registry,
        static_worker_addr="grpc://static-worker:7001",
    )

    # Should return static address without querying registry
    addr = await router.select_worker()

    assert addr == "grpc://static-worker:7001"
    mock_registry.get_workers.assert_not_called()


@pytest.mark.asyncio
async def test_static_routing_ignores_session_affinity(mock_registry: Any) -> None:
    """Test static routing ignores session affinity."""
    router = Router(
        registry=mock_registry,
        static_worker_addr="grpc://static-worker:7001",
        redis_url="redis://localhost:6379",
        affinity_enabled=True,
    )

    # Should still return static address
    addr = await router.select_worker(session_id="test-session")

    assert addr == "grpc://static-worker:7001"


# =============================================================================
# Capability Filtering Tests
# =============================================================================


@pytest.mark.asyncio
async def test_capability_filter_language(mock_registry: Any, sample_workers: Any) -> None:
    """Test filtering workers by language capability."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(registry=mock_registry)

    # Request Chinese language support
    worker = await router.select_worker_dynamic(language="zh")

    # Only worker-1 supports Chinese
    assert worker.name == "worker-1"


@pytest.mark.asyncio
async def test_capability_filter_streaming(mock_registry: Any, sample_workers: Any) -> None:
    """Test filtering workers by streaming capability."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(registry=mock_registry)

    # Request streaming capability
    worker = await router.select_worker_dynamic(capabilities={"streaming": True})

    # All workers support streaming
    assert worker.name in ["worker-0", "worker-1", "worker-2"]


@pytest.mark.asyncio
async def test_capability_filter_zero_shot(mock_registry: Any, sample_workers: Any) -> None:
    """Test filtering workers by zero-shot capability."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(registry=mock_registry)

    # Request zero-shot capability
    worker = await router.select_worker_dynamic(capabilities={"zero_shot": True})

    # Only worker-1 supports zero-shot
    assert worker.name == "worker-1"


@pytest.mark.asyncio
async def test_capability_filter_multiple(mock_registry: Any, sample_workers: Any) -> None:
    """Test filtering workers by multiple capabilities."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(registry=mock_registry)

    # Request streaming + zero-shot + Chinese
    worker = await router.select_worker_dynamic(
        language="zh",
        capabilities={"streaming": True, "zero_shot": True},
    )

    # Only worker-1 matches all requirements
    assert worker.name == "worker-1"


@pytest.mark.asyncio
async def test_no_workers_match_capability(mock_registry: Any, sample_workers: Any) -> None:
    """Test error when no workers match capability."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(registry=mock_registry)

    # Request unsupported language
    with pytest.raises(RuntimeError, match="No workers support language"):
        await router.select_worker_dynamic(language="fr")


# =============================================================================
# Health Filtering Tests
# =============================================================================


@pytest.mark.asyncio
async def test_health_filter_excludes_stale_workers(
    mock_registry: Any, sample_workers: Any
) -> None:
    """Test health filter excludes workers with stale heartbeats."""
    # Make worker-1 unhealthy (heartbeat 2 minutes ago)
    sample_workers[1].last_heartbeat_ts = time.time() - 120

    mock_registry.get_workers.return_value = sample_workers

    router = Router(
        registry=mock_registry,
        health_check_interval=30,  # 30s interval, 60s threshold
    )

    # Should only select healthy workers
    worker = await router.select_worker_dynamic()

    assert worker.name in ["worker-0", "worker-2"]


@pytest.mark.asyncio
async def test_all_workers_unhealthy_error(mock_registry: Any, sample_workers: Any) -> None:
    """Test error when all workers are unhealthy."""
    # Make all workers unhealthy
    for worker in sample_workers:
        worker.last_heartbeat_ts = time.time() - 120

    mock_registry.get_workers.return_value = sample_workers

    router = Router(
        registry=mock_registry,
        health_check_interval=30,
    )

    # Should raise error
    with pytest.raises(RuntimeError, match="All workers are unhealthy"):
        await router.select_worker_dynamic()


# =============================================================================
# Load Balancing Strategy Tests
# =============================================================================


@pytest.mark.asyncio
async def test_least_loaded_strategy(mock_registry: Any, sample_workers: Any) -> None:
    """Test least-loaded strategy selects worker with lowest queue depth."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(
        registry=mock_registry,
        load_balance_strategy="least_loaded",
    )

    # worker-0 has queue_depth=0 (lowest)
    worker = await router.select_worker_dynamic()

    assert worker.name == "worker-0"
    assert worker.metrics["queue_depth"] == 0


@pytest.mark.asyncio
async def test_least_latency_strategy(mock_registry: Any, sample_workers: Any) -> None:
    """Test least-latency strategy selects worker with lowest RTF."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(
        registry=mock_registry,
        load_balance_strategy="least_latency",
    )

    # worker-0 has rtf=0.3 (lowest)
    worker = await router.select_worker_dynamic(strategy=RoutingStrategy.LEAST_LATENCY)

    assert worker.name == "worker-0"
    assert worker.metrics["rtf"] == 0.3


@pytest.mark.asyncio
async def test_round_robin_strategy(mock_registry: Any, sample_workers: Any) -> None:
    """Test round-robin strategy distributes requests evenly."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(
        registry=mock_registry,
        load_balance_strategy="round_robin",
    )

    # Make 6 requests (2 full cycles)
    selected_workers = []
    for _ in range(6):
        worker = await router.select_worker_dynamic(strategy=RoutingStrategy.ROUND_ROBIN)
        selected_workers.append(worker.name)

    # Should cycle through workers in order
    assert selected_workers == [
        "worker-0",
        "worker-1",
        "worker-2",
        "worker-0",
        "worker-1",
        "worker-2",
    ]


@pytest.mark.asyncio
async def test_random_strategy(mock_registry: Any, sample_workers: Any) -> None:
    """Test random strategy selects workers."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(
        registry=mock_registry,
        load_balance_strategy="random",
    )

    # Make 10 requests
    selected_workers = set()
    for _ in range(10):
        worker = await router.select_worker_dynamic(strategy=RoutingStrategy.RANDOM)
        selected_workers.add(worker.name)

    # Should select multiple different workers (probabilistic test)
    assert len(selected_workers) >= 2  # At least 2 different workers


# =============================================================================
# Session Affinity Tests
# =============================================================================


@pytest.mark.asyncio
async def test_session_affinity_miss_creates_mapping(
    router_with_redis: Any, mock_registry: Any, sample_workers: Any
) -> None:
    """Test session affinity creates mapping on first request."""
    mock_registry.get_workers.return_value = sample_workers

    # Mock Redis to return no existing affinity
    router_with_redis._redis.get.return_value = None

    # First request creates affinity
    worker = await router_with_redis.select_worker_dynamic(
        session_id="session-123",
        strategy=RoutingStrategy.ROUND_ROBIN,
    )

    # Should create session affinity mapping
    router_with_redis._redis.setex.assert_called_once()
    call_args = router_with_redis._redis.setex.call_args

    assert call_args[0][0] == "routing:affinity:session-123"  # key
    assert call_args[0][2] == worker.name  # value

    # Metrics should show affinity miss
    metrics = router_with_redis.get_metrics()
    assert metrics["affinity_misses"] == 1
    assert metrics["affinity_hits"] == 0


@pytest.mark.asyncio
async def test_session_affinity_hit_reuses_worker(
    router_with_redis: Any, mock_registry: Any, sample_workers: Any
) -> None:
    """Test session affinity reuses same worker."""
    mock_registry.get_workers.return_value = sample_workers

    # Mock Redis to return existing affinity
    router_with_redis._redis.get.return_value = "worker-1"

    # Request should use affinity worker
    worker = await router_with_redis.select_worker_dynamic(session_id="session-123")

    assert worker.name == "worker-1"

    # Metrics should show affinity hit
    metrics = router_with_redis.get_metrics()
    assert metrics["affinity_hits"] == 1
    assert metrics["affinity_misses"] == 0


@pytest.mark.asyncio
async def test_session_affinity_worker_unavailable(
    router_with_redis: Any, mock_registry: Any, sample_workers: Any
) -> None:
    """Test session affinity falls back when worker unavailable."""
    # Only return worker-0 and worker-2 (worker-1 is missing)
    mock_registry.get_workers.return_value = [sample_workers[0], sample_workers[2]]

    # Mock Redis to return unavailable worker
    router_with_redis._redis.get.return_value = "worker-1"

    # Should fall back to available worker
    worker = await router_with_redis.select_worker_dynamic(session_id="session-123")

    assert worker.name in ["worker-0", "worker-2"]

    # Metrics should show affinity miss
    metrics = router_with_redis.get_metrics()
    assert metrics["affinity_misses"] == 1


@pytest.mark.asyncio
async def test_session_affinity_disabled(mock_registry: Any, sample_workers: Any) -> None:
    """Test routing works with session affinity disabled."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(
        registry=mock_registry,
        affinity_enabled=False,
    )

    # Should select worker without checking affinity
    worker = await router.select_worker_dynamic(session_id="session-123")

    assert worker.name in ["worker-0", "worker-1", "worker-2"]


# =============================================================================
# Resident Model Preference Tests
# =============================================================================


@pytest.mark.asyncio
async def test_prefer_resident_model(mock_registry: Any, sample_workers: Any) -> None:
    """Test router prefers workers with resident model."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(
        registry=mock_registry,
        prefer_resident_models=True,
    )

    # Request worker with cosyvoice2-en-base model
    worker = await router.select_worker_dynamic(model_id="cosyvoice2-en-base")

    # Should select worker-1 (has model loaded)
    assert worker.name == "worker-1"
    assert "cosyvoice2-en-base" in worker.resident_models


@pytest.mark.asyncio
async def test_no_resident_model_fallback(mock_registry: Any, sample_workers: Any) -> None:
    """Test router falls back when no workers have resident model."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(
        registry=mock_registry,
        prefer_resident_models=True,
    )

    # Request model that no worker has loaded
    worker = await router.select_worker_dynamic(model_id="xtts-v2-en")

    # Should still select a worker (fallback)
    assert worker.name in ["worker-0", "worker-1", "worker-2"]


@pytest.mark.asyncio
async def test_resident_model_preference_disabled(mock_registry: Any, sample_workers: Any) -> None:
    """Test resident model preference can be disabled."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(
        registry=mock_registry,
        prefer_resident_models=False,
    )

    # Request worker with specific model
    worker = await router.select_worker_dynamic(model_id="cosyvoice2-en-base")

    # Should select based on load balancing, not residency
    assert worker.name in ["worker-0", "worker-1", "worker-2"]


# =============================================================================
# Metrics Tests
# =============================================================================


@pytest.mark.asyncio
async def test_routing_metrics_tracking(mock_registry: Any, sample_workers: Any) -> None:
    """Test routing metrics are tracked correctly."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(registry=mock_registry)

    # Make several routing decisions through select_worker()
    for _ in range(5):
        await router.select_worker()

    # Check metrics
    metrics = router.get_metrics()

    assert metrics["total_decisions"] == 5
    assert metrics["avg_latency_ms"] >= 0
    assert metrics["no_workers_errors"] == 0


@pytest.mark.asyncio
async def test_routing_latency_threshold(mock_registry: Any, sample_workers: Any) -> None:
    """Test routing decision latency meets SLA (<1ms p95)."""
    mock_registry.get_workers.return_value = sample_workers

    router = Router(registry=mock_registry)

    # Make 100 routing decisions
    for _ in range(100):
        await router.select_worker()

    # Check average latency
    metrics = router.get_metrics()

    # Average latency should be well under 1ms target
    assert metrics["avg_latency_ms"] < 1.0


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_no_workers_available(mock_registry: Any) -> None:
    """Test error when no workers are registered."""
    mock_registry.get_workers.return_value = []

    router = Router(registry=mock_registry)

    with pytest.raises(RuntimeError, match="No workers available"):
        await router.select_worker_dynamic()


@pytest.mark.asyncio
async def test_registry_connection_error(mock_registry: Any) -> None:
    """Test handling of registry connection errors."""
    mock_registry.get_workers.side_effect = ConnectionError("Redis unavailable")

    router = Router(
        registry=mock_registry,
        static_worker_addr=None,
    )

    with pytest.raises(RuntimeError, match="unavailable"):
        await router.select_worker()


@pytest.mark.asyncio
async def test_invalid_routing_strategy(mock_registry: Any) -> None:
    """Test invalid routing strategy falls back gracefully."""
    router = Router(
        registry=mock_registry,
        load_balance_strategy="invalid_strategy",
    )

    # Should fall back to LEAST_LOADED
    assert router.load_balance_strategy == RoutingStrategy.LEAST_LOADED


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_full_routing_pipeline(
    router_with_redis: Any, mock_registry: Any, sample_workers: Any
) -> None:
    """Test complete routing pipeline with all features."""
    mock_registry.get_workers.return_value = sample_workers

    # Mock Redis for session affinity
    router_with_redis._redis.get.return_value = None

    # Make routing decision with all features (use select_worker() for metrics)
    addr = await router_with_redis.select_worker(
        language="en",
        model_id="piper-en-us-lessac-medium",
        capabilities={"streaming": True},
        session_id="session-123",
        strategy=RoutingStrategy.LEAST_LOADED,
    )

    # Should select worker-0 (lowest queue, has resident model)
    assert addr == "grpc://localhost:7001"

    # Verify session affinity was created
    router_with_redis._redis.setex.assert_called_once()

    # Verify metrics
    metrics = router_with_redis.get_metrics()
    assert metrics["total_decisions"] == 1
    assert metrics["affinity_misses"] == 1


@pytest.mark.asyncio
async def test_multi_request_affinity_consistency(
    router_with_redis: Any, mock_registry: Any, sample_workers: Any
) -> None:
    """Test session affinity maintains consistency across multiple requests."""
    mock_registry.get_workers.return_value = sample_workers

    session_id = "session-456"

    # First request creates affinity
    router_with_redis._redis.get.return_value = None
    worker1 = await router_with_redis.select_worker_dynamic(session_id=session_id)

    # Second request uses affinity
    router_with_redis._redis.get.return_value = worker1.name
    worker2 = await router_with_redis.select_worker_dynamic(session_id=session_id)

    # Third request uses affinity
    router_with_redis._redis.get.return_value = worker1.name
    worker3 = await router_with_redis.select_worker_dynamic(session_id=session_id)

    # All requests should route to same worker
    assert worker1.name == worker2.name == worker3.name

    # Metrics should show 1 miss + 2 hits
    metrics = router_with_redis.get_metrics()
    assert metrics["affinity_misses"] == 1
    assert metrics["affinity_hits"] == 2
