"""Integration tests for /metrics and /metrics/summary endpoints.

Tests verify:
- Prometheus metrics endpoint functionality
- Metrics summary JSON endpoint
- Metrics format compliance
- HTTP response codes and headers
"""

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from src.orchestrator.health import setup_health_routes
from src.orchestrator.metrics import MetricsCollector, get_metrics_collector


@pytest.fixture
async def metrics_app() -> web.Application:
    """Create test app with metrics endpoints."""
    app = web.Application()
    setup_health_routes(app)
    return app


@pytest.fixture
async def client(metrics_app: web.Application) -> AsyncGenerator[Any, None]:
    """Create test client."""
    async with TestClient(TestServer(metrics_app)) as client:
        # Reset metrics singleton before each test to avoid leakage
        MetricsCollector._instance = None
        yield client


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_endpoint_exists(client: Any) -> None:
    """Test /metrics endpoint is accessible."""
    resp = await client.get("/metrics")
    assert resp.status == 200
    # Content-Type may or may not include charset depending on aiohttp version
    assert resp.content_type.startswith("text/plain")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_prometheus_format(client: Any) -> None:
    """Test /metrics returns valid Prometheus format."""
    # Record some metrics
    collector = get_metrics_collector()
    collector.record_synthesis_start("test1")
    collector.record_synthesis_complete("test1")

    resp = await client.get("/metrics")
    assert resp.status == 200

    text = await resp.text()

    # Validate Prometheus format
    assert "# HELP" in text
    assert "# TYPE" in text

    # Check for key metrics
    assert "synthesis_total" in text
    assert "synthesis_latency_seconds" in text
    assert "workers_active" in text
    assert "sessions_active" in text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_histogram_format(client: Any) -> None:
    """Test histogram metrics are properly formatted."""
    # Add histogram data
    collector = get_metrics_collector()
    collector._histograms["synthesis_latency_seconds"].observe(0.100)

    resp = await client.get("/metrics")
    text = await resp.text()

    # Check histogram format
    assert "synthesis_latency_seconds_bucket" in text
    assert "synthesis_latency_seconds_sum" in text
    assert "synthesis_latency_seconds_count" in text

    # Check bucket labels (e.g., le="0.1")
    assert 'le="0.1"' in text or 'le="0.05"' in text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_summary_endpoint(client: Any) -> None:
    """Test /metrics/summary endpoint returns JSON."""
    resp = await client.get("/metrics/summary")
    assert resp.status == 200
    # Content-Type may or may not include charset depending on aiohttp version
    assert resp.content_type.startswith("application/json")

    data = await resp.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "metrics" in data

    metrics = data["metrics"]
    assert "synthesis_total" in metrics
    assert "workers_active" in metrics
    assert "sessions_active" in metrics


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_summary_percentiles(client: Any) -> None:
    """Test /metrics/summary includes percentile calculations."""
    # Add some latency data
    collector = get_metrics_collector()
    for i in range(100):
        collector.record_synthesis_start(f"req{i}")
        collector.record_synthesis_complete(f"req{i}")

    resp = await client.get("/metrics/summary")
    data = await resp.json()

    metrics = data["metrics"]
    # Check count is as expected (metrics should be reset between tests)
    assert metrics["synthesis_total"] >= 100.0

    # Percentiles should be calculated (may be None if no data)
    # In this case we have data so they should exist
    assert "synthesis_latency_p50_ms" in metrics
    assert "synthesis_latency_p95_ms" in metrics
    assert "synthesis_latency_p99_ms" in metrics


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_endpoint_still_works(client: Any) -> None:
    """Test /health endpoint still works after metrics added."""
    resp = await client.get("/health")
    assert resp.status in (200, 503)  # May be unhealthy if Redis not available

    data = await resp.json()
    assert "status" in data
    assert "uptime_seconds" in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_liveness_endpoint_still_works(client: Any) -> None:
    """Test /liveness endpoint still works."""
    resp = await client.get("/liveness")
    assert resp.status == 200

    data = await resp.json()
    assert data["status"] == "alive"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_readiness_endpoint_still_works(client: Any) -> None:
    """Test /readiness endpoint still works."""
    resp = await client.get("/readiness")
    assert resp.status in (200, 503)

    data = await resp.json()
    assert "status" in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_thread_safety(client: Any) -> None:
    """Test metrics endpoint with concurrent requests."""
    import asyncio

    # Record metrics from multiple tasks
    async def record_metrics(task_id: int) -> None:
        collector = get_metrics_collector()
        for i in range(10):
            collector.record_synthesis_start(f"task{task_id}_req{i}")
            await asyncio.sleep(0.001)
            collector.record_synthesis_complete(f"task{task_id}_req{i}")

    # Run 5 tasks concurrently
    await asyncio.gather(*[record_metrics(i) for i in range(5)])

    # Fetch metrics (should not error)
    resp = await client.get("/metrics")
    assert resp.status == 200

    # Verify data integrity - check count is at least 50
    # (may be higher if metrics leak from other tests, but fixture should prevent that)
    text = await resp.text()
    assert "synthesis_total" in text
    # Extract the value and verify it's at least 50
    for line in text.split("\n"):
        if line.startswith("synthesis_total "):
            count = float(line.split()[1])
            assert count >= 50.0, f"Expected at least 50 synthesis requests, got {count}"
            break


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_export_performance(client: Any) -> None:
    """Test /metrics endpoint responds quickly even with lots of data."""
    import time

    # Add substantial metrics data
    collector = get_metrics_collector()
    for i in range(1000):
        collector.record_synthesis_start(f"perf{i}")
        collector.record_synthesis_complete(f"perf{i}")

    # Measure response time
    start = time.perf_counter()
    resp = await client.get("/metrics")
    duration_ms = (time.perf_counter() - start) * 1000.0

    assert resp.status == 200

    # Should respond in <100ms even with 1000 samples
    assert duration_ms < 100.0, f"Metrics export took {duration_ms:.1f}ms (target: <100ms)"
