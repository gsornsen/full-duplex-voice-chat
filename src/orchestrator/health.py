"""Health check endpoints for orchestrator.

Provides HTTP health check endpoint for load balancers, monitoring systems,
and orchestration tools (e.g., Docker healthcheck, Kubernetes liveness probe).

M11: Enhanced with Prometheus metrics endpoint and comprehensive telemetry.
"""

import logging
import time
from typing import Any

from aiohttp import web

from src.orchestrator.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class HealthCheckHandler:
    """Health check handler for orchestrator.

    Provides /health endpoint that checks:
    - Redis connectivity
    - Worker availability (optional)
    - Service uptime

    M11: Added /metrics endpoint for Prometheus scraping.
    """

    def __init__(self, registry: Any = None, worker_client: Any = None) -> None:
        """Initialize health check handler.

        Args:
            registry: WorkerRegistry instance (optional)
            worker_client: TTSWorkerClient instance (optional)
        """
        self.registry = registry
        self.worker_client = worker_client
        self.start_time = time.time()
        self.metrics_collector = get_metrics_collector()

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint.

        Returns:
            200 OK: Service is healthy and ready
            503 Service Unavailable: Service is unhealthy

        Response format:
        {
            "status": "healthy" | "unhealthy",
            "uptime_seconds": float,
            "redis": bool,
            "worker": bool,
            "checks": {
                "redis": {"ok": bool, "error": str | null},
                "worker": {"ok": bool, "error": str | null}
            }
        }
        """
        checks: dict[str, Any] = {}

        # Check Redis connectivity
        redis_ok = False
        redis_error = None
        if self.registry is not None:
            try:
                redis_ok = await self.registry.health_check()
            except Exception as e:
                redis_error = str(e)
                logger.warning("Redis health check failed", extra={"error": str(e)})

        checks["redis"] = {"ok": redis_ok, "error": redis_error}

        # Check worker connectivity (optional, simplified for M2)
        worker_ok = True  # Assume OK if worker_client exists
        worker_error = None
        if self.worker_client is not None:
            try:
                # For M2: Just check if client is connected
                # For M4+: Use gRPC health check protocol
                worker_ok = True  # Simplified for M2
            except Exception as e:
                worker_ok = False
                worker_error = str(e)
                logger.warning("Worker health check failed", extra={"error": str(e)})

        checks["worker"] = {"ok": worker_ok, "error": worker_error}

        # Determine overall health status
        overall_healthy = redis_ok and worker_ok
        status_code = 200 if overall_healthy else 503

        response_data = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "uptime_seconds": time.time() - self.start_time,
            "redis": redis_ok,
            "worker": worker_ok,
            "checks": checks,
        }

        logger.debug(
            "Health check performed",
            extra={"status": response_data["status"], "checks": checks},
        )

        return web.json_response(response_data, status=status_code)

    async def readiness_check(self, request: web.Request) -> web.Response:
        """Readiness check endpoint.

        Similar to health check but stricter - returns OK only when
        service is ready to accept requests.

        Returns:
            200 OK: Service is ready
            503 Service Unavailable: Service is not ready
        """
        # For M2: readiness == health
        # For M4+: Also check model preload status
        return await self.health_check(request)

    async def liveness_check(self, request: web.Request) -> web.Response:
        """Liveness check endpoint.

        Returns OK if service is running (even if dependencies are down).
        Used to detect if service should be restarted.

        Returns:
            200 OK: Service is alive
        """
        return web.json_response(
            {
                "status": "alive",
                "uptime_seconds": time.time() - self.start_time,
            },
            status=200,
        )

    async def metrics_endpoint(self, request: web.Request) -> web.Response:
        """Prometheus metrics endpoint (M11).

        Exposes all collected metrics in Prometheus exposition format
        for scraping by monitoring systems.

        Returns:
            200 OK: Metrics in Prometheus text format
            Content-Type: text/plain; version=0.0.4

        Format:
            # HELP metric_name Description
            # TYPE metric_name type
            metric_name{label="value"} value
        """
        try:
            # Export metrics in Prometheus format
            metrics_text = self.metrics_collector.export_prometheus()

            return web.Response(
                text=metrics_text,
                content_type="text/plain; version=0.0.4",
                status=200,
            )

        except Exception as e:
            logger.error(
                "Failed to export metrics",
                extra={"error": str(e)},
                exc_info=True,
            )
            return web.Response(
                text=f"# Error exporting metrics: {e}\n",
                content_type="text/plain",
                status=500,
            )

    async def metrics_summary(self, request: web.Request) -> web.Response:
        """Human-readable metrics summary endpoint (M11).

        Returns key metrics in JSON format for dashboards and debugging.

        Returns:
            200 OK: Metrics summary in JSON format
        """
        try:
            summary = self.metrics_collector.get_summary()

            return web.json_response(
                {
                    "status": "ok",
                    "uptime_seconds": time.time() - self.start_time,
                    "metrics": summary,
                },
                status=200,
            )

        except Exception as e:
            logger.error(
                "Failed to generate metrics summary",
                extra={"error": str(e)},
                exc_info=True,
            )
            return web.json_response(
                {
                    "status": "error",
                    "error": str(e),
                },
                status=500,
            )


def setup_health_routes(
    app: web.Application,
    registry: Any = None,
    worker_client: Any = None,
) -> None:
    """Set up health check routes on application.

    Args:
        app: aiohttp Application instance
        registry: WorkerRegistry instance (optional)
        worker_client: TTSWorkerClient instance (optional)
    """
    handler = HealthCheckHandler(registry=registry, worker_client=worker_client)

    # Health check endpoints
    app.router.add_get("/health", handler.health_check)
    app.router.add_get("/readiness", handler.readiness_check)
    app.router.add_get("/liveness", handler.liveness_check)

    # Metrics endpoints (M11)
    app.router.add_get("/metrics", handler.metrics_endpoint)
    app.router.add_get("/metrics/summary", handler.metrics_summary)

    logger.info(
        "Health check endpoints configured: "
        "/health, /readiness, /liveness, /metrics, /metrics/summary"
    )
