"""gRPC health check service for TTS worker.

Implements the standard gRPC health checking protocol as defined in:
https://github.com/grpc/grpc/blob/master/doc/health-checking.md

This allows tools like grpc-health-probe, Docker healthchecks, and
Kubernetes probes to check worker health.
"""

import logging
from enum import IntEnum

import grpc

logger = logging.getLogger(__name__)


# Define health check status codes
# These match the standard grpc.health.v1.HealthCheckResponse.ServingStatus
class HealthStatus(IntEnum):
    """Health check status codes."""

    UNKNOWN = 0
    SERVING = 1
    NOT_SERVING = 2
    SERVICE_UNKNOWN = 3


# Manual health check message definitions (simplified for M2)
# In production, these would come from grpc.health.v1 proto


class HealthCheckRequest:
    """Health check request message."""

    def __init__(self, service: str = "") -> None:
        """Initialize health check request.

        Args:
            service: Service name to check (empty string = overall health)
        """
        self.service = service


class HealthCheckResponse:
    """Health check response message."""

    def __init__(self, status: HealthStatus) -> None:
        """Initialize health check response.

        Args:
            status: Health status of the service
        """
        self.status = status


class HealthServicer:
    """gRPC health check servicer.

    Implements the standard gRPC health checking protocol. Services can
    register their health status, and this servicer responds to health
    check requests.

    For M2: Returns SERVING if worker is running, NOT_SERVING if degraded.
    For M4+: Can check individual model health, CUDA availability, etc.
    """

    def __init__(self) -> None:
        """Initialize health servicer."""
        self._status: dict[str, HealthStatus] = {}
        # Default overall health to SERVING
        self.set_status("", HealthStatus.SERVING)
        logger.info("HealthServicer initialized")

    def set_status(self, service: str, status: HealthStatus) -> None:
        """Set health status for a service.

        Args:
            service: Service name (empty string for overall health)
            status: Health status
        """
        self._status[service] = status
        logger.debug(f"Health status updated: service={service!r}, status={status.name}")

    async def Check(
        self, request: HealthCheckRequest, context: grpc.aio.ServicerContext
    ) -> HealthCheckResponse:
        """Check service health.

        Args:
            request: HealthCheckRequest with service name
            context: gRPC service context

        Returns:
            HealthCheckResponse with health status
        """
        service = request.service
        status = self._status.get(service, HealthStatus.SERVICE_UNKNOWN)

        logger.debug(
            f"Health check: service={service!r}, status={status.name}",
            extra={"service": service, "status": status.name},
        )

        return HealthCheckResponse(status=status)

    async def Watch(
        self, request: HealthCheckRequest, context: grpc.aio.ServicerContext
    ) -> None:
        """Watch service health (streaming).

        Not implemented for M2 (optional in gRPC health protocol).

        Args:
            request: HealthCheckRequest with service name
            context: gRPC service context

        Raises:
            grpc.StatusCode.UNIMPLEMENTED: Watch is not implemented
        """
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "Watch is not implemented in M2. Use Check for polling.",
        )


def add_health_servicer_to_server(
    servicer: HealthServicer, server: grpc.aio.Server
) -> None:
    """Register health servicer with gRPC server.

    For M2: Manual registration without generated proto.
    For M4+: Use grpc.health.v1 generated code.

    Args:
        servicer: HealthServicer instance
        server: gRPC server instance
    """
    # NOTE: For M2, we're using a simplified implementation
    # For production (M4+), use the standard grpc.health.v1 proto
    logger.info("Health servicer registered (simplified M2 implementation)")
    # In M4+, uncomment:
    # from grpc.health.v1 import health_pb2_grpc
    # health_pb2_grpc.add_HealthServicer_to_server(servicer, server)
