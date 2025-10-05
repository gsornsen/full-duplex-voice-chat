"""Orchestrator server with WebSocket transport and gRPC TTS integration.

Main server implementation that:
1. Starts WebSocket transport
2. Provides HTTP health check endpoints
3. Accepts client sessions
4. Forwards text to TTS worker via gRPC
5. Streams audio back to clients
"""

import asyncio
import logging
from pathlib import Path

from aiohttp import web

from src.orchestrator.config import OrchestratorConfig
from src.orchestrator.grpc_client import TTSWorkerClient
from src.orchestrator.health import setup_health_routes
from src.orchestrator.registry import WorkerRegistry
from src.orchestrator.session import SessionManager, SessionState
from src.orchestrator.transport.websocket_transport import WebSocketTransport

logger = logging.getLogger(__name__)


async def handle_session(
    session_manager: SessionManager, worker_client: TTSWorkerClient
) -> None:
    """Handle a single client session.

    Args:
        session_manager: Session manager instance
        worker_client: gRPC client connected to TTS worker
    """
    session_id = session_manager.session_id

    try:
        logger.info("Session handler started", extra={"session_id": session_id})

        # Start TTS worker session
        success = await worker_client.start_session(session_id, model_id="mock-440hz")
        if not success:
            logger.error("Failed to start TTS session", extra={"session_id": session_id})
            return

        # Start audio sender loop in background
        audio_task = asyncio.create_task(session_manager.audio_sender_loop())

        # Transition to listening state
        session_manager.transition_state(SessionState.LISTENING)

        # Process text from client
        async for text in session_manager.transport.receive_text():
            logger.info(
                "Received text from client",
                extra={"session_id": session_id, "text": text[:50]},
            )

            # Transition to speaking state
            session_manager.transition_state(SessionState.SPEAKING)

            # Synthesize text to audio
            async for frame in worker_client.synthesize([text]):
                # Queue audio frame for sending
                await session_manager.queue_audio_frame(frame.audio_data)

            # Return to listening state
            session_manager.transition_state(SessionState.LISTENING)

    except Exception as e:
        logger.error(
            "Error in session handler",
            extra={"session_id": session_id, "error": str(e)},
        )
    finally:
        # Cleanup
        logger.info("Session handler ending", extra={"session_id": session_id})

        # Cancel audio sender
        if "audio_task" in locals():
            audio_task.cancel()
            try:
                await audio_task
            except asyncio.CancelledError:
                pass

        # End TTS session
        try:
            await worker_client.end_session()
        except Exception as e:
            logger.warning(
                "Error ending TTS session",
                extra={"session_id": session_id, "error": str(e)},
            )

        # Shutdown session
        await session_manager.shutdown()

        # Log metrics
        metrics = session_manager.get_metrics_summary()
        logger.info("Session completed", extra={"metrics": metrics})


async def start_health_server(
    host: str,
    port: int,
    registry: WorkerRegistry | None = None,
    worker_client: TTSWorkerClient | None = None,
) -> web.AppRunner:
    """Start HTTP server for health check endpoints.

    Args:
        host: HTTP server bind host
        port: HTTP server bind port
        registry: WorkerRegistry for Redis health checks
        worker_client: TTSWorkerClient for worker health checks

    Returns:
        AppRunner instance for graceful shutdown
    """
    app = web.Application()

    # Set up health check routes
    setup_health_routes(app, registry=registry, worker_client=worker_client)

    # Start HTTP server
    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info(f"Health check server started on http://{host}:{port}")
    return runner


async def start_server(config_path: Path | None = None) -> None:
    """Start the orchestrator server with WebSocket transport and health checks.

    Args:
        config_path: Optional path to configuration file
    """
    # Load configuration
    config = OrchestratorConfig.from_yaml_with_defaults(config_path)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting orchestrator server", extra={"config": config.model_dump()})

    # Initialize Redis registry (optional for M2, required for M9+)
    registry: WorkerRegistry | None = None
    try:
        registry = WorkerRegistry(
            redis_url=config.redis.url,
            db=config.redis.db,
            worker_key_prefix=config.redis.worker_key_prefix,
            worker_ttl_seconds=config.redis.worker_ttl_seconds,
            connection_pool_size=config.redis.connection_pool_size,
        )
        await registry.connect()
        logger.info("Connected to Redis", extra={"url": config.redis.url})
    except Exception as e:
        logger.warning(
            "Failed to connect to Redis (optional for M2)",
            extra={"error": str(e)},
        )
        registry = None

    # Initialize WebSocket transport
    ws_config = config.transport.websocket
    transport = WebSocketTransport(
        host=ws_config.host,
        port=ws_config.port,
        max_connections=ws_config.max_connections,
    )

    # Start transport
    await transport.start()
    logger.info("WebSocket transport started", extra={"port": ws_config.port})

    # Parse worker address
    worker_addr = config.routing.static_worker_addr
    if not worker_addr:
        raise RuntimeError(
            "Worker address not configured. "
            "Cause: routing.static_worker_addr is empty in orchestrator.yaml. "
            "Resolution: Set routing.static_worker_addr to your TTS worker gRPC URL "
            "(e.g., grpc://localhost:7001). "
            "See: docs/CONFIGURATION_REFERENCE.md#routing"
        )

    # Extract host:port from grpc:// URL
    if worker_addr.startswith("grpc://"):
        worker_addr = worker_addr[7:]

    logger.info("Connecting to TTS worker", extra={"address": worker_addr})

    # Create worker client (shared across sessions for now)
    worker_client = TTSWorkerClient(worker_addr)
    try:
        await worker_client.connect()
        logger.info("Connected to TTS worker", extra={"address": worker_addr})
    except Exception as e:
        raise ConnectionError(
            f"gRPC worker unavailable at {worker_addr}. "
            f"Cause: Worker process not running or network issue. "
            f"Resolution: 1) Check worker is running: docker ps | grep tts-worker. "
            f"2) Verify network connectivity: nc -zv localhost 7001. "
            f"3) Check worker logs: docker logs tts-worker-0. "
            f"See: docs/runbooks/GRPC_WORKER.md"
        ) from e

    # Start health check HTTP server (port 8081 by default)
    health_port = ws_config.port + 1  # Use next port after WebSocket
    health_runner = await start_health_server(
        host=ws_config.host,
        port=health_port,
        registry=registry,
        worker_client=worker_client,
    )

    # Session handling tasks
    session_tasks: set[asyncio.Task[None]] = set()

    try:
        # Accept client sessions
        while transport.is_running:
            try:
                # Accept next session
                transport_session = await transport.accept_session()

                logger.info(
                    "Accepted new session",
                    extra={"session_id": transport_session.session_id},
                )

                # Create session manager
                session_manager = SessionManager(transport_session)

                # Handle session in background task
                task = asyncio.create_task(
                    handle_session(session_manager, worker_client)
                )
                session_tasks.add(task)

                # Remove task when done
                task.add_done_callback(session_tasks.discard)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error accepting session", extra={"error": str(e)})

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        # Shutdown
        logger.info("Shutting down orchestrator server")

        # Stop health check server
        await health_runner.cleanup()

        # Stop transport
        await transport.stop()

        # Disconnect from worker
        await worker_client.disconnect()

        # Disconnect from Redis
        if registry is not None:
            await registry.disconnect()

        # Wait for session tasks to complete
        if session_tasks:
            logger.info(
                "Waiting for sessions to complete", extra={"count": len(session_tasks)}
            )
            await asyncio.gather(*session_tasks, return_exceptions=True)

        logger.info("Orchestrator server stopped")


def main() -> None:
    """Entry point for orchestrator server."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "orchestrator.yaml"

    try:
        asyncio.run(start_server(config_path))
    except KeyboardInterrupt:
        logger.info("Orchestrator server interrupted")


if __name__ == "__main__":
    main()
