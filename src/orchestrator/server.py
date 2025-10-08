"""Orchestrator server with WebSocket transport and gRPC TTS integration.

Main server implementation that:
1. Starts WebSocket transport
2. Provides HTTP health check endpoints
3. Accepts client sessions
4. Forwards text to TTS worker via gRPC
5. Streams audio back to clients
"""

import argparse
import asyncio
import logging
from pathlib import Path

# from aiohttp import web
from aiohttp.web import Application, AppRunner, TCPSite

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

    Processes text input from the session, forwards it to the TTS worker,
    and streams audio back to the client. Handles session lifecycle and
    error cases.

    Args:
        session_manager: Manager for this session
        worker_client: TTS worker gRPC client

    Notes:
        This coroutine runs for the duration of the session, from client
        connection through to disconnect or error. It maintains the session
        state machine (LISTENING → SPEAKING) and coordinates with the worker.
    """
    session_id = session_manager.session_id

    try:
        logger.info("Starting session handler", extra={"session_id": session_id})

        # Start TTS session with worker
        model_id = "mock-440hz"  # M2: hardcoded model
        await worker_client.start_session(session_id, model_id)
        logger.info(
            "TTS session started",
            extra={"session_id": session_id, "model_id": model_id},
        )

        # Main session loop: consume text input, stream audio output
        while True:
            # Wait for text input
            logger.debug("Waiting for text input", extra={"session_id": session_id})
            text = await session_manager.transport.receive_text().__anext__()

            if text is None:
                logger.info("Client disconnected (no text)", extra={"session_id": session_id})
                break

            if not text.strip():
                logger.debug(
                    "Received empty text, skipping",
                    extra={"session_id": session_id},
                )
                continue

            logger.info(
                "Received text from client",
                extra={"session_id": session_id, "text_length": len(text)},
            )

            # Transition to SPEAKING state
            session_manager.state = SessionState.SPEAKING

            # Stream synthesis
            frame_count = 0
            async for audio_frame in worker_client.synthesize([text]):
                frame_count += 1

                # Send audio to client
                await session_manager.transport.send_audio_frame(audio_frame.audio_data)

                if audio_frame.is_final:
                    logger.debug(
                        "Final audio frame sent",
                        extra={
                            "session_id": session_id,
                            "total_frames": frame_count,
                        },
                    )
                    break

            # Return to LISTENING state
            session_manager.state = SessionState.LISTENING
            logger.info(
                "Synthesis complete",
                extra={"session_id": session_id, "frames_sent": frame_count},
            )

    except asyncio.CancelledError:
        logger.info("Session handler cancelled", extra={"session_id": session_id})
        raise
    except Exception as e:
        logger.exception(
            "Session handler error",
            extra={"session_id": session_id, "error": str(e)},
        )
        raise
    finally:
        # Clean up TTS session
        try:
            await worker_client.end_session()
            logger.info("TTS session ended", extra={"session_id": session_id})
        except Exception as e:
            logger.warning(
                "Error ending TTS session",
                extra={"session_id": session_id, "error": str(e)},
            )


async def start_server(config_path: Path) -> None:
    """Start orchestrator server with configured transports.

    Initializes all components (config, transport, worker client, health checks)
    and runs the main server loop until interrupted or error.

    Args:
        config_path: Path to YAML config file

    Raises:
        RuntimeError: If configuration is invalid
        ConnectionError: If worker connection fails
    """
    # Load config
    config = OrchestratorConfig.from_yaml(config_path)
    logger.info("Loaded configuration", extra={"config_path": str(config_path)})

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # M9: Initialize Redis worker registry (optional)
    registry = None
    if config.redis and config.redis.url:
        logger.info("Initializing worker registry", extra={"redis_url": config.redis.url})
        registry = WorkerRegistry(
            redis_url=config.redis.url,
            db=config.redis.db,
            worker_key_prefix=config.redis.worker_key_prefix,
            worker_ttl_seconds=config.redis.worker_ttl_seconds,
            connection_pool_size=config.redis.connection_pool_size,
        )
        await registry.connect()
        logger.info("Worker registry connected")

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

    # Initialize LiveKit transport if enabled
    livekit_transport = None
    logger.info("LiveKit config check", extra={
        "enabled": config.transport.livekit.enabled,
        "url": config.transport.livekit.url
    })
    if config.transport.livekit.enabled:
        try:
            from src.orchestrator.transport.livekit_transport import LiveKitTransport
            
            lk_config = config.transport.livekit
            logger.info("Initializing LiveKit transport", extra={"url": lk_config.url})
            livekit_transport = LiveKitTransport(lk_config)
            
            await livekit_transport.start()
            logger.info("LiveKit transport started", extra={"url": lk_config.url})
        except Exception as e:
            logger.error("Failed to start LiveKit transport", extra={"error": str(e)})
            logger.exception("LiveKit transport error")

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
    health_app = Application()
    setup_health_routes(health_app, worker_client, transport)

    runner = AppRunner(health_app)
    await runner.setup()
    site = TCPSite(runner, "127.0.0.1", health_port)
    await site.start()
    logger.info("Health check server started", extra={"port": health_port})

    # Main server loop: accept sessions and spawn handlers
    session_tasks = []
    try:
        logger.info("Orchestrator server ready")
        
        # Create tasks for both transports
        transport_tasks = []
        
        # WebSocket transport task
        async def websocket_loop():
            while True:
                session = await transport.accept_session()
                logger.info(
                    "New WebSocket session accepted",
                    extra={"session_id": session.session_id},
                )
                session_manager = SessionManager(session)
                task = asyncio.create_task(handle_session(session_manager, worker_client))
                session_tasks.append(task)
        
        transport_tasks.append(asyncio.create_task(websocket_loop()))
        
        # LiveKit transport task (if enabled)
        if livekit_transport:
            async def livekit_loop():
                while True:
                    session = await livekit_transport.accept_session()
                    logger.info(
                        "New LiveKit session accepted",
                        extra={"session_id": session.session_id},
                    )
                    session_manager = SessionManager(session)
                    task = asyncio.create_task(handle_session(session_manager, worker_client))
                    session_tasks.append(task)
            
            transport_tasks.append(asyncio.create_task(livekit_loop()))
        
        # Wait for any transport task to complete (they should run forever)
        await asyncio.gather(*transport_tasks)

    except asyncio.CancelledError:
        logger.info("Server loop cancelled")
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.exception("Server error", extra={"error": str(e)})
    finally:
        logger.info("Shutting down orchestrator server")

        # Stop transports
        await transport.stop()
        logger.info("WebSocket transport stopped")
        
        if livekit_transport:
            await livekit_transport.stop()
            logger.info("LiveKit transport stopped")

        # Stop health server
        await runner.cleanup()
        logger.info("Health check server stopped")

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
    parser = argparse.ArgumentParser(description="Orchestrator server")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent.parent / "configs" / "orchestrator.yaml",
        help="Path to orchestrator config YAML file",
    )
    args = parser.parse_args()

    try:
        asyncio.run(start_server(args.config))
    except KeyboardInterrupt:
        logger.info("Orchestrator server interrupted")


if __name__ == "__main__":
    main()
