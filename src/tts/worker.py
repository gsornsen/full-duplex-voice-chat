"""TTS worker gRPC server implementation.

This module implements the gRPC server that hosts TTS adapters and serves
text-to-speech requests. It handles session management, streaming synthesis,
control commands, and model lifecycle operations via the ModelManager.
"""

import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import grpc
import yaml

from src.rpc.generated import tts_pb2, tts_pb2_grpc
from src.tts.model_manager import ModelManager

logger = logging.getLogger(__name__)


def validate_audio_frame(frame: tts_pb2.AudioFrame) -> bool:
    """Validate audio frame format according to protocol specification.

    Validates frame according to the rules defined in tts.proto:
    - Empty frame with is_final=False is INVALID (protocol error)
    - Empty frame with is_final=True is VALID (End Marker)
    - Non-empty frame with any is_final value is VALID

    Args:
        frame: AudioFrame to validate

    Returns:
        True if frame is valid, False if invalid

    Notes:
        Logs warning for invalid frames with diagnostic information.
        Expected frame size is 1920 bytes for 20ms @ 48kHz mono 16-bit PCM.
    """
    # Empty frame without final marker is invalid
    if len(frame.audio_data) == 0 and not frame.is_final:
        logger.warning(
            "Invalid frame: empty audio_data without is_final marker",
            extra={
                "session_id": frame.session_id,
                "sequence_number": frame.sequence_number,
            },
        )
        return False

    # Non-empty frame should have expected size (1920 bytes for 20ms@48kHz)
    if len(frame.audio_data) > 0:
        expected_size = frame.sample_rate * frame.frame_duration_ms // 1000 * 2
        if len(frame.audio_data) != expected_size:
            logger.warning(
                "Frame size mismatch (may be valid for some codecs)",
                extra={
                    "session_id": frame.session_id,
                    "sequence_number": frame.sequence_number,
                    "expected_bytes": expected_size,
                    "actual_bytes": len(frame.audio_data),
                },
            )
            # Don't fail - partial frames might be valid for some codecs

    return True


class TTSWorkerServicer(tts_pb2_grpc.TTSServiceServicer):
    """gRPC servicer for TTS worker.

    This servicer implements the TTSService interface, managing TTS sessions
    and routing requests to the appropriate adapter instances via the ModelManager.
    Each session maintains its own adapter instance for isolation and concurrent processing.

    Attributes:
        model_manager: ModelManager instance for model lifecycle
        sessions: Dictionary mapping session_id to session metadata
        adapters: Dictionary mapping session_id to adapter instances
    """

    def __init__(self, model_manager: ModelManager) -> None:
        """Initialize the TTS worker servicer.

        Args:
            model_manager: ModelManager instance for model lifecycle
        """
        self.model_manager = model_manager
        self.sessions: dict[str, dict[str, Any]] = {}
        self.adapters: dict[str, Any] = {}
        logger.info("TTSWorkerServicer initialized")

    async def StartSession(
        self, request: tts_pb2.StartSessionRequest, context: grpc.aio.ServicerContext
    ) -> tts_pb2.StartSessionResponse:
        """Start a new TTS session.

        Creates a new session with a dedicated adapter instance from the ModelManager.
        Sessions are isolated to support concurrent requests from multiple clients.

        Args:
            request: StartSessionRequest with session_id, model_id, and options
            context: gRPC service context

        Returns:
            StartSessionResponse indicating success or failure
        """
        session_id = request.session_id
        model_id = request.model_id or self.model_manager.default_model_id

        logger.info(
            "Starting session",
            extra={"session_id": session_id, "model_id": model_id},
        )

        try:
            # Load model (increments refcount, returns existing if already loaded)
            adapter = await self.model_manager.load(model_id)

            self.adapters[session_id] = adapter
            self.sessions[session_id] = {
                "model_id": model_id,
                "options": dict(request.options),
                "adapter": adapter,
            }

            logger.info(
                "Session started successfully",
                extra={"session_id": session_id, "active_sessions": len(self.sessions)},
            )

            return tts_pb2.StartSessionResponse(success=True, message="Session started")

        except Exception as e:
            logger.error(
                "Failed to start session",
                extra={"session_id": session_id, "model_id": model_id, "error": str(e)},
            )
            return tts_pb2.StartSessionResponse(
                success=False, message=f"Failed to start session: {e}"
            )

    async def EndSession(
        self, request: tts_pb2.EndSessionRequest, context: grpc.aio.ServicerContext
    ) -> tts_pb2.EndSessionResponse:
        """End a TTS session.

        Cleans up session resources and releases the model reference count.
        This should be called when the client is done with the session.

        Args:
            request: EndSessionRequest with session_id
            context: gRPC service context

        Returns:
            EndSessionResponse indicating success
        """
        session_id = request.session_id

        logger.info("Ending session", extra={"session_id": session_id})

        # Release model reference
        if session_id in self.sessions:
            model_id = self.sessions[session_id]["model_id"]
            try:
                await self.model_manager.release(model_id)
            except Exception as e:
                logger.warning(
                    "Error releasing model",
                    extra={"session_id": session_id, "model_id": model_id, "error": str(e)},
                )

        # Clean up session
        if session_id in self.adapters:
            del self.adapters[session_id]
        if session_id in self.sessions:
            del self.sessions[session_id]

        logger.info(
            "Session ended successfully",
            extra={"session_id": session_id, "active_sessions": len(self.sessions)},
        )

        return tts_pb2.EndSessionResponse(success=True)

    async def Synthesize(
        self,
        request_iterator: AsyncIterator[tts_pb2.TextChunk],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[tts_pb2.AudioFrame]:
        """Stream text chunks to audio frames.

        Main streaming synthesis endpoint. Receives text chunks from the client,
        passes them to the adapter, and streams back 20ms audio frames at 48kHz.

        Protocol Implementation:
        - Uses FINAL DATA FRAME pattern (Option A from tts.proto)
        - Marks the last data frame with is_final=True
        - Does NOT send separate empty end marker
        - All frames have non-empty audio_data

        Args:
            request_iterator: Async iterator of TextChunk messages
            context: gRPC service context

        Yields:
            AudioFrame messages with 20ms PCM audio at 48kHz

        Notes:
            - First chunk must contain session_id
            - Adapter must output 20ms frames at 48kHz
            - Last frame has is_final=True with audio data (Final Data Frame)
        """
        session_id: str | None = None
        sequence_number = 0

        try:
            # Extract session_id from first chunk
            first_chunk = await anext(request_iterator)
            session_id = first_chunk.session_id

            if not session_id:
                logger.error("Missing session_id in first chunk")
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Missing session_id")
                return

            if session_id not in self.adapters:
                logger.error("Session not found", extra={"session_id": session_id})
                await context.abort(grpc.StatusCode.NOT_FOUND, f"Session {session_id} not found")
                return

            adapter = self.adapters[session_id]

            logger.info(
                "Starting synthesis stream",
                extra={"session_id": session_id},
            )

            async def text_generator() -> AsyncIterator[str]:
                """Generate text from request chunks.

                Yields:
                    Text strings from TextChunk messages
                """
                # Yield first chunk text
                yield first_chunk.text

                # Yield remaining chunks
                async for chunk in request_iterator:
                    yield chunk.text

            # Collect all frames to identify the last one (Final Data Frame pattern)
            frames_list: list[bytes] = []
            async for audio_data in adapter.synthesize_stream(text_generator()):
                frames_list.append(audio_data)

            # Yield frames, marking the last one as final (Option A: Final Data Frame)
            frame_count = 0
            for idx, audio_data in enumerate(frames_list):
                sequence_number += 1
                frame_count += 1
                is_last = idx == len(frames_list) - 1

                frame = tts_pb2.AudioFrame(
                    session_id=session_id,
                    audio_data=audio_data,
                    sample_rate=48000,
                    frame_duration_ms=20,
                    sequence_number=sequence_number,
                    is_final=is_last,  # Mark last frame as final
                )

                # Validate frame before sending
                if not validate_audio_frame(frame):
                    logger.error(
                        "Generated invalid frame, skipping",
                        extra={"session_id": session_id, "seq": sequence_number},
                    )
                    continue

                yield frame

            logger.info(
                "Synthesis stream completed",
                extra={"session_id": session_id, "frames_sent": frame_count},
            )

        except StopAsyncIteration:
            # Normal end of stream
            logger.info(
                "Client closed stream",
                extra={"session_id": session_id, "frames_sent": sequence_number},
            )
        except Exception as e:
            logger.exception(
                "Error in synthesis stream",
                extra={"session_id": session_id, "error": str(e)},
            )
            raise

    async def Control(
        self, request: tts_pb2.ControlRequest, context: grpc.aio.ServicerContext
    ) -> tts_pb2.ControlResponse:
        """Handle control commands.

        Processes runtime control commands for a session. Supports PAUSE, RESUME,
        STOP, and RELOAD commands with < 50ms response time.

        Args:
            request: ControlRequest with session_id and command
            context: gRPC service context

        Returns:
            ControlResponse with success status and timestamp

        Notes:
            - Commands must respond within 50ms for barge-in SLA
            - Unknown commands return success=False
        """
        session_id = request.session_id
        command_enum = request.command

        # Convert enum to string
        command_map = {
            tts_pb2.PAUSE: "PAUSE",
            tts_pb2.RESUME: "RESUME",
            tts_pb2.STOP: "STOP",
            tts_pb2.RELOAD: "RELOAD",
        }
        command = command_map.get(command_enum, "UNKNOWN")

        logger.info(
            "Control command received",
            extra={"session_id": session_id, "command": command},
        )

        if command == "UNKNOWN":
            logger.error(
                "Unknown control command",
                extra={"session_id": session_id, "command_enum": command_enum},
            )
            return tts_pb2.ControlResponse(
                success=False,
                message=f"Unknown command: {command_enum}",
                timestamp_ms=int(time.time() * 1000),
            )

        # Forward to adapter
        if session_id in self.adapters:
            adapter = self.adapters[session_id]
            try:
                await adapter.control(command)
                logger.info(
                    "Control command executed",
                    extra={"session_id": session_id, "command": command},
                )
            except ValueError as e:
                logger.error(
                    "Control command failed",
                    extra={"session_id": session_id, "command": command, "error": str(e)},
                )
                return tts_pb2.ControlResponse(
                    success=False,
                    message=str(e),
                    timestamp_ms=int(time.time() * 1000),
                )
        else:
            logger.warning(
                "Session not found for control command",
                extra={"session_id": session_id, "command": command},
            )

        return tts_pb2.ControlResponse(
            success=True,
            message=f"Command {command} executed",
            timestamp_ms=int(time.time() * 1000),
        )

    async def ListModels(
        self, request: tts_pb2.ListModelsRequest, context: grpc.aio.ServicerContext
    ) -> tts_pb2.ListModelsResponse:
        """List available models.

        Returns a list of all loaded models with their metadata.

        Args:
            request: ListModelsRequest (empty)
            context: gRPC service context

        Returns:
            ListModelsResponse with available models
        """
        logger.debug("ListModels called")

        try:
            loaded_models = await self.model_manager.list_models()
            model_info = await self.model_manager.get_model_info()

            models = []
            for model_id in loaded_models:
                info = model_info.get(model_id, {})
                model = tts_pb2.ModelInfo(
                    model_id=model_id,
                    family="mock",  # M4: all models are mock, M5+ will have real families
                    is_loaded=True,
                    languages=["en"],
                    metadata={
                        "refcount": str(info.get("refcount", 0)),
                        "idle_seconds": str(info.get("idle_seconds", 0)),
                    },
                )
                models.append(model)

            return tts_pb2.ListModelsResponse(models=models)

        except Exception as e:
            logger.error("Error listing models", extra={"error": str(e)})
            return tts_pb2.ListModelsResponse(models=[])

    async def LoadModel(
        self, request: tts_pb2.LoadModelRequest, context: grpc.aio.ServicerContext
    ) -> tts_pb2.LoadModelResponse:
        """Load a model.

        Dynamically loads a TTS model via the ModelManager.

        Args:
            request: LoadModelRequest with model_id
            context: gRPC service context

        Returns:
            LoadModelResponse with success status and load duration
        """
        model_id = request.model_id
        logger.info("LoadModel called", extra={"model_id": model_id})

        try:
            start_time = time.time()
            await self.model_manager.load(model_id)
            load_duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "Model loaded successfully",
                extra={"model_id": model_id, "load_duration_ms": load_duration_ms},
            )

            return tts_pb2.LoadModelResponse(
                success=True,
                message="Model loaded",
                load_duration_ms=load_duration_ms,
            )

        except Exception as e:
            logger.error(
                "Failed to load model",
                extra={"model_id": model_id, "error": str(e)},
            )
            return tts_pb2.LoadModelResponse(
                success=False,
                message=f"Failed to load model: {e}",
                load_duration_ms=0,
            )

    async def UnloadModel(
        self, request: tts_pb2.UnloadModelRequest, context: grpc.aio.ServicerContext
    ) -> tts_pb2.UnloadModelResponse:
        """Unload a model.

        Unloads a TTS model to free resources. The model will only be unloaded
        if it's not currently in use (refcount == 0).

        Args:
            request: UnloadModelRequest with model_id
            context: gRPC service context

        Returns:
            UnloadModelResponse with success status
        """
        model_id = request.model_id
        logger.info("UnloadModel called", extra={"model_id": model_id})

        try:
            # Release model reference (this doesn't force unload, just decrements refcount)
            # For force unload, we'd need to check refcount and call _unload_model directly
            # For M4, we'll implement a simple release that respects refcounting
            await self.model_manager.release(model_id)

            logger.info("Model unload initiated", extra={"model_id": model_id})

            return tts_pb2.UnloadModelResponse(
                success=True,
                message="Model unload initiated (will unload when idle)",
            )

        except Exception as e:
            logger.error(
                "Failed to unload model",
                extra={"model_id": model_id, "error": str(e)},
            )
            return tts_pb2.UnloadModelResponse(
                success=False,
                message=f"Failed to unload model: {e}",
            )

    async def GetCapabilities(
        self, request: tts_pb2.GetCapabilitiesRequest, context: grpc.aio.ServicerContext
    ) -> tts_pb2.GetCapabilitiesResponse:
        """Get worker capabilities.

        Returns the capabilities of this worker, including supported features,
        resident models, and performance metrics.

        Args:
            request: GetCapabilitiesRequest (empty)
            context: gRPC service context

        Returns:
            GetCapabilitiesResponse with capabilities, models, and metrics
        """
        logger.debug("GetCapabilities called")

        try:
            loaded_models = await self.model_manager.list_models()

            capabilities = tts_pb2.Capabilities(
                streaming=True,
                zero_shot=False,  # M5 baseline: no zero-shot support yet
                lora=False,
                cpu_ok=True,
                languages=["en"],
                emotive_zero_prompt=False,
                max_concurrent_sessions=10,
            )

            return tts_pb2.GetCapabilitiesResponse(
                capabilities=capabilities,
                resident_models=loaded_models,
                metrics={"rtf": 0.1, "queue_depth": float(len(self.sessions))},
            )

        except Exception as e:
            logger.error("Error getting capabilities", extra={"error": str(e)})
            # Return minimal capabilities on error
            capabilities = tts_pb2.Capabilities(
                streaming=True,
                zero_shot=False,
                lora=False,
                cpu_ok=True,
                languages=["en"],
                emotive_zero_prompt=False,
                max_concurrent_sessions=10,
            )
            return tts_pb2.GetCapabilitiesResponse(
                capabilities=capabilities,
                resident_models=[],
                metrics={},
            )


def load_config(config_path: str = "configs/worker.yaml") -> dict[str, Any]:
    """Load worker configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        result = yaml.safe_load(f)
        assert isinstance(result, dict)
        return result


async def start_worker(config: dict[str, Any]) -> None:
    """Start TTS worker gRPC server.

    Initializes and starts the async gRPC server on the configured port.
    The server runs until interrupted (Ctrl+C) or terminated.

    Args:
        config: Worker configuration dictionary with 'port' key

    Notes:
        - Uses grpc.aio for async server support
        - Listens on all interfaces ([::]:{port})
        - Graceful shutdown with 5 second timeout
    """
    # Extract configuration
    # Support both legacy flat structure and new nested structure
    if "worker" in config:
        # New nested structure
        worker_config = config["worker"]
        port = worker_config.get("grpc_port", 7001)
    else:
        # Legacy flat structure (from __main__.py)
        port = config.get("port", 7001)

    mm_config = config.get("model_manager", {})

    # Resolve default model ID with environment variable fallback
    default_model_id = mm_config.get("default_model_id", "mock-440hz")
    model_source = mm_config.get("default_model_source", "config")

    # Environment variable fallback (if not already set via CLI or __main__.py)
    if model_source == "config" and (env_model := os.getenv("DEFAULT_MODEL")):
        default_model_id = env_model
        model_source = "env"

    # Create ModelManager
    model_manager = ModelManager(
        default_model_id=default_model_id,
        preload_model_ids=mm_config.get("preload_model_ids", []),
        ttl_ms=mm_config.get("ttl_ms", 600000),
        min_residency_ms=mm_config.get("min_residency_ms", 120000),
        resident_cap=mm_config.get("resident_cap", 3),
        max_parallel_loads=mm_config.get("max_parallel_loads", 1),
        warmup_enabled=mm_config.get("warmup_enabled", True),
        warmup_text=mm_config.get("warmup_text", "This is a warmup test."),
        evict_check_interval_ms=mm_config.get("evict_check_interval_ms", 30000),
    )

    # Log model configuration source
    logger.info(
        "ModelManager configured",
        extra={
            "default_model_id": default_model_id,
            "model_source": model_source,
            "preload_model_ids": mm_config.get("preload_model_ids", []),
        },
    )

    # Initialize ModelManager (load default/preload models, start eviction)
    try:
        await model_manager.initialize()
    except Exception as e:
        logger.error("Failed to initialize ModelManager", extra={"error": str(e)})
        raise

    # Create async gRPC server
    server = grpc.aio.server()
    servicer = TTSWorkerServicer(model_manager)
    tts_pb2_grpc.add_TTSServiceServicer_to_server(servicer, server)  # type: ignore[no-untyped-call]

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    logger.info("Starting TTS worker", extra={"address": listen_addr})
    await server.start()
    logger.info("TTS worker started successfully", extra={"address": listen_addr})

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down")
    finally:
        logger.info("Stopping TTS worker")
        await server.stop(grace=5.0)
        await model_manager.shutdown()
        logger.info("TTS worker stopped")
