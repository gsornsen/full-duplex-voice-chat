"""Orchestrator server with WebSocket transport and gRPC TTS integration.

Main server implementation that:
1. Starts WebSocket transport
2. Provides HTTP health check endpoints
3. Accepts client sessions
4. Forwards text to TTS worker via gRPC
5. Streams audio back to clients
6. Processes incoming audio through VAD for barge-in detection (M3+)
7. Transcribes speech via ASR and forwards to TTS (M10+)
"""

import argparse
import asyncio
import logging
import time
from pathlib import Path

import grpc
from aiohttp.web import Application, AppRunner, TCPSite

from src.asr.adapters import WhisperAdapter, WhisperXAdapter
from src.orchestrator.audio.buffer import AudioBuffer
from src.orchestrator.audio.resampler import AudioResampler
from src.orchestrator.config import OrchestratorConfig
from src.orchestrator.grpc_client import TTSWorkerClient
from src.orchestrator.health import setup_health_routes
from src.orchestrator.registry import WorkerRegistry
from src.orchestrator.session import SessionManager, SessionState
from src.orchestrator.transport.websocket_transport import WebSocketTransport
from src.orchestrator.vad_processor import VADAudioProcessor
from src.rpc.generated import tts_pb2

logger = logging.getLogger(__name__)


def _validate_audio_frame(frame: tts_pb2.AudioFrame) -> bool:
    """Validate audio frame from worker.

    Validates frame according to protocol specification in tts.proto:
    - Empty frame with is_final=False is INVALID (protocol error)
    - Empty frame with is_final=True is VALID (End Marker)
    - Non-empty frame with any is_final value is VALID

    Args:
        frame: AudioFrame to validate

    Returns:
        True if frame is valid, False if invalid
    """
    # Empty frame without final marker is invalid
    if len(frame.audio_data) == 0 and not frame.is_final:
        return False

    # All other cases are valid (will be handled appropriately)
    return True


class OrchestratorServer:
    """Orchestrator server with ASR integration.

    Manages ASR adapter lifecycle and per-session audio buffers for
    speech-to-text transcription pipeline.

    Thread-safety: This class is NOT thread-safe. Use from a single async task.
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        """Initialize orchestrator server.

        Args:
            config: Server configuration
        """
        self.config = config

        # ASR adapter (shared across sessions)
        self.asr_adapter: WhisperAdapter | WhisperXAdapter | None = None

        # Audio buffers per session (for ASR)
        self.audio_buffers: dict[str, AudioBuffer] = {}

        # Audio resamplers per session (48kHz → 16kHz for Whisper)
        self.audio_resamplers: dict[str, AudioResampler] = {}

        logger.info(
            f"Orchestrator server initialized: ASR enabled={config.asr.enabled}"
        )

    async def initialize(self) -> None:
        """Initialize server components including ASR adapter.

        Raises:
            RuntimeError: If ASR initialization fails
        """
        # Initialize ASR adapter if enabled
        if self.config.asr.enabled:
            logger.info("Initializing ASR adapter...")

            try:
                # Select adapter based on configuration
                adapter_type = self.config.asr.adapter.lower()

                if adapter_type == "whisper":
                    logger.info("Creating WhisperAdapter (standard faster-whisper)")
                    self.asr_adapter = WhisperAdapter(
                        model_size=self.config.asr.model_size,
                        device=self.config.asr.device,
                        language=self.config.asr.language,
                        compute_type=self.config.asr.compute_type,
                        model_path=self.config.asr.model_path,
                    )
                elif adapter_type == "whisperx":
                    logger.info("Creating WhisperXAdapter (4x faster CTranslate2)")
                    self.asr_adapter = WhisperXAdapter(
                        model_size=self.config.asr.model_size,
                        device=self.config.asr.device,
                        language=self.config.asr.language,
                        compute_type=self.config.asr.compute_type,
                        model_path=self.config.asr.model_path,
                    )
                else:
                    raise ValueError(
                        f"Unsupported ASR adapter: '{adapter_type}'. "
                        f"Supported adapters: whisper, whisperx"
                    )

                await self.asr_adapter.initialize()
                logger.info("ASR adapter initialized and ready")

                # Log model info
                model_info = self.asr_adapter.get_model_info()
                logger.info(f"ASR model info: {model_info}")

            except Exception as e:
                logger.error(f"Failed to initialize ASR adapter: {e}")
                raise RuntimeError(f"ASR initialization failed: {e}") from e
        else:
            logger.info("ASR disabled, using text-only mode")

    async def shutdown(self) -> None:
        """Shutdown server and cleanup resources."""
        logger.info("Shutting down orchestrator server...")

        # Shutdown ASR adapter
        if self.asr_adapter:
            logger.info("Shutting down ASR adapter...")
            try:
                await self.asr_adapter.shutdown()
                logger.info("ASR adapter shutdown complete")
            except Exception as e:
                logger.warning(f"Error shutting down ASR adapter: {e}")

        # Clear audio buffers
        for session_id in list(self.audio_buffers.keys()):
            await self._cleanup_session(session_id)

        logger.info("Orchestrator server shutdown complete")

    async def create_session_resources(self, session_id: str) -> None:
        """Create per-session resources (audio buffer, resampler).

        Args:
            session_id: Session identifier
        """
        if not self.config.asr.enabled or not self.asr_adapter:
            return

        # Create audio buffer for this session
        self.audio_buffers[session_id] = AudioBuffer(
            sample_rate=16000,  # Whisper requires 16kHz
            channels=1,
            max_duration_s=self.config.asr.buffer_max_duration_s,
        )

        # Create audio resampler (48kHz → 16kHz)
        self.audio_resamplers[session_id] = AudioResampler(
            source_rate=48000,
            target_rate=16000,
        )

        logger.info(
            f"Created ASR resources for session {session_id}: "
            f"buffer max_duration={self.config.asr.buffer_max_duration_s}s"
        )

    async def _cleanup_session(self, session_id: str) -> None:
        """Cleanup session resources.

        Args:
            session_id: Session identifier
        """
        # Clear audio buffer
        if session_id in self.audio_buffers:
            await self.audio_buffers[session_id].clear()
            del self.audio_buffers[session_id]

        # Remove resampler
        if session_id in self.audio_resamplers:
            del self.audio_resamplers[session_id]

        logger.debug(f"Cleaned up ASR resources for session {session_id}")

    async def on_vad_speech_start(self, session_id: str) -> None:
        """Handle VAD speech start event.

        Args:
            session_id: Session identifier
        """
        logger.debug(f"Speech started for session {session_id}")

        # Clear audio buffer for new utterance
        if session_id in self.audio_buffers:
            await self.audio_buffers[session_id].clear()

    async def on_vad_audio_frame(self, session_id: str, audio_frame_48khz: bytes) -> None:
        """Handle incoming audio frame during speech.

        Args:
            session_id: Session identifier
            audio_frame_48khz: Audio frame at 48kHz (1920 bytes for 20ms)
        """
        # Buffer audio if ASR enabled and buffer exists
        if session_id not in self.audio_buffers or session_id not in self.audio_resamplers:
            return

        try:
            # Resample 48kHz → 16kHz
            resampler = self.audio_resamplers[session_id]
            audio_frame_16khz = resampler.process_frame(audio_frame_48khz)

            # Append to buffer
            await self.audio_buffers[session_id].append(audio_frame_16khz)

        except Exception as e:
            logger.error(f"Failed to buffer audio for session {session_id}: {e}")

    async def on_vad_speech_end(
        self,
        session_id: str,
        session_manager: SessionManager,
        worker_client: TTSWorkerClient,
    ) -> None:
        """Handle VAD speech end event - trigger ASR transcription.

        Args:
            session_id: Session identifier
            session_manager: Session manager for sending responses
            worker_client: TTS worker client for synthesis
        """
        logger.critical(
            "[ASR PIPELINE] on_vad_speech_end called",
            extra={
                "session_id": session_id,
                "asr_adapter_available": self.asr_adapter is not None,
                "session_in_audio_buffers": session_id in self.audio_buffers,
            }
        )

        # Transcribe buffered audio if ASR enabled
        if self.asr_adapter and session_id in self.audio_buffers:
            logger.critical(
                "[ASR PIPELINE] Calling _transcribe_and_synthesize",
                extra={"session_id": session_id}
            )
            await self._transcribe_and_synthesize(session_id, session_manager, worker_client)
        else:
            logger.critical(
                "[ASR PIPELINE] NOT transcribing - conditions not met",
                extra={
                    "session_id": session_id,
                    "asr_adapter": self.asr_adapter is not None,
                    "in_buffers": session_id in self.audio_buffers,
                }
            )

    async def _transcribe_and_synthesize(
        self,
        session_id: str,
        session_manager: SessionManager,
        worker_client: TTSWorkerClient,
    ) -> None:
        """Transcribe buffered audio and send to TTS.

        Args:
            session_id: Session identifier
            session_manager: Session manager for state and transport
            worker_client: TTS worker client for synthesis
        """
        try:
            logger.critical(
                "[ASR PIPELINE] _transcribe_and_synthesize entered",
                extra={"session_id": session_id}
            )

            # Get buffered audio
            audio_buffer = self.audio_buffers.get(session_id)
            if not audio_buffer or await audio_buffer.is_empty():
                logger.critical(
                    "[ASR PIPELINE] No audio to transcribe - buffer empty or missing",
                    extra={
                        "session_id": session_id,
                        "buffer_exists": audio_buffer is not None,
                    }
                )
                return

            audio_bytes = await audio_buffer.get_audio()
            duration_ms = audio_buffer.duration_ms()

            logger.critical(
                "[ASR PIPELINE] Starting transcription",
                extra={
                    "session_id": session_id,
                    "audio_duration_ms": duration_ms,
                    "audio_bytes": len(audio_bytes),
                }
            )

            # Transcribe audio (audio is already at 16kHz from buffering)
            if not self.asr_adapter:
                logger.critical("[ASR PIPELINE] ASR adapter not available!")
                return

            logger.critical("[ASR PIPELINE] Calling asr_adapter.transcribe()")
            result = await self.asr_adapter.transcribe(audio_bytes, sample_rate=16000)
            logger.critical(
                "[ASR PIPELINE] Transcription completed",
                extra={
                    "session_id": session_id,
                    "text": result.text[:100] + "..." if len(result.text) > 100 else result.text,
                    "confidence": result.confidence,
                }
            )

            if not result.text.strip():
                logger.warning(
                    f"Transcription returned empty text "
                    f"(confidence={result.confidence:.2f})"
                )
                # Optionally send message to user
                await self._send_text_to_client(
                    session_manager, "I didn't catch that. Could you please repeat?"
                )
                return

            logger.info(
                f"Transcription result: '{result.text}' "
                f"(confidence={result.confidence:.2f}, "
                f"language={result.language})"
            )

            # Send transcribed text to TTS worker (reuse existing synthesis path)
            await self._handle_text_synthesis(
                session_id, result.text, session_manager, worker_client
            )

        except Exception as e:
            logger.error(
                f"Transcription failed for session {session_id}: {e}",
                exc_info=True,
            )
            # Fall back to text mode
            await self._send_text_to_client(
                session_manager,
                "Speech recognition is temporarily unavailable. Please type your message.",
            )

    async def _handle_text_synthesis(
        self,
        session_id: str,
        text: str,
        session_manager: SessionManager,
        worker_client: TTSWorkerClient,
    ) -> None:
        """Handle text input and send to TTS for synthesis.

        Args:
            session_id: Session identifier
            text: Text to synthesize
            session_manager: Session manager
            worker_client: TTS worker client
        """
        # This method synthesizes text through the TTS worker
        # The actual synthesis happens in handle_session() main loop
        # For now, we'll implement a simple direct synthesis

        logger.info(f"Synthesizing text for session {session_id}: '{text[:50]}...'")

        # Transition to SPEAKING state
        session_manager.transition_state(SessionState.SPEAKING)

        try:
            # Stream synthesis
            frame_count = 0
            async for audio_frame in worker_client.synthesize([text]):
                # Check if we were interrupted by barge-in
                if session_manager.state == SessionState.BARGED_IN:
                    logger.info(
                        "Synthesis interrupted by barge-in, stopping stream",
                        extra={"session_id": session_id, "frames_sent": frame_count},
                    )
                    await worker_client.control("STOP")
                    break

                # Validate frame
                if not _validate_audio_frame(audio_frame):
                    logger.warning(
                        "Skipping invalid frame (empty without is_final)",
                        extra={
                            "session_id": session_id,
                            "sequence_number": audio_frame.sequence_number,
                        },
                    )
                    continue

                # Skip empty frames (End Markers)
                if len(audio_frame.audio_data) == 0:
                    if audio_frame.is_final:
                        logger.debug(
                            "Received end marker (empty final frame), completing synthesis",
                            extra={"session_id": session_id},
                        )
                        break
                    else:
                        continue

                # Send non-empty audio to client
                frame_count += 1
                await session_manager.transport.send_audio_frame(audio_frame.audio_data)

                # Check if this is the final data frame
                if audio_frame.is_final:
                    logger.debug(
                        "Received final data frame, completing synthesis",
                        extra={
                            "session_id": session_id,
                            "total_frames": frame_count,
                        },
                    )
                    break

            # Return to LISTENING state if not interrupted
            if session_manager.state != SessionState.BARGED_IN:
                session_manager.transition_state(SessionState.LISTENING)
                logger.info(
                    "Synthesis complete, ready for next message",
                    extra={"session_id": session_id, "frames_sent": frame_count},
                )

        except grpc.RpcError as e:
            logger.error(
                "gRPC error during synthesis",
                extra={"session_id": session_id, "error": str(e)},
            )
            session_manager.transition_state(SessionState.LISTENING)
        except Exception as e:
            logger.exception(
                "Unexpected error during synthesis",
                extra={"session_id": session_id, "error": str(e)},
            )
            session_manager.transition_state(SessionState.LISTENING)

    async def _send_text_to_client(
        self,
        session_manager: SessionManager,
        message: str,
    ) -> None:
        """Send text message to client (for fallback/error messages).

        Args:
            session_manager: Session manager
            message: Text message to send
        """
        # TODO: Implement text message sending via transport
        # For now, just log it
        logger.info(f"Message to client: {message}")


async def handle_session(
    session_manager: SessionManager,
    worker_client: TTSWorkerClient,
    config: OrchestratorConfig,
    orchestrator: OrchestratorServer,
) -> None:
    """Handle a single client session with VAD-based barge-in and ASR.

    Processes text input from the session, forwards it to the TTS worker,
    and streams audio back to the client. Handles session lifecycle,
    error cases, VAD-based barge-in interruption, and ASR transcription.

    Protocol Handling:
    - Supports both Final Data Frame and End Marker patterns
    - Skips empty frames (End Markers) without forwarding to client
    - Detects is_final=True to complete synthesis loop
    - Returns to LISTENING state after each synthesis for next message
    - Supports multiple messages per session

    Barge-in Handling (M3+):
    - Processes incoming audio through VAD
    - Detects speech_start → sends PAUSE to worker → transitions to BARGED_IN
    - Detects speech_end → sends RESUME to worker → transitions to LISTENING
    - Tracks barge-in latency metrics

    ASR Handling (M10+):
    - Buffers audio during VAD speech detection
    - Transcribes on speech_end via Whisper
    - Forwards transcribed text to TTS worker
    - Falls back to text mode on errors

    Args:
        session_manager: Manager for this session
        worker_client: TTS worker gRPC client
        config: Orchestrator configuration
        orchestrator: Orchestrator server instance for ASR integration

    Notes:
        This coroutine runs for the duration of the session, from client
        connection through to disconnect or error. It maintains the session
        state machine (LISTENING → SPEAKING → BARGED_IN → LISTENING) and
        coordinates with the worker.
    """
    session_id = session_manager.session_id

    # Session lifecycle tracking (M10+)
    session_start_time = time.monotonic()
    message_count = 0

    # Create session resources (audio buffer, resampler)
    await orchestrator.create_session_resources(session_id)

    # Initialize VAD processor if enabled
    vad_processor: VADAudioProcessor | None = None
    if config.vad.enabled:

        def on_speech_start(timestamp_ms: float) -> None:
            """Handle VAD speech detection (barge-in trigger or ASR start)."""
            # If ASR enabled, start buffering audio
            if config.asr.enabled:
                asyncio.create_task(orchestrator.on_vad_speech_start(session_id))

            # Only trigger barge-in if we're currently speaking
            if session_manager.state == SessionState.SPEAKING:
                logger.info(
                    "Barge-in detected (speech start)",
                    extra={
                        "session_id": session_id,
                        "vad_timestamp_ms": timestamp_ms,
                    },
                )

                # Record barge-in start time for latency tracking
                asyncio.create_task(_handle_barge_in_pause(session_manager, worker_client))

        def on_speech_end(timestamp_ms: float) -> None:
            """Handle VAD silence detection (resume trigger or ASR transcribe)."""
            # CRITICAL LOGGING: Track speech end events for debugging
            logger.critical(
                "[ASR TRIGGER] Speech ended, firing on_speech_end handler",
                extra={
                    "session_id": session_id,
                    "timestamp_ms": timestamp_ms,
                    "asr_enabled": config.asr.enabled,
                    "session_state": session_manager.state.name,
                    "audio_buffer_size": (
                        session_manager.audio_buffer.qsize()
                        if hasattr(session_manager, "audio_buffer")
                        else 0
                    ),
                },
            )

            # If ASR enabled, transcribe buffered audio
            if config.asr.enabled:
                logger.critical(
                    "[ASR TRIGGER] Triggering ASR transcription task",
                    extra={"session_id": session_id}
                )
                asyncio.create_task(
                    orchestrator.on_vad_speech_end(session_id, session_manager, worker_client)
                )

            # Only resume if we're in BARGED_IN state
            if session_manager.state == SessionState.BARGED_IN:
                logger.info(
                    "Silence detected after barge-in (speech end)",
                    extra={
                        "session_id": session_id,
                        "vad_timestamp_ms": timestamp_ms,
                    },
                )

                # Note: We transition back to LISTENING, not RESUME synthesis
                # This allows the user to provide new input
                session_manager.transition_state(SessionState.LISTENING)

        vad_processor = VADAudioProcessor(
            config=config.vad,
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
        )

        logger.info(
            "VAD processor initialized for session",
            extra={
                "session_id": session_id,
                "enabled": vad_processor.is_enabled,
                "asr_enabled": config.asr.enabled,
            },
        )

    # Set up audio frame callback for LiveKit sessions (to receive participant audio)
    audio_processor_task = None
    try:
        from src.orchestrator.transport.livekit_transport import LiveKitSession

        if isinstance(session_manager.transport, LiveKitSession) and vad_processor:
            logger.info(
                "Setting up audio frame callback for LiveKit session",
                extra={"session_id": session_id},
            )

            # Create non-blocking audio queue (drops frames if full)
            audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=10)
            frames_received = 0
            frames_dropped = 0
            last_log_time = time.monotonic()

            def on_audio_frame(audio_data: bytes) -> None:
                """Fast non-blocking callback - just queue frames."""
                nonlocal frames_received, frames_dropped, last_log_time

                try:
                    # Non-blocking queue put (O(1), no I/O)
                    audio_queue.put_nowait(audio_data)
                    frames_received += 1
                except asyncio.QueueFull:
                    # Drop frame rather than blocking (graceful degradation)
                    frames_dropped += 1

                # Log stats every 10 seconds (not every frame!)
                now = time.monotonic()
                if now - last_log_time > 10.0:
                    logger.info(
                        "Audio callback stats",
                        extra={
                            "session_id": session_id,
                            "frames_received": frames_received,
                            "frames_dropped": frames_dropped,
                            "queue_size": audio_queue.qsize(),
                        },
                    )
                    last_log_time = now

            async def process_audio_queue() -> None:
                """Process queued audio frames asynchronously."""
                try:
                    while session_manager.is_active:
                        try:
                            # Get frame with timeout to allow shutdown checks
                            audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)

                            # Process through VAD
                            vad_processor.process_frame(
                                audio_data, session_state=session_manager.state
                            )

                            # Buffer for ASR if speaking
                            if vad_processor.is_speaking and config.asr.enabled:
                                await orchestrator.on_vad_audio_frame(session_id, audio_data)

                        except TimeoutError:
                            # No frame available, continue to check shutdown
                            continue
                        except Exception as e:
                            logger.error(
                                "Error processing audio frame",
                                extra={"session_id": session_id, "error": str(e)},
                            )
                except asyncio.CancelledError:
                    logger.info(
                        "Audio processing task cancelled",
                        extra={"session_id": session_id},
                    )

            # Start audio processor task
            audio_processor_task = asyncio.create_task(process_audio_queue())

            session_manager.transport.set_audio_frame_callback(on_audio_frame)
            logger.info(
                "Audio frame callback registered (non-blocking queue pattern)",
                extra={"session_id": session_id},
            )
    except ImportError:
        # LiveKit transport not available
        pass

    try:
        logger.info("Starting session handler", extra={"session_id": session_id})

        # Start TTS session with worker
        model_id = "piper-en-us-lessac-medium"  # M5: Use Piper TTS
        await worker_client.start_session(session_id, model_id)
        logger.info(
            "TTS session started",
            extra={"session_id": session_id, "model_id": model_id},
        )

        # Transition from IDLE to LISTENING (ready for input)
        session_manager.transition_state(SessionState.LISTENING)
        logger.info(
            "Session ready for input",
            extra={"session_id": session_id, "state": session_manager.state.value},
        )

        # Main session loop: consume text input, stream audio output
        # Supports multiple messages per session with idle timeout (M10+)
        while True:
            # Check session limits (M10+)
            session_duration = time.monotonic() - session_start_time
            if session_duration > config.session.max_session_duration_seconds:
                logger.info(
                    "Session max duration reached",
                    extra={
                        "session_id": session_id,
                        "duration_s": session_duration,
                        "limit_s": config.session.max_session_duration_seconds,
                    },
                )
                break

            if message_count >= config.session.max_messages_per_session:
                logger.info(
                    "Session max messages reached",
                    extra={
                        "session_id": session_id,
                        "message_count": message_count,
                        "limit": config.session.max_messages_per_session,
                    },
                )
                break

            # Transition to WAITING_FOR_INPUT (ready for next turn) - M10+
            if session_manager.state == SessionState.LISTENING:
                session_manager.transition_state(SessionState.WAITING_FOR_INPUT)

            # Wait for text input with idle timeout (M10+)
            logger.debug(
                "Waiting for next turn (with timeout)",
                extra={
                    "session_id": session_id,
                    "timeout_s": config.session.idle_timeout_seconds,
                },
            )

            try:
                # Wait for next text with timeout
                text = await asyncio.wait_for(
                    session_manager.transport.receive_text().__anext__(),
                    timeout=config.session.idle_timeout_seconds,
                )
            except TimeoutError:
                # No user input within timeout (idle timeout reached)
                logger.info(
                    "Session idle timeout (no user input)",
                    extra={
                        "session_id": session_id,
                        "idle_timeout_s": config.session.idle_timeout_seconds,
                    },
                )
                break
            except StopAsyncIteration:
                # Client closed text stream
                logger.info("Client closed text stream", extra={"session_id": session_id})
                break
            except ConnectionError:
                # Transport connection lost
                logger.info("Transport connection lost", extra={"session_id": session_id})
                break

            if text is None:
                logger.info("Client disconnected (no text)", extra={"session_id": session_id})
                break

            if not text.strip():
                logger.debug(
                    "Received empty text, skipping",
                    extra={"session_id": session_id},
                )
                continue

            # Increment message count (M10+)
            message_count += 1

            logger.info(
                "Received text from client",
                extra={
                    "session_id": session_id,
                    "text_length": len(text),
                    "message_count": message_count,
                },
            )

            # Transition to SPEAKING state
            session_manager.transition_state(SessionState.SPEAKING)

            # Reset VAD state for new synthesis
            if vad_processor:
                vad_processor.reset()

            try:
                # Stream synthesis
                frame_count = 0
                async for audio_frame in worker_client.synthesize([text]):
                    # Check if we were interrupted by barge-in
                    if session_manager.state == SessionState.BARGED_IN:
                        logger.info(
                            "Synthesis interrupted by barge-in, stopping stream",
                            extra={"session_id": session_id, "frames_sent": frame_count},
                        )
                        # Stop synthesis - we're now in BARGED_IN state waiting for user
                        await worker_client.control("STOP")
                        break

                    # Validate frame
                    if not _validate_audio_frame(audio_frame):
                        logger.warning(
                            "Skipping invalid frame (empty without is_final)",
                            extra={
                                "session_id": session_id,
                                "sequence_number": audio_frame.sequence_number,
                            },
                        )
                        continue

                    # Skip empty frames (End Markers) - don't send to client
                    if len(audio_frame.audio_data) == 0:
                        if audio_frame.is_final:
                            logger.debug(
                                "Received end marker (empty final frame), completing synthesis",
                                extra={"session_id": session_id},
                            )
                            break
                        else:
                            # Invalid: empty frame without final marker (already logged warning)
                            continue

                    # Send non-empty audio to client
                    frame_count += 1
                    await session_manager.transport.send_audio_frame(audio_frame.audio_data)

                    # Check if this is the final data frame
                    if audio_frame.is_final:
                        logger.debug(
                            "Received final data frame, completing synthesis",
                            extra={
                                "session_id": session_id,
                                "total_frames": frame_count,
                            },
                        )
                        break

                # Return to LISTENING state if not already interrupted
                if session_manager.state != SessionState.BARGED_IN:
                    session_manager.transition_state(SessionState.LISTENING)
                    logger.info(
                        "Synthesis complete, ready for next message",
                        extra={"session_id": session_id, "frames_sent": frame_count},
                    )

            except grpc.RpcError as e:
                logger.error(
                    "gRPC error during synthesis",
                    extra={"session_id": session_id, "error": str(e)},
                )
                # Don't break session - return to LISTENING for retry
                session_manager.transition_state(SessionState.LISTENING)
                continue
            except Exception as e:
                logger.exception(
                    "Unexpected error during synthesis",
                    extra={"session_id": session_id, "error": str(e)},
                )
                # Break session on unexpected errors
                break

            # Loop continues for next message...

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
        # Cancel audio processor task if it exists
        if audio_processor_task and not audio_processor_task.done():
            audio_processor_task.cancel()
            try:
                await audio_processor_task
            except asyncio.CancelledError:
                pass

        # Clean up TTS session
        try:
            await worker_client.end_session()
            logger.info("TTS session ended", extra={"session_id": session_id})
        except Exception as e:
            logger.warning(
                "Error ending TTS session",
                extra={"session_id": session_id, "error": str(e)},
            )

        # Clean up ASR resources
        await orchestrator._cleanup_session(session_id)

        # Log final metrics including barge-in stats
        metrics = session_manager.get_metrics_summary()
        logger.info("Session metrics", extra={"session_id": session_id, **metrics})

        if vad_processor and vad_processor.is_enabled:
            vad_stats = vad_processor.stats
            logger.info("VAD statistics", extra={"session_id": session_id, **vad_stats})


async def _handle_barge_in_pause(
    session_manager: SessionManager,
    worker_client: TTSWorkerClient,
) -> None:
    """Handle barge-in PAUSE command with latency tracking.

    Args:
        session_manager: Session manager for state and metrics
        worker_client: Worker client for sending PAUSE command
    """
    session_id = session_manager.session_id
    barge_in_start = time.monotonic()

    try:
        # Send PAUSE command to worker
        success = await worker_client.control("PAUSE")

        if success:
            # Calculate latency
            latency_ms = (time.monotonic() - barge_in_start) * 1000.0

            # Transition to BARGED_IN state
            session_manager.transition_state(SessionState.BARGED_IN)

            # Record metrics
            session_manager.metrics.record_barge_in(latency_ms)

            logger.info(
                "Barge-in PAUSE completed",
                extra={
                    "session_id": session_id,
                    "latency_ms": latency_ms,
                },
            )

            # Check if we met SLA (<50ms p95)
            if latency_ms > 50:
                logger.warning(
                    "Barge-in latency exceeded SLA",
                    extra={
                        "session_id": session_id,
                        "latency_ms": latency_ms,
                        "sla_ms": 50,
                    },
                )
        else:
            logger.error(
                "Barge-in PAUSE command failed",
                extra={"session_id": session_id},
            )

    except Exception as e:
        logger.error(
            "Error handling barge-in PAUSE",
            extra={"session_id": session_id, "error": str(e)},
        )


async def start_server(config_path: Path, orchestrator: OrchestratorServer | None = None) -> None:
    """Start orchestrator server with configured transports.

    Initializes all components (config, transport, worker client, health checks)
    and runs the main server loop until interrupted or error.

    Args:
        config_path: Path to YAML config file
        orchestrator: Optional pre-created orchestrator (for testing)

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

    # Create or use provided orchestrator
    if orchestrator is None:
        orchestrator = OrchestratorServer(config)

    # Initialize orchestrator (including ASR)
    await orchestrator.initialize()

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
    logger.info(
        "LiveKit config check",
        extra={"enabled": config.transport.livekit.enabled, "url": config.transport.livekit.url},
    )
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
    setup_health_routes(health_app, registry, worker_client)

    runner = AppRunner(health_app)
    await runner.setup()
    site = TCPSite(runner, "127.0.0.1", health_port)
    await site.start()
    logger.info("Health check server started", extra={"port": health_port})

    # Main server loop: accept sessions and spawn handlers
    session_tasks = []
    try:
        logger.info(
            "Orchestrator server ready",
            extra={"vad_enabled": config.vad.enabled, "asr_enabled": config.asr.enabled},
        )

        # Create tasks for both transports
        transport_tasks = []

        # WebSocket transport task
        async def websocket_loop() -> None:
            while True:
                session = await transport.accept_session()
                logger.info(
                    "New WebSocket session accepted",
                    extra={"session_id": session.session_id},
                )
                session_manager = SessionManager(session)
                task = asyncio.create_task(
                    handle_session(session_manager, worker_client, config, orchestrator)
                )
                session_tasks.append(task)

        transport_tasks.append(asyncio.create_task(websocket_loop()))

        # LiveKit transport task (if enabled)
        if livekit_transport:

            async def livekit_loop() -> None:
                while True:
                    session = await livekit_transport.accept_session()
                    logger.info(
                        "New LiveKit session accepted",
                        extra={"session_id": session.session_id},
                    )
                    session_manager = SessionManager(session)
                    task = asyncio.create_task(
                        handle_session(session_manager, worker_client, config, orchestrator)
                    )
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

        # Shutdown orchestrator (including ASR)
        await orchestrator.shutdown()

        # Wait for session tasks to complete
        if session_tasks:
            logger.info("Waiting for sessions to complete", extra={"count": len(session_tasks)})
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
