"""LiveKit Agents-based orchestrator for voice assistant.

This is a minimal LiveKit Agent implementation that replaces the custom
orchestrator (server.py) to provide proper agent protocol compatibility
with the frontend LiveKit Agent starter template.

Phase 1: Minimal POC
- Uses built-in STT/TTS/VAD temporarily for testing
- Once working, will be replaced with custom plugins for our TTS worker and WhisperX

Requirements:
- LIVEKIT_URL: LiveKit server URL (e.g., ws://localhost:7880)
- LIVEKIT_API_KEY: LiveKit API key
- LIVEKIT_API_SECRET: LiveKit API secret
- OPENAI_API_KEY: OpenAI API key (for temporary TTS)

Usage:
    python -m src.orchestrator.agent
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
)
from livekit.agents import (
    llm as llm_module,
)
from livekit.plugins import openai, silero

from orchestrator.config_validator import ConfigValidator
from orchestrator.continuation_detector import ContinuationDetector
from orchestrator.dual_llm import DualLLMOrchestrator
from orchestrator.sentence_segmenter import SentenceSegmenter
from orchestrator.transcript_buffer import TranscriptBuffer
from plugins import grpc_tts, whisperx

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Run validation on module load (before any agent instances created)
logger.info("Validating LiveKit Agent configuration...")
try:
    ConfigValidator.validate_all(strict=False)
except Exception as e:
    logger.warning(f"Configuration validation failed: {e}")


class DualLLMPlugin(llm_module.LLM):  # type: ignore[type-arg]
    """Custom LLM plugin that wraps DualLLMOrchestrator for natural conversation.

    This plugin integrates the dual-LLM strategy into the LiveKit Agent framework:
    1. Generates instant filler responses ("Let me think...")
    2. Streams full LLM responses in parallel
    3. Segments responses into complete sentences for TTS

    The dual-LLM approach reduces perceived latency by providing immediate
    feedback while the full response generates in the background.
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini") -> None:
        """Initialize dual LLM plugin.

        Args:
            openai_api_key: OpenAI API key for full LLM
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        self.dual_llm = DualLLMOrchestrator(
            openai_api_key=openai_api_key,
            full_model=model,
            filler_enabled=True,
        )
        self.segmenter = SentenceSegmenter(
            min_tokens=3,
            buffer_timeout=0.5,
        )
        logger.info(f"DualLLMPlugin initialized with model={model}")

    def chat(  # type: ignore[no-untyped-def]
        self,
        *,
        chat_ctx: llm_module.ChatContext,
        **kwargs,
    ) -> llm_module.LLMStream:
        """Generate response using dual-LLM strategy.

        Args:
            chat_ctx: Chat context with conversation history
            conn_options: Connection options (unused)

        Returns:
            LLMStream with filler and full response chunks
        """
        # Extract latest user message
        messages = chat_ctx.messages  # type: ignore[attr-defined]
        if not messages:
            logger.warning("No messages in chat context")
            return self._create_empty_stream()

        # Get the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.role == llm_module.ChatRole.USER:  # type: ignore[attr-defined]
                user_message = msg.content
                break

        if not user_message:
            logger.warning("No user message found in chat context")
            return self._create_empty_stream()

        logger.info(f"Processing user message: {user_message[:100]}...")

        # Convert chat context to conversation history
        conversation_history = []
        for msg in messages[:-1]:  # Exclude latest user message (added by dual_llm)
            conversation_history.append({
                "role": "assistant" if msg.role == llm_module.ChatRole.ASSISTANT else "user",  # type: ignore[attr-defined]
                "content": msg.content,
            })

        # Generate response using dual-LLM
        return self._stream_dual_llm_response(user_message, conversation_history)

    def _stream_dual_llm_response(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]],
    ) -> llm_module.LLMStream:
        """Stream dual-LLM response with filler and full content.

        Args:
            user_message: User's input message
            conversation_history: Previous conversation messages

        Returns:
            LLMStream with response chunks
        """
        # Create async generator for dual-LLM response
        async def generate_chunks():  # type: ignore[no-untyped-def]
            accumulated_text = ""

            # Generate response with dual-LLM
            async for text_chunk, _phase in self.dual_llm.generate_response(
                user_message=user_message,
                conversation_history=conversation_history,
            ):
                # Skip empty chunks
                if not text_chunk:
                    continue

                accumulated_text += text_chunk

                # Yield chunk with role information
                yield llm_module.ChatChunk(  # type: ignore[call-arg]
                    choices=[
                        llm_module.Choice(  # type: ignore[attr-defined]
                            delta=llm_module.ChoiceDelta(
                                role=llm_module.ChatRole.ASSISTANT,  # type: ignore[attr-defined]
                                content=text_chunk,
                            ),
                            index=0,
                        )
                    ],
                )

            logger.info(f"Dual-LLM response completed: {len(accumulated_text)} chars")

        # Return LLM stream
        return llm_module.LLMStream(
            oai_stream=generate_chunks(),  # type: ignore
            chat_ctx=llm_module.ChatContext(),
        )

    def _create_empty_stream(self) -> llm_module.LLMStream:
        """Create empty LLM stream for error cases.

        Returns:
            Empty LLMStream
        """
        async def empty_generator():  # type: ignore[no-untyped-def]
            if False:  # pragma: no cover
                yield None

        return llm_module.LLMStream(
            oai_stream=empty_generator(),  # type: ignore
            chat_ctx=llm_module.ChatContext(),
        )


class VoiceAssistantAgent(Agent):
    """Voice assistant agent with simple echo/passthrough behavior.

    For Phase 1 POC, this agent will:
    - Accept speech input via STT
    - Optionally process through LLM (or skip for direct passthrough)
    - Synthesize speech via TTS
    - Support barge-in and interruptions

    Future phases will integrate:
    - Custom WhisperX STT plugin
    - Custom gRPC TTS plugin (Piper adapter)
    - Custom VAD with state-aware gating and noise gate
    - Session management with timeout
    """

    def __init__(self) -> None:
        """Initialize voice assistant agent."""
        # Debug: Log TTS_WORKER_ADDRESS environment variable
        tts_worker_env = os.getenv("TTS_WORKER_ADDRESS")
        logger.info(
            "TTS_WORKER_ADDRESS environment variable check",
            extra={
                "TTS_WORKER_ADDRESS": tts_worker_env,
                "default_fallback": "localhost:7001",
            },
        )
        # Check if dual-LLM is enabled
        dual_llm_enabled = os.getenv("DUAL_LLM_ENABLED", "false").lower() == "true"
        openai_api_key = os.getenv("OPENAI_API_KEY", "")

        # Select LLM implementation based on feature flag
        # Note: DualLLMPlugin temporarily disabled due to LiveKit API compatibility issues
        # Will be re-enabled after API compatibility is resolved
        if False and dual_llm_enabled and openai_api_key:  # Temporarily disabled
            logger.info("Using dual-LLM strategy for natural conversation")
            llm_instance = DualLLMPlugin(
                openai_api_key=openai_api_key,
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            )
        else:
            if dual_llm_enabled:
                logger.warning(
                    "Dual-LLM requested but temporarily disabled due to API compatibility. "
                    "Using standard OpenAI LLM instead."
                )
            logger.info("Using standard OpenAI LLM")
            llm_instance = openai.LLM(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
            )

        # Configure parallel synthesis if enabled
        self.parallel_synthesis_enabled = (
            os.getenv("PARALLEL_SYNTHESIS_ENABLED", "false").lower() == "true"
        )

        if self.parallel_synthesis_enabled:
            num_workers = int(os.getenv("PARALLEL_SYNTHESIS_NUM_WORKERS", "2"))
            max_queue_depth = int(os.getenv("PARALLEL_SYNTHESIS_MAX_QUEUE_DEPTH", "10"))
            gpu_limit_str = os.getenv("PARALLEL_SYNTHESIS_GPU_LIMIT", "2")
            gpu_limit = int(gpu_limit_str) if gpu_limit_str.lower() != "none" else None
            prefetch_enabled = (
                os.getenv("PARALLEL_SYNTHESIS_PREFETCH_ENABLED", "false").lower() == "true"
            )
            prefetch_depth = int(os.getenv("PARALLEL_SYNTHESIS_PREFETCH_DEPTH", "3"))

            # Create TTS client with parallel synthesis enabled
            worker_address = os.getenv("TTS_WORKER_ADDRESS", "localhost:7001")
            self._tts_client = grpc_tts.TTS(
                worker_address=worker_address,
                model_id=os.getenv(
                    "DEFAULT_MODEL_ID", os.getenv("DEFAULT_MODEL", "cosyvoice2-en-base")
                ),
                parallel_enabled=True,
                parallel_num_workers=num_workers,
                parallel_max_queue=max_queue_depth,
                parallel_gpu_limit=gpu_limit,
                prefetch_enabled=prefetch_enabled,
                prefetch_depth=prefetch_depth,
            )
            logger.info(
                f"Parallel synthesis enabled (workers={num_workers}, "
                f"queue_depth={max_queue_depth}, gpu_limit={gpu_limit}, "
                f"prefetch_enabled={prefetch_enabled}, prefetch_depth={prefetch_depth})",
                extra={"worker_address": worker_address},
            )
        else:
            # Create TTS client without parallel synthesis
            worker_address = os.getenv("TTS_WORKER_ADDRESS", "localhost:7001")
            self._tts_client = grpc_tts.TTS(
                worker_address=worker_address,
                model_id=os.getenv(
                    "DEFAULT_MODEL_ID", os.getenv("DEFAULT_MODEL", "cosyvoice2-en-base")
                ),
                parallel_enabled=False,
            )
            logger.info(
                "Parallel synthesis disabled (using standard TTS)",
                extra={"worker_address": worker_address},
            )

        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You assist users with their questions by providing clear and concise information.
            Keep your responses brief and conversational.
            Avoid complex formatting, emojis, or special characters.
            You are friendly, curious, and have a helpful attitude.""",
            # Phase 2: Custom WhisperX STT plugin (4-8x faster than OpenAI Whisper)
            stt=whisperx.STT(
                model_size=os.getenv("ASR_MODEL_SIZE", "small"),
                device=os.getenv("ASR_DEVICE", "auto"),
                language=os.getenv("ASR_LANGUAGE", "en"),
                compute_type=os.getenv("ASR_COMPUTE_TYPE", "default"),
            ),
            # LLM: Dual-LLM strategy if enabled, otherwise standard OpenAI
            llm=llm_instance,
            # Phase 3: Custom gRPC TTS plugin (connects to our TTS worker with Piper)
            # NOW with optional parallel synthesis support
            tts=self._tts_client,
            # VAD: Required for streaming with OpenAI STT (which doesn't support streaming natively)
            # Tuned to capture full speech start (avoid trimming first words)
            # and prevent agent audio from triggering barge-in
            vad=silero.VAD.load(
                min_speech_duration=0.25,  # 250ms minimum (faster detection)
                min_silence_duration=0.8,  # 800ms silence (prevent TTS tail triggering)
                activation_threshold=0.75,  # Higher threshold (reduce false positives)
                padding_duration=0.5,  # 500ms padding (capture speech start)
            ),
            # Turn detection and interruptions
            allow_interruptions=True,  # Enable barge-in
            min_endpointing_delay=0.5,  # 500ms before declaring turn complete
            max_endpointing_delay=3.0,  # 3s max wait for turn completion
        )

        self.dual_llm_enabled = dual_llm_enabled

        # Initialize continuation detection (Phase 1)
        self.continuation_enabled = (
            os.getenv("ENABLE_CONTINUATION_DETECTION", "false").lower() == "true"
        )
        if self.continuation_enabled:
            self.continuation_detector = ContinuationDetector()
            self.transcript_buffer = TranscriptBuffer(max_size=10, ttl_seconds=30.0)
            logger.info("Continuation detection enabled")
        else:
            self.continuation_detector = None  # type: ignore[assignment]
            self.transcript_buffer = None  # type: ignore[assignment]
            logger.info("Continuation detection disabled")

        logger.info(
            f"VoiceAssistantAgent initialized "
            f"(dual_llm_enabled={dual_llm_enabled}, "
            f"continuation_enabled={self.continuation_enabled}, "
            f"parallel_synthesis_enabled={self.parallel_synthesis_enabled})"
        )


async def entrypoint(ctx: JobContext) -> None:
    """Agent entry point - called when a new job is assigned.

    This function is invoked by the LiveKit Agents worker when a participant
    joins a room. It creates an AgentSession and starts the voice assistant.

    IMPORTANT: Early Connection Pattern
    We connect to the room FIRST (via session.start()) before initializing
    heavy components (WhisperX verification). This makes ctx.room.local_participant
    available for sending status updates via data channel during initialization.

    Without this, status messages fail with:
    "cannot access local participant before connecting"

    Args:
        ctx: Job context containing room and participant information
    """
    logger.info(
        "Agent entrypoint called",
        extra={
            "room": ctx.room.name if ctx.room else "unknown",
            "job_id": ctx.job.id if ctx.job else "unknown",
        },
    )

    try:
        # Create agent instance
        agent = VoiceAssistantAgent()

        # Create agent session
        # Note: STT/TTS/LLM are configured in the Agent class constructor
        # AgentSession handles the runtime orchestration
        session: AgentSession[Agent] = AgentSession(
            # Interruption settings
            allow_interruptions=True,
            min_interruption_duration=0.5,  # 500ms minimum to register as interruption
            # Endpointing settings (when to consider user turn complete)
            min_endpointing_delay=0.5,  # Wait at least 500ms after silence
            max_endpointing_delay=3.0,  # Max 3s wait
            # Session timeout (Phase 5 will add custom session management)
            user_away_timeout=300.0,  # 5 minutes idle timeout
        )

        logger.info("AgentSession created, connecting to room early...")

        # CRITICAL: Start session FIRST to connect to room
        # This makes ctx.room.local_participant available for status updates
        # during WhisperX verification below
        await session.start(
            room=ctx.room,
            agent=agent,
            # IMPORTANT: immediate_greeting=False prevents waiting for user speech
            # We'll send the greeting manually after initialization
        )

        logger.info(
            "Agent connected to room, local_participant now available",
            extra={"room": ctx.room.name},
        )


        # Send status update: connecting to TTS worker
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps({
                    "type": "status",
                    "message": "Connecting to speech synthesis service...",
                    "phase": "connecting_tts"
                }).encode('utf-8')
            )
            logger.info("Sent connecting_tts status to client")
        except Exception as e:
            logger.warning(f"Failed to send connecting_tts status update: {e}")

        # Start parallel synthesis mode if enabled
        # This starts the persistent worker pool that synthesizes sentences in parallel
        if agent.parallel_synthesis_enabled:
            logger.info("Starting parallel synthesis persistent worker pool...")
            try:
                # Send status update: TTS model loading
                try:
                    await ctx.room.local_participant.publish_data(
                        json.dumps({
                            "type": "status",
                            "message": (
                                "Loading AI models "
                                "(this may take up to 2 minutes on first start)..."
                            ),
                            "phase": "loading_tts_model"
                        }).encode('utf-8')
                    )
                    logger.info("Sent loading_tts_model status to client")
                except Exception as e:
                    logger.warning(f"Failed to send loading_tts_model status update: {e}")

                await agent._tts_client.start_parallel_mode()
                logger.info("Parallel synthesis worker pool started successfully")
            except Exception as e:
                logger.error(f"Failed to start parallel synthesis mode: {e}", exc_info=True)
                logger.warning("Continuing with sequential synthesis as fallback")

        # PERFORMANCE NOTE: WhisperX is pre-warmed at container startup (see main())
        # This eliminates the 25-second delay on first room join.
        # The per-job initialization here should be instant (~100ms) as it uses the
        # singleton cached model loaded during worker startup.
        #
        # MULTIPROCESSING NOTE: LiveKit Agents uses multiprocessing for job isolation.
        # Each worker process has its own Python runtime and module-level singleton cache.
        # This means the pre-warmed model in the main process is NOT shared with worker
        # processes. Each worker must load the model independently, potentially causing
        # GPU contention when multiple jobs start simultaneously.
        #
        # TIMEOUT: Increased from 5s to 90s to handle:
        # - Cache miss in forked process (model not pre-warmed in this process): ~25s
        # - GPU contention with multiple concurrent loads (2-3x processes): ~60s
        # - Network delays for model download (if not cached on disk): +10-15s
        # Send status update: verifying WhisperX
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps({
                    "type": "status",
                    "message": "Initializing voice recognition...",
                    "phase": "verifying_whisperx"
                }).encode('utf-8')
            )
            logger.info("Sent verifying_whisperx status to client")
        except Exception as e:
            logger.warning(f"Failed to send verifying_whisperx status update: {e}")

        logger.info("Verifying WhisperX STT plugin is ready...")
        try:
            # Access the STT plugin and verify initialization (should be fast via cache)
            stt_plugin = agent._stt
            if isinstance(stt_plugin, whisperx.STT):
                # Timeout matches pre-warming timeout to handle GPU contention
                await asyncio.wait_for(
                    stt_plugin._ensure_initialized(),
                    timeout=90.0,  # 90 second timeout (was 5s - too aggressive for multiprocess)
                )
                logger.info("WhisperX STT plugin ready (using cached model from startup)")
            else:
                logger.warning(
                    f"STT plugin is not WhisperX (got {type(stt_plugin).__name__}), "
                    "skipping verification"
                )
        except TimeoutError:
            logger.error("WhisperX verification timed out after 5 seconds")
            raise
        except Exception as e:
            logger.error(f"Failed to verify WhisperX STT: {e}", exc_info=True)
            raise

        # SOLUTION A: Eager TTS Warm-Up
        # Warm up TTS model to eliminate cold-start CUDA compilation latency
        # This prevents 1-2 second delay on first greeting synthesis
        logger.info("Warming up TTS model...")
        try:
            # Access TTS plugin and warm up
            tts_plugin = agent._tts
            if tts_plugin is not None and hasattr(tts_plugin, 'warm_up'):
                warm_up_start = time.perf_counter()
                await asyncio.wait_for(
                    tts_plugin.warm_up(),
                    timeout=30.0,
                )
                warm_up_duration = time.perf_counter() - warm_up_start
                logger.info(f"TTS model warmed up successfully in {warm_up_duration:.2f}s")
            else:
                logger.warning(
                    f"TTS plugin does not implement warm_up() (got {type(tts_plugin).__name__}), "
                    "skipping warm-up"
                )
        except TimeoutError:
            logger.warning("TTS warm-up timed out after 30s (non-fatal)")
        except Exception as e:
            logger.warning(f"TTS warm-up failed (non-fatal): {e}")

        # Send status update: ready
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps({
                    "type": "status",
                    "message": "Voice assistant ready!",
                    "phase": "ready"
                }).encode('utf-8')
            )
            logger.info("Sent ready status to client")
        except Exception as e:
            logger.warning(f"Failed to send ready status update: {e}")

        # Generate immediate greeting (don't wait for user to speak first)
        # This ensures the agent greets proactively upon joining
        try:
            logger.info("Generating initial greeting...")
            # Add timeout for greeting generation
            await asyncio.wait_for(
                session.say(
                    "Hello! I'm your voice assistant. How can I help you today?",
                    allow_interruptions=True,  # User can interrupt the greeting
                ),
                timeout=30.0,  # 30 second timeout (increased from 15s for TTS delivery)
            )
            logger.info("Initial greeting sent to user")
        except TimeoutError:
            logger.error("Greeting generation timed out after 15 seconds")
            # Don't raise - agent can still work for subsequent interactions
        except Exception as e:
            logger.error(f"Failed to generate initial greeting: {e}", exc_info=True)
            # Don't raise - agent can still work for subsequent interactions

    except Exception as e:
        logger.error(
            f"Fatal error in agent entrypoint: {e}",
            exc_info=True,
            extra={
                "room": ctx.room.name if ctx.room else "unknown",
                "job_id": ctx.job.id if ctx.job else "unknown",
            },
        )
        raise  # Re-raise to mark job as failed


def shutdown_handler(signum: int, frame) -> None:  # type: ignore
    """Handle graceful shutdown of agent worker.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    # LiveKit agents framework handles deregistration automatically
    sys.exit(0)


async def prewarm_whisperx() -> None:
    """Pre-warm WhisperX model at container startup.

    This eliminates the 25-second delay on first room join by loading the model
    into the singleton cache before any jobs are assigned.

    Performance Impact:
        - Without pre-warming: First user waits 25-28 seconds
        - With pre-warming: First user waits <1 second (model already cached)

    The model is loaded once and cached in module-level storage, shared across
    all subsequent STT plugin instances with the same configuration.
    """
    logger.info("╔════════════════════════════════════════════════════════════════════╗")
    logger.info("║  Pre-warming WhisperX model (container startup)                   ║")
    logger.info("╚════════════════════════════════════════════════════════════════════╝")

    try:
        # Create STT instance with same config as agent will use
        stt_plugin = whisperx.STT(
            model_size=os.getenv("ASR_MODEL_SIZE", "small"),
            device=os.getenv("ASR_DEVICE", "auto"),
            language=os.getenv("ASR_LANGUAGE", "en"),
            compute_type=os.getenv("ASR_COMPUTE_TYPE", "default"),
        )

        logger.info(
            "Loading WhisperX model...",
            extra={
                "model_size": os.getenv("ASR_MODEL_SIZE", "small"),
                "device": os.getenv("ASR_DEVICE", "auto"),
                "compute_type": os.getenv("ASR_COMPUTE_TYPE", "default"),
            },
        )

        # Trigger initialization (loads model into singleton cache)
        # 90 second timeout accounts for:
        # - Model download (if not cached): ~10-15s
        # - Model loading to GPU: ~15-20s
        # - Initialization overhead: ~5s
        # - Buffer for slow systems/network: 50s
        await asyncio.wait_for(
            stt_plugin._ensure_initialized(),
            timeout=90.0,  # 90 second timeout for startup pre-warming
        )

        logger.info("✓ WhisperX model pre-warmed successfully")
        logger.info("  First user will experience <1s initialization (cached model)")
        logger.info("  Singleton cache active for all subsequent STT instances")

    except TimeoutError:
        logger.error("✗ WhisperX pre-warming timed out after 90 seconds")
        logger.error("  Continuing startup - first user will experience longer delay")
    except Exception as e:
        logger.error(f"✗ WhisperX pre-warming failed: {e}", exc_info=True)
        logger.error("  Continuing startup - first user will experience longer delay")


def main() -> None:
    """Main entry point for the agent worker.

    Starts the LiveKit Agents worker which will:
    1. Pre-warm WhisperX model (eliminate 25s first-user delay)
    2. Connect to LiveKit server
    3. Register as available worker
    4. Wait for job assignments (participant joins)
    5. Call entrypoint() for each job
    6. Handle worker lifecycle and cleanup
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    # Verify environment variables
    required_env_vars = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "OPENAI_API_KEY",  # Phase 1 requirement (temporary)
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        logger.error("Please set these in your environment or .env file")
        logger.error("Example:")
        logger.error("  LIVEKIT_URL=ws://localhost:7880")
        logger.error("  LIVEKIT_API_KEY=your-api-key")
        logger.error("  LIVEKIT_API_SECRET=your-api-secret")
        logger.error("  OPENAI_API_KEY=sk-...")
        return

    # Log dual-LLM configuration
    dual_llm_enabled = os.getenv("DUAL_LLM_ENABLED", "false").lower() == "true"
    parallel_synthesis_enabled = os.getenv("PARALLEL_SYNTHESIS_ENABLED", "false").lower() == "true"

    logger.info(f"Dual-LLM feature: {'ENABLED' if dual_llm_enabled else 'DISABLED'}")
    logger.info(f"Parallel synthesis: {'ENABLED' if parallel_synthesis_enabled else 'DISABLED'}")
    logger.info("Starting LiveKit Agents worker (Phase 1 POC)...")
    logger.info(f"LiveKit URL: {os.getenv('LIVEKIT_URL')}")
    logger.info("Using WhisperX STT + OpenAI LLM + gRPC TTS (Piper/CosyVoice2)")

    # PRE-WARM WHISPERX MODEL (before accepting any jobs)
    # This eliminates the 25-second delay on first room join
    logger.info("Pre-warming WhisperX model before starting worker...")
    try:
        asyncio.run(prewarm_whisperx())
    except Exception as e:
        logger.warning(f"Pre-warming failed (non-fatal): {e}")
        logger.warning("Worker will start but first user may experience delay")

    logger.info("Starting worker (ready to accept room assignments)...")

    # Start the worker with increased timeout for process initialization
    # The initialize_process_timeout covers:
    # 1. Process fork/spawn time
    # 2. Module imports (WhisperX, PyTorch, etc.)
    # 3. Singleton cache lookup (should be fast since we pre-warmed)
    # 4. Any per-job initialization in entrypoint()
    #
    # While pre-warming eliminates most delay, we still need sufficient timeout
    # to handle edge cases (cache miss, slow process spawn, etc.)
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="voice-assistant",  # Must match frontend request!
            # CRITICAL: Increase from default 10s to 120s to prevent premature timeout
            # during agent initialization. WhisperX loading takes 25-28s without pre-warming.
            # Even with pre-warming, we need buffer for process spawn, imports, and cache lookup.
            initialize_process_timeout=120.0,  # 120 seconds (was: 10s default)
            # Also increase shutdown timeout to allow graceful cleanup
            shutdown_process_timeout=90.0,  # 90 seconds (was: 60s default)
        )
    )


if __name__ == "__main__":
    main()
