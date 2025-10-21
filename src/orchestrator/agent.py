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

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
)
from livekit.plugins import openai, silero

from src.orchestrator.config_validator import ConfigValidator
from src.plugins import grpc_tts, whisperx

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Run validation on module load (before any agent instances created)
logger.info("Validating LiveKit Agent configuration...")
try:
    ConfigValidator.validate_all(strict=False)
except Exception as e:
    logger.warning(f"Configuration validation failed: {e}")


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
            # Phase 1: Still using OpenAI LLM (Phase 4 will make this optional)
            llm=openai.LLM(model="gpt-4o-mini"),
            # Phase 3: Custom gRPC TTS plugin (connects to our TTS worker with Piper)
            tts=grpc_tts.TTS(
                worker_address=os.getenv("TTS_WORKER_ADDRESS", "localhost:7001"),
                model_id=os.getenv(
                    "DEFAULT_MODEL_ID", os.getenv("DEFAULT_MODEL", "cosyvoice2-en-base")
                ),
            ),
            # VAD: Required for streaming with OpenAI STT (which doesn't support streaming natively)
            # Configure with higher threshold to reduce false positives from background noise
            vad=silero.VAD.load(
                min_speech_duration=0.3,  # 300ms minimum speech to trigger
                min_silence_duration=0.5,  # 500ms silence before considering speech ended
                activation_threshold=0.6,  # Higher threshold (0.5 default) - less sensitive
                padding_duration=0.2,  # 200ms padding around speech segments
            ),
            # Turn detection and interruptions
            allow_interruptions=True,  # Enable barge-in
            min_endpointing_delay=0.5,  # 500ms before declaring turn complete
            max_endpointing_delay=3.0,  # 3s max wait for turn completion
        )

        logger.info("VoiceAssistantAgent initialized (Phase 1 POC)")


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

        # Send status update: initialization starting
        # NOW this works because session.start() has connected to the room
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps({
                    "type": "status",
                    "message": "Initializing voice recognition...",
                    "phase": "whisperx_init"
                }).encode('utf-8')
            )
            logger.info("Sent initialization status to client")
        except Exception as e:
            logger.warning(f"Failed to send initialization status update: {e}")

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
                timeout=15.0,  # 15 second timeout for greeting
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
