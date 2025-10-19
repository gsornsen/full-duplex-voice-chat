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

import logging
import os

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
)
from livekit.plugins import openai, silero

from src.plugins import grpc_tts, whisperx

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


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
                model_size="small",  # Good balance of speed and accuracy
                device="cpu",  # Use CPU (GPU has cuDNN version mismatch in WSL2)
                language="en",  # English language
            ),
            # Phase 1: Still using OpenAI LLM (Phase 4 will make this optional)
            llm=openai.LLM(model="gpt-4o-mini"),
            # Phase 3: Custom gRPC TTS plugin (connects to our TTS worker with Piper)
            tts=grpc_tts.TTS(
                worker_address=os.getenv("TTS_WORKER_ADDRESS", "localhost:7001"),
                model_id=os.getenv("DEFAULT_MODEL", "piper-en-us-lessac-medium"),
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

    logger.info("AgentSession created, starting agent...")

    # Start the session with the room and agent
    await session.start(
        room=ctx.room,
        agent=VoiceAssistantAgent(),
        # IMPORTANT: immediate_greeting=False prevents waiting for user speech
        # The greeting will be sent as soon as the agent joins the room
    )

    logger.info(
        "Agent session started successfully",
        extra={"room": ctx.room.name},
    )

    # Generate immediate greeting (don't wait for user to speak first)
    # This ensures the agent greets proactively upon joining
    try:
        # Use say() instead of generate_reply() for immediate speech
        await session.say(
            "Hello! I'm your voice assistant. How can I help you today?",
            allow_interruptions=True,  # User can interrupt the greeting
        )
        logger.info("Initial greeting sent to user")
    except Exception as e:
        logger.warning(f"Failed to generate initial greeting: {e}")
        # Non-fatal - agent will still work for subsequent interactions


def main() -> None:
    """Main entry point for the agent worker.

    Starts the LiveKit Agents worker which will:
    1. Connect to LiveKit server
    2. Register as available worker
    3. Wait for job assignments (participant joins)
    4. Call entrypoint() for each job
    5. Handle worker lifecycle and cleanup
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

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
    logger.info("Using OpenAI STT/LLM/TTS (temporary - will be replaced in Phases 2-3)")

    # Start the worker
    # This will block until interrupted
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="voice-assistant",  # Must match frontend request!
        )
    )


if __name__ == "__main__":
    main()
