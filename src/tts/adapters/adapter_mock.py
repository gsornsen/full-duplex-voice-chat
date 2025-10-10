"""Mock TTS adapter for testing the TTS worker system.

This adapter generates synthetic sine wave audio for testing purposes, following
the TTSAdapter protocol. It provides realistic streaming behavior with proper
state management for pause/resume/stop control commands.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from enum import Enum
from typing import Final

from src.tts.audio.synthesis import generate_sine_wave_frames

# Constants
SINE_FREQUENCY_HZ: Final[int] = 440  # A4 note
AUDIO_DURATION_PER_CHUNK_MS: Final[int] = 500  # 500ms per text chunk
SAMPLE_RATE_HZ: Final[int] = 48000  # 48kHz
FRAME_DURATION_MS: Final[int] = 20  # 20ms frames
INTER_FRAME_DELAY_MS: Final[float] = 5.0  # 5ms delay to simulate streaming


logger = logging.getLogger(__name__)


class AdapterState(Enum):
    """State machine for the mock adapter."""

    IDLE = "idle"
    SYNTHESIZING = "synthesizing"
    PAUSED = "paused"
    STOPPED = "stopped"


class MockTTSAdapter:
    """Mock TTS adapter that generates synthetic sine wave audio for testing.

    This adapter implements the TTSAdapter protocol, providing realistic streaming
    behavior with proper state management. It generates 440Hz sine waves for each
    text chunk and supports pause/resume/stop control commands with < 50ms response time.

    Attributes:
        state: Current adapter state (IDLE, SYNTHESIZING, PAUSED, STOPPED)
        pause_event: Event for pause/resume signaling
        stop_event: Event for stop signaling
        lock: Async lock for protecting state transitions
    """

    def __init__(self) -> None:
        """Initialize the mock adapter."""
        self.state = AdapterState.IDLE
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Start unpaused
        self.stop_event = asyncio.Event()
        self.lock = asyncio.Lock()
        logger.info("MockTTSAdapter initialized")

    async def synthesize_stream(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Generate synthetic audio for each text chunk.

        For each text chunk, generates 500ms of 440Hz sine wave audio framed into
        20ms PCM chunks at 48kHz. Yields frames one at a time with a 5ms delay to
        simulate realistic streaming behavior.

        Args:
            text_chunks: Async iterator of text chunks to synthesize

        Yields:
            20ms PCM audio frames at 48kHz (bytes)

        Notes:
            - Respects PAUSE commands (stops yielding frames immediately)
            - Respects RESUME commands (continues yielding frames)
            - Respects STOP commands (terminates streaming)
            - All control commands respond within < 50ms
        """
        async with self.lock:
            self.state = AdapterState.SYNTHESIZING
            logger.info("Starting synthesis stream", extra={"state": self.state.value})

        try:
            chunk_count = 0
            async for text in text_chunks:
                chunk_count += 1
                logger.debug(
                    "Processing text chunk",
                    extra={"chunk_id": chunk_count, "text_length": len(text)},
                )

                # Generate frames for this text chunk
                frames = generate_sine_wave_frames(
                    frequency=SINE_FREQUENCY_HZ,
                    duration_ms=AUDIO_DURATION_PER_CHUNK_MS,
                    sample_rate=SAMPLE_RATE_HZ,
                    frame_duration_ms=FRAME_DURATION_MS,
                )

                logger.debug(
                    "Generated frames for chunk",
                    extra={"chunk_id": chunk_count, "frame_count": len(frames)},
                )

                # Yield frames with streaming delay
                for frame_idx, frame in enumerate(frames):
                    # Check if stopped (immediate termination)
                    if self.stop_event.is_set():
                        logger.info(
                            "Synthesis stopped by STOP command",
                            extra={"chunk_id": chunk_count, "frame_idx": frame_idx},
                        )
                        return

                    # Wait if paused (blocks until RESUME)
                    await self.pause_event.wait()

                    # Small delay to simulate streaming behavior
                    await asyncio.sleep(INTER_FRAME_DELAY_MS / 1000.0)

                    yield frame

            logger.info("Synthesis stream completed", extra={"total_chunks": chunk_count})

        finally:
            async with self.lock:
                # Only reset to IDLE if we weren't stopped
                if self.state != AdapterState.STOPPED:
                    self.state = AdapterState.IDLE
                    logger.info("Synthesis stream ended", extra={"state": self.state.value})

    async def control(self, command: str) -> None:
        """Handle control commands with < 50ms response time.

        Args:
            command: Control command string (PAUSE, RESUME, STOP)

        Raises:
            ValueError: If command is not recognized

        Notes:
            - PAUSE: Stops yielding frames immediately, state → PAUSED
            - RESUME: Continues yielding frames, state → SYNTHESIZING
            - STOP: Terminates streaming, state → STOPPED
            - All commands use asyncio.Event for immediate response
        """
        async with self.lock:
            previous_state = self.state

            if command == "PAUSE":
                if self.state == AdapterState.SYNTHESIZING:
                    self.state = AdapterState.PAUSED
                    self.pause_event.clear()  # Block synthesize_stream
                    logger.info(
                        "Adapter paused",
                        extra={"command": command, "previous_state": previous_state.value},
                    )
                else:
                    logger.warning(
                        "PAUSE command ignored (not synthesizing)",
                        extra={"current_state": self.state.value},
                    )

            elif command == "RESUME":
                if self.state == AdapterState.PAUSED:
                    self.state = AdapterState.SYNTHESIZING
                    self.pause_event.set()  # Unblock synthesize_stream
                    logger.info(
                        "Adapter resumed",
                        extra={"command": command, "previous_state": previous_state.value},
                    )
                else:
                    logger.warning(
                        "RESUME command ignored (not paused)",
                        extra={"current_state": self.state.value},
                    )

            elif command == "STOP":
                self.state = AdapterState.STOPPED
                self.stop_event.set()
                self.pause_event.set()  # Unblock if paused
                logger.info(
                    "Adapter stopped",
                    extra={"command": command, "previous_state": previous_state.value},
                )

            else:
                logger.error("Unknown control command", extra={"command": command})
                raise ValueError(f"Unknown control command: {command}")

    async def load_model(self, model_id: str) -> None:
        """Mock model load (no-op for testing).

        Args:
            model_id: Model identifier (ignored)

        Notes:
            This is a no-op for the mock adapter. Real adapters would load
            model weights and initialize inference engines here.
        """
        logger.info("Mock load_model called (no-op)", extra={"model_id": model_id})

    async def unload_model(self, model_id: str) -> None:
        """Mock model unload (no-op for testing).

        Args:
            model_id: Model identifier (ignored)

        Notes:
            This is a no-op for the mock adapter. Real adapters would free
            model memory and release GPU resources here.
        """
        logger.info("Mock unload_model called (no-op)", extra={"model_id": model_id})

    def get_state(self) -> AdapterState:
        """Get the current adapter state.

        Returns:
            Current adapter state

        Notes:
            This method is provided for testing purposes to inspect adapter state
            without accessing internal attributes directly.
        """
        return self.state

    async def reset(self) -> None:
        """Reset the adapter to initial state.

        This is a testing utility method to reset the adapter between test cases.
        It clears all events and resets the state to IDLE.

        Notes:
            Not part of the TTSAdapter protocol - for testing only.
        """
        async with self.lock:
            self.state = AdapterState.IDLE
            self.pause_event.set()
            self.stop_event.clear()
            logger.info("Adapter reset to initial state")
