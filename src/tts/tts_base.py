"""Base protocol/interface for TTS adapters."""

from collections.abc import AsyncIterator
from enum import Enum
from typing import Protocol


class AdapterState(Enum):
    """State machine for TTS adapters.

    All TTS adapters follow this state machine for consistent behavior:

    State Transitions:
        IDLE → SYNTHESIZING: Start synthesis
        SYNTHESIZING → PAUSED: PAUSE command received
        PAUSED → SYNTHESIZING: RESUME command received
        SYNTHESIZING → STOPPED: STOP command received
        PAUSED → STOPPED: STOP command received
        STOPPED → IDLE: Reset/cleanup complete

    States:
        IDLE: Adapter ready, no active synthesis
        SYNTHESIZING: Actively generating and emitting audio frames
        PAUSED: Synthesis paused, waiting for RESUME
        STOPPED: Synthesis terminated, cleanup in progress
    """

    IDLE = "idle"
    SYNTHESIZING = "synthesizing"
    PAUSED = "paused"
    STOPPED = "stopped"


class TTSAdapter(Protocol):
    """Protocol defining the unified TTS adapter interface."""

    async def synthesize_stream(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Synthesize text chunks to audio frames.

        Args:
            text_chunks: Async iterator of text chunks

        Yields:
            20ms PCM audio frames at 48kHz
        """
        ...

    async def control(self, command: str) -> None:
        """Handle control commands (PAUSE, RESUME, STOP).

        Args:
            command: Control command string
        """
        ...

    async def load_model(self, model_id: str) -> None:
        """Load a specific model.

        Args:
            model_id: Model identifier
        """
        ...

    async def unload_model(self, model_id: str) -> None:
        """Unload a specific model.

        Args:
            model_id: Model identifier
        """
        ...
