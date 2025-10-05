"""Base protocol/interface for TTS adapters."""

from collections.abc import AsyncIterator
from typing import Protocol


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
