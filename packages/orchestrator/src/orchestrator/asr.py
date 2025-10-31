"""Automatic Speech Recognition using Whisper."""

from collections.abc import AsyncIterator


class ASR:
    """Whisper-based speech recognition."""

    def __init__(self, model_name: str = "small") -> None:
        """Initialize ASR with specified Whisper model.

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_name = model_name

    async def transcribe_stream(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Transcribe streaming audio to text.

        Args:
            audio_stream: Async iterator of audio frames

        Yields:
            Transcribed text chunks
        """
        if False:
            yield ""
