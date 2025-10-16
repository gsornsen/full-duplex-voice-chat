"""WhisperX STT plugin for LiveKit Agents.

This module provides a LiveKit Agents-compatible STT plugin that wraps our
custom WhisperX adapter. It bridges the LiveKit Agents STT protocol with our
optimized WhisperX implementation.

Features:
- Non-streaming STT (requires VAD wrapper for streaming)
- 4-8x faster than standard Whisper
- CPU and GPU support with auto-selection
- Multiple model sizes (tiny/base/small/medium/large)
- Automatic audio resampling to 16kHz
"""

import asyncio
import logging

from livekit.agents import APIConnectOptions, stt, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGiven

from src.asr.adapters.adapter_whisperx import WhisperXAdapter

logger = logging.getLogger(__name__)


class STT(stt.STT[None]):
    """WhisperX STT plugin for LiveKit Agents.

    Wraps our custom WhisperX adapter to provide LiveKit Agents STT interface.
    This is a non-streaming STT, so it must be wrapped with VAD for real-time use.

    Args:
        model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        device: Device for inference ("cpu", "cuda", or "auto" for auto-detection)
        compute_type: Compute precision ("int8", "float16", "float32", or "default")
        language: Default language code (e.g., "en") or None for auto-detection

    Example:
        ```python
        from src.plugins.whisperx import STT

        # Create STT instance
        stt_plugin = STT(
            model_size="small",
            device="auto",
            language="en"
        )

        # Use with VAD in LiveKit Agent
        agent = Agent(
            stt=stt_plugin,
            vad=silero.VAD.load(),  # Required for streaming
            ...
        )
        ```
    """

    def __init__(
        self,
        *,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = "en",
    ):
        """Initialize WhisperX STT plugin.

        Args:
            model_size: Model size ("tiny", "base", "small", "medium", "large")
            device: Device selection ("cpu", "cuda", "auto")
            compute_type: Precision ("int8", "float16", "float32", "default")
            language: Language code or None for auto-detection
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,  # Non-streaming - requires VAD wrapper
                interim_results=False,
            )
        )

        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language

        # WhisperX adapter (initialized lazily)
        self._adapter: WhisperXAdapter | None = None
        self._initialization_lock = asyncio.Lock()

        logger.info(
            "WhisperX STT plugin created",
            extra={
                "model_size": model_size,
                "device": device,
                "compute_type": compute_type,
                "language": language,
            },
        )

    async def _ensure_initialized(self) -> None:
        """Ensure the WhisperX adapter is initialized (lazy loading)."""
        if self._adapter is not None:
            return

        async with self._initialization_lock:
            # Double-check after acquiring lock
            if self._adapter is not None:
                return

            logger.info("Initializing WhisperX adapter...")

            # Create adapter
            self._adapter = WhisperXAdapter(
                model_size=self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                language=self._language,
            )

            # Initialize (loads model)
            await self._adapter.initialize()

            logger.info(
                "WhisperX adapter initialized successfully",
                extra={
                    "model_size": self._model_size,
                    "device": self._adapter._device,
                    "compute_type": self._adapter._compute_type,
                },
            )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | NotGiven = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Recognize speech from audio buffer.

        This is called by LiveKit Agents framework when VAD detects speech.

        Args:
            buffer: Audio buffer from VAD (contains complete utterance)
            language: Optional language override

        Returns:
            SpeechEvent with transcription result
        """
        # Ensure adapter is initialized
        await self._ensure_initialized()
        assert self._adapter is not None

        # Convert AudioBuffer to PCM bytes
        # LiveKit audio is 16-bit PCM, same as our adapter expects
        audio_frame = utils.merge_frames(buffer)
        audio_data = bytes(audio_frame.data)  # Convert memoryview to bytes

        # Get sample rate from merged audio frame
        sample_rate = audio_frame.sample_rate

        # Determine language to use
        effective_language = self._language if isinstance(language, NotGiven) else language

        logger.debug(
            "Transcribing audio",
            extra={
                "duration_ms": audio_frame.duration * 1000,
                "sample_rate": sample_rate,
                "language": effective_language,
            },
        )

        # Transcribe using WhisperX adapter
        result = await self._adapter.transcribe(
            audio=audio_data,
            sample_rate=sample_rate,
            language=effective_language,
        )

        logger.debug(
            "Transcription complete",
            extra={
                "text": result.text,
                "confidence": result.confidence,
                "language": result.language,
            },
        )

        # Convert to LiveKit SpeechEvent
        # WhisperX provides final results only (no streaming)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=result.text,
                    language=result.language or effective_language or "",
                    confidence=result.confidence,
                )
            ],
        )

    async def aclose(self) -> None:
        """Close the STT plugin and release resources."""
        if self._adapter is not None:
            logger.info("Shutting down WhisperX adapter...")
            await self._adapter.shutdown()
            self._adapter = None
            logger.info("WhisperX adapter shut down successfully")
