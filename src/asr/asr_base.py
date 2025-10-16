"""Base protocol/interface for ASR (Automatic Speech Recognition) adapters.

This module defines the abstract interface that all ASR adapters must implement,
following the same pattern as TTS adapters. It provides a unified API for
speech recognition regardless of the underlying model (Whisper, Vosk, etc.).

Key design principles:
- Protocol-based interface for structural typing
- Async/await throughout for non-blocking I/O
- Type-safe with strict mypy compliance
- Comprehensive error handling
- Resource lifecycle management (initialize/shutdown)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class ASRError(Exception):
    """Base exception for ASR-related errors."""

    pass


class ModelNotLoadedError(ASRError):
    """Raised when trying to use ASR before model is loaded."""

    pass


class InvalidAudioError(ASRError):
    """Raised when audio data is invalid or cannot be processed."""

    pass


class TranscriptionError(ASRError):
    """Raised when transcription fails."""

    pass


class ASRLanguage(str, Enum):
    """Supported ASR languages.

    Using enum for type safety and autocomplete support.
    """

    EN = "en"  # English
    ES = "es"  # Spanish
    FR = "fr"  # French
    DE = "de"  # German
    ZH = "zh"  # Chinese
    JA = "ja"  # Japanese
    KO = "ko"  # Korean
    AUTO = "auto"  # Auto-detect language


@dataclass(frozen=True)
class TranscriptionResult:
    """Result of an ASR transcription operation.

    Immutable dataclass containing transcription output and metadata.

    Attributes:
        text: Transcribed text (UTF-8 string)
        confidence: Overall confidence score (0.0 = no confidence, 1.0 = perfect)
        language: Detected or specified language code (ISO 639-1)
        duration_ms: Duration of audio that was transcribed (milliseconds)
        word_timestamps: Optional list of (word, start_ms, end_ms) tuples
        is_partial: Whether this is a partial (streaming) result
    """

    text: str
    confidence: float
    language: str | None = None
    duration_ms: int = 0
    word_timestamps: list[tuple[str, int, int]] | None = None
    is_partial: bool = False

    def __post_init__(self) -> None:
        """Validate field values after initialization."""
        # Validate confidence range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")

        # Validate duration
        if self.duration_ms < 0:
            raise ValueError(f"Duration must be non-negative, got {self.duration_ms}")


class ASRAdapterBase(ABC):
    """Abstract base class for ASR adapters.

    All ASR implementations must inherit from this class and implement
    the required abstract methods. This ensures a consistent interface
    across different ASR models (Whisper, Vosk, etc.).

    Lifecycle:
        1. Create adapter instance
        2. Call initialize() to load model
        3. Call transcribe() as needed
        4. Call shutdown() to cleanup resources

    Thread-safety: Implementations should be thread-safe for transcribe()
    calls after initialization completes.

    Example:
        ```python
        # Create and initialize adapter
        adapter = WhisperAdapter(model_size="small", device="cuda")
        await adapter.initialize()

        # Transcribe audio
        audio_data = load_audio_bytes()  # 16-bit PCM, mono
        result = await adapter.transcribe(audio_data, sample_rate=16000)

        print(f"Transcription: {result.text}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Language: {result.language}")

        # Cleanup
        await adapter.shutdown()
        ```
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the ASR adapter and load model.

        This method performs:
        - Model loading (from disk or download)
        - Resource allocation (GPU memory, etc.)
        - Warmup inference (optional but recommended)

        Must be called before transcribe() can be used.

        Raises:
            ASRError: If initialization fails (model not found, OOM, etc.)
            RuntimeError: If already initialized
        """
        ...

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        sample_rate: int,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio bytes to text.

        Args:
            audio: Raw PCM audio bytes (16-bit signed int, little endian, mono)
            sample_rate: Sample rate in Hz (e.g., 16000, 22050, 48000)
            language: Optional language hint (ISO 639-1 code like "en", "es")
                     If None, adapter may auto-detect or use default

        Returns:
            TranscriptionResult with text, confidence, and metadata

        Raises:
            ModelNotLoadedError: If initialize() hasn't been called
            InvalidAudioError: If audio data is malformed or empty
            TranscriptionError: If transcription fails

        Performance:
            - Target latency: p95 < 1.5s for GPU (Whisper small)
            - Target latency: p95 < 5s for CPU
            - Implementations should handle audio up to 30 seconds

        Example:
            ```python
            # 1 second of 16kHz audio (16-bit PCM, mono)
            sample_rate = 16000
            audio_bytes = b"\\x00\\x01" * sample_rate  # 32000 bytes

            result = await adapter.transcribe(audio_bytes, sample_rate)
            assert len(result.text) > 0
            assert 0.0 <= result.confidence <= 1.0
            ```
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown adapter and release resources.

        This method performs:
        - Model unloading
        - GPU memory release
        - Cleanup of temporary files

        After calling shutdown(), initialize() must be called again
        before transcribe() can be used.

        Raises:
            ASRError: If shutdown fails (should be rare)
        """
        ...

    @abstractmethod
    def get_supported_sample_rates(self) -> list[int]:
        """Get list of sample rates supported by this adapter.

        Returns:
            List of supported sample rates in Hz (e.g., [8000, 16000, 48000])

        Note:
            If adapter supports arbitrary sample rates via resampling,
            should return common rates like [8000, 16000, 22050, 44100, 48000]
        """
        ...

    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if adapter is initialized and ready for transcription.

        Returns:
            True if initialize() has been called and succeeded, False otherwise
        """
        ...

    @abstractmethod
    def get_model_info(self) -> dict[str, str | int | float]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model metadata:
            - name: Model name (e.g., "whisper-small")
            - version: Model version (e.g., "20240930")
            - language: Primary language or "multilingual"
            - parameters: Number of parameters (e.g., 244_000_000)
            - device: Device model is loaded on ("cpu", "cuda", "cuda:0")

        Raises:
            ModelNotLoadedError: If initialize() hasn't been called
        """
        ...


class StreamingASRAdapter(Protocol):
    """Protocol for streaming ASR adapters.

    Optional protocol for adapters that support real-time streaming
    transcription. Not all ASR models support this (e.g., Whisper doesn't).

    This is a Protocol (not ABC) to allow duck typing - adapters can
    implement these methods without explicit inheritance.

    Example:
        ```python
        # Check if adapter supports streaming
        if isinstance(adapter, StreamingASRAdapter):
            async for result in adapter.transcribe_stream(audio_stream):
                print(f"Partial: {result.text}")
        ```
    """

    async def transcribe_stream(
        self,
        audio_stream: bytes,
        sample_rate: int,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe streaming audio with incremental results.

        Args:
            audio_stream: Audio chunk (16-bit PCM, mono)
            sample_rate: Sample rate in Hz
            language: Optional language hint

        Returns:
            TranscriptionResult with is_partial=True for intermediate results

        Note:
            This is called for each audio chunk. Final result has is_partial=False.
        """
        ...

    async def finalize_stream(self) -> TranscriptionResult:
        """Finalize streaming transcription and get final result.

        Returns:
            Final TranscriptionResult with is_partial=False
        """
        ...


class ASRCapabilities:
    """Capabilities descriptor for ASR adapters.

    Used for adapter selection and routing decisions.

    Attributes:
        supports_streaming: Whether adapter supports real-time streaming
        supports_timestamps: Whether adapter provides word-level timestamps
        supports_language_detection: Whether adapter can auto-detect language
        supported_languages: List of supported language codes
        max_audio_duration_s: Maximum audio duration supported (seconds)
        requires_gpu: Whether GPU is required (vs optional)
        model_size_mb: Approximate model size in megabytes
    """

    def __init__(
        self,
        supports_streaming: bool = False,
        supports_timestamps: bool = False,
        supports_language_detection: bool = False,
        supported_languages: list[str] | None = None,
        max_audio_duration_s: float = 30.0,
        requires_gpu: bool = False,
        model_size_mb: int = 0,
    ) -> None:
        """Initialize capabilities descriptor.

        Args:
            supports_streaming: Real-time streaming support
            supports_timestamps: Word-level timestamp support
            supports_language_detection: Auto language detection
            supported_languages: List of ISO 639-1 language codes
            max_audio_duration_s: Max audio duration
            requires_gpu: GPU requirement
            model_size_mb: Model size in MB
        """
        self.supports_streaming = supports_streaming
        self.supports_timestamps = supports_timestamps
        self.supports_language_detection = supports_language_detection
        self.supported_languages = supported_languages or ["en"]
        self.max_audio_duration_s = max_audio_duration_s
        self.requires_gpu = requires_gpu
        self.model_size_mb = model_size_mb

    def to_dict(self) -> dict[str, bool | list[str] | float | int]:
        """Convert capabilities to dictionary for serialization."""
        return {
            "supports_streaming": self.supports_streaming,
            "supports_timestamps": self.supports_timestamps,
            "supports_language_detection": self.supports_language_detection,
            "supported_languages": self.supported_languages,
            "max_audio_duration_s": self.max_audio_duration_s,
            "requires_gpu": self.requires_gpu,
            "model_size_mb": self.model_size_mb,
        }
