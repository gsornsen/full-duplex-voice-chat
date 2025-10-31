"""Whisper ASR adapter implementation.

This module provides an ASR adapter for OpenAI's Whisper model using the
faster-whisper library. It implements the ASRAdapterBase protocol for
consistent integration with the orchestrator.

Key features:
- CPU and GPU inference support with auto-selection
- Multiple model sizes (tiny, base, small, medium, large)
- Language auto-detection or explicit specification
- High-quality transcription for duplex voice chat
- Async interface with non-blocking inference

Performance targets (M10):
- GPU (small model): p95 < 1.5s for <30s audio
- CPU (small model): p95 < 5s for <30s audio
"""

import asyncio
import logging
import struct

import numpy as np
from numpy.typing import NDArray

from asr.asr_base import (
    ASRAdapterBase,
    ASRCapabilities,
    InvalidAudioError,
    ModelNotLoadedError,
    TranscriptionError,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_SIZE = "small"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "float32"
BYTES_PER_SAMPLE = 2  # 16-bit PCM
WHISPER_SAMPLE_RATE = 16000  # Whisper requires 16kHz audio


class WhisperAdapter(ASRAdapterBase):
    """Whisper ASR adapter using faster-whisper library.

    Provides high-quality speech recognition for duplex voice chat using
    OpenAI's Whisper model. Supports multiple model sizes and devices.

    Thread-safety: Safe for concurrent transcribe() calls after initialization.

    Example:
        ```python
        # Initialize adapter with auto device selection
        adapter = WhisperAdapter(
            model_size="small",
            device="auto",  # Auto-selects GPU if available, else CPU
            language="en"
        )
        await adapter.initialize()

        # Transcribe audio (16kHz, 16-bit PCM, mono)
        audio_bytes = load_audio()
        result = await adapter.transcribe(audio_bytes, sample_rate=16000)

        print(f"Text: {result.text}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Language: {result.language}")

        # Cleanup
        await adapter.shutdown()
        ```
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = DEFAULT_DEVICE,
        language: str | None = "en",
        compute_type: str = DEFAULT_COMPUTE_TYPE,
        model_path: str | None = None,
    ) -> None:
        """Initialize Whisper adapter.

        Args:
            model_size: Model size (tiny, base, small, medium, large)
            device: Device for inference (cpu, cuda, cuda:0, auto)
            language: Target language (ISO 639-1 code or None for auto-detect)
            compute_type: Compute precision (float32, float16, int8)
            model_path: Optional custom model path (uses default if None)

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate model size
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        if model_size not in valid_sizes:
            raise ValueError(f"Model size must be one of {valid_sizes}, got {model_size}")

        # Validate device
        valid_device_prefixes = ["cpu", "cuda", "auto"]
        if not any(device.startswith(prefix) for prefix in valid_device_prefixes):
            raise ValueError(f"Device must start with 'cpu', 'cuda', or 'auto', got {device}")

        # Validate compute type
        valid_compute_types = ["float32", "float16", "int8"]
        if compute_type not in valid_compute_types:
            raise ValueError(
                f"Compute type must be one of {valid_compute_types}, got {compute_type}"
            )

        self._model_size = model_size
        self._device = device
        self._language = language
        self._compute_type = compute_type
        self._model_path = model_path

        # Model state
        self._model = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self._actual_device = device  # Will be set after initialization

        logger.info(
            f"WhisperAdapter created: model_size={model_size}, "
            f"device={device}, language={language or 'auto'}, "
            f"compute_type={compute_type}"
        )

    async def initialize(self) -> None:
        """Initialize Whisper model and load weights.

        Downloads model if not cached, loads into memory, and performs
        warmup inference to prepare for transcription.

        The initialization process:
        1. Auto-selects device (GPU if available, else CPU) if device="auto"
        2. Loads model with faster-whisper
        3. Performs warmup inference (~300ms silence)

        Raises:
            ASRError: If model loading fails
            RuntimeError: If already initialized
        """
        if self._initialized:
            raise RuntimeError("Adapter already initialized")

        logger.info(f"Initializing Whisper model: {self._model_size} on {self._device}")

        try:
            # Import faster-whisper (lazy import for better startup)
            from faster_whisper import WhisperModel

            # Determine actual device
            device = self._device
            if device == "auto":
                # Try to detect CUDA availability
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"Auto-detected device: {device}")
                except ImportError:
                    device = "cpu"
                    logger.info("PyTorch not available, using CPU")

            self._actual_device = device

            # Load model (will download if needed)
            model_name = self._model_path or self._model_size

            logger.info(
                f"Loading Whisper model: {model_name} "
                f"(device={device}, compute_type={self._compute_type})"
            )

            # Run model loading in thread pool to avoid blocking event loop
            self._model = await asyncio.get_event_loop().run_in_executor(  # type: ignore[func-returns-value]
                None,
                lambda: WhisperModel(
                    model_name,
                    device=device,
                    compute_type=self._compute_type,
                ),
            )

            self._initialized = True
            logger.info(
                f"Whisper model loaded successfully: {self._model_size} "
                f"({device}, {self._compute_type})"
            )

            # Perform warmup inference (300ms of silence)
            await self._warmup()

        except ImportError as e:
            raise TranscriptionError(
                "Failed to import faster-whisper. Install with: pip install faster-whisper"
            ) from e
        except Exception as e:
            raise TranscriptionError(f"Failed to initialize Whisper model: {e}") from e

    async def _warmup(self) -> None:
        """Perform warmup inference to prepare model.

        Transcribes a short silence clip to ensure model is fully loaded
        and ready for low-latency inference.
        """
        if not self._model:
            return

        logger.debug("Performing Whisper warmup inference...")

        try:
            # Create 300ms of silence (4800 samples at 16kHz)
            warmup_samples = int(WHISPER_SAMPLE_RATE * 0.3)
            warmup_audio = np.zeros(warmup_samples, dtype=np.float32)

            # Run warmup transcription
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(
                    self._model.transcribe(
                        warmup_audio,
                        language=self._language,
                        beam_size=5,
                        word_timestamps=False,
                    )
                ),
            )

            logger.info("Whisper warmup complete")

        except Exception as e:
            logger.warning(f"Warmup inference failed (non-fatal): {e}")

    async def transcribe(
        self,
        audio: bytes,
        sample_rate: int,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio bytes to text.

        Args:
            audio: Raw PCM audio bytes (16-bit signed int, little endian, mono)
            sample_rate: Sample rate in Hz (will be resampled to 16kHz if needed)
            language: Optional language override (uses adapter default if None)

        Returns:
            TranscriptionResult with text, confidence, and metadata

        Raises:
            ModelNotLoadedError: If initialize() hasn't been called
            InvalidAudioError: If audio data is malformed or empty
            TranscriptionError: If transcription fails

        Example:
            ```python
            # Transcribe 1 second of 16kHz audio
            audio_bytes = b"\\x00\\x01" * 16000  # 32000 bytes
            result = await adapter.transcribe(audio_bytes, 16000)
            print(result.text)
            ```
        """
        if not self._initialized or not self._model:
            raise ModelNotLoadedError("Model not initialized. Call initialize() first.")

        if not audio:
            raise InvalidAudioError("Audio data is empty")

        if len(audio) % BYTES_PER_SAMPLE != 0:
            raise InvalidAudioError(
                f"Audio size must be multiple of {BYTES_PER_SAMPLE} bytes (16-bit samples), "
                f"got {len(audio)} bytes"
            )

        # Use specified language or adapter default
        lang = language or self._language

        try:
            # Convert PCM bytes to numpy float32 array
            audio_array = self._bytes_to_float32(audio, sample_rate)

            # Calculate duration
            duration_ms = int((len(audio) / BYTES_PER_SAMPLE) / sample_rate * 1000)

            logger.debug(
                f"Transcribing {duration_ms}ms of audio "
                f"(sample_rate={sample_rate}Hz, language={lang or 'auto'})"
            )

            # Run transcription in thread pool (CPU-intensive operation)
            segments, info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._model.transcribe(
                    audio_array,
                    language=lang,
                    beam_size=5,
                    word_timestamps=False,
                    vad_filter=True,  # Enable VAD to filter silence
                    vad_parameters={
                        "threshold": 0.5,
                        "min_speech_duration_ms": 250,
                        "min_silence_duration_ms": 100,
                    },
                ),
            )

            # Convert generator to list (also in executor to avoid blocking)
            segments_list = await asyncio.get_event_loop().run_in_executor(
                None, lambda: list(segments)
            )

            # Combine all segment texts
            text = " ".join(segment.text.strip() for segment in segments_list)

            # Calculate average confidence
            if segments_list:
                avg_confidence = sum(
                    segment.avg_logprob for segment in segments_list
                ) / len(segments_list)
                # Convert log probability to confidence score (0.0 - 1.0)
                # avg_logprob ranges from ~-1.0 (bad) to 0.0 (perfect)
                confidence = max(0.0, min(1.0, 1.0 + avg_confidence))
            else:
                confidence = 0.0

            # Get detected language
            detected_language = info.language if hasattr(info, "language") else lang

            logger.info(
                f"Transcription complete: '{text[:50]}...' "
                f"(confidence={confidence:.2f}, language={detected_language}, "
                f"duration={duration_ms}ms)"
            )

            return TranscriptionResult(
                text=text,
                confidence=confidence,
                language=detected_language,
                duration_ms=duration_ms,
                word_timestamps=None,  # TODO: Add word timestamps support
                is_partial=False,
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise TranscriptionError(f"Transcription failed: {e}") from e

    async def shutdown(self) -> None:
        """Shutdown adapter and release model resources.

        Unloads model from memory and releases GPU/CPU resources.

        Raises:
            ASRError: If shutdown fails (rare)
        """
        if not self._initialized:
            logger.debug("Adapter not initialized, skipping shutdown")
            return

        logger.info("Shutting down Whisper adapter...")

        async with self._lock:
            try:
                # Release model reference (garbage collection will free memory)
                self._model = None
                self._initialized = False

                logger.info("Whisper adapter shutdown complete")

            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
                raise TranscriptionError(f"Shutdown failed: {e}") from e

    def get_supported_sample_rates(self) -> list[int]:
        """Get list of supported sample rates.

        Whisper natively uses 16kHz, but we support arbitrary rates via
        resampling in the _bytes_to_float32 method.

        Returns:
            List of commonly supported sample rates
        """
        return [8000, 16000, 22050, 44100, 48000]

    def is_initialized(self) -> bool:
        """Check if adapter is initialized and ready.

        Returns:
            True if model is loaded and ready for transcription
        """
        return self._initialized and self._model is not None

    def get_model_info(self) -> dict[str, str | int | float]:
        """Get information about loaded model.

        Returns:
            Dictionary with model metadata

        Raises:
            ModelNotLoadedError: If model not initialized
        """
        if not self._initialized:
            raise ModelNotLoadedError("Model not initialized")

        # Whisper model parameters (approximate)
        model_params = {
            "tiny": 39_000_000,
            "base": 74_000_000,
            "small": 244_000_000,
            "medium": 769_000_000,
            "large": 1_550_000_000,
        }

        return {
            "name": f"whisper-{self._model_size}",
            "version": "faster-whisper",
            "language": self._language or "multilingual",
            "parameters": model_params.get(self._model_size, 0),
            "device": self._actual_device,
            "compute_type": self._compute_type,
        }

    def get_capabilities(self) -> ASRCapabilities:
        """Get adapter capabilities for routing decisions.

        Returns:
            ASRCapabilities describing this adapter's features
        """
        # Whisper model sizes (MB)
        model_sizes_mb = {
            "tiny": 75,
            "base": 145,
            "small": 488,
            "medium": 1540,
            "large": 3100,
        }

        return ASRCapabilities(
            supports_streaming=False,  # Whisper is not streaming (full utterance)
            supports_timestamps=True,  # Can provide word timestamps (not implemented yet)
            supports_language_detection=True,  # Whisper auto-detects language
            supported_languages=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "nl",
                "pl",
                "ru",
                "zh",
                "ja",
                "ko",
            ],
            max_audio_duration_s=30.0,  # Recommended max for real-time
            requires_gpu=self._actual_device.startswith("cuda"),
            model_size_mb=model_sizes_mb.get(self._model_size, 0),
        )

    def _bytes_to_float32(self, audio: bytes, sample_rate: int) -> NDArray[np.float32]:
        """Convert PCM bytes to float32 numpy array for Whisper.

        Whisper expects float32 audio normalized to [-1.0, 1.0] at 16kHz.
        This method converts 16-bit PCM to float32 and resamples if needed.

        Args:
            audio: Raw PCM bytes (16-bit signed int, little endian, mono)
            sample_rate: Input sample rate in Hz

        Returns:
            Float32 numpy array normalized to [-1.0, 1.0] at 16kHz

        Raises:
            InvalidAudioError: If audio data is malformed
        """
        try:
            # Convert bytes to int16 array
            sample_count = len(audio) // BYTES_PER_SAMPLE
            audio_int16: NDArray[np.int16] = np.array(
                struct.unpack(f"<{sample_count}h", audio), dtype=np.int16
            )

            # Convert to float32 and normalize to [-1.0, 1.0]
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # Resample to 16kHz if needed
            if sample_rate != WHISPER_SAMPLE_RATE:
                audio_float32 = self._resample_audio(
                    audio_float32, sample_rate, WHISPER_SAMPLE_RATE
                )

            return audio_float32

        except Exception as e:
            raise InvalidAudioError(f"Failed to convert audio: {e}") from e

    def _resample_audio(
        self,
        audio: NDArray[np.float32],
        source_rate: int,
        target_rate: int,
    ) -> NDArray[np.float32]:
        """Resample audio to target sample rate.

        Uses scipy's high-quality resampling for best audio quality.

        Args:
            audio: Input audio array
            source_rate: Source sample rate in Hz
            target_rate: Target sample rate in Hz

        Returns:
            Resampled audio array
        """
        from scipy import signal

        # Calculate output length
        output_length = int(len(audio) * target_rate / source_rate)

        # Resample using scipy (FFT-based method)
        resampled: NDArray[np.float32] = signal.resample(audio, output_length).astype(
            np.float32
        )

        return resampled

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"WhisperAdapter(model_size={self._model_size}, "
            f"device={self._actual_device}, language={self._language or 'auto'}, "
            f"initialized={self._initialized})"
        )
