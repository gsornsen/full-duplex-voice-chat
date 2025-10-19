"""CosyVoice 2 TTS adapter - GPU-based neural TTS for realtime synthesis.

This adapter integrates CosyVoice 2 (https://github.com/FunAudioLLM/CosyVoice) as the
first GPU-accelerated TTS model in the system (M6). CosyVoice 2 is a high-quality,
streaming-capable neural TTS system optimized for low-latency realtime synthesis.

Key features:
- Zero-shot voice cloning with 3-10s reference audio
- Streaming synthesis with incremental text processing
- GPU acceleration for sub-300ms First Audio Latency (FAL)
- Multi-speaker support with voice embeddings
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray

# Import CosyVoice 2 with graceful fallback for testing
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
except ImportError:
    CosyVoice2 = None  # Graceful degradation for testing

from src.tts.audio.framing import repacketize_to_20ms
from src.tts.audio.resampling import resample_audio
from src.tts.tts_base import AdapterState

# Constants
TARGET_SAMPLE_RATE_HZ: Final[int] = 48000  # Required output sample rate
FRAME_DURATION_MS: Final[int] = 20  # 20ms frames
SAMPLES_PER_FRAME: Final[int] = 960  # 48000 Hz * 0.020 sec = 960 samples
INTER_FRAME_DELAY_MS: Final[float] = 2.0  # Small delay to simulate streaming
WARMUP_TEXT: Final[str] = "Testing warmup synthesis for model initialization."

# CosyVoice 2 specific constants
COSYVOICE_NATIVE_SAMPLE_RATE: Final[int] = 24000  # CosyVoice 2 native rate
COSYVOICE_DEFAULT_SPEAKER: Final[str] = "default"  # Default speaker ID

logger = logging.getLogger(__name__)


class CosyVoiceAdapter:
    """CosyVoice 2 TTS adapter implementing the TTSAdapter protocol.

    This adapter loads CosyVoice 2 models from voicepacks and provides streaming
    synthesis with 20ms PCM frames at 48kHz. It supports pause/resume/stop control
    commands with <50ms response time.

    GPU acceleration enables:
    - First Audio Latency (FAL): p95 < 300ms
    - Streaming synthesis: incremental text processing
    - Zero-shot voice cloning: 3-10s reference audio
    - Multi-speaker support: voice embeddings

    Attributes:
        model_id: Identifier for the model instance
        model_path: Path to the voicepack directory
        device: CUDA device for model inference
        model: Loaded CosyVoice model instance
        native_sample_rate: Sample rate of the CosyVoice model (24000 Hz)
        state: Current adapter state (IDLE, SYNTHESIZING, PAUSED, STOPPED)
        pause_event: Event for pause/resume signaling
        stop_event: Event for stop signaling
        lock: Async lock for protecting state transitions

    Example:
        >>> adapter = CosyVoiceAdapter(
        ...     model_id="cosyvoice2-en-base",
        ...     model_path="voicepacks/cosyvoice/en-base"
        ... )
        >>> async def text_gen():
        ...     yield "Hello, world!"
        >>> async for frame in adapter.synthesize_stream(text_gen()):
        ...     # Process 20ms audio frame at 48kHz
        ...     pass
    """

    def __init__(self, model_id: str, model_path: str | Path) -> None:
        """Initialize the CosyVoice adapter.

        Args:
            model_id: Model identifier (e.g., "cosyvoice2-en-base")
            model_path: Path to the voicepack directory containing model files

        Raises:
            FileNotFoundError: If model files are missing
            ValueError: If model configuration is invalid
            ImportError: If CosyVoice package is not installed
        """
        self.model_id = model_id
        self.model_path = Path(model_path)
        self.state = AdapterState.IDLE
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Start unpaused
        self.stop_event = asyncio.Event()
        self.lock = asyncio.Lock()

        # Check CUDA availability (warning only, not fatal)
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA is not available - CosyVoice requires GPU acceleration",
                extra={"model_id": model_id},
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            "Initializing CosyVoice adapter",
            extra={
                "model_id": model_id,
                "model_path": str(self.model_path),
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
                ),
            },
        )

        # Load CosyVoice 2 model
        if CosyVoice2 is None:
            logger.error(
                "CosyVoice2 package not installed - cannot load model",
                extra={"model_id": model_id},
            )
            raise ImportError(
                "CosyVoice2 package not installed. "
                "Install with: pip install cosyvoice"
            )

        try:
            self.model = CosyVoice2(
                str(self.model_path),
                load_jit=False,  # Optional optimization
                load_trt=False,  # Optional TensorRT (GPU-specific)
                load_vllm=False,  # Optional vLLM acceleration
                fp16=True,  # FP16 for faster inference
            )
            logger.info(
                "CosyVoice 2 model loaded successfully",
                extra={"model_id": model_id, "model_path": str(self.model_path)},
            )
        except Exception as e:
            logger.error(
                "Failed to load CosyVoice 2 model",
                extra={
                    "model_id": model_id,
                    "model_path": str(self.model_path),
                    "error": str(e),
                },
            )
            raise

        # CosyVoice 2 has fixed 24kHz sample rate
        self.native_sample_rate = COSYVOICE_NATIVE_SAMPLE_RATE

        logger.info(
            "CosyVoiceAdapter initialized",
            extra={
                "model_id": model_id,
                "native_sample_rate": self.native_sample_rate,
                "target_sample_rate": TARGET_SAMPLE_RATE_HZ,
                "device": str(self.device),
            },
        )

    async def synthesize_stream(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Generate TTS audio for each text chunk.

        For each text chunk, synthesizes audio using CosyVoice 2, resamples to 48kHz,
        and repacketizes into 20ms PCM frames. Yields frames one at a time with
        minimal delay to simulate realistic streaming behavior.

        Args:
            text_chunks: Async iterator of text chunks to synthesize

        Yields:
            20ms PCM audio frames at 48kHz (bytes, int16 little-endian)

        Notes:
            - Respects PAUSE commands (stops yielding frames immediately)
            - Respects RESUME commands (continues yielding frames)
            - Respects STOP commands (terminates streaming)
            - All control commands respond within < 50ms
            - Uses GPU acceleration for synthesis (CUDA)
        """
        async with self.lock:
            self.state = AdapterState.SYNTHESIZING
            logger.info(
                "Starting synthesis stream",
                extra={"state": self.state.value, "model_id": self.model_id},
            )

        try:
            chunk_count = 0
            async for text in text_chunks:
                chunk_count += 1
                logger.debug(
                    "Processing text chunk",
                    extra={
                        "chunk_id": chunk_count,
                        "text_length": len(text),
                        "text_preview": text[:50],
                        "model_id": self.model_id,
                    },
                )

                # Synthesize audio using CosyVoice (blocking call, run in executor)
                audio = await asyncio.to_thread(self._synthesize_cosyvoice, text)

                # Resample to 48kHz if needed
                if self.native_sample_rate != TARGET_SAMPLE_RATE_HZ:
                    audio = await asyncio.to_thread(
                        resample_audio, audio, self.native_sample_rate, TARGET_SAMPLE_RATE_HZ
                    )

                # Repacketize to 20ms frames
                frames = await asyncio.to_thread(
                    repacketize_to_20ms, audio, sample_rate=TARGET_SAMPLE_RATE_HZ
                )

                logger.debug(
                    "Generated frames for chunk",
                    extra={
                        "chunk_id": chunk_count,
                        "frame_count": len(frames),
                        "audio_duration_ms": len(audio) / TARGET_SAMPLE_RATE_HZ * 1000,
                        "model_id": self.model_id,
                    },
                )

                # Yield frames with streaming delay
                for frame_idx, frame in enumerate(frames):
                    # Check if stopped (immediate termination)
                    if self.stop_event.is_set():
                        logger.info(
                            "Synthesis stopped by STOP command",
                            extra={
                                "chunk_id": chunk_count,
                                "frame_idx": frame_idx,
                                "model_id": self.model_id,
                            },
                        )
                        return

                    # Wait if paused (blocks until RESUME)
                    await self.pause_event.wait()

                    # Small delay to simulate streaming behavior
                    await asyncio.sleep(INTER_FRAME_DELAY_MS / 1000.0)

                    # Check again before yielding (race condition fix)
                    if self.stop_event.is_set():
                        return
                    await self.pause_event.wait()

                    yield frame

            logger.info(
                "Synthesis stream completed",
                extra={"total_chunks": chunk_count, "model_id": self.model_id},
            )

        finally:
            async with self.lock:
                # Only reset to IDLE if we weren't stopped
                if self.state != AdapterState.STOPPED:
                    self.state = AdapterState.IDLE
                    logger.info(
                        "Synthesis stream ended",
                        extra={"state": self.state.value, "model_id": self.model_id},
                    )

    def _synthesize_cosyvoice(self, text: str) -> NDArray[np.int16]:
        """Synthesize audio using CosyVoice 2 (synchronous).

        Args:
            text: Text to synthesize

        Returns:
            Audio samples as int16 numpy array at native sample rate (24kHz)

        Raises:
            RuntimeError: If synthesis fails

        Notes:
            This is a blocking call that runs on GPU. It should be called via
            asyncio.to_thread() to avoid blocking the event loop.

            For M6, uses batch mode with default voice. Future enhancements:
            - Voice cloning with reference audio
            - True streaming mode (stream=True)
            - Multi-speaker support
        """
        if self.model is None:
            logger.error(
                "Cannot synthesize - model not loaded",
                extra={"model_id": self.model_id},
            )
            raise RuntimeError("Model not loaded")

        try:
            with torch.no_grad():
                # CosyVoice 2 streaming API
                # inference_zero_shot() returns a generator of audio chunks
                audio_chunks: list[torch.Tensor] = []

                # For M6: Use batch mode with default voice (no reference audio)
                # Empty prompt_text and silence reference audio for default voice
                for chunk in self.model.inference_zero_shot(
                    tts_text=text,
                    prompt_text="",  # Empty for default voice
                    prompt_speech_16k=np.zeros(16000, dtype=np.float32),  # Silence ref
                    stream=False,  # Use batch mode for simplicity (M6)
                    speed=1.0,
                ):
                    # Extract audio tensor from chunk dict
                    # Expected format: {'tts_speech': torch.Tensor}
                    audio_tensor = chunk["tts_speech"]  # Shape: (1, samples)
                    audio_chunks.append(audio_tensor)

                # Concatenate chunks
                if not audio_chunks:
                    logger.warning(
                        "No audio generated for text",
                        extra={"model_id": self.model_id, "text_length": len(text)},
                    )
                    return np.zeros(0, dtype=np.int16)

                # Concatenate along sample dimension: (1, total_samples)
                audio_tensor = torch.cat(audio_chunks, dim=1)

                # Convert to numpy: (1, samples) -> (samples,)
                audio_float = audio_tensor.squeeze(0).cpu().numpy()

                # Convert float32 [-1, 1] to int16 [-32768, 32767]
                audio_int16 = (audio_float * 32767).clip(-32768, 32767).astype(np.int16)

                logger.debug(
                    "CosyVoice synthesis complete",
                    extra={
                        "model_id": self.model_id,
                        "text_length": len(text),
                        "audio_samples": len(audio_int16),
                        "audio_duration_ms": len(audio_int16) / self.native_sample_rate * 1000,
                    },
                )

                return audio_int16

        except Exception as e:
            logger.error(
                "CosyVoice synthesis failed",
                extra={
                    "model_id": self.model_id,
                    "text_length": len(text),
                    "error": str(e),
                },
            )
            raise RuntimeError(f"CosyVoice synthesis failed: {e}") from e

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
                        extra={
                            "command": command,
                            "previous_state": previous_state.value,
                            "model_id": self.model_id,
                        },
                    )
                else:
                    logger.warning(
                        "PAUSE command ignored (not synthesizing)",
                        extra={"current_state": self.state.value, "model_id": self.model_id},
                    )

            elif command == "RESUME":
                if self.state == AdapterState.PAUSED:
                    self.state = AdapterState.SYNTHESIZING
                    self.pause_event.set()  # Unblock synthesize_stream
                    logger.info(
                        "Adapter resumed",
                        extra={
                            "command": command,
                            "previous_state": previous_state.value,
                            "model_id": self.model_id,
                        },
                    )
                else:
                    logger.warning(
                        "RESUME command ignored (not paused)",
                        extra={"current_state": self.state.value, "model_id": self.model_id},
                    )

            elif command == "STOP":
                self.state = AdapterState.STOPPED
                self.stop_event.set()
                self.pause_event.set()  # Unblock if paused
                logger.info(
                    "Adapter stopped",
                    extra={
                        "command": command,
                        "previous_state": previous_state.value,
                        "model_id": self.model_id,
                    },
                )

            else:
                logger.error(
                    "Unknown control command",
                    extra={"command": command, "model_id": self.model_id},
                )
                raise ValueError(f"Unknown control command: {command}")

    async def load_model(self, model_id: str) -> None:
        """Load a specific CosyVoice model.

        Args:
            model_id: Model identifier

        Notes:
            For M6, this is simplified - the model is loaded at initialization.
            Future implementations may support dynamic model loading for the
            Model Manager (M4) to manage multiple models with TTL-based eviction.

            GPU memory management:
            - Model size: ~1-2GB VRAM (typical for CosyVoice)
            - Consider unloading unused models to free VRAM
            - Use torch.cuda.empty_cache() after unloading
        """
        logger.info(
            "CosyVoice load_model called (model already loaded at init)",
            extra={"model_id": model_id, "adapter_model_id": self.model_id},
        )

        # TODO: Implement dynamic model loading for M4+ if needed
        # if self.model is None:
        #     self.model = CosyVoice2(
        #         str(self.model_path),
        #         load_jit=False,
        #         load_trt=False,
        #         load_vllm=False,
        #         fp16=True,
        #     )

    async def unload_model(self, model_id: str) -> None:
        """Unload a specific CosyVoice model.

        Args:
            model_id: Model identifier

        Notes:
            For M6, this is simplified - model lifecycle is managed by adapter
            instance lifecycle. Future implementations may support dynamic unloading
            for GPU memory management (M4+).

            GPU memory cleanup:
            - Delete model reference
            - Call torch.cuda.empty_cache() to free VRAM
            - Monitor VRAM usage with torch.cuda.memory_allocated()
        """
        logger.info(
            "CosyVoice unload_model called (model lifecycle managed by instance)",
            extra={"model_id": model_id, "adapter_model_id": self.model_id},
        )

        # TODO: Implement dynamic model unloading for M4+ if needed
        # if self.model is not None:
        #     del self.model
        #     self.model = None
        #     torch.cuda.empty_cache()

    async def warm_up(self) -> None:
        """Warm up the model by synthesizing a test utterance.

        This method synthesizes a short test sentence to ensure the model
        is fully loaded, cached, and GPU kernels are compiled for faster
        first-real-synthesis latency.

        Notes:
            Target: <1s warmup time on modern GPU (RTX 3090/4090, A100)
            Discards output audio, measures duration for telemetry
            GPU-specific optimizations:
            - Compile CUDA kernels on first run
            - Cache embeddings for default speaker
            - Allocate GPU memory upfront
        """
        import time

        logger.info("Starting warmup synthesis", extra={"model_id": self.model_id})

        start_time = time.perf_counter()

        # Synthesize warmup text
        audio = await asyncio.to_thread(self._synthesize_cosyvoice, WARMUP_TEXT)

        # Resample and repacketize (same as normal synthesis)
        if self.native_sample_rate != TARGET_SAMPLE_RATE_HZ:
            audio = await asyncio.to_thread(
                resample_audio, audio, self.native_sample_rate, TARGET_SAMPLE_RATE_HZ
            )

        # Just measure, don't repacketize
        warmup_duration_ms = (time.perf_counter() - start_time) * 1000

        # Log GPU memory usage
        if torch.cuda.is_available():
            gpu_memory_allocated_mb = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            gpu_memory_reserved_mb = torch.cuda.memory_reserved(self.device) / 1024 / 1024
        else:
            gpu_memory_allocated_mb = 0.0
            gpu_memory_reserved_mb = 0.0

        logger.info(
            "Warmup synthesis complete",
            extra={
                "model_id": self.model_id,
                "warmup_duration_ms": warmup_duration_ms,
                "audio_duration_ms": len(audio) / TARGET_SAMPLE_RATE_HZ * 1000,
                "gpu_memory_allocated_mb": gpu_memory_allocated_mb,
                "gpu_memory_reserved_mb": gpu_memory_reserved_mb,
            },
        )

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
            Does NOT clear GPU memory or reload the model.
        """
        async with self.lock:
            self.state = AdapterState.IDLE
            self.pause_event.set()
            self.stop_event.clear()
            logger.info("Adapter reset to initial state", extra={"model_id": self.model_id})
