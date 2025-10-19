"""Piper TTS adapter - CPU-based neural TTS using ONNX Runtime.

This adapter integrates Piper TTS (https://github.com/rhasspy/piper) as the first
real TTS model in the system, proving the adapter abstraction layer and establishing
the baseline for future GPU adapters (M6-M8).

Piper is a fast, CPU-only neural TTS system using ONNX Runtime, making it ideal for
edge deployments and low-latency inference.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from piper import PiperVoice

from src.tts.audio.framing import repacketize_to_20ms
from src.tts.audio.resampling import resample_audio
from src.tts.tts_base import AdapterState

# Constants
TARGET_SAMPLE_RATE_HZ: Final[int] = 48000  # Required output sample rate
FRAME_DURATION_MS: Final[int] = 20  # 20ms frames
SAMPLES_PER_FRAME: Final[int] = 960  # 48000 Hz * 0.020 sec = 960 samples
INTER_FRAME_DELAY_MS: Final[float] = 2.0  # Small delay to simulate streaming
WARMUP_TEXT: Final[str] = "Testing warmup synthesis for model initialization."

logger = logging.getLogger(__name__)


class PiperTTSAdapter:
    """Piper TTS adapter implementing the TTSAdapter protocol.

    This adapter loads Piper ONNX models from voicepacks and provides streaming
    synthesis with 20ms PCM frames at 48kHz. It supports pause/resume/stop control
    commands with <50ms response time.

    Attributes:
        model_id: Identifier for the model instance
        model_path: Path to the voicepack directory
        voice: Loaded Piper voice instance
        native_sample_rate: Sample rate of the Piper model
        state: Current adapter state (IDLE, SYNTHESIZING, PAUSED, STOPPED)
        pause_event: Event for pause/resume signaling
        stop_event: Event for stop signaling
        lock: Async lock for protecting state transitions

    Example:
        >>> adapter = PiperTTSAdapter(
        ...     model_id="piper-en-us-lessac-medium",
        ...     model_path="voicepacks/piper/en-us-lessac-medium"
        ... )
        >>> async def text_gen():
        ...     yield "Hello, world!"
        >>> async for frame in adapter.synthesize_stream(text_gen()):
        ...     # Process 20ms audio frame at 48kHz
        ...     pass
    """

    def __init__(self, model_id: str, model_path: str | Path) -> None:
        """Initialize the Piper adapter.

        Args:
            model_id: Model identifier (e.g., "piper-en-us-lessac-medium")
            model_path: Path to the voicepack directory containing model files

        Raises:
            FileNotFoundError: If model files are missing
            ValueError: If model configuration is invalid
        """
        self.model_id = model_id
        self.model_path = Path(model_path)
        self.state = AdapterState.IDLE
        self.pause_event = asyncio.Event()
        self.pause_event.set()  # Start unpaused
        self.stop_event = asyncio.Event()
        self.lock = asyncio.Lock()

        # Find ONNX model and config files
        onnx_files = list(self.model_path.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX model found in {self.model_path}")

        onnx_path = onnx_files[0]
        config_path = onnx_path.with_suffix(".onnx.json")

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load Piper voice
        logger.info(
            "Loading Piper voice",
            extra={
                "model_id": model_id,
                "onnx_path": str(onnx_path),
                "config_path": str(config_path),
            },
        )

        self.voice = PiperVoice.load(str(onnx_path), str(config_path), use_cuda=False)

        # Read native sample rate from config
        with open(config_path) as f:
            config = json.load(f)
            self.native_sample_rate = config["audio"]["sample_rate"]

        logger.info(
            "PiperTTSAdapter initialized",
            extra={
                "model_id": model_id,
                "native_sample_rate": self.native_sample_rate,
                "target_sample_rate": TARGET_SAMPLE_RATE_HZ,
            },
        )

    async def synthesize_stream(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Generate TTS audio for each text chunk.

        For each text chunk, synthesizes audio using Piper, resamples to 48kHz,
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

                # Synthesize audio using Piper (blocking call, run in executor)
                audio = await asyncio.to_thread(self._synthesize_piper, text)

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

    def _synthesize_piper(self, text: str) -> NDArray[np.int16]:
        """Synthesize audio using Piper (synchronous).

        Args:
            text: Text to synthesize

        Returns:
            Audio samples as int16 numpy array at native sample rate
        """
        # Piper's synthesize() returns an iterator of AudioChunk objects
        audio_chunks: list[NDArray[np.int16]] = []

        for audio_chunk in self.voice.synthesize(text):
            # Extract int16 array from AudioChunk
            audio_chunks.append(audio_chunk.audio_int16_array)

        # Concatenate all chunks
        if not audio_chunks:
            # Return silence if no audio generated
            return np.zeros(0, dtype=np.int16)

        return np.concatenate(audio_chunks)

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
        """Load a specific Piper model.

        Args:
            model_id: Model identifier

        Notes:
            For M5, this is simplified - the model is loaded at initialization.
            Future implementations (M4+) may support dynamic model loading.
        """
        logger.info(
            "Piper load_model called (model already loaded at init)",
            extra={"model_id": model_id, "adapter_model_id": self.model_id},
        )

    async def unload_model(self, model_id: str) -> None:
        """Unload a specific Piper model.

        Args:
            model_id: Model identifier

        Notes:
            For M5, this is simplified - model lifecycle is managed by adapter
            instance lifecycle. Future implementations (M4+) may support dynamic unloading.
        """
        logger.info(
            "Piper unload_model called (model lifecycle managed by instance)",
            extra={"model_id": model_id, "adapter_model_id": self.model_id},
        )

    async def warm_up(self) -> None:
        """Warm up the model by synthesizing a test utterance.

        This method synthesizes a short test sentence to ensure the model
        is fully loaded and cached for faster first-real-synthesis latency.

        Notes:
            Target: <1s warmup time on modern CPU
            Discards output audio, measures duration for telemetry
        """
        import time

        logger.info("Starting warmup synthesis", extra={"model_id": self.model_id})

        start_time = time.perf_counter()

        # Synthesize warmup text
        audio = await asyncio.to_thread(self._synthesize_piper, WARMUP_TEXT)

        # Resample and repacketize (same as normal synthesis)
        if self.native_sample_rate != TARGET_SAMPLE_RATE_HZ:
            audio = await asyncio.to_thread(
                resample_audio, audio, self.native_sample_rate, TARGET_SAMPLE_RATE_HZ
            )

        # Just measure, don't repacketize
        warmup_duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Warmup synthesis complete",
            extra={
                "model_id": self.model_id,
                "warmup_duration_ms": warmup_duration_ms,
                "audio_duration_ms": len(audio) / TARGET_SAMPLE_RATE_HZ * 1000,
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
        """
        async with self.lock:
            self.state = AdapterState.IDLE
            self.pause_event.set()
            self.stop_event.clear()
            logger.info("Adapter reset to initial state", extra={"model_id": self.model_id})
