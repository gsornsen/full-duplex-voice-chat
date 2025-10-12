"""VAD audio processor for real-time barge-in detection.

Integrates VAD with audio resampling to process incoming 48kHz audio
and detect speech events for barge-in functionality.
"""

import logging
import struct
import time
from collections.abc import Callable

from src.orchestrator.audio.resampler import AudioResampler, create_vad_resampler
from src.orchestrator.config import VADConfig
from src.orchestrator.vad import VADProcessor

logger = logging.getLogger(__name__)


class VADAudioProcessor:
    """Real-time VAD processor with resampling for barge-in detection.

    Combines audio resampling (48kHz → 16kHz) with VAD processing to
    enable real-time speech detection from client audio streams.

    Thread-safety: NOT thread-safe. Use from a single async task.

    Example:
        ```python
        # Create processor
        config = VADConfig(aggressiveness=2, min_speech_duration_ms=100)
        processor = VADAudioProcessor(config)

        # Set event callbacks
        processor.on_speech_start = lambda ts: handle_barge_in()
        processor.on_speech_end = lambda ts: handle_resume()

        # Process incoming audio frames (48kHz, 20ms)
        async for audio_frame in client_audio_stream():
            processor.process_frame(audio_frame)
        ```
    """

    def __init__(
        self,
        config: VADConfig,
        on_speech_start: Callable[[float], None] | None = None,
        on_speech_end: Callable[[float], None] | None = None,
    ) -> None:
        """Initialize VAD audio processor.

        Args:
            config: VAD configuration
            on_speech_start: Optional callback for speech detection events
            on_speech_end: Optional callback for silence detection events
        """
        self._config = config
        self._enabled = config.enabled

        # Create resampler (48kHz → 16kHz for VAD)
        self._resampler: AudioResampler | None = None
        if self._enabled:
            self._resampler = create_vad_resampler()

        # Create VAD processor
        self._vad: VADProcessor | None = None
        if self._enabled:
            self._vad = VADProcessor(
                config=config,
                min_speech_duration_ms=config.min_speech_duration_ms,
                min_silence_duration_ms=config.min_silence_duration_ms,
            )

            # Set event callbacks
            if on_speech_start:
                self._vad.on_speech_start = on_speech_start
            if on_speech_end:
                self._vad.on_speech_end = on_speech_end

        # Statistics
        self._frames_processed = 0
        self._speech_events = 0
        self._silence_events = 0
        self._last_error_ts: float | None = None

        logger.info(
            f"VAD audio processor initialized: enabled={self._enabled}, "
            f"aggressiveness={config.aggressiveness}, "
            f"min_speech={config.min_speech_duration_ms}ms, "
            f"min_silence={config.min_silence_duration_ms}ms"
        )

    def process_frame(self, frame_48khz: bytes) -> bool:
        """Process a 48kHz audio frame through VAD pipeline.

        Args:
            frame_48khz: PCM audio frame at 48kHz (1920 bytes for 20ms)

        Returns:
            True if frame contains speech (after debouncing), False otherwise

        Raises:
            ValueError: If frame size is invalid
        """
        if not self._enabled or not self._vad or not self._resampler:
            return False

        try:
            # DEBUG: Calculate audio level before resampling
            audio_level_48k = self._calculate_audio_level(frame_48khz)
            is_silent_48k = audio_level_48k < 100

            logger.info(
                f"[VAD PROCESSOR DEBUG] Incoming frame: "
                f"size={len(frame_48khz)} bytes (48kHz), "
                f"audio_level={audio_level_48k:.1f}, "
                f"is_silent={is_silent_48k}"
            )

            # Resample 48kHz → 16kHz
            frame_16khz = self._resampler.process_frame(frame_48khz)

            # DEBUG: Calculate audio level after resampling
            audio_level_16k = self._calculate_audio_level(frame_16khz)
            is_silent_16k = audio_level_16k < 100

            logger.info(
                f"[VAD PROCESSOR DEBUG] Resampled frame: "
                f"size={len(frame_16khz)} bytes (16kHz), "
                f"audio_level={audio_level_16k:.1f}, "
                f"is_silent={is_silent_16k}"
            )

            # Process through VAD
            is_speech = self._vad.process_frame(frame_16khz)

            self._frames_processed += 1

            return is_speech

        except ValueError as e:
            # Log frame size errors but don't crash
            now = time.monotonic()
            if self._last_error_ts is None or (now - self._last_error_ts) > 5.0:
                logger.warning(f"VAD frame processing error: {e}")
                self._last_error_ts = now
            return False

        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error in VAD processing: {e}", exc_info=True)
            return False

    def _calculate_audio_level(self, frame: bytes) -> float:
        """Calculate RMS audio level for debugging.

        Args:
            frame: Raw PCM audio frame (16-bit signed int, little endian)

        Returns:
            RMS audio level (0-32768 range for 16-bit audio)
        """
        # Unpack bytes as signed 16-bit integers
        sample_count = len(frame) // 2
        samples = struct.unpack(f"<{sample_count}h", frame)

        # Calculate RMS
        sum_squares = sum(sample * sample for sample in samples)
        rms_value: float = float((sum_squares / sample_count) ** 0.5)
        return rms_value

    def reset(self) -> None:
        """Reset VAD state.

        Clears all VAD state and statistics. Use when starting a new
        audio stream or after barge-in handling.
        """
        if self._vad:
            self._vad.reset()

        self._frames_processed = 0
        self._last_error_ts = None

        logger.debug("VAD audio processor reset")

    @property
    def is_enabled(self) -> bool:
        """Check if VAD processing is enabled."""
        return self._enabled

    @property
    def is_speaking(self) -> bool:
        """Check if currently in speaking state (after debouncing).

        Returns:
            True if speech is currently detected
        """
        if not self._vad:
            return False
        return self._vad.is_speaking

    @property
    def stats(self) -> dict[str, int | float]:
        """Get processing statistics.

        Returns:
            Dictionary with processing stats:
            - frames_processed: Total frames processed
            - vad_speech_frames: VAD-detected speech frames
            - vad_silence_frames: VAD-detected silence frames
            - vad_speech_ratio: Ratio of speech to total frames
        """
        base_stats = {
            "frames_processed": self._frames_processed,
            "vad_speech_frames": 0,
            "vad_silence_frames": 0,
            "vad_speech_ratio": 0.0,
        }

        if self._vad:
            vad_stats = self._vad.stats
            base_stats.update(
                {
                    "vad_speech_frames": vad_stats["speech_frames"],
                    "vad_silence_frames": vad_stats["silence_frames"],
                    "vad_speech_ratio": vad_stats["speech_ratio"],
                }
            )

        return base_stats

    def set_callbacks(
        self,
        on_speech_start: Callable[[float], None] | None = None,
        on_speech_end: Callable[[float], None] | None = None,
    ) -> None:
        """Update VAD event callbacks.

        Args:
            on_speech_start: Callback for speech detection events
            on_speech_end: Callback for silence detection events
        """
        if self._vad:
            if on_speech_start:
                self._vad.on_speech_start = on_speech_start
            if on_speech_end:
                self._vad.on_speech_end = on_speech_end
