"""Voice Activity Detection module.

Implements speech/silence detection using webrtcvad library for real-time
audio processing. This is a stub implementation for M2 - full barge-in
integration happens in M3.

Key features:
- Process 20ms frames at 16kHz (webrtcvad requirement)
- Configurable aggressiveness (0-3)
- Event callbacks for speech_start and speech_end
- Debouncing to avoid spurious detections
- Type-safe with strict mypy compliance
"""

import logging
from collections.abc import Callable
from typing import Protocol

import webrtcvad

from src.orchestrator.config import VADConfig

logger = logging.getLogger(__name__)


class VADEventHandler(Protocol):
    """Protocol for VAD event callbacks.

    Implementing classes should provide these methods to receive
    speech detection events.
    """

    def on_speech_start(self, timestamp_ms: float) -> None:
        """Called when speech is detected.

        Args:
            timestamp_ms: Timestamp in milliseconds when speech started
        """
        ...

    def on_speech_end(self, timestamp_ms: float) -> None:
        """Called when silence is detected after speech.

        Args:
            timestamp_ms: Timestamp in milliseconds when speech ended
        """
        ...


class VADProcessor:
    """Voice Activity Detection processor using webrtcvad.

    Processes 20ms audio frames at 16kHz and emits events when speech
    is detected or ends. Includes debouncing to avoid spurious detections.

    Thread-safety: This class is NOT thread-safe. Use from a single thread.

    Example:
        ```python
        config = VADConfig(aggressiveness=2, sample_rate=16000)
        vad = VADProcessor(config)

        # Set event handlers
        vad.on_speech_start = lambda ts: print(f"Speech started at {ts}ms")
        vad.on_speech_end = lambda ts: print(f"Speech ended at {ts}ms")

        # Process frames
        for frame in audio_frames:
            vad.process_frame(frame)
        ```
    """

    def __init__(
        self,
        config: VADConfig,
        min_speech_duration_ms: float = 100,
        min_silence_duration_ms: float = 300,
    ) -> None:
        """Initialize VAD processor.

        Args:
            config: VAD configuration with sample rate and aggressiveness
            min_speech_duration_ms: Minimum speech duration to trigger event (debouncing)
            min_silence_duration_ms: Minimum silence duration to trigger event (debouncing)

        Raises:
            ValueError: If configuration is invalid
        """
        self._config = config
        self._vad = webrtcvad.Vad(config.aggressiveness)
        self._min_speech_duration_ms = min_speech_duration_ms
        self._min_silence_duration_ms = min_silence_duration_ms

        # State tracking
        self._is_speaking = False
        self._speech_start_time_ms: float | None = None
        self._last_speech_time_ms: float | None = None
        self._current_time_ms: float = 0.0

        # Frame tracking
        self._frames_processed = 0
        self._speech_frames = 0
        self._silence_frames = 0

        # Event callbacks (set these to receive events)
        self.on_speech_start: Callable[[float], None] | None = None
        self.on_speech_end: Callable[[float], None] | None = None

        logger.info(
            f"VAD initialized: aggressiveness={config.aggressiveness}, "
            f"sample_rate={config.sample_rate}Hz, "
            f"frame_duration={config.frame_duration_ms}ms, "
            f"debounce: speech={min_speech_duration_ms}ms, "
            f"silence={min_silence_duration_ms}ms"
        )

    def process_frame(self, frame: bytes) -> bool:
        """Process a single audio frame and detect speech.

        Args:
            frame: Raw PCM audio frame (16-bit signed int, little endian)
                   Must be exactly the right size for configured sample rate
                   and frame duration (e.g., 640 bytes for 20ms @ 16kHz)

        Returns:
            True if frame contains speech, False otherwise

        Raises:
            ValueError: If frame size is incorrect
        """
        # Validate frame size
        expected_size = self._calculate_expected_frame_size()
        if len(frame) != expected_size:
            raise ValueError(
                f"Invalid frame size: expected {expected_size} bytes "
                f"({self._config.frame_duration_ms}ms @ {self._config.sample_rate}Hz), "
                f"got {len(frame)} bytes"
            )

        # Detect speech in frame (webrtcvad.is_speech returns bool, type: ignore for mypy)
        is_speech: bool = bool(self._vad.is_speech(frame, self._config.sample_rate))

        # Update frame counters
        self._frames_processed += 1
        if is_speech:
            self._speech_frames += 1
        else:
            self._silence_frames += 1

        # Update state machine
        self._update_state(is_speech)

        # Advance time
        self._current_time_ms += self._config.frame_duration_ms

        return is_speech

    def _update_state(self, is_speech: bool) -> None:
        """Update speech/silence state machine and fire events.

        Args:
            is_speech: True if current frame contains speech
        """
        if is_speech:
            self._last_speech_time_ms = self._current_time_ms

            # Transition: silence → speech
            if not self._is_speaking:
                if self._speech_start_time_ms is None:
                    # First speech frame detected
                    self._speech_start_time_ms = self._current_time_ms
                else:
                    # Check if we've accumulated enough speech
                    speech_duration = (
                        self._current_time_ms - self._speech_start_time_ms
                    )
                    if speech_duration >= self._min_speech_duration_ms:
                        # Confirmed speech start
                        self._is_speaking = True
                        if self.on_speech_start:
                            self.on_speech_start(self._speech_start_time_ms)
                        logger.debug(
                            f"Speech started at {self._speech_start_time_ms:.1f}ms "
                            f"(debounced over {speech_duration:.1f}ms)"
                        )

        else:  # silence
            # If we were tracking potential speech, reset
            if not self._is_speaking and self._speech_start_time_ms is not None:
                logger.debug(
                    f"Speech candidate discarded (too short): "
                    f"{self._current_time_ms - self._speech_start_time_ms:.1f}ms"
                )
                self._speech_start_time_ms = None

            # Transition: speech → silence
            if self._is_speaking and self._last_speech_time_ms is not None:
                silence_duration = self._current_time_ms - self._last_speech_time_ms
                if silence_duration >= self._min_silence_duration_ms:
                    # Confirmed speech end
                    self._is_speaking = False
                    if self.on_speech_end:
                        self.on_speech_end(self._last_speech_time_ms)
                    logger.debug(
                        f"Speech ended at {self._last_speech_time_ms:.1f}ms "
                        f"(debounced over {silence_duration:.1f}ms)"
                    )
                    self._speech_start_time_ms = None

    def _calculate_expected_frame_size(self) -> int:
        """Calculate expected frame size in bytes.

        Returns:
            Expected frame size based on sample rate and frame duration
        """
        samples_per_frame = (
            self._config.sample_rate * self._config.frame_duration_ms // 1000
        )
        bytes_per_sample = 2  # 16-bit PCM
        return samples_per_frame * bytes_per_sample

    def reset(self) -> None:
        """Reset VAD state.

        Clears all state tracking and counters. Useful when starting
        a new audio stream or session.
        """
        self._is_speaking = False
        self._speech_start_time_ms = None
        self._last_speech_time_ms = None
        self._current_time_ms = 0.0
        self._frames_processed = 0
        self._speech_frames = 0
        self._silence_frames = 0
        logger.debug("VAD state reset")

    @property
    def is_speaking(self) -> bool:
        """Check if currently in speaking state.

        Returns:
            True if speech is currently detected (after debouncing)
        """
        return self._is_speaking

    @property
    def stats(self) -> dict[str, int | float]:
        """Get processing statistics.

        Returns:
            Dictionary with processing stats:
            - frames_processed: Total frames processed
            - speech_frames: Frames containing speech
            - silence_frames: Frames containing silence
            - speech_ratio: Ratio of speech frames to total frames
        """
        total = self._frames_processed
        return {
            "frames_processed": total,
            "speech_frames": self._speech_frames,
            "silence_frames": self._silence_frames,
            "speech_ratio": self._speech_frames / total if total > 0 else 0.0,
        }


class DefaultVADEventHandler:
    """Default VAD event handler that logs events.

    This is a simple implementation for testing and debugging.
    In production, you would implement custom handlers that trigger
    barge-in logic, update UI, etc.
    """

    def on_speech_start(self, timestamp_ms: float) -> None:
        """Log when speech starts.

        Args:
            timestamp_ms: Timestamp when speech started
        """
        logger.info(f"[VAD Event] Speech started at {timestamp_ms:.1f}ms")

    def on_speech_end(self, timestamp_ms: float) -> None:
        """Log when speech ends.

        Args:
            timestamp_ms: Timestamp when speech ended
        """
        logger.info(f"[VAD Event] Speech ended at {timestamp_ms:.1f}ms")
