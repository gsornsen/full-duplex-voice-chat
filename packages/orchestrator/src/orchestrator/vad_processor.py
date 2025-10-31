"""VAD audio processor for real-time barge-in detection.

Integrates VAD with audio resampling to process incoming 48kHz audio
and detect speech events for barge-in functionality.
"""

import logging
import struct
import time
from collections.abc import Callable

import numpy as np

from orchestrator.audio.buffer import RMSBuffer
from orchestrator.audio.resampler import AudioResampler, create_vad_resampler
from orchestrator.config import VADConfig
from orchestrator.session import SessionState
from orchestrator.vad import VADProcessor

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
            processor.process_frame(audio_frame, session_state=SessionState.SPEAKING)
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

        # Adaptive noise gate components
        self._noise_gate_enabled = config.noise_gate.enabled
        self._noise_gate_config = config.noise_gate
        self._rms_buffer = RMSBuffer(size=config.noise_gate.window_size)
        self._noise_floor = 0.0
        self._adaptive_threshold = config.noise_gate.min_threshold
        self._frames_since_update = 0

        # State-aware VAD gating components (M10 Polish Task 3)
        self._state_aware_gating = config.state_aware_gating
        self._speaking_multiplier = config.speaking_threshold_multiplier
        self._listening_multiplier = config.listening_threshold_multiplier
        self._barged_in_multiplier = config.barged_in_threshold_multiplier
        self._current_state: SessionState | None = None
        self._current_state_multiplier = self._listening_multiplier

        # Statistics
        self._frames_processed = 0
        self._frames_gated = 0  # Frames blocked by noise gate
        self._frames_state_gated = 0  # Frames blocked by state-aware gating
        self._speech_events = 0
        self._silence_events = 0
        self._last_error_ts: float | None = None

        logger.info(
            f"VAD audio processor initialized: enabled={self._enabled}, "
            f"aggressiveness={config.aggressiveness}, "
            f"min_speech={config.min_speech_duration_ms}ms, "
            f"min_silence={config.min_silence_duration_ms}ms, "
            f"noise_gate_enabled={self._noise_gate_enabled}, "
            f"state_aware_gating={self._state_aware_gating}"
        )

        if self._noise_gate_enabled:
            logger.info(
                f"Adaptive noise gate initialized: "
                f"window={config.noise_gate.window_size} frames, "
                f"percentile={config.noise_gate.percentile}, "
                f"multiplier={config.noise_gate.threshold_multiplier}x, "
                f"min_threshold={config.noise_gate.min_threshold}"
            )

        if self._state_aware_gating:
            logger.info(
                f"State-aware VAD gating initialized: "
                f"speaking_multiplier={self._speaking_multiplier}x, "
                f"listening_multiplier={self._listening_multiplier}x, "
                f"barged_in_multiplier={self._barged_in_multiplier}x"
            )

    def _calculate_rms(self, audio_data: np.ndarray) -> float:
        """Calculate RMS energy of audio frame.

        Args:
            audio_data: Audio samples as numpy array (int16)

        Returns:
            RMS energy (0-32768 range for 16-bit audio)
        """
        return float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))

    def _update_noise_floor(self) -> None:
        """Update noise floor estimate using percentile method.

        Updates adaptive threshold based on percentile of recent RMS values.
        Only updates when buffer is full (enough data for reliable estimation).
        """
        if self._rms_buffer.is_full():
            self._noise_floor = self._rms_buffer.get_percentile(
                self._noise_gate_config.percentile
            )
            self._adaptive_threshold = max(
                self._noise_floor * self._noise_gate_config.threshold_multiplier,
                self._noise_gate_config.min_threshold,
            )
            logger.debug(
                f"[NOISE GATE] Updated: floor={self._noise_floor:.1f}, "
                f"threshold={self._adaptive_threshold:.1f}"
            )

    def _get_state_multiplier(self, session_state: SessionState | None) -> float:
        """Get threshold multiplier for current session state.

        Args:
            session_state: Current session state

        Returns:
            Threshold multiplier for the given state
        """
        if not self._state_aware_gating or session_state is None:
            return 1.0

        # Update cached multiplier only if state changed
        if self._current_state != session_state:
            self._current_state = session_state
            if session_state == SessionState.SPEAKING:
                self._current_state_multiplier = self._speaking_multiplier
            elif session_state == SessionState.BARGED_IN:
                self._current_state_multiplier = self._barged_in_multiplier
            else:
                # LISTENING, WAITING_FOR_INPUT, IDLE, TERMINATED
                self._current_state_multiplier = self._listening_multiplier

        return self._current_state_multiplier

    def process_frame(
        self, frame_48khz: bytes, session_state: SessionState | None = None
    ) -> bool:
        """Process a 48kHz audio frame through VAD pipeline.

        Pipeline stages:
        1. Calculate RMS energy at 48kHz
        2. Apply adaptive noise gate (optional)
        3. Apply state-aware threshold gating (optional)
        4. Resample 48kHz → 16kHz
        5. Process through VAD

        Args:
            frame_48khz: PCM audio frame at 48kHz (1920 bytes for 20ms)
            session_state: Current session state for state-aware gating

        Returns:
            True if frame contains speech (after debouncing), False otherwise

        Raises:
            ValueError: If frame size is invalid
        """
        if not self._enabled or not self._vad or not self._resampler:
            return False

        try:
            # Convert to numpy array for processing
            audio_data_48k = np.frombuffer(frame_48khz, dtype=np.int16)

            # Calculate RMS before resampling
            rms = self._calculate_rms(audio_data_48k)
            is_silent_48k = rms < 100

            logger.info(
                f"[VAD PROCESSOR DEBUG] Incoming frame: "
                f"size={len(frame_48khz)} bytes (48kHz), "
                f"audio_level={rms:.1f}, "
                f"is_silent={is_silent_48k}, "
                f"session_state={session_state.value if session_state else 'None'}"
            )

            # Apply adaptive noise gate BEFORE resampling
            if self._noise_gate_enabled:
                # Update RMS buffer
                self._rms_buffer.push(rms)
                self._frames_since_update += 1

                # Periodic noise floor update
                if self._frames_since_update >= self._noise_gate_config.update_interval_frames:
                    self._update_noise_floor()
                    self._frames_since_update = 0

                # Apply noise gate: block frames below adaptive threshold
                if rms < self._adaptive_threshold:
                    logger.debug(
                        f"[NOISE GATE] Blocked: rms={rms:.1f} < "
                        f"threshold={self._adaptive_threshold:.1f}"
                    )
                    self._frames_gated += 1
                    self._frames_processed += 1
                    return False  # Below noise floor - not speech

            # Apply state-aware threshold gating AFTER noise gate
            if self._state_aware_gating and session_state is not None:
                state_multiplier = self._get_state_multiplier(session_state)

                # Apply state-aware threshold (multiplies the adaptive threshold)
                state_threshold = self._adaptive_threshold * state_multiplier

                if rms < state_threshold:
                    logger.debug(
                        f"[STATE GATING] Blocked: rms={rms:.1f} < "
                        f"state_threshold={state_threshold:.1f} "
                        f"(base={self._adaptive_threshold:.1f} * "
                        f"{state_multiplier:.1f}x for {session_state.value})"
                    )
                    self._frames_state_gated += 1
                    self._frames_processed += 1
                    return False  # Below state-aware threshold - not strong enough speech

            # Resample 48kHz → 16kHz
            frame_16khz = self._resampler.process_frame(frame_48khz)

            # DEBUG: Calculate audio level after resampling
            audio_data_16k = np.frombuffer(frame_16khz, dtype=np.int16)
            audio_level_16k = self._calculate_rms(audio_data_16k)
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
        """Calculate RMS audio level for debugging (legacy method).

        DEPRECATED: Use _calculate_rms() instead.

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

        # Reset noise gate state
        if self._noise_gate_enabled:
            self._rms_buffer.clear()
            self._noise_floor = 0.0
            self._adaptive_threshold = self._noise_gate_config.min_threshold
            self._frames_since_update = 0

        # Reset state-aware gating
        if self._state_aware_gating:
            self._current_state = None
            self._current_state_multiplier = self._listening_multiplier

        self._frames_processed = 0
        self._frames_gated = 0
        self._frames_state_gated = 0
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
    def noise_floor(self) -> float:
        """Get current noise floor estimate.

        Returns:
            Noise floor RMS value (0-32768 range)
        """
        return self._noise_floor

    @property
    def adaptive_threshold(self) -> float:
        """Get current adaptive threshold.

        Returns:
            Adaptive threshold RMS value (0-32768 range)
        """
        return self._adaptive_threshold

    @property
    def current_state_multiplier(self) -> float:
        """Get current state-aware threshold multiplier.

        Returns:
            Current threshold multiplier based on session state
        """
        return self._current_state_multiplier

    @property
    def stats(self) -> dict[str, int | float]:
        """Get processing statistics.

        Returns:
            Dictionary with processing stats:
            - frames_processed: Total frames processed
            - frames_gated: Frames blocked by noise gate
            - frames_state_gated: Frames blocked by state-aware gating
            - gating_ratio: Ratio of noise-gated to total frames
            - state_gating_ratio: Ratio of state-gated to total frames
            - noise_floor: Current noise floor estimate
            - adaptive_threshold: Current adaptive threshold
            - current_state_multiplier: Current state threshold multiplier
            - vad_speech_frames: VAD-detected speech frames
            - vad_silence_frames: VAD-detected silence frames
            - vad_speech_ratio: Ratio of speech to total frames
        """
        base_stats = {
            "frames_processed": self._frames_processed,
            "frames_gated": self._frames_gated,
            "frames_state_gated": self._frames_state_gated,
            "gating_ratio": (
                self._frames_gated / self._frames_processed if self._frames_processed > 0 else 0.0
            ),
            "state_gating_ratio": (
                self._frames_state_gated / self._frames_processed
                if self._frames_processed > 0
                else 0.0
            ),
            "noise_floor": self._noise_floor,
            "adaptive_threshold": self._adaptive_threshold,
            "current_state_multiplier": self._current_state_multiplier,
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
