"""VAD and barge-in testing utilities.

Provides utilities for testing Voice Activity Detection and barge-in functionality:
- VAD event recording and analysis
- Synthetic audio generation (speech and silence)
- Latency measurement helpers
- Test configuration factories
- Audio frame generation conforming to webrtcvad requirements

These utilities support the M3 barge-in test suite.
"""

import logging
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from orchestrator.config import VADConfig

logger = logging.getLogger(__name__)


# ============================================================================
# VAD Event Recording
# ============================================================================


@dataclass
class VADEvent:
    """Represents a VAD detection event with timing information."""

    event_type: str  # "speech_start" or "speech_end"
    timestamp_ms: float  # VAD-reported timestamp
    wall_clock_time: float  # Actual wall clock time when event fired
    frame_count: int  # Number of frames processed before this event


@dataclass
class VADTestRecorder:
    """Records VAD events for testing and analysis.

    Captures speech_start and speech_end events with precise timing
    information for latency analysis and validation.

    Example:
        ```python
        recorder = VADTestRecorder()
        vad.on_speech_start = recorder.on_speech_start
        vad.on_speech_end = recorder.on_speech_end

        # Process audio...
        # Analyze events
        assert len(recorder.speech_start_events) > 0
        ```
    """

    # Event lists
    speech_start_events: list[VADEvent] = field(default_factory=list)
    speech_end_events: list[VADEvent] = field(default_factory=list)
    all_events: list[VADEvent] = field(default_factory=list)

    # Frame tracking
    frames_processed: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def on_speech_start(self, timestamp_ms: float) -> None:
        """Record a speech start event.

        Args:
            timestamp_ms: VAD-reported timestamp when speech started
        """
        event = VADEvent(
            event_type="speech_start",
            timestamp_ms=timestamp_ms,
            wall_clock_time=time.perf_counter(),
            frame_count=self.frames_processed,
        )
        self.speech_start_events.append(event)
        self.all_events.append(event)
        logger.debug(
            f"VAD Event: speech_start at {timestamp_ms:.1f}ms "
            f"(frame {self.frames_processed})"
        )

    def on_speech_end(self, timestamp_ms: float) -> None:
        """Record a speech end event.

        Args:
            timestamp_ms: VAD-reported timestamp when speech ended
        """
        event = VADEvent(
            event_type="speech_end",
            timestamp_ms=timestamp_ms,
            wall_clock_time=time.perf_counter(),
            frame_count=self.frames_processed,
        )
        self.speech_end_events.append(event)
        self.all_events.append(event)
        logger.debug(
            f"VAD Event: speech_end at {timestamp_ms:.1f}ms "
            f"(frame {self.frames_processed})"
        )

    def increment_frame_count(self) -> None:
        """Increment frame counter. Call after each process_frame()."""
        self.frames_processed += 1

    def reset(self) -> None:
        """Reset all recorded events and counters."""
        self.speech_start_events.clear()
        self.speech_end_events.clear()
        self.all_events.clear()
        self.frames_processed = 0
        self.start_time = time.perf_counter()

    def get_event_count(self) -> dict[str, int]:
        """Get event counts.

        Returns:
            Dictionary with event counts:
            - speech_starts: Number of speech start events
            - speech_ends: Number of speech end events
            - total_events: Total events
        """
        return {
            "speech_starts": len(self.speech_start_events),
            "speech_ends": len(self.speech_end_events),
            "total_events": len(self.all_events),
        }

    def get_event_timings(self) -> dict[str, list[float]]:
        """Get event timing data.

        Returns:
            Dictionary with timing lists:
            - start_timestamps_ms: VAD timestamps for speech starts
            - end_timestamps_ms: VAD timestamps for speech ends
            - start_latencies_ms: Wall clock latencies for starts
            - end_latencies_ms: Wall clock latencies for ends
        """
        return {
            "start_timestamps_ms": [e.timestamp_ms for e in self.speech_start_events],
            "end_timestamps_ms": [e.timestamp_ms for e in self.speech_end_events],
            "start_latencies_ms": [
                (e.wall_clock_time - self.start_time) * 1000
                for e in self.speech_start_events
            ],
            "end_latencies_ms": [
                (e.wall_clock_time - self.start_time) * 1000
                for e in self.speech_end_events
            ],
        }


# ============================================================================
# Barge-in Latency Measurement
# ============================================================================


@dataclass
class BargeInLatencyMeasurement:
    """Records timing for a single barge-in event."""

    speech_detected_ts: float  # When speech was detected by VAD
    pause_sent_ts: float | None = None  # When PAUSE command was sent
    pause_confirmed_ts: float | None = None  # When PAUSE was acknowledged
    audio_stopped_ts: float | None = None  # When audio frames actually stopped

    def get_detection_to_pause_latency_ms(self) -> float | None:
        """Calculate latency from detection to pause sent.

        Returns:
            Latency in milliseconds, or None if pause not sent
        """
        if self.pause_sent_ts is None:
            return None
        return (self.pause_sent_ts - self.speech_detected_ts) * 1000

    def get_pause_to_confirmation_latency_ms(self) -> float | None:
        """Calculate latency from pause sent to confirmation.

        Returns:
            Latency in milliseconds, or None if not confirmed
        """
        if self.pause_sent_ts is None or self.pause_confirmed_ts is None:
            return None
        return (self.pause_confirmed_ts - self.pause_sent_ts) * 1000

    def get_total_barge_in_latency_ms(self) -> float | None:
        """Calculate total barge-in latency (detection to confirmation).

        Returns:
            Total latency in milliseconds, or None if not complete
        """
        if self.pause_confirmed_ts is None:
            return None
        return (self.pause_confirmed_ts - self.speech_detected_ts) * 1000

    def get_audio_stop_latency_ms(self) -> float | None:
        """Calculate latency from detection to audio actually stopping.

        Returns:
            Latency in milliseconds, or None if audio stop not recorded
        """
        if self.audio_stopped_ts is None:
            return None
        return (self.audio_stopped_ts - self.speech_detected_ts) * 1000


def measure_barge_in_latency(
    on_speech_detected: Callable[[], None],
    on_pause_sent: Callable[[], None],
    on_pause_confirmed: Callable[[], None],
) -> BargeInLatencyMeasurement:
    """Create a latency measurement context for barge-in testing.

    Args:
        on_speech_detected: Callback to invoke when speech detected
        on_pause_sent: Callback to invoke when PAUSE sent
        on_pause_confirmed: Callback to invoke when PAUSE confirmed

    Returns:
        BargeInLatencyMeasurement instance for recording timestamps

    Example:
        ```python
        measurement = measure_barge_in_latency(
            on_speech_detected=lambda: vad_detected(),
            on_pause_sent=lambda: send_pause(),
            on_pause_confirmed=lambda: pause_ack(),
        )
        # Use measurement.speech_detected_ts, etc.
        latency = measurement.get_total_barge_in_latency_ms()
        ```
    """
    measurement = BargeInLatencyMeasurement(speech_detected_ts=time.perf_counter())

    # Wrap callbacks to record timestamps
    def wrapped_on_speech_detected() -> None:
        measurement.speech_detected_ts = time.perf_counter()
        on_speech_detected()

    def wrapped_on_pause_sent() -> None:
        measurement.pause_sent_ts = time.perf_counter()
        on_pause_sent()

    def wrapped_on_pause_confirmed() -> None:
        measurement.pause_confirmed_ts = time.perf_counter()
        on_pause_confirmed()

    return measurement


# ============================================================================
# Synthetic Audio Generation
# ============================================================================


def generate_speech_audio(
    duration_ms: int = 20,
    sample_rate: int = 16000,
    amplitude: float = 0.8,
) -> bytes:
    """Generate synthetic speech-like audio for VAD testing.

    Creates multi-frequency audio with noise to simulate speech characteristics
    that will be reliably detected by webrtcvad.

    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz (must be 8000, 16000, 32000, or 48000)
        amplitude: Amplitude scaling (0.0 - 1.0)

    Returns:
        PCM audio data (16-bit signed int, little endian, mono)

    Example:
        ```python
        # Generate 20ms speech frame at 16kHz
        frame = generate_speech_audio(duration_ms=20, sample_rate=16000)
        assert len(frame) == 640  # 320 samples × 2 bytes
        ```
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)

    # Mix multiple frequencies for speech-like characteristics
    # Use fundamental + harmonics typical of human speech (100-300 Hz fundamental)
    audio = (
        0.3 * np.sin(2 * np.pi * 150 * t)  # Fundamental
        + 0.25 * np.sin(2 * np.pi * 300 * t)  # 2nd harmonic
        + 0.2 * np.sin(2 * np.pi * 450 * t)  # 3rd harmonic
        + 0.15 * np.sin(2 * np.pi * 700 * t)  # Formant 1
        + 0.1 * np.sin(2 * np.pi * 1200 * t)  # Formant 2
    )

    # Add noise for realism (helps VAD detect as speech)
    noise = np.random.normal(0, 0.08, num_samples)
    audio += noise

    # Normalize and apply amplitude scaling
    audio = audio / np.max(np.abs(audio))
    audio = audio * amplitude

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)

    result: bytes = audio_int16.tobytes()
    return result


def generate_silence_audio(
    duration_ms: int = 20,
    sample_rate: int = 16000,
    noise_floor: float = 0.0,
) -> bytes:
    """Generate silent audio frames for VAD testing.

    Creates near-zero audio with optional low noise floor.

    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz (must be 8000, 16000, 32000, or 48000)
        noise_floor: Optional noise floor amplitude (0.0 - 0.1)

    Returns:
        PCM audio data (16-bit signed int, little endian, mono)

    Example:
        ```python
        # Generate 20ms silence at 16kHz
        frame = generate_silence_audio(duration_ms=20, sample_rate=16000)
        assert len(frame) == 640  # 320 samples × 2 bytes
        ```
    """
    num_samples = int(sample_rate * duration_ms / 1000)

    if noise_floor > 0:
        # Add very low noise floor
        noise = np.random.normal(0, noise_floor * 100, num_samples).astype(np.int16)
        return noise.tobytes()

    # Pure silence
    return b"\x00\x00" * num_samples


def generate_audio_sequence(
    pattern: list[tuple[str, int]],
    sample_rate: int = 16000,
) -> list[bytes]:
    """Generate a sequence of audio frames following a pattern.

    Args:
        pattern: List of (type, duration_ms) tuples
                 type can be "speech" or "silence"
        sample_rate: Sample rate in Hz

    Returns:
        List of 20ms audio frames

    Example:
        ```python
        # Generate: 100ms speech, 200ms silence, 100ms speech
        frames = generate_audio_sequence([
            ("speech", 100),
            ("silence", 200),
            ("speech", 100),
        ])
        # Returns 20 frames (400ms total @ 20ms per frame)
        ```
    """
    frames = []
    frame_duration_ms = 20

    for audio_type, total_duration_ms in pattern:
        num_frames = total_duration_ms // frame_duration_ms

        for _ in range(num_frames):
            if audio_type == "speech":
                frame = generate_speech_audio(
                    duration_ms=frame_duration_ms,
                    sample_rate=sample_rate,
                )
            elif audio_type == "silence":
                frame = generate_silence_audio(
                    duration_ms=frame_duration_ms,
                    sample_rate=sample_rate,
                )
            else:
                raise ValueError(f"Unknown audio type: {audio_type}")

            frames.append(frame)

    return frames


# ============================================================================
# Test Configuration Factories
# ============================================================================


def create_test_vad_config(
    aggressiveness: int = 2,
    sample_rate: int = 16000,
    frame_duration_ms: int = 20,
    min_speech_duration_ms: float = 100,
    min_silence_duration_ms: float = 300,
) -> VADConfig:
    """Create a VAD configuration for testing.

    Args:
        aggressiveness: VAD aggressiveness (0-3)
        sample_rate: Sample rate in Hz (8000, 16000, 32000, or 48000)
        frame_duration_ms: Frame duration (10, 20, or 30)
        min_speech_duration_ms: Min speech duration for debouncing
        min_silence_duration_ms: Min silence duration for debouncing

    Returns:
        VADConfig instance

    Example:
        ```python
        config = create_test_vad_config(aggressiveness=3)
        vad = VADProcessor(config)
        ```
    """
    return VADConfig(
        enabled=True,
        aggressiveness=aggressiveness,
        sample_rate=sample_rate,
        frame_duration_ms=frame_duration_ms,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )


# ============================================================================
# Frame Validation Utilities
# ============================================================================


def validate_vad_frame_size(frame: bytes, sample_rate: int, frame_duration_ms: int) -> None:
    """Validate that a frame has the correct size for VAD processing.

    Args:
        frame: PCM audio frame
        sample_rate: Expected sample rate
        frame_duration_ms: Expected frame duration

    Raises:
        ValueError: If frame size is incorrect
    """
    expected_samples = sample_rate * frame_duration_ms // 1000
    expected_bytes = expected_samples * 2  # 16-bit PCM

    if len(frame) != expected_bytes:
        raise ValueError(
            f"Invalid VAD frame size: expected {expected_bytes} bytes "
            f"({frame_duration_ms}ms @ {sample_rate}Hz), got {len(frame)} bytes"
        )


def bytes_to_int16_array(data: bytes) -> list[int]:
    """Convert PCM bytes to list of int16 samples.

    Args:
        data: PCM audio bytes (16-bit signed int, little endian)

    Returns:
        List of int16 sample values

    Example:
        ```python
        frame = generate_speech_audio(duration_ms=20)
        samples = bytes_to_int16_array(frame)
        assert len(samples) == 320  # 20ms @ 16kHz
        ```
    """
    sample_count = len(data) // 2
    return list(struct.unpack(f"<{sample_count}h", data))


def int16_array_to_bytes(samples: list[int]) -> bytes:
    """Convert list of int16 samples to PCM bytes.

    Args:
        samples: List of int16 sample values

    Returns:
        PCM audio bytes (16-bit signed int, little endian)

    Example:
        ```python
        samples = [0] * 320  # 320 samples of silence
        frame = int16_array_to_bytes(samples)
        assert len(frame) == 640  # 320 samples × 2 bytes
        ```
    """
    return struct.pack(f"<{len(samples)}h", *samples)


# ============================================================================
# Statistical Analysis Helpers
# ============================================================================


def calculate_p95_latency(latencies_ms: list[float]) -> float:
    """Calculate 95th percentile latency.

    Args:
        latencies_ms: List of latency measurements in milliseconds

    Returns:
        P95 latency in milliseconds

    Raises:
        ValueError: If latencies list is empty
    """
    if not latencies_ms:
        raise ValueError("Cannot calculate P95 on empty latency list")

    sorted_latencies = sorted(latencies_ms)
    p95_index = int(len(sorted_latencies) * 0.95)
    return sorted_latencies[min(p95_index, len(sorted_latencies) - 1)]


def calculate_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    """Calculate comprehensive latency statistics.

    Args:
        latencies_ms: List of latency measurements in milliseconds

    Returns:
        Dictionary with statistics:
        - mean: Mean latency
        - median: Median latency
        - p95: 95th percentile
        - p99: 99th percentile
        - min: Minimum latency
        - max: Maximum latency
        - std: Standard deviation
    """
    if not latencies_ms:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }

    latencies_array = np.array(latencies_ms)
    return {
        "mean": float(np.mean(latencies_array)),
        "median": float(np.median(latencies_array)),
        "p95": float(np.percentile(latencies_array, 95)),
        "p99": float(np.percentile(latencies_array, 99)),
        "min": float(np.min(latencies_array)),
        "max": float(np.max(latencies_array)),
        "std": float(np.std(latencies_array)),
    }
