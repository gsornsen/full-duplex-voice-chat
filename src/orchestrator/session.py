"""Transport-agnostic session management.

Manages the lifecycle and state of individual client sessions, handling
audio buffering, text queuing, and metrics collection independently of
the underlying transport mechanism.
"""

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from src.orchestrator.transport.base import TransportSession


class SessionState(Enum):
    """Session state machine states."""

    IDLE = "idle"  # Initial state, no activity
    LISTENING = "listening"  # Waiting for user speech
    SPEAKING = "speaking"  # Playing TTS audio
    BARGED_IN = "barged_in"  # User interrupted during playback (M3)
    TERMINATED = "terminated"  # Session ended


@dataclass
class SessionMetrics:
    """Session performance and activity metrics."""

    # Timing metrics
    first_audio_latency_ms: float | None = None  # Text received → first audio frame
    frame_count: int = 0  # Total audio frames sent
    text_chunks_received: int = 0  # Total text chunks received

    # Timing tracking
    last_text_received_ts: float | None = None  # Timestamp of last text chunk
    first_audio_sent_ts: float | None = None  # Timestamp of first audio frame
    session_start_ts: float = field(default_factory=time.monotonic)
    session_end_ts: float | None = None

    # Frame timing (for jitter analysis)
    frame_send_times: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    # Future: barge-in metrics (M3+)
    barge_in_count: int = 0
    barge_in_latencies_ms: list[float] = field(default_factory=list)

    def record_text_received(self) -> None:
        """Record that a text chunk was received."""
        self.text_chunks_received += 1
        self.last_text_received_ts = time.monotonic()

    def record_audio_sent(self) -> None:
        """Record that an audio frame was sent."""
        self.frame_count += 1
        now = time.monotonic()

        # Track first audio latency
        if self.first_audio_sent_ts is None:
            self.first_audio_sent_ts = now
            if self.last_text_received_ts is not None:
                self.first_audio_latency_ms = (
                    (now - self.last_text_received_ts) * 1000.0
                )

        # Track frame timing for jitter analysis
        self.frame_send_times.append(now)

    def compute_frame_jitter_ms(self) -> float | None:
        """Compute frame timing jitter (standard deviation of inter-frame intervals).

        Returns:
            float: Jitter in milliseconds, or None if insufficient data
        """
        if len(self.frame_send_times) < 2:
            return None

        intervals = [
            (self.frame_send_times[i] - self.frame_send_times[i - 1]) * 1000.0
            for i in range(1, len(self.frame_send_times))
        ]

        # Compute standard deviation
        mean = sum(intervals) / len(intervals)
        variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        return math.sqrt(variance)

    def finalize(self) -> None:
        """Mark session as complete and record end time."""
        self.session_end_ts = time.monotonic()


class SessionManager:
    """Manages a single client session with transport-agnostic logic.

    Handles audio buffering, text queuing, state tracking, and metrics
    collection for a single client session, regardless of transport type.
    """

    def __init__(self, transport_session: TransportSession):
        """Initialize session manager.

        Args:
            transport_session: Underlying transport session
        """
        self.transport = transport_session
        self.state: SessionState = SessionState.IDLE
        self.metrics = SessionMetrics()

        # Audio buffer (jitter buffer)
        # Stores frames waiting to be sent to minimize timing variance
        self.audio_buffer: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)

        # Text queue for incoming text chunks
        self.text_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)

        # Shutdown coordination
        self._shutdown_event = asyncio.Event()

    @property
    def session_id(self) -> str:
        """Get session ID from transport."""
        return self.transport.session_id

    @property
    def is_active(self) -> bool:
        """Check if session is active (connected and not terminated)."""
        return (
            self.transport.is_connected
            and self.state != SessionState.TERMINATED
            and not self._shutdown_event.is_set()
        )

    async def queue_audio_frame(self, frame: bytes) -> None:
        """Add an audio frame to the send buffer.

        Args:
            frame: PCM audio frame (20ms @ 48kHz, 1920 bytes)

        Raises:
            asyncio.QueueFull: If buffer is full (backpressure)
        """
        if not self.is_active:
            return

        try:
            await asyncio.wait_for(self.audio_buffer.put(frame), timeout=1.0)
        except TimeoutError:
            # Log warning about backpressure (future: implement proper handling)
            pass

    async def queue_text(self, text: str) -> None:
        """Add a text chunk to the receive queue.

        Args:
            text: Text chunk from client

        Raises:
            asyncio.QueueFull: If queue is full
        """
        if not self.is_active:
            return

        self.metrics.record_text_received()
        await self.text_queue.put(text)

    async def audio_sender_loop(self) -> None:
        """Continuously send buffered audio frames to client.

        This coroutine runs for the lifetime of the session, sending
        frames from the buffer to the client with proper timing.
        """
        try:
            while self.is_active:
                try:
                    # Get next frame with timeout to allow shutdown checks
                    frame = await asyncio.wait_for(
                        self.audio_buffer.get(), timeout=0.1
                    )

                    # Send to client via transport
                    await self.transport.send_audio_frame(frame)
                    self.metrics.record_audio_sent()

                except TimeoutError:
                    # No frame available, continue to check shutdown
                    continue
                except ConnectionError:
                    # Transport connection lost
                    break

        except asyncio.CancelledError:
            # Clean shutdown
            pass

    async def text_receiver_loop(self) -> None:
        """Continuously receive text from client and queue it.

        This coroutine runs for the lifetime of the session, receiving
        text chunks from the transport and queuing them for processing.
        """
        try:
            async for text in self.transport.receive_text():
                if not self.is_active:
                    break

                await self.queue_text(text)

        except ConnectionError:
            # Transport connection lost
            pass
        except asyncio.CancelledError:
            # Clean shutdown
            pass

    def transition_state(self, new_state: SessionState) -> None:
        """Transition session to a new state.

        Args:
            new_state: Target state
        """
        # old_state = self.state  # TODO: Use when logging is implemented
        self.state = new_state

        # Log state transition (future: add proper logging)
        # logger.info(f"Session {self.session_id}: {self.state.value} → {new_state.value}")

    async def shutdown(self) -> None:
        """Gracefully shut down the session.

        Closes transport, finalizes metrics, and cleans up resources.
        """
        if self._shutdown_event.is_set():
            return

        self._shutdown_event.set()
        self.transition_state(SessionState.TERMINATED)

        # Close transport
        await self.transport.close()

        # Finalize metrics
        self.metrics.finalize()

        # Clear queues
        while not self.audio_buffer.empty():
            try:
                self.audio_buffer.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def get_metrics_summary(self) -> dict[str, str | float | int | None]:
        """Get session metrics summary for logging/monitoring.

        Returns:
            Dictionary of metric names to values
        """
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "first_audio_latency_ms": self.metrics.first_audio_latency_ms,
            "frame_count": self.metrics.frame_count,
            "text_chunks": self.metrics.text_chunks_received,
            "frame_jitter_ms": self.metrics.compute_frame_jitter_ms(),
            "session_duration_s": (
                (self.metrics.session_end_ts or time.monotonic())
                - self.metrics.session_start_ts
            ),
        }
