"""Common type aliases for the M2 project.

These aliases improve code readability and type safety across the codebase.
They represent domain concepts and common patterns used throughout the system.

The type system distinguishes between:
- Audio types: Raw PCM data and streaming interfaces
- Text types: Text chunks for TTS synthesis
- Worker types: Capability flags and performance metrics
- Session types: Session identifiers and addresses

Example:
    >>> from shared.types import AudioFrame, WorkerCapabilities
    >>> frame: AudioFrame = b'\\x00' * 1920  # 20ms at 48kHz mono
    >>> caps: WorkerCapabilities = {
    ...     "streaming": True,
    ...     "zero_shot": False,
    ...     "lora": False,
    ...     "cpu_ok": True,
    ...     "languages": ["en"],
    ...     "emotive_zero_prompt": False,
    ... }
"""

from collections.abc import AsyncIterator
from typing import TypedDict

# Audio types
type AudioFrame = bytes
"""20ms PCM audio frame at 48kHz mono (960 samples = 1920 bytes).

Each frame contains exactly 960 samples (20ms at 48kHz) of 16-bit PCM audio
in little-endian format, resulting in 1920 bytes per frame. This frame size
is optimized for low-latency streaming while maintaining audio quality.

Example:
    >>> frame: AudioFrame = b'\\x00' * 1920  # Silence
    >>> len(frame)
    1920
    >>> samples = len(frame) // 2  # 16-bit samples
    >>> duration_ms = (samples / 48000) * 1000
    >>> assert duration_ms == 20.0
"""

type AudioFrameStream = AsyncIterator[AudioFrame]
"""Async stream of audio frames.

An async iterator that yields AudioFrame objects, typically used for
streaming TTS synthesis output from workers to the orchestrator and
ultimately to the client.

Example:
    >>> async def process_audio(stream: AudioFrameStream) -> None:
    ...     async for frame in stream:
    ...         # Process each 20ms frame
    ...         assert len(frame) == 1920
"""


# Text types
type TextChunk = str
"""Text chunk for TTS synthesis.

Variable-length text strings that are synthesized into speech. Chunks can
represent words, sentences, or arbitrary text segments depending on the
synthesis strategy (streaming vs. buffered).

Example:
    >>> chunk: TextChunk = "Hello, world!"
    >>> # Synthesize chunk to audio...
"""

type TextChunkStream = AsyncIterator[str]
"""Async stream of text chunks.

An async iterator that yields text strings for incremental TTS synthesis.
Used when text is generated or received incrementally (e.g., from an LLM
or streamed from a client).

Example:
    >>> async def synthesize(chunks: TextChunkStream) -> AudioFrameStream:
    ...     async for chunk in chunks:
    ...         # Synthesize each chunk to audio frames
    ...         yield audio_frame
"""


class WorkerCapabilities(TypedDict):
    """Worker capability flags.

    Describes the features and capabilities supported by a TTS worker.
    Used for routing decisions and compatibility checks in the orchestrator.

    Attributes:
        streaming: Supports streaming synthesis (vs. batch-only)
        zero_shot: Supports zero-shot voice cloning from reference audio (M7+)
        lora: Supports LoRA adapter loading for voice customization (M8+)
        cpu_ok: Can run on CPU without GPU (affects latency/throughput)
        languages: List of supported language codes (ISO 639-1: "en", "zh", etc.)
        emotive_zero_prompt: Supports emotional synthesis without reference audio (M6+)

    Example:
        >>> caps: WorkerCapabilities = {
        ...     "streaming": True,
        ...     "zero_shot": False,
        ...     "lora": False,
        ...     "cpu_ok": True,
        ...     "languages": ["en"],
        ...     "emotive_zero_prompt": False,
        ... }
        >>> if "en" in caps["languages"] and caps["streaming"]:
        ...     print("Worker supports English streaming synthesis")
    """

    streaming: bool
    zero_shot: bool
    lora: bool
    cpu_ok: bool
    languages: list[str]
    emotive_zero_prompt: bool


class WorkerMetrics(TypedDict):
    """Worker performance metrics.

    Real-time performance indicators used for load balancing and monitoring.
    Metrics are periodically reported by workers and stored in Redis.

    Attributes:
        rtf: Real-time factor (< 1.0 means faster than real-time)
            - 0.5 = synthesizes 2x faster than real-time
            - 1.0 = synthesizes at real-time speed
            - 2.0 = synthesizes 2x slower than real-time (problematic)
        queue_depth: Number of queued synthesis requests waiting for processing
        active_sessions: Number of currently active synthesis sessions
        models_loaded: Number of models currently loaded in VRAM/memory

    Example:
        >>> metrics: WorkerMetrics = {
        ...     "rtf": 0.3,  # Fast synthesis
        ...     "queue_depth": 0.0,  # No queue
        ...     "active_sessions": 2,  # Two concurrent sessions
        ...     "models_loaded": 1,  # One model in VRAM
        ... }
        >>> # Prefer workers with low queue depth and low RTF
        >>> is_available = metrics["queue_depth"] < 5 and metrics["rtf"] < 1.0
    """

    rtf: float
    queue_depth: float
    active_sessions: int
    models_loaded: int


# Session types
type SessionID = str
"""Unique session identifier (UUID v4 format).

UUIDs are used to identify TTS sessions across the orchestrator and workers.
Each client connection may have multiple sessions over its lifetime.

Example:
    >>> from uuid import uuid4
    >>> session_id: SessionID = str(uuid4())
    >>> assert len(session_id) == 36  # Standard UUID format
    >>> # Example: "123e4567-e89b-12d3-a456-426614174000"
"""

type WorkerAddress = str
"""Worker gRPC address in URI format.

Full gRPC endpoint address including protocol, host, and port. Used by
the orchestrator to connect to workers for TTS synthesis requests.

Format: "grpc://host:port" or "grpcs://host:port" (TLS)

Example:
    >>> addr: WorkerAddress = "grpc://localhost:7001"
    >>> # Parse components:
    >>> protocol, rest = addr.split("://")
    >>> host, port_str = rest.split(":")
    >>> assert protocol == "grpc"
    >>> assert host == "localhost"
    >>> assert int(port_str) == 7001
"""

type ModelID = str
"""Model identifier string.

Identifies a specific TTS model in the voicepacks directory structure.
Format is typically "family/variant" or just the model name.

Example:
    >>> model_id: ModelID = "cosyvoice2-en-base"
    >>> # Or with family prefix:
    >>> model_id_full: ModelID = "cosyvoice2/en-base"
    >>> # Maps to: voicepacks/cosyvoice2/en-base/
"""
