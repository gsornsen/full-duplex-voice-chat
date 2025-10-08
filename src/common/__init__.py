"""Common utilities and type definitions.

This package provides shared types, utilities, and constants used across
the M2 realtime duplex voice demo system.
"""

from src.common.types import (
    AudioFrame,
    AudioFrameStream,
    ModelID,
    SessionID,
    TextChunk,
    TextChunkStream,
    WorkerAddress,
    WorkerCapabilities,
    WorkerMetrics,
)

__all__ = [
    "AudioFrame",
    "AudioFrameStream",
    "ModelID",
    "SessionID",
    "TextChunk",
    "TextChunkStream",
    "WorkerAddress",
    "WorkerCapabilities",
    "WorkerMetrics",
]
