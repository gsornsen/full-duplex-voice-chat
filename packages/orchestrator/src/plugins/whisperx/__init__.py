"""WhisperX STT plugin for LiveKit Agents.

This plugin wraps our custom WhisperX adapter to provide a LiveKit Agents-compatible
STT interface. It enables using our highly-optimized WhisperX implementation
(4-8x faster than standard Whisper) within the LiveKit Agents framework.
"""

from .stt import STT

__all__ = ["STT"]
