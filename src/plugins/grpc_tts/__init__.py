"""gRPC TTS plugin for LiveKit Agents.

This plugin wraps our custom gRPC TTS worker (with Piper and future adapters)
to provide a LiveKit Agents-compatible TTS interface.
"""

from .tts import TTS

__all__ = ["TTS"]
