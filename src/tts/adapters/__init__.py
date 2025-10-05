"""TTS model adapters implementing the unified streaming interface."""

from src.tts.adapters.adapter_mock import AdapterState, MockTTSAdapter

__all__ = ["MockTTSAdapter", "AdapterState"]
