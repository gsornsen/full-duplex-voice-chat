"""LiveKit integration module.

Provides LiveKit/WebRTC support for browser-based clients with real-time
audio streaming and text messaging via data channels.
"""

from src.orchestrator.livekit.room_manager import LiveKitRoomManager

__all__ = ["LiveKitRoomManager"]
