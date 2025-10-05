"""WebSocket message protocol definitions.

Defines Pydantic models for WebSocket message serialization/deserialization.
Messages are JSON-encoded for text-based WebSocket transport.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TextMessage(BaseModel):
    """Client → Server: Text input message.

    Sent by the client to provide text for TTS synthesis.
    """

    type: Literal["text"] = "text"
    text: str = Field(..., min_length=1, description="Text to synthesize")
    is_final: bool = Field(
        default=True, description="Whether this is the final chunk in a sequence"
    )


class AudioMessage(BaseModel):
    """Server → Client: Audio frame message.

    Contains a 20ms PCM audio frame encoded in base64.
    """

    type: Literal["audio"] = "audio"
    pcm: str = Field(..., description="Base64-encoded PCM audio (1920 bytes)")
    sample_rate: int = Field(default=48000, description="Sample rate in Hz")
    frame_ms: int = Field(default=20, description="Frame duration in milliseconds")
    sequence: int = Field(..., ge=1, description="Sequence number for frame ordering")


class SessionStartMessage(BaseModel):
    """Server → Client: Session start notification.

    Sent when a new session is established.
    """

    type: Literal["session_start"] = "session_start"
    session_id: str = Field(..., description="Unique session identifier")


class SessionEndMessage(BaseModel):
    """Server → Client: Session end notification.

    Sent when a session terminates (normal or error).
    """

    type: Literal["session_end"] = "session_end"
    session_id: str = Field(..., description="Session identifier")
    reason: str = Field(default="completed", description="Reason for session end")


class ControlMessage(BaseModel):
    """Client → Server: Control command.

    Used for session control (pause, resume, stop).
    """

    type: Literal["control"] = "control"
    command: Literal["PAUSE", "RESUME", "STOP"] = Field(
        ..., description="Control command"
    )


class ErrorMessage(BaseModel):
    """Server → Client: Error notification.

    Sent when an error occurs during the session.
    """

    type: Literal["error"] = "error"
    message: str = Field(..., description="Error description")
    code: str = Field(default="INTERNAL_ERROR", description="Error code")


# Union type for all server → client messages
ServerMessage = (
    AudioMessage | SessionStartMessage | SessionEndMessage | ErrorMessage
)

# Union type for all client → server messages
ClientMessage = TextMessage | ControlMessage
