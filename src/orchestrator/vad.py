"""Voice Activity Detection module."""



class VAD:
    """Voice Activity Detection using webrtcvad."""

    def __init__(self, sample_rate: int = 48000, frame_duration_ms: int = 20) -> None:
        """Initialize VAD.

        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration_ms: Frame duration in milliseconds
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms

    def is_speech(self, audio_frame: bytes) -> bool:
        """Detect if audio frame contains speech.

        Args:
            audio_frame: Raw audio bytes

        Returns:
            True if speech detected, False otherwise
        """
        return False
