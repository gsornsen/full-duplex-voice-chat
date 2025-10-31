"""ASR (Automatic Speech Recognition) module.

This module provides a unified interface for speech recognition across
different ASR implementations (Whisper, Vosk, etc.).

Key components:
- ASRAdapterBase: Abstract base class for all ASR adapters
- TranscriptionResult: Dataclass for transcription results
- ASRError hierarchy: Exception types for error handling

Example:
    ```python
    from asr import ASRAdapterBase, TranscriptionResult
    from asr.adapters import WhisperAdapter

    # Create and initialize adapter
    adapter = WhisperAdapter(model_size="small", device="cuda")
    await adapter.initialize()

    # Transcribe audio
    audio_data = load_audio()  # 16-bit PCM, mono
    result = await adapter.transcribe(audio_data, sample_rate=16000)

    print(f"Transcription: {result.text}")
    print(f"Confidence: {result.confidence:.2f}")

    # Cleanup
    await adapter.shutdown()
    ```
"""

from asr.asr_base import (
    ASRAdapterBase,
    ASRCapabilities,
    ASRError,
    ASRLanguage,
    InvalidAudioError,
    ModelNotLoadedError,
    StreamingASRAdapter,
    TranscriptionError,
    TranscriptionResult,
)

__all__ = [
    # Base classes
    "ASRAdapterBase",
    "StreamingASRAdapter",
    # Data types
    "TranscriptionResult",
    "ASRCapabilities",
    "ASRLanguage",
    # Exceptions
    "ASRError",
    "ModelNotLoadedError",
    "InvalidAudioError",
    "TranscriptionError",
]
