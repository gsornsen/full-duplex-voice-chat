"""ASR adapter implementations for different speech recognition backends.

This package provides concrete implementations of the ASRAdapterBase interface
for various speech recognition engines.

Available adapters:
- WhisperAdapter: OpenAI Whisper models (CPU/GPU)
- WhisperXAdapter: WhisperX with speaker diarization and VAD

Example:
    ```python
    from asr.adapters import WhisperAdapter, WhisperXAdapter

    # Use basic Whisper
    whisper = WhisperAdapter(model_size="small", device="cuda")
    await whisper.initialize()

    # Or use WhisperX with advanced features
    whisperx = WhisperXAdapter(model_size="small", compute_type="float16")
    await whisperx.initialize()
    ```
"""

from asr.adapters.adapter_whisper import WhisperAdapter
from asr.adapters.adapter_whisperx import WhisperXAdapter

__all__ = [
    "WhisperAdapter",
    "WhisperXAdapter",
]
