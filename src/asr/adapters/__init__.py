"""ASR adapter implementations.

This package contains concrete implementations of ASR adapters
for different speech recognition engines.

Available adapters:
- WhisperAdapter (M10): OpenAI Whisper-based ASR using faster-whisper
- WhisperXAdapter (M10 enhancement): 4x faster Whisper with CTranslate2 backend

Example:
    ```python
    from src.asr.adapters import WhisperAdapter, WhisperXAdapter

    # Standard Whisper adapter
    adapter = WhisperAdapter(model_size="small", device="cuda")
    await adapter.initialize()
    result = await adapter.transcribe(audio_bytes, sample_rate=16000)

    # WhisperX adapter (4x faster)
    adapter_x = WhisperXAdapter(model_size="small", device="auto")
    await adapter_x.initialize()
    result = await adapter_x.transcribe(audio_bytes, sample_rate=16000)
    ```
"""

from src.asr.adapters.adapter_whisper import WhisperAdapter
from src.asr.adapters.adapter_whisperx import WhisperXAdapter

__all__ = [
    "WhisperAdapter",
    "WhisperXAdapter",
]
