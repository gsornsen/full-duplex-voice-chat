---
title: "Automatic Speech Recognition (ASR)"
tags: ["asr", "whisper", "whisperx", "transcription", "orchestrator", "m10"]
related_files:
  - "src/asr/**/*.py"
  - "tests/unit/asr/**/*.py"
  - "tests/integration/test_whisper*.py"
dependencies:
  - ".claude/modules/architecture.md#orchestrator-layer"
  - "docs/WHISPER_ADAPTER.md"
estimated_tokens: 450
priority: "high"
keywords: ["ASR", "automatic speech recognition", "whisper", "whisperx", "transcription", "speech-to-text"]
---

# Automatic Speech Recognition (ASR)

**Last Updated**: 2025-10-17

ASR enables speech-to-text transcription for true speech↔speech conversations using Whisper and WhisperX adapters.

> 📖 **Quick Summary**: See [CLAUDE.md#architecture-summary](../../../CLAUDE.md#architecture-summary)
>
> 📖 **Detailed Guide**: See [docs/WHISPER_ADAPTER.md](../../../docs/WHISPER_ADAPTER.md)

## Overview

**Implementation**: `src/asr/` module (M10)

**Purpose**: Transcribe user speech to text for LLM processing or direct TTS synthesis.

**Adapters**:
- **Whisper**: Standard OpenAI Whisper (CPU/GPU)
- **WhisperX**: 4-8x faster with CTranslate2 optimization

## Quick Start

```yaml
# configs/orchestrator.yaml
asr:
  enabled: true
  adapter: "whisper"  # or "whisperx"
  model_size: "small"  # tiny/base/small/medium/large
  language: "en"
  device: "cpu"  # or "cuda"
  compute_type: "float32"  # or "int8", "float16"
```

```python
from src.asr.adapters.adapter_whisper import WhisperAdapter

adapter = WhisperAdapter(model_size="small", device="cpu")
await adapter.initialize()

result = await adapter.transcribe(audio_bytes, sample_rate=16000)
print(f"Text: {result.text}, Confidence: {result.confidence}")

await adapter.shutdown()
```

## Performance

| Metric | Target | Whisper (CPU) | WhisperX (CPU) | WhisperX (GPU) |
|--------|--------|---------------|----------------|----------------|
| Transcription latency (p95) | <1.5s (CPU), <1.0s (GPU) | 1.2s ✅ | 0.48s ✅ | 0.35s ✅ |
| Real-Time Factor (RTF) | <1.0 (CPU), <0.5 (GPU) | 0.36 ✅ | 0.095 ✅ | 0.048 ✅ |
| Memory usage | <2GB (CPU), <1GB (GPU) | 1.5GB ✅ | 1.2GB ✅ | 920MB ✅ |
| Initialization | <5s | ~3s ✅ | ~4s ✅ | ~2s ✅ |

## Test Coverage

**Total**: 103/103 passing (M10)
- Unit: 64/64 tests (ASR base 23 + audio buffer 41)
- Integration: 39/39 tests (Whisper 28 + performance 11)

## Implementation Files

- `src/asr/asr_base.py`: ASR adapter protocol
- `src/asr/adapters/adapter_whisper.py`: Whisper adapter
- `src/asr/adapters/adapter_whisperx.py`: WhisperX adapter (4-8x faster)
- `src/orchestrator/audio/buffer.py`: Audio buffering for ASR
- `docs/WHISPER_ADAPTER.md`: Complete usage guide

## References

- **Whisper Guide**: [docs/WHISPER_ADAPTER.md](../../../docs/WHISPER_ADAPTER.md)
- **Architecture**: [.claude/modules/architecture.md](../architecture.md)
- **Core Documentation**: [CLAUDE.md](../../../CLAUDE.md)

---

**Last Updated**: 2025-10-17
**Status**: Complete (M10)
