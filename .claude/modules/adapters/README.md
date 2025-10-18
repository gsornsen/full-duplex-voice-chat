---
title: "TTS Adapters Overview"
tags: ["adapters", "tts", "overview", "architecture"]
related_files:
  - "src/tts/adapters/*.py"
  - "voicepacks/**/*"
dependencies:
  - ".claude/modules/architecture.md#tts-worker-layer"
estimated_tokens: 900
priority: "medium"
keywords: ["TTS adapters", "adapter architecture", "voicepack", "streaming ABI"]
---

# TTS Adapters

**Last Updated**: 2025-10-17

This directory contains documentation for TTS adapter implementations and usage.

> 📖 **Quick Summary**: See [CLAUDE.md#architecture-summary](../../../CLAUDE.md#architecture-summary)

## Overview

TTS adapters provide a unified interface for different text-to-speech models while conforming to the streaming ABI defined in `src/rpc/tts.proto`.

**Key Responsibilities**:
- Model-specific inference logic
- Repacketization to 20ms frames at 48kHz
- Native sample rate → 48kHz resampling
- PAUSE/RESUME/STOP control (<50ms response time)

## Available Adapters

| Adapter | Status | Device | Sample Rate | Description |
|---------|--------|--------|-------------|-------------|
| [Mock](../../../src/tts/adapters/adapter_mock.py) | ✅ M1 | CPU | 48kHz | Sine wave generator for testing |
| [Piper](piper.md) | ✅ M5 | CPU | 22050Hz | ONNX CPU baseline |
| CosyVoice2 | 🔄 M6 | GPU | 24kHz | Expressive GPU TTS |
| XTTS-v2 | 📝 M7 | GPU | - | Voice cloning |
| Sesame | 📝 M8 | GPU | - | Base + LoRA variants |

## Adapter Architecture

```
┌─────────────────────────────────────────┐
│ Adapter (implements TTSAdapterBase)    │
├─────────────────────────────────────────┤
│ - Model loading & initialization       │
│ - Inference (model-specific)           │
│ - Audio resampling (native → 48kHz)    │
│ - Frame repacketization (→ 20ms)       │
│ - Control flow (PAUSE/RESUME/STOP)     │
│ - State machine (IDLE/SYNTHESIZING/...)│
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ Shared Audio Utilities (M6+)           │
├─────────────────────────────────────────┤
│ - resampling.py: High-quality resample │
│ - framing.py: Frame duration calc      │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│ gRPC Server (worker.py)                │
├─────────────────────────────────────────┤
│ - Synthesize(stream TextChunk) →       │
│   stream AudioFrame                     │
│ - Control(PAUSE|RESUME|STOP|RELOAD)    │
└─────────────────────────────────────────┘
```

## Voicepack Structure

Models are stored in `voicepacks/<family>/<model_id>/`:

```
voicepacks/
├─ piper/en-us-lessac-medium/         # M5 ✅
│  ├─ en-us-lessac-medium.onnx        # ONNX model
│  ├─ en-us-lessac-medium.onnx.json   # Config
│  └─ metadata.yaml                    # Tags
├─ cosyvoice2/en-base/                 # M6 🔄
│  ├─ model.safetensors
│  ├─ config.json
│  └─ metadata.yaml
└─ xtts-v2/en-demo/                    # M7 📝
   ├─ model.safetensors
   ├─ config.json
   ├─ metadata.yaml
   └─ ref/seed.wav          # Reference audio for cloning
```

**metadata.yaml** format:
```yaml
family: piper
model_id: piper-en-us-lessac-medium
tags:
  lang: en
  cpu_ok: true
  sample_rate: 22050  # Native sample rate
  streaming: true
  zero_shot: false
  expressive: false
```

## Creating a New Adapter

**Quick Start**: Follow the [template guide](template.md)

**Steps**:
1. Create `src/tts/adapters/adapter_<name>.py`
2. Inherit from `TTSAdapterBase`
3. Implement required methods (initialize, synthesize_stream, pause, resume, stop, shutdown)
4. Use shared utilities (`resampling.py`, `framing.py`) for audio processing
5. Add voicepack to `voicepacks/<family>/<model_id>/`
6. Write unit and integration tests
7. Update ModelManager prefix routing

**Example** (minimal adapter):
```python
from src.tts.tts_base import TTSAdapterBase, AdapterState
from src.tts.audio.resampling import resample_audio
from src.tts.audio.framing import frames_for_duration

class MyTTSAdapter(TTSAdapterBase):
    async def initialize(self):
        # Load model from self.model_path
        self.native_sample_rate = 22050

    async def synthesize_stream(self, text_stream):
        # 1. Synthesize with model
        # 2. Resample to 48kHz
        # 3. Repacketize to 20ms frames
        # 4. Respect PAUSE/RESUME/STOP
        # 5. Yield AudioFrame objects
        pass
```

## Performance Requirements

| Metric | Target | Notes |
|--------|--------|-------|
| First Audio Latency (FAL) | p95 <300ms (GPU), <500ms (CPU) | Time to first frame |
| Real-Time Factor (RTF) | <1.0 (realtime or faster) | Synthesis speed |
| Frame jitter | p95 <10ms | Frame timing stability |
| Control latency | p95 <50ms | PAUSE/RESUME/STOP response |
| Warmup time | <1s | Model initialization |

## Testing Requirements

**Unit Tests** (per adapter):
- Initialization and configuration
- Streaming synthesis
- Audio resampling (native → 48kHz)
- Frame repacketization (→ 20ms)
- Control commands (PAUSE/RESUME/STOP)
- Empty audio edge cases
- State machine transitions
- Performance validation

**Integration Tests** (per adapter):
- ModelManager integration
- Voicepack discovery and loading
- End-to-end synthesis pipeline
- Multi-session concurrency

## References

- **Adapter Template**: [template.md](template.md) - Implementation guide
- **Piper Reference**: [piper.md](piper.md) - CPU baseline example
- **Model Manager**: [../features/model-manager.md](../features/model-manager.md)
- **Architecture**: [../architecture.md](../architecture.md)
- **Core Documentation**: [CLAUDE.md](../../../CLAUDE.md)

---

**Last Updated**: 2025-10-17
**Status**: 2 adapters implemented (Mock, Piper), 1 in progress (CosyVoice2), 2 planned (XTTS, Sesame)
