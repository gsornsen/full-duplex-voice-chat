---
title: "Piper TTS Adapter"
tags: ["piper", "tts", "adapter", "onnx", "cpu", "m5"]
related_files:
  - "src/tts/adapters/adapter_piper.py"
  - "tests/unit/tts/adapters/test_adapter_piper.py"
  - "tests/integration/test_piper*.py"
  - "voicepacks/piper/**/*"
dependencies:
  - ".claude/modules/adapters/template.md"
  - ".claude/modules/features/model-manager.md"
estimated_tokens: 900
priority: "medium"
keywords: ["piper", "ONNX", "CPU baseline", "22050Hz", "voicepack", "edge deployment"]
---

# Piper TTS Adapter

**Last Updated**: 2025-10-17

Piper is a fast, local neural text-to-speech system using ONNX Runtime (CPU baseline adapter for M5).

> 📖 **Quick Summary**: See [CLAUDE.md#important-patterns](../../../CLAUDE.md#important-patterns)

## Overview

**Implementation**: `src/tts/adapters/adapter_piper.py` (M5)

**Purpose**: Fast CPU-only TTS for edge deployment without GPU requirements.

**Key Features**:
- ONNX Runtime for CPU-only inference
- Native sample rate: 22050Hz, resampled to 48kHz
- 20ms frame output with strict repacketization
- PAUSE/RESUME/STOP control (<50ms response time)
- Warmup synthesis (~800ms on modern CPU)

## Performance

| Metric | Target | Actual |
|--------|--------|--------|
| First Audio Latency (FAL) | p95 <500ms | ✅ <450ms |
| Real-Time Factor (RTF) | - | 0.4 (2.5x faster than realtime) |
| Frame jitter | p95 <10ms | ✅ <8ms |
| Control latency | p95 <50ms | ✅ <40ms |
| Warmup time | <1s | ~800ms ✅ |

## Configuration

```yaml
# configs/worker.yaml
adapter:
  type: "piper"
  model_path: "voicepacks/piper/en-us-lessac-medium"
  config:
    sample_rate: 48000  # Output sample rate
    channels: 1

model_manager:
  default_model_id: "piper-en-us-lessac-medium"
```

## Usage

```python
from src.tts.adapters.adapter_piper import PiperTTSAdapter

adapter = PiperTTSAdapter(
    model_id="piper-en-us-lessac-medium",
    model_path="voicepacks/piper/en-us-lessac-medium"
)

# Warmup (~800ms)
await adapter.warm_up()

# Synthesize
async def text_gen():
    yield "Hello, this is Piper TTS speaking."

async for frame in adapter.synthesize_stream(text_gen()):
    # Process 20ms PCM frame at 48kHz
    # frame.audio = 960 samples (1920 bytes)
    play_audio(frame.audio)

# Control
await adapter.pause()   # <50ms response
await adapter.resume()
await adapter.stop()
```

## Voicepack Structure

```
voicepacks/piper/en-us-lessac-medium/
├── en-us-lessac-medium.onnx        # ONNX model
├── en-us-lessac-medium.onnx.json   # Config file
└── metadata.yaml                    # tags: lang, cpu_ok, sample_rate
```

**metadata.yaml** example:
```yaml
family: piper
model_id: piper-en-us-lessac-medium
tags:
  lang: en
  cpu_ok: true
  sample_rate: 22050
  streaming: true
```

## Test Coverage

**Total**: 25/25 passing (M5)
- Unit: 25/25 tests (`test_adapter_piper.py`)
  - Initialization and configuration
  - Streaming synthesis
  - Audio resampling (22050Hz → 48kHz)
  - Frame repacketization (20ms frames)
  - Control commands (PAUSE/RESUME/STOP)
  - Empty audio edge case handling
  - Race-condition-free pause timing
  - Performance validation

## Implementation Files

- `src/tts/adapters/adapter_piper.py`: Piper adapter (M5)
- `src/tts/audio/resampling.py`: Shared resampling utility (M6)
- `src/tts/audio/framing.py`: Shared framing utility (M6)
- `voicepacks/piper/`: Voicepack directory

## References

- **Adapter Template**: [.claude/modules/adapters/template.md](template.md)
- **Model Manager**: [.claude/modules/features/model-manager.md](../features/model-manager.md)
- **Architecture**: [.claude/modules/architecture.md](../architecture.md)
- **Core Documentation**: [CLAUDE.md](../../../CLAUDE.md)

---

**Last Updated**: 2025-10-17
**Status**: Complete (M5)
