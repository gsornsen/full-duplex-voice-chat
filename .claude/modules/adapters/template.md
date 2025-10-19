---
title: "TTS Adapter Implementation Template"
tags: ["adapter", "template", "tts", "implementation-guide"]
related_files:
  - "src/tts/adapters/*.py"
  - "src/tts/tts_base.py"
  - "src/tts/audio/resampling.py"
  - "src/tts/audio/framing.py"
dependencies: []
estimated_tokens: 1500
priority: "medium"
keywords: ["TTS adapter", "template", "implementation", "synthesize", "resampling", "framing", "control flow"]
---

# TTS Adapter Implementation Template

**Last Updated**: 2025-10-17

This guide provides a template for implementing new TTS adapters that conform to the unified streaming ABI.

> 📖 **Reference Implementations**: See [piper.md](piper.md) (CPU) and [COSYVOICE_PYTORCH_CONFLICT.md](../../../docs/COSYVOICE_PYTORCH_CONFLICT.md) (GPU)

## Adapter Protocol

All adapters must inherit from `TTSAdapterBase` and implement the required methods.

### Required Interface

```python
from src.tts.tts_base import TTSAdapterBase, AdapterState
from typing import AsyncGenerator

class MyTTSAdapter(TTSAdapterBase):
    """
    MyTTS adapter implementation.

    Native sample rate: XXXXX Hz
    Output: 20ms frames at 48kHz (960 samples per frame)
    """

    def __init__(self, model_id: str, model_path: str, **kwargs):
        """
        Initialize adapter.

        Args:
            model_id: Unique model identifier (e.g., "mytts-en-base")
            model_path: Path to voicepack directory
            **kwargs: Adapter-specific configuration
        """
        super().__init__(model_id, model_path)
        # Initialize adapter-specific state

    async def initialize(self) -> None:
        """
        Load model and initialize inference engine.

        Called once during startup or model load.
        """
        # Load model from self.model_path
        # Initialize inference engine
        # Set self.native_sample_rate
        pass

    async def warm_up(self) -> None:
        """
        Warmup model with synthetic utterance (~300ms).

        Improves first-synthesis latency.
        """
        synthetic_text = "Warmup synthesis."
        async for _ in self.synthesize_stream(async_text_gen(synthetic_text)):
            pass  # Discard frames

    async def synthesize_stream(
        self,
        text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[AudioFrame, None]:
        """
        Streaming synthesis: text chunks → 20ms audio frames at 48kHz.

        IMPORTANT Requirements:
        1. Yield exactly 20ms frames (960 samples @ 48kHz)
        2. Respect PAUSE/RESUME/STOP control (<50ms)
        3. Resample from native sample rate to 48kHz
        4. Handle empty audio edge cases

        Args:
            text_stream: Async generator yielding text chunks

        Yields:
            AudioFrame: 20ms PCM frames (960 samples, 48kHz, mono)
        """
        self.state = AdapterState.SYNTHESIZING

        # Accumulate text chunks
        full_text = ""
        async for chunk in text_stream:
            full_text += chunk

        # Model-specific synthesis
        # native_audio = self.model.synthesize(full_text)
        # native_sr = self.native_sample_rate (e.g., 22050 Hz)

        # Resample to 48kHz
        from src.tts.audio.resampling import resample_audio
        audio_48k = resample_audio(
            native_audio,
            orig_sr=self.native_sample_rate,
            target_sr=48000
        )

        # Repacketize to 20ms frames
        from src.tts.audio.framing import frames_for_duration
        frame_size = frames_for_duration(duration_ms=20, sample_rate=48000)

        for i in range(0, len(audio_48k), frame_size):
            # Check control state
            if self.state == AdapterState.STOPPED:
                break

            while self.state == AdapterState.PAUSED:
                await asyncio.sleep(0.01)  # Wait for RESUME

            frame_audio = audio_48k[i:i + frame_size]

            # Pad last frame if needed
            if len(frame_audio) < frame_size:
                frame_audio = np.pad(
                    frame_audio,
                    (0, frame_size - len(frame_audio)),
                    mode='constant'
                )

            yield AudioFrame(
                audio=frame_audio.tobytes(),
                sample_rate=48000,
                channels=1
            )

        self.state = AdapterState.IDLE

    async def pause(self) -> None:
        """Pause synthesis. Must respond within <50ms."""
        self.state = AdapterState.PAUSED

    async def resume(self) -> None:
        """Resume synthesis. Must respond within <50ms."""
        if self.state == AdapterState.PAUSED:
            self.state = AdapterState.SYNTHESIZING

    async def stop(self) -> None:
        """Stop synthesis. Must respond within <50ms."""
        self.state = AdapterState.STOPPED

    async def shutdown(self) -> None:
        """Release resources (model, VRAM, etc.)."""
        # Cleanup model, inference engine
        pass
```

## Implementation Checklist

### 1. Model Loading
- [ ] Load model from `model_path` directory
- [ ] Initialize inference engine (ONNX, PyTorch, etc.)
- [ ] Set `self.native_sample_rate` correctly

### 2. Audio Processing
- [ ] Resample from native sample rate to 48kHz
- [ ] Repacketize to exactly 20ms frames (960 samples)
- [ ] Handle empty audio arrays (avoid ZeroDivisionError)
- [ ] Pad last frame if needed

### 3. Control Flow
- [ ] PAUSE stops new frames within <50ms
- [ ] RESUME continues synthesis
- [ ] STOP terminates synthesis immediately
- [ ] Check control state before each frame yield

### 4. State Machine
- [ ] Use `AdapterState` enum (IDLE/SYNTHESIZING/PAUSED/STOPPED)
- [ ] Transition states correctly during synthesis
- [ ] Handle race conditions (check state before yield)

### 5. Testing
- [ ] Unit tests for initialization, synthesis, control
- [ ] Integration tests with ModelManager
- [ ] Performance validation (FAL, RTF, frame jitter)

## Voicepack Structure

```
voicepacks/my-tts/model-name/
├── model_files/           # Model weights, config
│   ├── model.safetensors
│   └── config.json
├── metadata.yaml          # Model metadata (tags, capabilities)
└── ref/                   # Optional reference audio (for cloning)
    └── seed.wav
```

**metadata.yaml** example:
```yaml
family: my-tts
model_id: mytts-en-base
tags:
  lang: en
  cpu_ok: false  # GPU required
  sample_rate: 24000  # Native sample rate
  streaming: true
  zero_shot: true  # Optional: voice cloning
  expressive: true  # Optional: emotional TTS
```

## Common Patterns

### Shared Utilities (M6+)

Use shared utilities for audio processing:

```python
from src.tts.audio.resampling import resample_audio
from src.tts.audio.framing import frames_for_duration

# Resample
audio_48k = resample_audio(audio, orig_sr=22050, target_sr=48000)

# Calculate frame size
frame_size = frames_for_duration(duration_ms=20, sample_rate=48000)  # 960
```

### Control Flow Pattern

```python
# Inside synthesize_stream loop
for frame_data in model_output:
    # Check STOP
    if self.state == AdapterState.STOPPED:
        break

    # Handle PAUSE (blocking)
    while self.state == AdapterState.PAUSED:
        await asyncio.sleep(0.01)

    # Yield frame
    yield AudioFrame(...)
```

### Empty Audio Handling

```python
# Before resampling
if len(native_audio) == 0:
    return  # No frames to yield

# After resampling
if len(audio_48k) < frame_size:
    audio_48k = np.pad(audio_48k, (0, frame_size - len(audio_48k)))
```

## References

- **Piper Adapter**: [piper.md](piper.md) - CPU baseline example
- **CosyVoice2 Design**: [../../../docs/COSYVOICE2_ADAPTER_DESIGN.md](../../../project_documentation/m6_design/COSYVOICE2_ADAPTER_DESIGN.md) - GPU example
- **tts_base.py**: [../../../src/tts/tts_base.py](../../../src/tts/tts_base.py) - Base protocol
- **Architecture**: [../architecture.md](../architecture.md)

---

**Last Updated**: 2025-10-17
**Status**: Template for M6-M8 adapters
