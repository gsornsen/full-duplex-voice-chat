---
title: "Voice Activity Detection (VAD)"
tags: ["vad", "barge-in", "noise-gate", "orchestrator", "m3", "m10-polish"]
related_files:
  - "src/orchestrator/vad.py"
  - "src/orchestrator/vad_processor.py"
  - "src/orchestrator/audio/buffer.py"
  - "tests/unit/orchestrator/test_vad.py"
dependencies:
  - ".claude/modules/architecture.md#orchestrator-layer"
estimated_tokens: 1200
priority: "high"
keywords: ["VAD", "voice activity detection", "barge-in", "noise gate", "adaptive threshold", "webrtcvad", "RMS buffer"]
---

# Voice Activity Detection (VAD)

**Last Updated**: 2025-10-17

Voice Activity Detection enables real-time speech detection for barge-in interruption with <50ms latency.

> 📖 **Quick Summary**: See [CLAUDE.md#architecture-summary](../../../CLAUDE.md#architecture-summary)

## Overview

**Implementation**: `src/orchestrator/vad.py` (M3), `src/orchestrator/vad_processor.py` (M10 Polish)

**Purpose**: Detect when user is speaking to enable barge-in (pause TTS playback mid-utterance).

**Key Features**:
- Real-time speech detection using webrtcvad library
- <50ms latency for barge-in detection
- Audio resampling (48kHz → 16kHz for VAD processing)
- **M10 Polish Enhancements**:
  - State-aware VAD gating (threshold multipliers by session state)
  - Adaptive noise gate (percentile-based noise floor estimation)
  - 70-90% reduction in false barge-ins

## Configuration

```yaml
vad:
  enabled: true
  aggressiveness: 2  # 0=least aggressive, 3=most aggressive
  sample_rate: 16000  # Required by webrtcvad (8k, 16k, 32k, 48k)
  frame_duration_ms: 20  # 10, 20, or 30
  min_speech_duration_ms: 100  # Debounce threshold for speech start
  min_silence_duration_ms: 300  # Debounce threshold for speech end

  # M10 Polish: State-aware gating
  state_aware_gating: true
  speaking_threshold_multiplier: 2.0  # Higher threshold during TTS playback
  listening_threshold_multiplier: 1.0  # Normal sensitivity when waiting
  barged_in_threshold_multiplier: 1.2  # Slightly elevated after barge-in

  # M10 Polish: Adaptive noise gate
  noise_gate:
    enabled: true
    window_size: 100  # 2 seconds @ 50fps
    percentile: 0.25  # 25th percentile = noise floor
    threshold_multiplier: 2.5  # 2.5x noise floor
    min_threshold: 200.0  # Absolute minimum RMS threshold
    update_interval_frames: 10  # Update every 200ms
```

## Usage

```python
from src.orchestrator.vad import VADProcessor
from src.orchestrator.vad_processor import VADAudioProcessor
from src.orchestrator.config import VADConfig
from src.orchestrator.session import SessionState

# Basic VAD (M3)
config = VADConfig(aggressiveness=2, sample_rate=16000)
vad = VADProcessor(config, min_speech_duration_ms=100)

# Set event handlers
vad.on_speech_start = lambda ts: handle_speech_start(ts)
vad.on_speech_end = lambda ts: handle_speech_end(ts)

# Process audio frames (16kHz, 16-bit PCM)
is_speech = vad.process_frame(audio_frame)

# Enhanced VAD with noise gate (M10 Polish)
vad_processor = VADAudioProcessor(config)

# Process with session state awareness
is_speech = vad_processor.process_frame(
    audio_frame_48khz,  # 48kHz input
    session_state=SessionState.SPEAKING
)

# Get statistics
stats = vad_processor.stats
# {
#   "frames_processed": 5000,
#   "frames_gated": 2100,  # Noise gate filtering
#   "gating_ratio": 0.42,
#   "noise_floor": 150.3,
#   "adaptive_threshold": 375.75,
#   ...
# }
```

## Architecture

### Audio Processing Pipeline

```
48kHz Audio Frame
    ↓
Calculate RMS Energy (48kHz)
    ↓
[NOISE GATE] Push RMS to buffer
    ↓
[NOISE GATE] Update noise floor (every 10 frames)
    ↓
[NOISE GATE] Apply threshold gate (rms < threshold → block frame)
    ↓
[STATE-AWARE] Apply session state multiplier
    ↓
Resample to 16kHz
    ↓
Process through webrtcvad
    ↓
Debounce (min_speech_duration_ms / min_silence_duration_ms)
    ↓
Fire Event (on_speech_start / on_speech_end)
```

### State-Aware VAD Gating (M10 Polish)

**Problem**: VAD too sensitive during TTS playback, causing false barge-ins from TTS audio leaking into microphone.

**Solution**: Adjust VAD threshold multipliers based on session state.

**Multipliers**:
- **LISTENING**: 1.0x (normal sensitivity)
- **SPEAKING**: 2.0x (reduced sensitivity, prevents false positives)
- **BARGED_IN**: 1.2x (slightly elevated)
- **WAITING_FOR_INPUT**: 1.0x (normal sensitivity)

**Impact**: 70-80% reduction in false positives during TTS playback.

### Adaptive Noise Gate (M10 Polish)

**Problem**: Background noise (fan hum, typing, distant conversation) triggers false VAD events.

**Solution**: Percentile-based noise floor estimation with adaptive threshold.

**How it works**:
1. **Warmup** (2 seconds): Collect RMS energy values to establish baseline
2. **Noise floor estimation**: 25th percentile of recent RMS values
3. **Adaptive threshold**: `max(noise_floor * 2.5, 200.0)`
4. **Filtering**: Block frames with RMS below threshold before VAD processing
5. **Updates**: Recalculate noise floor every 200ms (10 frames)

**Impact**: Additional 30-40% reduction in false positives from background noise.

**Combined Impact**: 70-90% total reduction in false barge-ins.

## Performance Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Barge-in pause latency | p95 <50ms | ✅ <50ms |
| VAD processing latency | p95 <5ms/frame | ✅ <5ms |
| Noise gate overhead | p95 <1ms/frame | ✅ <1ms |
| False positive reduction | 70-90% | ✅ 70-90% |

## Test Coverage

**M3 VAD Tests**: 37/37 passing
- Unit: 29/29 tests (`test_vad.py`)
  - Configuration validation
  - Speech/silence detection
  - Event callbacks
  - Debouncing logic
  - Audio resampling
- Integration: 8/8 tests (`test_vad_integration.py`)
  - Speech detection validation
  - Aggressiveness levels
  - Processing latency measurement

**M10 Polish Tests**: 31/31 passing
- RMS buffer tests (`test_buffer.py`)
- VAD processor tests with noise gate
- State-aware gating validation
- Statistics tracking

## Implementation Files

- `src/orchestrator/vad.py`: Basic VAD processor (M3)
- `src/orchestrator/vad_processor.py`: Enhanced VAD with noise gate (M10 Polish)
- `src/orchestrator/audio/buffer.py`: RMS energy buffer for noise estimation
- `src/orchestrator/audio/resampler.py`: Audio resampling (48kHz ↔ 16kHz)
- `src/orchestrator/config.py`: VADConfig, NoiseGateConfig
- `tests/unit/orchestrator/test_vad.py`: VAD unit tests
- `tests/integration/test_vad_integration.py`: VAD integration tests

## References

- **Architecture**: [.claude/modules/architecture.md](../architecture.md)
- **Session Management**: [.claude/modules/features/session-management.md](session-management.md)
- **Core Documentation**: [CLAUDE.md](../../../CLAUDE.md)

---

**Last Updated**: 2025-10-17
**Status**: Complete (M3 + M10 Polish)
