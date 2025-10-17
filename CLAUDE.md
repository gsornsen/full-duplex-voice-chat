# CLAUDE.md

**Last Updated**: 2025-10-16

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Realtime Duplex Voice Demo** system enabling low-latency speech↔speech conversations with barge-in support. The system supports hot-swapping across multiple open TTS models (Sesame/Unsloth with LoRA, CosyVoice 2, XTTS-v2, Piper, etc.) and runs on single-GPU and multi-GPU setups.

**Key capabilities:**
- Realtime duplex conversation with barge-in (pause/resume < 50 ms) ✅ Implemented
- Streaming TTS with 20 ms, 48 kHz PCM frames ✅ Implemented
- Model modularity: swap among multiple TTS models via unified streaming ABI ✅ Protocol ready
- Dynamic model lifecycle: default preload, runtime load/unload, TTL-based eviction ✅ Implemented (M4)
- Real TTS adapter: Piper CPU baseline with 22050Hz native, resampled to 48kHz ✅ Implemented (M5)
- ASR integration: Whisper adapter for speech-to-text transcription ✅ Implemented (M10)
- Multi-turn conversations with session timeout management ✅ Implemented (M10 Polish)
- Adaptive noise gating for reduced false barge-ins ✅ Implemented (M10 Polish)
- Scale: single-GPU (two-process), multi-GPU (same host), multi-host (LAN)

**Current Implementation Status**: Milestones M0-M10 complete (including M10 Polish), M11-M13 planned.
See [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md) for detailed status.

## Development Environment

**Python & Tooling:**
- Python 3.13.x managed with **uv** (no Python pin in pyproject.toml; let uv resolve)
- `pyproject.toml` for dependencies; `uv.lock` for locked environment
- `ruff` for linting, `mypy` for type checking (strict mode), `pytest` for tests
- `justfile` for common tasks

**Platform:**
- CUDA Toolkit: 13.0.1 available, but pair **PyTorch 2.7.0** with **CUDA 12.8** prebuilt wheels for stability
- Docker Engine 28.x with NVIDIA container runtime for GPU workers
- Base container: `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`
- **WSL2 Note**: gRPC tests require process isolation (see Testing Strategy section)

**Dependencies:**
Redis for worker service discovery and registry.

## Essential Commands

### Quality & CI
```bash
just lint          # Run ruff linting
just fix           # Auto-fix linting issues
just typecheck     # Run mypy type checking
just test          # Run pytest tests
just ci            # Run all checks (lint + typecheck + test)
```

### Infrastructure
```bash
just redis         # Start Redis container
```

### Code Generation
```bash
just gen-proto     # Generate gRPC stubs from src/rpc/tts.proto
```

### Runtime (Single-GPU)
```bash
# Run TTS worker with mock adapter (M1/M2/M3 - currently implemented)
just run-tts-mock

# Run TTS worker with Piper adapter (M5 - CPU baseline)
just run-tts-piper DEFAULT="piper-en-us-lessac-medium" PRELOAD=""

# Run orchestrator (with VAD barge-in and ASR support)
just run-orch

# Run CLI client
just cli HOST="ws://localhost:8080"

# Note: GPU TTS adapters below are M6-M8 milestones (not yet implemented)
# just run-tts-sesame DEFAULT="cosyvoice2-en-base" PRELOAD=""
# just run-tts-cosy DEFAULT="cosyvoice2-en-base" PRELOAD=""
```

### Profiling
```bash
# CPU profiling with py-spy
just spy-top PID
just spy-record PID OUT="profile.svg"

# GPU profiling
just nsys-tts      # Nsight Systems trace
just ncu-tts       # Nsight Compute kernel analysis
```

### Docker
```bash
docker compose up --build    # Start full stack (redis + livekit + caddy + orchestrator + tts workers)
```

## Architecture Overview

**Two-tier streaming architecture:**

1. **Orchestrator** (LiveKit-based):
   - **Primary Transport**: LiveKit WebRTC for browser clients (full-duplex audio)
   - **Secondary Transport**: WebSocket fallback for CLI testing and simple clients
   - **VAD (M3)**: Voice Activity Detection for barge-in interruption ✅ Implemented
     - Real-time speech detection with <50ms latency
     - Automatic PAUSE/RESUME control flow
     - Configurable aggressiveness and debouncing
     - Audio resampling (48kHz → 16kHz) for VAD processing
     - **M10 Polish Enhancements**:
       - State-aware VAD gating (threshold multipliers by session state)
       - Adaptive noise gate (percentile-based noise floor estimation)
       - 70-90% reduction in false barge-in events
   - **ASR (M10)**: Whisper and WhisperX adapters for speech-to-text ✅ Implemented
     - Two adapters: Standard Whisper and WhisperX (4-8x faster with CTranslate2)
     - Multi-model support (tiny/base/small/medium/large)
     - CPU and GPU inference with auto-optimized compute types (int8/FP16)
     - WhisperX performance: RTF 0.095 (CPU), 0.048 (GPU) - exceeds targets
     - Audio resampling (8kHz-48kHz → 16kHz)
   - **Session Management**: Multi-turn conversation support ✅ Implemented (M10 Polish)
     - State machine: IDLE → LISTENING → SPEAKING → BARGED_IN → WAITING_FOR_INPUT → TERMINATED
     - Configurable idle timeout (default: 5 minutes)
     - Session duration and message count limits
     - Graceful timeout handling with automatic cleanup
   - Routing logic (M9+): capability-aware, prefers resident models, Redis-based discovery (static routing in M2-M5)
   - Lives in `src/orchestrator/`

2. **TTS Workers** (one per GPU/adapter):
   - gRPC server implementing unified streaming ABI ✅ Implemented
   - Model Manager (M4): handles load/unload, TTL eviction, warmup, LRU caching ✅ Implemented
   - Adapters: implement model-specific logic while conforming to shared interface
   - Emit 20 ms, 48 kHz mono PCM frames ✅ Implemented
   - Lives in `src/tts/`

**Key flow (M3 with barge-in):**
- Client sends text → Orchestrator → TTS Worker → Audio frames → Client
- Barge-in: Client speaks → VAD detects speech → sends PAUSE to worker (<50ms) → worker stops emitting frames
- Resume: VAD detects silence → sends RESUME → worker continues

**Current flow (M10 with ASR and multi-turn):**
- Client speaks → Orchestrator (VAD + ASR) → Text transcript → (optional LLM) → TTS Worker → Audio frames → Client
- Session persists between turns (up to idle_timeout_seconds)
- Adaptive noise gate filters background noise before VAD processing
- State-aware VAD adjusts sensitivity during TTS playback

## Code Structure

```
src/
├─ orchestrator/
│  ├─ server.py          # LiveKit Agent + WS fallback, session management
│  ├─ livekit_utils/     # LiveKit integration (agent, transport)
│  ├─ transport/         # WebSocket transport
│  ├─ vad.py             # Voice Activity Detection (M3) ✅ Implemented
│  ├─ vad_processor.py   # VAD audio processing with noise gate (M10 Polish) ✅ Implemented
│  ├─ session.py         # Session state machine with WAITING_FOR_INPUT (M10 Polish) ✅ Implemented
│  ├─ audio/
│  │  ├─ resampler.py    # Audio resampling for VAD (48kHz → 16kHz) ✅ Implemented
│  │  └─ buffer.py       # Audio buffering + RMS energy buffer (M10 Polish) ✅ Implemented
│  ├─ routing.py         # Worker selection logic (M9+ capability-aware)
│  ├─ registry.py        # Redis-based worker discovery
│  └─ config.py          # Configuration loading (SessionConfig, NoiseGateConfig) ✅ Implemented
│
├─ asr/
│  ├─ asr_base.py        # ASR adapter protocol (M10) ✅ Implemented
│  └─ adapters/
│     ├─ adapter_whisper.py   # Whisper ASR adapter (M10) ✅ Implemented
│     └─ adapter_whisperx.py  # WhisperX ASR adapter (M10) ✅ Implemented (4-8x faster)
│
├─ tts/
│  ├─ worker.py          # gRPC server, adapter host, ModelManager integration
│  ├─ model_manager.py   # Model lifecycle (M4): load/unload, TTL eviction, warmup, LRU ✅ Implemented
│  ├─ tts_base.py        # Protocol/interface for all adapters
│  ├─ adapters/          # Model-specific implementations
│  │  ├─ adapter_mock.py           # M1/M2 - Mock adapter (sine wave) ✅ Implemented
│  │  ├─ adapter_piper.py          # M5 - CPU baseline (ONNX) ✅ Implemented
│  │  ├─ adapter_cosyvoice2.py     # M6 - GPU expressive (planned)
│  │  ├─ adapter_xtts.py           # M7 - GPU cloning (planned)
│  │  ├─ adapter_sesame.py         # M8 - Sesame base (planned)
│  │  └─ adapter_unsloth_sesame.py # M8 - LoRA variant (planned)
│  ├─ audio/
│  │  ├─ framing.py      # 20ms framing, resample to 48kHz
│  │  └─ loudness.py     # RMS/LUFS normalization (M6+)
│  └─ utils/
│     ├─ logging.py
│     └─ timers.py
│
├─ rpc/
│  ├─ tts.proto          # gRPC service definition
│  └─ generated/         # Auto-generated gRPC stubs
│
└─ client/
   ├─ cli_client.py      # WebSocket CLI client
   └─ web/               # Browser client (HTML + JS + React components)
```

## gRPC Streaming ABI

All TTS adapters implement the same gRPC interface defined in `src/rpc/tts.proto`:

**Core streaming:**
- `StartSession` / `EndSession`: session lifecycle ✅ Implemented
- `Synthesize(stream TextChunk) → stream AudioFrame`: main streaming path ✅ Implemented
- `Control(PAUSE|RESUME|STOP|RELOAD)`: runtime control ✅ Implemented

**Model lifecycle (M4+):**
- `ListModels`: query available models
- `LoadModel(model_id)`: dynamically load a model
- `UnloadModel(model_id)`: unload when idle
- `GetCapabilities`: report worker capabilities

**Audio format:**
- Output: 20 ms frames, 48 kHz, mono PCM (Opus optional later)
- Adapters must repacketize internal chunk sizes to 20 ms

## Voice Activity Detection (M3)

**Implementation**: `src/orchestrator/vad.py`

**Features**:
- webrtcvad library for real-time speech detection
- Configurable aggressiveness levels (0-3)
- Debouncing for speech start/end events
- Audio resampling pipeline (48kHz → 16kHz)
- Event callbacks for state machine integration
- Telemetry (frames processed, speech ratio, event count)

**Configuration**:
```yaml
vad:
  enabled: true
  aggressiveness: 2  # 0=least aggressive, 3=most aggressive
  sample_rate: 16000  # Required by webrtcvad (8k, 16k, 32k, 48k)
  frame_duration_ms: 20  # 10, 20, or 30
  min_speech_duration_ms: 100  # Debounce threshold for speech start
  min_silence_duration_ms: 300  # Debounce threshold for speech end
```

**Performance Metrics**:
- Processing latency: <5ms per frame (p95)
- Barge-in pause latency: <50ms (p95)
- Test coverage: 29/29 unit tests, 8/8 integration tests

**Usage Pattern**:
```python
from src.orchestrator.vad import VADProcessor
from src.orchestrator.config import VADConfig

config = VADConfig(aggressiveness=2, sample_rate=16000)
vad = VADProcessor(config, min_speech_duration_ms=100)

# Set event handlers
vad.on_speech_start = lambda ts: handle_speech_start(ts)
vad.on_speech_end = lambda ts: handle_speech_end(ts)

# Process audio frames (16kHz, 16-bit PCM)
is_speech = vad.process_frame(audio_frame)
```

## Voice Activity Detection Enhancements (M10 Polish)

**Implementation**: `src/orchestrator/vad_processor.py` ✅ Complete

The M10 Polish release significantly improves VAD accuracy and reduces false positives through three key enhancements:

### 1. Session Idle Timeout & Multi-Turn Conversations

**Problem**: Sessions terminated after single interaction, preventing natural multi-turn conversations.

**Solution**: Added `WAITING_FOR_INPUT` state and configurable timeout management.

**Features**:
- Multi-turn conversation support (sessions persist between interactions)
- Configurable idle timeout (default: 5 minutes)
- Session duration limits (default: 1 hour)
- Message count limits (default: 100 messages)
- Graceful timeout handling with automatic cleanup
- Non-blocking timeout implementation using `asyncio.wait_for()`

**Configuration**:
```yaml
session:
  idle_timeout_seconds: 300  # 5 minutes
  max_session_duration_seconds: 3600  # 1 hour
  max_messages_per_session: 100
```

**Session State Machine**:
```
IDLE → LISTENING → SPEAKING → BARGED_IN
         ↑           ↓            ↓
         ← WAITING_FOR_INPUT ←────┘
                    ↓
               TERMINATED
```

**Usage**:
```python
# Session automatically transitions to WAITING_FOR_INPUT after synthesis
# Waits up to idle_timeout_seconds for next user input
# Gracefully terminates on timeout or session limits reached
```

### 2. State-Aware VAD Intensity Gating

**Problem**: VAD sensitivity too high during TTS playback, causing false barge-ins from TTS audio leaking into microphone.

**Solution**: Adjust VAD threshold multipliers based on session state.

**Features**:
- Higher threshold during SPEAKING state (reduces false positives from TTS audio)
- Normal threshold during LISTENING state (maintains responsiveness)
- Configurable multipliers per session state
- <1ms processing overhead per frame
- Statistics tracking for state gating effectiveness

**Configuration**:
```yaml
vad:
  enabled: true
  aggressiveness: 2
  state_aware_gating: true  # Enable state-aware thresholds
  speaking_threshold_multiplier: 2.0  # 2x higher threshold during TTS playback
  listening_threshold_multiplier: 1.0  # Normal sensitivity when waiting for speech
  barged_in_threshold_multiplier: 1.2  # Slightly elevated after barge-in
```

**Expected Impact**: 70-80% reduction in false positives during TTS playback.

**Usage**:
```python
from src.orchestrator.vad_processor import VADAudioProcessor
from src.orchestrator.session import SessionState

# Pass session state when processing frames
is_speech = vad_processor.process_frame(
    audio_frame,
    session_state=SessionState.SPEAKING
)
```

### 3. Adaptive Noise Gate

**Problem**: Background noise (fan hum, typing, distant conversation) triggers false VAD events.

**Solution**: Percentile-based noise floor estimation with adaptive threshold.

**Features**:
- Automatic noise floor calibration (2-second warmup)
- Percentile-based estimation (25th percentile = noise floor)
- Adaptive threshold = max(noise_floor * 2.5, 200.0)
- Updates every 200ms (10 frames @ 50fps)
- Filters frames below threshold before VAD processing
- Minimal latency (<1ms per frame)
- Statistics tracking (frames_gated, gating_ratio, noise_floor)

**Configuration**:
```yaml
vad:
  enabled: true
  noise_gate:
    enabled: true  # Enable adaptive noise gate
    window_size: 100  # 2 seconds @ 50fps
    percentile: 0.25  # 25th percentile = noise floor
    threshold_multiplier: 2.5  # 2.5x noise floor
    min_threshold: 200.0  # Absolute minimum RMS threshold
    update_interval_frames: 10  # Update every 200ms
```

**Expected Impact**: Additional 30-40% reduction in false positives from background noise.

**Combined Impact**: 70-90% total reduction in false barge-ins (state-aware + noise gate).

**Implementation Details**:
```python
# Noise gate processing pipeline (src/orchestrator/vad_processor.py)
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
Process through VAD
```

**RMS Buffer** (`src/orchestrator/audio/buffer.py`):
- Circular buffer storing recent RMS energy values
- Efficient percentile calculation using numpy
- Fixed-size buffer (100 floats = 400 bytes)
- Thread-safe design for single-task access

**Statistics Available**:
```python
stats = vad_processor.stats
# {
#   "frames_processed": 5000,
#   "frames_gated": 2100,  # Frames blocked by noise gate
#   "gating_ratio": 0.42,  # 42% of frames filtered
#   "noise_floor": 150.3,  # Current noise floor estimate
#   "adaptive_threshold": 375.75,  # Current threshold (2.5x noise floor)
#   "state_gating_ratio": 0.15,  # State-aware filtering ratio
#   ...
# }
```

## Model Manager (M4)

`src/tts/model_manager.py` handles all model lifecycle ✅ Implemented (M4):

**Startup behavior:**
- Load **default_model_id** (required, must exist)
- Optionally load **preload_model_ids** list
- Warmup each model (~300 ms synthetic utterance)

**Runtime operations:**
- `load(model_id)`: respects max_parallel_loads semaphore, increments refcount
- `release(model_id)`: decrements refcount, updates last_used_ts
- `evict_idle()`: background task unloads models with refcount==0 and idle > ttl_ms
- LRU eviction when resident models exceed resident_cap

**Configuration (worker.yaml):**
```yaml
model_manager:
  default_model_id: "piper-en-us-lessac-medium"
  preload_model_ids: []
  ttl_ms: 600000            # 10 min idle → unload
  min_residency_ms: 120000  # keep at least 2 min
  evict_check_interval_ms: 30000
  resident_cap: 3
  max_parallel_loads: 1
```

## Voice Packs (M5-M8)

Models are stored in `voicepacks/<family>/<model_id>/`:

```
voicepacks/
├─ piper/en-us-lessac-medium/         # M5 ✅ Implemented
│  ├─ en-us-lessac-medium.onnx        # ONNX model
│  ├─ en-us-lessac-medium.onnx.json   # Config file
│  └─ metadata.yaml                    # tags: lang, cpu_ok, sample_rate, etc.
├─ cosyvoice2/en-base/
│  ├─ model.safetensors
│  ├─ config.json
│  └─ metadata.yaml        # tags: lang, expressive, cpu_ok, lora, domain, etc.
└─ xtts-v2/en-demo/
   ├─ model.safetensors
   ├─ config.json
   ├─ metadata.yaml
   └─ ref/seed.wav          # optional reference audio for cloning
```

## Piper TTS Adapter (M5)

**Implementation**: `src/tts/adapters/adapter_piper.py` ✅ Complete

**Features**:
- ONNX Runtime for CPU-only inference (fast edge deployment)
- Native sample rate: 22050Hz, resampled to 48kHz
- 20ms frame output with strict repacketization
- PAUSE/RESUME/STOP control with <50ms response time
- Warmup synthesis for model initialization

**Configuration**:
```yaml
adapter:
  type: "piper"
  model_path: "voicepacks/piper/en-us-lessac-medium"
  config:
    sample_rate: 48000  # Output sample rate
    channels: 1
```

**Performance Metrics** (M5 validation):
- First Audio Latency (FAL): p95 < 450ms (CPU baseline)
- Real-time Factor (RTF): ~0.4 (2.5x faster than realtime on modern CPU)
- Frame jitter: p95 < 8ms
- Control latency: p95 < 40ms (PAUSE/RESUME/STOP)
- Warmup time: ~800ms

**Usage Example**:
```python
from src.tts.adapters.adapter_piper import PiperTTSAdapter

adapter = PiperTTSAdapter(
    model_id="piper-en-us-lessac-medium",
    model_path="voicepacks/piper/en-us-lessac-medium"
)

# Warmup
await adapter.warm_up()

# Synthesize
async def text_gen():
    yield "Hello, this is Piper TTS speaking."

async for frame in adapter.synthesize_stream(text_gen()):
    # Process 20ms PCM frame at 48kHz
    play_audio(frame)
```

## ASR Integration (M10)

**Implementation**: `src/asr/` module with Whisper adapter ✅ Complete

**Features**:
- OpenAI Whisper for speech-to-text transcription
- Multi-model support (tiny/base/small/medium/large)
- CPU and GPU inference with FP16 optimization
- Real-time processing with low latency
- Audio resampling pipeline (8kHz-48kHz → 16kHz)

**Configuration**:
```yaml
asr:
  enabled: true
  adapter: "whisper"
  model_size: "small"
  language: "en"
  device: "cpu"
  compute_type: "float32"
```

**Performance Metrics** (M10 validation):
- Transcription Latency: p95 < 1.5s (CPU small model)
- Real-Time Factor: 0.36 (CPU), ~0.2 (GPU)
- Memory Usage: ~1.5GB (CPU), ~920MB (GPU)
- Initialization: ~2-5s (cached), ~10-30s (first download)

**Usage Example**:
```python
from src.asr.adapters.adapter_whisper import WhisperAdapter

adapter = WhisperAdapter(model_size="small", device="cpu")
await adapter.initialize()

result = await adapter.transcribe(audio_bytes, sample_rate=16000)
print(f"Text: {result.text}, Confidence: {result.confidence}")

await adapter.shutdown()
```

## Routing & Worker Discovery (M9+)

Workers announce capabilities to Redis:
```json
{
  "name": "tts-piper@0",
  "addr": "grpc://tts-piper:7001",
  "capabilities": {
    "streaming": true,
    "zero_shot": false,
    "lora": false,
    "cpu_ok": true,
    "languages": ["en"],
    "emotive_zero_prompt": false
  },
  "resident_models": ["piper-en-us-lessac-medium"],
  "metrics": {"rtf": 0.4, "queue_depth": 0}
}
```

**Selection logic (M9+):**
1. Filter by language, capabilities, and sample rate
2. **Prefer resident models** (already loaded in VRAM/RAM)
3. Pick lowest queue_depth, then best p50 latency
4. If requested model not resident, optionally trigger async LoadModel and route to fallback

**Current (M2-M5):** Static worker address configuration for testing.

## Performance Targets

**Latency SLAs:**
- Barge-in pause latency: p95 < 50 ms ✅ Validated (M3)
- VAD processing latency: p95 < 5 ms per frame ✅ Validated (M3)
- VAD processing with noise gate: p95 < 1 ms overhead ✅ Validated (M10 Polish)
- First Audio Latency (FAL): p95 < 300 ms for GPU adapters, < 500 ms for Piper CPU ✅ Validated (M5: 450ms)
- Frame jitter: p95 < 10 ms under 3 concurrent sessions ✅ Validated (M5: 8ms)
- ASR transcription latency: p95 < 1.5s (CPU), < 1.0s (GPU) ✅ Validated (M10: 1.2s)

**Metrics tracked:**
- FAL, RTF (real-time factor), frame jitter, queue depth
- Barge-in events, active sessions
- Model load/unload durations, eviction counts
- VAD statistics (speech ratio, event count, gating ratio, noise floor)
- ASR transcription latency, RTF, accuracy
- Session timeout events, multi-turn conversation counts

## Implementation Milestones

The project follows a phased implementation plan (see `project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md`):

1. **M0**: ✅ Repo scaffold + CI skeleton (Complete)
2. **M1**: ✅ gRPC ABI + Mock worker (Complete - 16/16 tests passing)
3. **M2**: ✅ Orchestrator transport + WS fallback (Enhanced - LiveKit WebRTC primary, exceeds original scope)
4. **M3**: ✅ Barge-in end-to-end (Complete - VAD integration, <50ms pause latency, 37/37 tests passing)
5. **M4**: ✅ Model Manager v1 (Complete - 20 unit + 15 integration tests passing, TTL/LRU eviction, warmup, refcounting)
6. **M5**: ✅ Piper adapter (CPU baseline) - Complete (15 unit + 10 integration tests passing, 22050Hz→48kHz resampling)
7. **M6**: 📝 CosyVoice 2 adapter (GPU) - Planned
8. **M7**: 📝 XTTS-v2 adapter (GPU + cloning) - Planned
9. **M8**: 📝 Sesame / Unsloth (+LoRA) adapter - Planned
10. **M9**: 📝 Routing v1 (capabilities + prefer resident) - Planned
11. **M10**: ✅ ASR integration (Complete - Whisper + WhisperX adapters, 128 tests passing)
    - **M10 Polish**: ✅ Complete (Session timeout, state-aware VAD, adaptive noise gate, frontend feedback)
12. **M11**: 📝 Observability & profiling - Planned
13. **M12**: 📝 Docker/Compose smoke; docs polish - Planned
14. **M13**: 📝 Multi-GPU & multi-host scale-out - Planned

**Legend**:
- ✅ Complete: Fully implemented and tested
- 🔄 Partial: Some implementation, needs completion
- 📝 Planned: Not yet started

**Note**: M2 exceeded original scope - LiveKit was implemented as PRIMARY transport (not just fallback), with comprehensive WebRTC support, Caddy reverse proxy, and TLS infrastructure.

**M10 Polish Features**:
1. ✅ Session idle timeout (multi-turn conversations, graceful cleanup) - Task 7 Complete
2. ✅ State-aware VAD gating (threshold multipliers by session state) - Task 3 Complete
3. ✅ Adaptive noise gate (percentile-based noise floor estimation) - Task 4 Complete
4. ✅ Frontend visual feedback (React components for state/audio/connection display) - Tasks 1-2 Complete
5. ✅ Comprehensive testing (71 unit + integration tests for Tasks 4 & 7) - Complete
6. 📝 Configuration optimization (tuning parameters based on test results) - Task 8 Pending

## Important Patterns

**Adapter implementation:**
- Inherit from base protocol in `tts_base.py`
- Implement streaming synthesis with repacketization to 20 ms frames
- Respect PAUSE/RESUME/STOP immediately (< 50 ms)
- Normalize loudness (~−16 LUFS target or RMS)
- Handle native sample rate → 48kHz resampling (e.g., Piper: 22050Hz → 48kHz)

**Worker process separation:**
- Orchestrator and TTS workers run as separate processes
- Enables single-GPU (two processes) and multi-GPU (N+1 processes) deployments
- Use `CUDA_VISIBLE_DEVICES` to pin workers to specific GPUs
- Piper adapter runs CPU-only (no GPU required)

**State machine (orchestrator):**
- IDLE: initial state
- LISTENING: waiting for user speech
- SPEAKING: playing TTS audio
- BARGED_IN: user interrupted, PAUSE sent to worker
- WAITING_FOR_INPUT: multi-turn conversation, waiting for next user input (M10 Polish)
- TERMINATED: session ended

**Session lifecycle (M10 Polish):**
- Sessions persist between interactions (multi-turn conversations)
- Automatic timeout after idle_timeout_seconds of inactivity
- Session limits enforced (max_duration, max_messages)
- Graceful cleanup on timeout or limit reached

**No mid-stream model switches:**
- Model changes require ending current session and starting new one
- Orchestrator handles UX transition

## Mandatory Acceptance Criteria for All Features and Milestones

**IMPORTANT**: Every completed feature, milestone, or code change MUST satisfy ALL of the following criteria before being considered complete:

### Code Quality Requirements

1. **✅ All Tests Pass**: `just test` must pass with no failures
   - Unit tests must pass
   - Integration tests must pass
   - No skipped tests without documented justification

2. **✅ Linting Clean**: `just lint` must pass with no errors or warnings
   - All ruff linting rules must pass
   - Code must follow project style guidelines
   - No unused imports, variables, or code

3. **✅ Type Checking Clean**: `just typecheck` must pass with no errors
   - mypy strict mode must pass
   - All type annotations must be correct and complete
   - No `type: ignore` comments without documented justification

4. **✅ CI Pipeline Green**: `just ci` must pass completely
   - Runs all three checks: lint + typecheck + test
   - This is the final gate before any milestone is considered complete

### Documentation Requirements

5. **✅ Documentation Updated**: All relevant documentation must be current
   - **CLAUDE.md**: Update milestone status, test counts, performance metrics
   - **docs/CURRENT_STATUS.md**: Mark milestone complete with implementation details
   - **project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md**: Update milestone status and exit criteria
   - **project_documentation/IMPLEMENTATION_MILESTONES_AND_TASKS_CHECKLIST.md**: Check off completed tasks
   - **README.md**: Update status badges and "Current Status" section
   - **Source code**: Complete docstrings with usage examples for all public APIs
   - **Configuration files**: Inline comments explaining all options

6. **✅ Documentation Audit**: Comprehensive review of all documentation
   - Coordinate @documentation-engineer, @devops-engineer, and @python-pro
   - Verify cross-document consistency (test counts, metrics, dates)
   - Check for missing documentation (usage guides, API docs)
   - Validate configuration examples are current
   - Ensure all file paths and references are correct

7. **✅ Commit and PR Documentation**: Professional commit messages and PR descriptions
   - Generate commit message in `/tmp/M{N}_commit.msg` using conventional commits format
   - Generate PR description in `/tmp/M{N}_pr.description` with comprehensive details
   - Include: summary, implementation details, test coverage, performance metrics, files changed
   - Follow industry best practices for git commit messages and GitHub PRs

**Enforcement**: No milestone, feature, or PR should be marked as complete unless ALL seven criteria are met. Use `just ci` for code quality validation and coordinate documentation audit with multi-agent team for documentation requirements.

## Testing Strategy

**Unit tests (`tests/unit/`):**
- VAD edge detection (M3) ✅ 29/29 passing
- Piper adapter logic (M5) ✅ 15/15 passing
- ASR base protocol (M10) ✅ 23/23 passing
- Audio buffer (M10) ✅ 41/41 passing
- RMS buffer / adaptive noise gate (M10 Polish Task 4) ✅ 31/31 passing
- Session timeout validation (M10 Polish Task 7) ✅ 18/18 passing
- Routing policy logic (M9+)
- TTS control semantics (PAUSE/RESUME/STOP)
- Model manager lifecycle (M4): load/unload/TTL/evict/LRU
- Audio framing (exact 20 ms cadence, 48 kHz)
- Audio resampling (22050Hz → 48kHz for Piper, 8kHz-48kHz → 16kHz for Whisper)

**Integration tests (`tests/integration/`):**
- M1 Worker Integration: 16/16 tests passing with `--forked` mode
- M3 VAD Integration: 8/8 tests passing
- M3 Barge-in Integration: 37/37 tests passing ✅ Complete
- M5 Piper Integration: 10/10 tests passing ✅ Complete
- M10 Whisper ASR Integration: 28/28 tests passing ✅ Complete
- M10 Whisper Performance: 11/11 tests passing ✅ Complete
- M10 Polish Multi-Turn Conversation: 22/22 tests passing ✅ Complete (Task 7)
- Full pipeline WebSocket tests: 6/8 passing (2 timeout - under investigation)
- Loopback WebSocket test (FAL + frame timing)
- Barge-in timing validation (< 50 ms) ✅ Validated
- Piper FAL validation (< 500ms CPU baseline) ✅ Validated
- Whisper transcription latency (< 1.5s CPU) ✅ Validated
- Preload defaults honored - M4+

**CI (`just ci`):**
- Runs ruff + mypy + pytest on all PRs
- GPU integration tests can be tagged/skipped on non-GPU runners
- Piper tests run on CPU-only environments
- Whisper tests run on CPU-only environments

**gRPC Testing in WSL2:**
- **Issue**: grpc-python has segfault issues in WSL2 during test teardown
- **Solution**: Use `just test-integration` which runs tests with `--forked` flag (process isolation)
- **Documentation**: See [GRPC_SEGFAULT_WORKAROUND.md](GRPC_SEGFAULT_WORKAROUND.md) for details
- **Status**: 100% mitigated with pytest-forked, tests reliable
- **Alternative**: Skip gRPC tests in WSL2 (automatic detection), run in Docker or native Linux

**Test Coverage Summary (M0-M10 + M10 Polish)**:
- **Total Tests**: 649 tests (as of 2025-10-16)
- **Test Breakdown**: 500 unit tests + 139 integration tests + 10 performance tests
- **M0-M5 Tests**: 113 tests (core infrastructure, VAD, Model Manager, Piper adapter)
- **M10 ASR Tests**: 128 tests (Whisper + WhisperX adapters, audio buffer, performance)
- **M10 Polish Tests**: 65 tests ✅ Complete (31 RMS buffer + 21 session timeout + 13 multi-turn conversation)
- **Other Tests**: ~343 tests (config validation, utilities, etc.)

## CI/CD Pipeline

The project uses a modern three-tier CI/CD strategy optimized for fast feedback and comprehensive quality gates.

### CI Architecture

**Three-Tier Strategy:**

1. **Feature Branch CI** (`.github/workflows/feature-ci.yml`)
   - **Triggers**: Push to `feature/*`, `feat/*`, `fix/*`, etc.
   - **Duration**: ~3-5 minutes (60-70% faster than full suite)
   - **Strategy**: Smart test selection based on changed files
   - **Status**: Informational (non-blocking)
   - **Purpose**: Fast feedback during development

2. **Pull Request CI** (`.github/workflows/pr-ci.yml`)
   - **Triggers**: PR creation/updates to main
   - **Duration**: ~10-15 minutes
   - **Strategy**: Full test suite (all 649 tests) + code coverage
   - **Status**: **REQUIRED** (blocks merge if failing)
   - **Purpose**: Comprehensive quality gate before merge

3. **Main Branch** (no CI)
   - **Rationale**: Quality guaranteed by PR gates
   - **Benefit**: 44% reduction in CI minutes

### Feature Branch CI (Smart Test Selection)

**Workflow**: `.github/workflows/feature-ci.yml`

**Change Detection Logic:**

```
Changed Files → Test Categories
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
src/orchestrator/** → orchestrator unit + integration tests
src/tts/** → TTS unit + integration tests
src/asr/** → ASR unit + integration tests
src/rpc/** → all integration tests
pyproject.toml → full test suite
uv.lock → full test suite
configs/** → all integration tests
*.md, docs/** → skip all (docs-only)
```

**Jobs:**

1. **detect-changes**: Analyzes git diff to determine which tests to run
2. **lint**: Runs ruff on all code (always enabled for code changes)
3. **typecheck**: Runs mypy with protobuf generation (always enabled)
4. **test**: Runs selected test suites based on change detection
5. **summary**: Aggregates results and provides feedback

**Performance:**
- With caching: 30 sec dependency install (vs 5 min cold)
- Selective tests: 1-3 min test execution (vs 8-10 min full suite)
- Total: 3-5 minutes end-to-end

**Example Output:**
```bash
# Orchestrator changes detected
✅ Lint (ruff) - 45s
✅ Typecheck (mypy) - 1m 20s
✅ Test (orchestrator + integration) - 2m 30s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: 4m 35s (vs 12m 15s full suite)
```

### Pull Request CI (Full Validation)

**Workflow**: `.github/workflows/pr-ci.yml`

**Quality Gates (ALL must pass):**

1. **Lint** (ruff)
   - All code must follow style guidelines
   - No unused imports, variables, or code
   - Enforces project conventions

2. **Type Check** (mypy)
   - Strict mode must pass
   - All type annotations correct
   - No `type: ignore` without justification

3. **Full Test Suite** (pytest)
   - All 649 tests must pass
   - Unit tests (500 tests)
   - Integration tests (139 tests)
   - Performance tests (10 tests)
   - Excludes gRPC tests on non-Docker runners

4. **Code Coverage** (codecov)
   - Overall: ≥80% coverage required
   - Patch (new code): ≥60% coverage required
   - Threshold: 2% drop allowed from base
   - Automatic PR comments with coverage diff

5. **Security Scan** (bandit)
   - Scans for common security issues
   - Reports vulnerabilities in artifacts
   - Informational (non-blocking)

6. **Dependency Check** (pip-audit)
   - Scans for known vulnerabilities in dependencies
   - Reports CVEs and security advisories
   - Informational (non-blocking)

7. **Build Check**
   - Verifies `uv.lock` is up-to-date
   - Tests frozen dependency installation
   - Ensures reproducible builds

**Coverage Integration:**

The PR CI workflow integrates with [Codecov](https://codecov.io) to provide:
- Automated coverage reporting on PRs
- Coverage diff visualization (shows exactly which lines need tests)
- Historical coverage tracking
- Branch coverage badges

**Configuration**: `codecov.yml`

```yaml
coverage:
  status:
    project:
      default:
        target: 80%  # Overall coverage target
        threshold: 2%  # Allow 2% drop
    patch:
      default:
        target: 60%  # New code coverage target
        threshold: 5%  # Allow 5% variance
```

### Caching Strategy

**uv Dependency Caching:**

```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v3
  with:
    version: "latest"
    enable-cache: true
    cache-dependency-glob: "uv.lock"
```

**Benefits:**
- **Cache hit** (same uv.lock): 30 sec install (90% faster)
- **Partial hit** (some deps changed): 2-3 min (50% faster)
- **Cache miss** (new deps): 5-6 min (baseline)

**Expected cache hit rates:**
- Feature branches: 90-95% (same dependencies as main)
- After dependency updates: 70-80% (partial cache reuse)

**Protobuf Stub Caching:**

```yaml
- name: Cache protobuf stubs
  uses: actions/cache@v4
  with:
    path: src/rpc/generated
    key: protobuf-${{ hashFiles('src/rpc/tts.proto') }}
```

**Benefits:**
- Skips protobuf generation if `.proto` file unchanged
- Saves 10-15 seconds per job (typecheck, test)

### Branch Protection Rules

**Required status checks** (configured in GitHub):

```
Branch: main
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Require pull request reviews: 1 reviewer
✅ Require status checks to pass:
   - PR CI / lint
   - PR CI / typecheck
   - PR CI / test
   - PR CI / build
✅ Require branches to be up to date
✅ Do not allow force pushes
✅ Do not allow deletions
```

**Setting up branch protection:**

1. Go to: Settings → Branches → Add rule
2. Branch name pattern: `main`
3. Enable "Require a pull request before merging"
4. Enable "Require status checks to pass before merging"
5. Select required checks:
   - `PR CI / lint`
   - `PR CI / typecheck`
   - `PR CI / test`
   - `PR CI / build`
6. Enable "Require branches to be up to date before merging"
7. Save changes

### Performance Metrics

**Baseline (old CI):**
- Duration: 10-15 minutes per run
- Runs on: every push (feature + main) + PRs
- Caching: none
- CI minutes/month: ~5,400 min

**Optimized (new CI):**
- Feature CI: 3-5 minutes (60-70% faster)
- PR CI: 10-15 minutes (same, but with coverage)
- Main CI: none (0 runs)
- Caching: 80-90% faster dependency install
- CI minutes/month: ~3,000 min (44% reduction)

**Cost savings:**
- Free tier (GitHub Actions): 2,000 min/month
- Estimated usage: 3,000 min/month
- Overage: 1,000 min @ $0.008/min = $8/month
- Previous: ~$27/month
- **Savings: $19/month (70% reduction)**

### Running CI Locally

**Feature CI equivalent:**

```bash
# Check what would run based on your changes
git diff --name-only main...HEAD

# Run affected test suites
just ci  # Runs: lint + typecheck + test
```

**PR CI equivalent:**

```bash
# Run full test suite with coverage
uv run pytest tests/ \
  -v \
  --cov=src \
  --cov-report=xml \
  --cov-report=term \
  --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

**Pre-commit hooks** (recommended):

```bash
# Install pre-commit hooks
uv pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

### Troubleshooting CI Failures

**Lint failures:**

```bash
# Auto-fix most issues
just fix

# Check remaining issues
just lint
```

**Type check failures:**

```bash
# Regenerate protobuf stubs
just gen-proto

# Run type check
just typecheck

# Common issues:
# - Missing type annotations: add type hints
# - Import errors: check module paths
# - Protobuf stubs: ensure gen-proto was run
```

**Test failures:**

```bash
# Run specific test file
uv run pytest tests/unit/orchestrator/test_session.py -v

# Run with verbose output
uv run pytest tests/ -vv --tb=long

# Run with pdb on failure
uv run pytest tests/ --pdb

# For gRPC tests in WSL2
just test-integration  # Uses --forked flag
```

**Coverage failures:**

```bash
# Generate coverage report locally
uv run pytest tests/ --cov=src --cov-report=term-missing

# Identify untested lines
uv run pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Common issues:
# - Add tests for new functionality
# - Remove dead code
# - Mark test-only code with # pragma: no cover
```

**Cache issues:**

If CI is slower than expected:

1. Check cache hit rate in CI logs:
   ```
   Cache restored successfully
   Cache restored from key: Linux-uv-abc123...
   ```

2. Clear cache if corrupted (GitHub UI):
   - Settings → Actions → Caches → Delete caches

3. Force cache refresh (temporary):
   - Update `uv.lock`: `uv lock --upgrade`

### CI Monitoring

**Key metrics to track:**

1. **CI Duration**
   - Feature CI: Target < 5 min
   - PR CI: Target < 15 min
   - Trend: Should decrease over time with caching

2. **Cache Hit Rate**
   - Target: >90% for feature branches
   - Monitor: CI logs show cache restore success

3. **Test Flakiness**
   - Target: <1% flaky tests
   - Monitor: GitHub Actions logs for intermittent failures
   - Action: Investigate and fix flaky tests immediately

4. **Coverage Trend**
   - Target: Maintain or increase coverage
   - Monitor: Codecov dashboard
   - Alert: If coverage drops >2% from main

**GitHub Actions insights:**

- Navigate to: Actions tab → Workflow → View workflow runs
- Filter by: branch, status, time period
- Analyze: duration trends, failure rates, cache effectiveness

### Codecov Setup

**Initial setup:**

1. Sign up at [codecov.io](https://codecov.io) with GitHub account
2. Enable Codecov GitHub App for your repository
3. Get upload token: Settings → Repository → Upload Token
4. Add secret to GitHub: Settings → Secrets → Actions → New secret
   - Name: `CODECOV_TOKEN`
   - Value: (token from Codecov)
5. Codecov will automatically comment on PRs with coverage reports

**Badge for README:**

```markdown
[![codecov](https://codecov.io/gh/YOUR_USERNAME/full-duplex-voice-chat/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/full-duplex-voice-chat)
```

**Configuration:**

See `codecov.yml` for detailed configuration:
- Coverage targets (80% overall, 60% patch)
- Ignored paths (generated code, tests)
- PR comment format and behavior

### Migration from Old CI

**Deprecation plan:**

1. ✅ Create `feature-ci.yml` and `pr-ci.yml`
2. ✅ Test workflows on feature branch
3. ✅ Rename old `ci.yml` → `ci.yml.old` (backup)
4. ⏳ Monitor new workflows for 1 week
5. ⏳ Adjust thresholds based on actual metrics
6. ⏳ Delete `ci.yml.old` after successful migration

**Rollback procedure:**

If new CI causes issues:

```bash
# Restore old workflow
git mv .github/workflows/ci.yml.old .github/workflows/ci.yml

# Delete new workflows
git rm .github/workflows/feature-ci.yml
git rm .github/workflows/pr-ci.yml

# Commit and push
git commit -m "chore(ci): rollback to old CI workflow"
git push
```

### Best Practices

**For developers:**

1. **Run local checks before pushing:**
   ```bash
   just ci  # Runs lint + typecheck + test
   ```

2. **Keep feature branches small:**
   - Smaller changes = faster CI
   - Easier to review and debug failures

3. **Fix CI failures immediately:**
   - Feature CI failures are warnings
   - PR CI failures block merge (fix required)

4. **Monitor coverage:**
   - Add tests for new functionality
   - Aim for >80% coverage on new code

5. **Update dependencies carefully:**
   - Run `uv lock` locally first
   - Test thoroughly before pushing
   - Expect longer CI times after updates

**For maintainers:**

1. **Monitor CI health:**
   - Check weekly: duration trends, failure rates
   - Investigate: flaky tests, slow jobs
   - Optimize: add caching, parallelize tests

2. **Keep workflows updated:**
   - Update actions versions regularly
   - Review GitHub Actions changelog
   - Test workflow changes on feature branches

3. **Manage secrets:**
   - Rotate Codecov token annually
   - Limit secret access to required jobs
   - Never log secrets in CI output

4. **Branch protection:**
   - Enforce required checks
   - Require code reviews
   - Keep main branch green always

### Future Improvements

**Planned enhancements:**

1. **Parallel test execution** (M11):
   - Use `pytest-xdist` for parallel testing
   - Target: 50% faster test execution
   - Risk: Requires thread-safe test fixtures

2. **Matrix testing** (M11):
   - Test multiple Python versions (3.12, 3.13)
   - Test multiple OS (Ubuntu, macOS, Windows)
   - Increases CI time but improves compatibility

3. **GPU CI runners** (M11):
   - Test GPU-dependent code (TTS adapters)
   - Requires self-hosted runners with NVIDIA GPUs
   - Cost: $1-2/hour for GPU instances

4. **Performance benchmarking** (M11):
   - Automated performance regression tests
   - Track metrics: latency, throughput, memory
   - Alert on >10% regression

5. **Deployment preview** (M12):
   - Automatic staging deployments for PRs
   - Test full stack in isolated environment
   - Requires Kubernetes or similar infrastructure

## Docker & Deployment

**Single-GPU quickstart:**
```bash
docker compose up --build
```

This starts:
- Redis (service discovery)
- LiveKit (WebRTC server)
- Caddy (HTTPS reverse proxy for WebRTC)
- Orchestrator (WebRTC/WS server with VAD, ASR, and multi-turn session support)
- TTS worker (mock adapter by default, or Piper with env var)

**Piper CPU deployment:**
```bash
# Set adapter type to Piper
export TTS_ADAPTER=piper
export TTS_MODEL_ID=piper-en-us-lessac-medium
docker compose up --build
```

**Multi-GPU (same host):**
```bash
CUDA_VISIBLE_DEVICES=0 just run-tts-mock  # Worker 0
CUDA_VISIBLE_DEVICES=1 just run-tts-mock  # Worker 1 (when available)
# Or run Piper on CPU (no GPU required)
just run-tts-piper
just run-orch
```

**Multi-host (LAN) - M13:**
- Run central Redis instance
- Workers announce with host:port reachable on LAN
- Orchestrator discovers via Redis

## Profiling & Debugging

**Python profiling:**
- Use `py-spy` for CPU profiling: `just spy-top PID` or `just spy-record PID`

**GPU profiling:**
- Nsight Systems: `just nsys-tts` for timeline traces
- Nsight Compute: `just ncu-tts` for kernel analysis
- PyTorch Profiler with NVTX ranges around key phases (ASR, TTS, repacketize)

**Piper CPU profiling:**
- Piper runs ONNX on CPU, no GPU profiling needed
- Use py-spy for inference bottleneck analysis
- Monitor ONNX Runtime performance with ORT profiling

**Whisper ASR profiling:**
- CPU and GPU profiling supported
- Use py-spy for CPU bottleneck analysis
- Monitor PyTorch operations with profiler

**Observability:**
- Prometheus counters for latency/jitter/queue metrics
- Structured JSON logs with session IDs
- Log redaction for PII/audio paths

## Security Notes

- **Audio retention**: off by default; opt-in with retention policy
- **Auth**: demo scope uses simple API key; mTLS optional for worker↔orchestrator
- **TLS**: terminate HTTPS/WSS at nginx/traefik if WAN-exposed
- **Isolation**: each worker pinned to device; crashes don't affect other sessions

## References

Full documentation in `project_documentation/`:
- `PRD.md`: Product requirements and scope
- `TDD.md`: Detailed technical design (v2.1)
- `INCREMENTAL_IMPLEMENTATION_PLAN.md`: Milestone breakdown
- `IMPLEMENTATION_MILESTONES_AND_TASKS_CHECKLIST.md`: Task checklists

Additional documentation:
- [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md): Current implementation status
- [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md): Testing strategy and commands
- [docs/WHISPER_ADAPTER.md](docs/WHISPER_ADAPTER.md): Whisper ASR adapter guide
- [GRPC_SEGFAULT_WORKAROUND.md](GRPC_SEGFAULT_WORKAROUND.md): WSL2 gRPC testing workaround
