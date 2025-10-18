---
title: "Architecture"
tags: ["architecture", "design", "orchestrator", "tts-worker", "grpc"]
related_files:
  - "src/orchestrator/**/*.py"
  - "src/tts/**/*.py"
  - "src/rpc/tts.proto"
dependencies: []
estimated_tokens: 1800
priority: "medium"
keywords: ["architecture", "two-tier", "orchestrator", "TTS worker", "gRPC", "routing", "LiveKit"]
---

# Architecture

**Last Updated**: 2025-10-17

This document provides detailed architectural information about the Realtime Duplex Voice Demo system.

> 📖 **Quick Summary**: See [CLAUDE.md#architecture-summary](../../CLAUDE.md#architecture-summary)

## Table of Contents

- [Two-Tier Streaming Architecture](#two-tier-streaming-architecture)
- [Orchestrator Layer](#orchestrator-layer)
- [TTS Worker Layer](#tts-worker-layer)
- [gRPC Streaming ABI](#grpc-streaming-abi)
- [Audio Flow](#audio-flow)
- [State Machine](#state-machine)
- [Code Structure](#code-structure)
- [Routing & Worker Discovery](#routing--worker-discovery)

## Two-Tier Streaming Architecture

The system uses a two-tier architecture that separates orchestration logic from TTS inference:

```
┌─────────────────────────────────────────────────────────────┐
│                      Orchestrator Layer                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ LiveKit WebRTC (Primary)                              │ │
│  │ - Full-duplex audio for browser clients               │ │
│  │ - Caddy reverse proxy with TLS                        │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ WebSocket (Secondary)                                  │ │
│  │ - CLI testing and simple clients                      │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ VAD (M3) - Voice Activity Detection                   │ │
│  │ - Real-time speech detection (<50ms latency)          │ │
│  │ - PAUSE/RESUME control flow                           │ │
│  │ - Adaptive noise gate (M10 Polish)                    │ │
│  │ - State-aware VAD gating (M10 Polish)                 │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ ASR (M10) - Automatic Speech Recognition              │ │
│  │ - Whisper and WhisperX adapters                       │ │
│  │ - CPU/GPU inference with auto-optimization            │ │
│  │ - Audio resampling (8kHz-48kHz → 16kHz)               │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Session Management (M10 Polish)                        │ │
│  │ - Multi-turn conversations                             │ │
│  │ - Idle timeout (default: 5 minutes)                   │ │
│  │ - Session limits (duration, message count)            │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Routing (M9+) - Worker Selection                      │ │
│  │ - Capability-aware routing                            │ │
│  │ - Prefers resident models                             │ │
│  │ - Redis-based discovery                               │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↓ gRPC
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      TTS Worker Layer                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ gRPC Server                                            │ │
│  │ - Unified streaming ABI (tts.proto)                   │ │
│  │ - Session lifecycle (StartSession/EndSession)         │ │
│  │ - Streaming synthesis (TextChunk → AudioFrame)        │ │
│  │ - Control flow (PAUSE/RESUME/STOP/RELOAD)             │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Model Manager (M4)                                     │ │
│  │ - Load/unload with refcounting                        │ │
│  │ - TTL eviction (idle > 10 min)                        │ │
│  │ - LRU eviction (resident_cap exceeded)                │ │
│  │ - Warmup synthesis (~300ms)                           │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ TTS Adapters                                           │ │
│  │ - Mock (M1): Sine wave generator ✅                   │ │
│  │ - Piper (M5): ONNX CPU baseline ✅                    │ │
│  │ - CosyVoice2 (M6): GPU expressive TTS (in progress)   │ │
│  │ - XTTS (M7): GPU voice cloning (planned)              │ │
│  │ - Sesame (M8): Sesame base (planned)                  │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Audio Processing                                       │ │
│  │ - 20ms framing at 48kHz                               │ │
│  │ - Resampling (native → 48kHz)                         │ │
│  │ - Loudness normalization (~-16 LUFS)                  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Orchestrator Layer

**Location**: `src/orchestrator/`

**Responsibilities**:

1. **Transport Management**
   - **Primary**: LiveKit WebRTC for browser clients (full-duplex audio)
   - **Secondary**: WebSocket fallback for CLI testing
   - TLS termination via Caddy reverse proxy
   - Connection lifecycle management

2. **Voice Activity Detection (VAD)**
   - Real-time speech detection using webrtcvad
   - <50ms latency for barge-in detection
   - Audio resampling (48kHz → 16kHz for VAD processing)
   - **M10 Polish Enhancements**:
     - State-aware VAD gating (threshold multipliers by session state)
     - Adaptive noise gate (percentile-based noise floor estimation)
     - 70-90% reduction in false barge-ins

3. **Automatic Speech Recognition (ASR)**
   - Whisper and WhisperX adapters for speech-to-text
   - Multi-model support (tiny/base/small/medium/large)
   - CPU and GPU inference with auto-optimized compute types
   - Audio resampling (8kHz-48kHz → 16kHz)

4. **Session Management**
   - Multi-turn conversation support (sessions persist between interactions)
   - Configurable idle timeout (default: 5 minutes)
   - Session duration and message count limits
   - Graceful timeout handling with automatic cleanup

5. **Worker Routing**
   - Capability-aware routing (M9+)
   - Prefers workers with model already resident in memory
   - Redis-based worker discovery
   - Load balancing based on queue depth and latency

**Key Files**:
- `server.py`: Main orchestrator with session management
- `vad.py`: Voice Activity Detection (M3)
- `vad_processor.py`: VAD with noise gate (M10 Polish)
- `session.py`: Session state machine with WAITING_FOR_INPUT
- `routing.py`: Worker selection logic (M9+)
- `registry.py`: Redis-based worker discovery

## TTS Worker Layer

**Location**: `src/tts/`

**Responsibilities**:

1. **gRPC Server**
   - Implements unified streaming ABI (defined in `src/rpc/tts.proto`)
   - Session lifecycle: StartSession / EndSession
   - Streaming synthesis: TextChunk → AudioFrame
   - Control flow: PAUSE/RESUME/STOP/RELOAD

2. **Model Manager**
   - Default model preloading at startup
   - Dynamic load/unload with refcounting
   - TTL-based eviction (idle > 10 min)
   - LRU eviction when resident_cap exceeded
   - Warmup synthesis (~300ms per model)

3. **TTS Adapters**
   - Model-specific inference logic
   - Repacketization to 20ms frames at 48kHz
   - Native sample rate → 48kHz resampling
   - PAUSE/RESUME/STOP control (<50ms response time)

4. **Audio Processing**
   - 20ms framing (960 samples @ 48kHz)
   - Sample rate conversion (e.g., Piper: 22050Hz → 48kHz)
   - Loudness normalization (~-16 LUFS target)

**Key Files**:
- `worker.py`: gRPC server + ModelManager integration
- `model_manager.py`: Model lifecycle (M4)
- `tts_base.py`: Adapter protocol/interface
- `adapters/`: Model-specific implementations
- `audio/framing.py`: 20ms framing + resampling

## gRPC Streaming ABI

**Protocol**: `src/rpc/tts.proto`

All TTS adapters implement the same gRPC interface:

### Core Streaming

```protobuf
// Session lifecycle
rpc StartSession(SessionRequest) returns (SessionResponse)
rpc EndSession(SessionRequest) returns (google.protobuf.Empty)

// Main streaming path
rpc Synthesize(stream TextChunk) returns (stream AudioFrame)

// Runtime control
rpc Control(ControlRequest) returns (ControlResponse)
  // Commands: PAUSE | RESUME | STOP | RELOAD
```

### Model Lifecycle (M4+)

```protobuf
// Query and manage models
rpc ListModels(google.protobuf.Empty) returns (ModelListResponse)
rpc LoadModel(LoadModelRequest) returns (LoadModelResponse)
rpc UnloadModel(UnloadModelRequest) returns (google.protobuf.Empty)
rpc GetCapabilities(google.protobuf.Empty) returns (CapabilitiesResponse)
```

### Audio Format

**Output Specification**:
- **Frame duration**: 20 ms
- **Sample rate**: 48 kHz
- **Channels**: mono (1 channel)
- **Format**: PCM (signed 16-bit integers)
- **Frame size**: 960 samples = 1920 bytes per frame
- **Optional**: Opus encoding for compression (future enhancement)

**Adapter Requirements**:
- Must repacketize internal chunk sizes to exactly 20ms
- Must handle native sample rate → 48kHz resampling
- Must respect PAUSE/RESUME/STOP within <50ms

## Audio Flow

### M3 Flow (with Barge-in)

```
1. Client sends text → Orchestrator
2. Orchestrator → TTS Worker (gRPC Synthesize request)
3. TTS Worker → Audio frames → Orchestrator
4. Orchestrator → Client (LiveKit/WebSocket)

Barge-in:
5. Client speaks → Orchestrator (VAD detects speech)
6. Orchestrator → TTS Worker (PAUSE command, <50ms)
7. TTS Worker stops emitting frames
8. VAD detects silence → Orchestrator → TTS Worker (RESUME)
9. TTS Worker continues synthesis
```

### M10 Flow (with ASR and Multi-Turn)

```
1. Client speaks → Orchestrator (audio input)
2. Orchestrator VAD → detects speech start
3. Orchestrator → Adaptive Noise Gate → filters background noise
4. Orchestrator → State-Aware VAD → adjusts sensitivity by session state
5. Orchestrator → ASR adapter → transcribes audio to text
6. Orchestrator → (optional LLM) → generates response text
7. Orchestrator → TTS Worker (gRPC Synthesize request)
8. TTS Worker → Audio frames → Orchestrator
9. Orchestrator → Client (LiveKit/WebSocket)
10. Session → WAITING_FOR_INPUT state (awaits next user input)
11. Idle timeout (5 min) → graceful session termination
```

## State Machine

**Session States** (defined in `src/orchestrator/session.py`):

```
IDLE
  ↓ (user connects)
LISTENING
  ↓ (user speaks, ASR transcribes)
WAITING_FOR_INPUT (optional LLM processing)
  ↓ (TTS synthesis starts)
SPEAKING
  ↓ (user interrupts)
BARGED_IN
  ↓ (VAD detects silence)
LISTENING or WAITING_FOR_INPUT
  ↓ (idle timeout or session limit)
TERMINATED
```

**State Transitions**:

- **IDLE → LISTENING**: User connects, session starts
- **LISTENING → SPEAKING**: TTS synthesis begins
- **SPEAKING → BARGED_IN**: VAD detects user speech (barge-in event)
- **BARGED_IN → LISTENING**: VAD detects silence, ready for next input
- **SPEAKING → WAITING_FOR_INPUT**: TTS synthesis completes, awaiting next user input
- **WAITING_FOR_INPUT → LISTENING**: User starts speaking
- **WAITING_FOR_INPUT → TERMINATED**: Idle timeout or session limit reached
- **Any → TERMINATED**: Session end (explicit or timeout)

**State-Aware VAD Thresholds** (M10 Polish):

- **LISTENING**: Normal sensitivity (1.0x multiplier)
- **SPEAKING**: Reduced sensitivity (2.0x multiplier) - prevents false barge-ins from TTS audio
- **BARGED_IN**: Slightly reduced sensitivity (1.2x multiplier)
- **WAITING_FOR_INPUT**: Normal sensitivity (1.0x multiplier)

## Code Structure

```
src/
├─ orchestrator/           # Orchestration layer
│  ├─ server.py            # Main entry point (LiveKit Agent + WS fallback)
│  ├─ livekit_utils/       # LiveKit integration
│  │  ├─ agent.py          # LiveKit agent implementation
│  │  └─ transport.py      # WebRTC transport
│  ├─ transport/           # WebSocket transport
│  │  └─ websocket.py      # WS server implementation
│  ├─ vad.py               # Voice Activity Detection (M3)
│  ├─ vad_processor.py     # VAD with noise gate (M10 Polish)
│  ├─ session.py           # Session state machine
│  ├─ audio/
│  │  ├─ resampler.py      # Audio resampling (48kHz ↔ 16kHz)
│  │  └─ buffer.py         # Audio buffering + RMS energy buffer
│  ├─ routing.py           # Worker selection logic (M9+)
│  ├─ registry.py          # Redis-based worker discovery
│  └─ config.py            # Configuration loading
│
├─ asr/                    # ASR adapters (M10)
│  ├─ asr_base.py          # ASR adapter protocol
│  └─ adapters/
│     ├─ adapter_whisper.py   # Whisper ASR adapter
│     └─ adapter_whisperx.py  # WhisperX ASR adapter (4-8x faster)
│
├─ tts/                    # TTS worker layer
│  ├─ worker.py            # gRPC server + adapter host
│  ├─ model_manager.py     # Model lifecycle (M4)
│  ├─ tts_base.py          # Adapter protocol/interface
│  ├─ adapters/            # Model-specific implementations
│  │  ├─ adapter_mock.py           # M1: Mock (sine wave)
│  │  ├─ adapter_piper.py          # M5: Piper (ONNX CPU)
│  │  ├─ adapter_cosyvoice2.py     # M6: CosyVoice2 (GPU)
│  │  ├─ adapter_xtts.py           # M7: XTTS (GPU cloning)
│  │  ├─ adapter_sesame.py         # M8: Sesame base
│  │  └─ adapter_unsloth_sesame.py # M8: Unsloth LoRA
│  ├─ audio/
│  │  ├─ framing.py        # 20ms framing + resampling
│  │  └─ loudness.py       # RMS/LUFS normalization
│  └─ utils/
│     ├─ logging.py
│     └─ timers.py
│
├─ rpc/                    # gRPC protocol
│  ├─ tts.proto            # Service definition
│  └─ generated/           # Auto-generated stubs
│
└─ client/                 # Client implementations
   ├─ cli_client.py        # WebSocket CLI client
   └─ web/                 # Browser client (HTML + JS + React)
```

## Routing & Worker Discovery

**M9+ Feature**: Capability-aware routing with Redis-based worker discovery.

### Worker Capabilities

Workers announce their capabilities to Redis on startup:

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
  "metrics": {
    "rtf": 0.4,
    "queue_depth": 0
  }
}
```

### Selection Logic

**Routing algorithm** (implemented in `src/orchestrator/routing.py`):

1. **Filter by requirements**:
   - Language match (e.g., `lang == "en"`)
   - Capability match (e.g., `streaming == true`)
   - Sample rate compatibility

2. **Prefer resident models**:
   - Workers with model already loaded in memory (VRAM/RAM)
   - Avoids model loading latency

3. **Load balancing**:
   - Pick worker with lowest `queue_depth`
   - Tie-break by best p50 latency

4. **Fallback handling**:
   - If requested model not resident, optionally trigger async `LoadModel`
   - Route to fallback worker while loading

**Current Status (M2-M5)**: Static worker address configuration for testing. Dynamic routing will be implemented in M9.

## References

- **Core Documentation**: [CLAUDE.md](../../CLAUDE.md)
- **gRPC Protocol**: [src/rpc/tts.proto](../../src/rpc/tts.proto)
- **VAD Implementation**: [.claude/modules/features/vad.md](features/vad.md)
- **ASR Integration**: [.claude/modules/features/asr.md](features/asr.md)
- **Model Manager**: [.claude/modules/features/model-manager.md](features/model-manager.md)
- **Session Management**: [.claude/modules/features/session-management.md](features/session-management.md)

---

**Last Updated**: 2025-10-17
**Status**: Complete for M0-M10, M9 routing planned
