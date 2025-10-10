# CLAUDE.md

**Last Updated**: 2025-10-09

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Realtime Duplex Voice Demo** system enabling low-latency speech↔speech conversations with barge-in support. The system supports hot-swapping across multiple open TTS models (Sesame/Unsloth with LoRA, CosyVoice 2, XTTS-v2, Piper, etc.) and runs on single-GPU and multi-GPU setups.

**Key capabilities:**
- Realtime duplex conversation with barge-in (pause/resume < 50 ms) ✅ Implemented
- Streaming TTS with 20 ms, 48 kHz PCM frames ✅ Implemented
- Model modularity: swap among multiple TTS models via unified streaming ABI ✅ Protocol ready
- Dynamic model lifecycle: default preload, runtime load/unload, TTL-based eviction (M4 planned)
- Scale: single-GPU (two-process), multi-GPU (same host), multi-host (LAN)

**Current Implementation Status**: Milestones M0-M3 complete, M4-M13 planned.
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

# Run orchestrator (with VAD barge-in support)
just run-orch

# Run CLI client
just cli HOST="ws://localhost:8080"

# Note: Real TTS adapters below are M5-M8 milestones (not yet implemented)
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
   - ASR (M10+): Whisper small/distil for speech-to-text (planned)
   - Session management and state machine (LISTENING → SPEAKING → BARGED_IN) ✅ Implemented
   - Routing logic (M9+): capability-aware, prefers resident models, Redis-based discovery (static routing in M2/M3)
   - Lives in `src/orchestrator/`

2. **TTS Workers** (one per GPU/adapter):
   - gRPC server implementing unified streaming ABI ✅ Implemented (M1)
   - Model Manager (M4+): handles load/unload, TTL eviction, warmup, LRU caching (planned)
   - Adapters: implement model-specific logic while conforming to shared interface
   - Emit 20 ms, 48 kHz mono PCM frames ✅ Implemented
   - Lives in `src/tts/`

**Key flow (M3 with barge-in):**
- Client sends text → Orchestrator → TTS Worker → Audio frames → Client
- Barge-in: Client speaks → VAD detects speech → sends PAUSE to worker (<50ms) → worker stops emitting frames
- Resume: VAD detects silence → sends RESUME → worker continues

**Future flow (M10+ with ASR):**
- Client speaks → Orchestrator (VAD + ASR) → (optional LLM) → TTS Worker → Audio frames → Client

## Code Structure

```
src/
├─ orchestrator/
│  ├─ server.py          # LiveKit Agent + WS fallback, session management
│  ├─ livekit_utils/     # LiveKit integration (agent, transport)
│  ├─ transport/         # WebSocket transport
│  ├─ vad.py             # Voice Activity Detection (M3) ✅ Implemented
│  ├─ audio/
│  │  └─ resampler.py    # Audio resampling for VAD (48kHz → 16kHz) ✅ Implemented
│  ├─ asr.py             # Automatic Speech Recognition (M10+)
│  ├─ routing.py         # Worker selection logic (M9+ capability-aware)
│  ├─ registry.py        # Redis-based worker discovery
│  └─ config.py          # Configuration loading
│
├─ tts/
│  ├─ worker.py          # gRPC server, adapter host, ModelManager integration
│  ├─ model_manager.py   # Model lifecycle (M4+): load/unload, TTL eviction, warmup, LRU
│  ├─ tts_base.py        # Protocol/interface for all adapters
│  ├─ adapters/          # Model-specific implementations
│  │  ├─ adapter_mock.py           # M1/M2 - Mock adapter (sine wave) ✅ Implemented
│  │  ├─ adapter_piper.py          # M5 - CPU-only baseline (planned)
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
   └─ web/               # Browser client (HTML + JS)
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

## Model Manager (M4+)

`src/tts/model_manager.py` handles all model lifecycle (planned for M4 milestone):

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
  default_model_id: "cosyvoice2-en-base"
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

## Routing & Worker Discovery (M9+)

Workers announce capabilities to Redis:
```json
{
  "name": "tts-cosyvoice2@0",
  "addr": "grpc://tts-cosy:7002",
  "capabilities": {
    "streaming": true,
    "zero_shot": true,
    "lora": false,
    "cpu_ok": false,
    "languages": ["en", "zh"],
    "emotive_zero_prompt": true
  },
  "resident_models": ["cosyvoice2-en-base"],
  "metrics": {"rtf": 0.2, "queue_depth": 0}
}
```

**Selection logic (M9+):**
1. Filter by language, capabilities, and sample rate
2. **Prefer resident models** (already loaded in VRAM)
3. Pick lowest queue_depth, then best p50 latency
4. If requested model not resident, optionally trigger async LoadModel and route to fallback

**Current (M2/M3):** Static worker address configuration for testing.

## Performance Targets

**Latency SLAs:**
- Barge-in pause latency: p95 < 50 ms ✅ Validated (M3)
- VAD processing latency: p95 < 5 ms per frame ✅ Validated (M3)
- First Audio Latency (FAL): p95 < 300 ms for GPU adapters, < 500 ms for Piper CPU (M5+ target)
- Frame jitter: p95 < 10 ms under 3 concurrent sessions

**Metrics tracked:**
- FAL, RTF (real-time factor), frame jitter, queue depth
- Barge-in events, active sessions
- Model load/unload durations, eviction counts
- VAD statistics (speech ratio, event count)

## Implementation Milestones

The project follows a phased implementation plan (see `project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md`):

1. **M0**: ✅ Repo scaffold + CI skeleton (Complete)
2. **M1**: ✅ gRPC ABI + Mock worker (Complete - 16/16 tests passing)
3. **M2**: ✅ Orchestrator transport + WS fallback (Enhanced - LiveKit WebRTC primary, exceeds original scope)
4. **M3**: ✅ Barge-in end-to-end (Complete - VAD integration, <50ms pause latency, 37/37 tests passing)
5. **M4**: 📝 Model Manager v1 (default/preload/TTL) - Planned
6. **M5**: 📝 Piper adapter (CPU baseline) - Planned
7. **M6**: 📝 CosyVoice 2 adapter (GPU) - Planned
8. **M7**: 📝 XTTS-v2 adapter (GPU + cloning) - Planned
9. **M8**: 📝 Sesame / Unsloth (+LoRA) adapter - Planned
10. **M9**: 📝 Routing v1 (capabilities + prefer resident) - Planned
11. **M10**: 📝 ASR integration; full speech↔speech - Planned
12. **M11**: 📝 Observability & profiling - Planned
13. **M12**: 📝 Docker/Compose smoke; docs polish - Planned
14. **M13**: 📝 Multi-GPU & multi-host scale-out - Planned

**Legend**:
- ✅ Complete: Fully implemented and tested
- 🔄 Partial: Some implementation, needs completion
- 📝 Planned: Not yet started

**Note**: M2 exceeded original scope - LiveKit was implemented as PRIMARY transport (not just fallback), with comprehensive WebRTC support, Caddy reverse proxy, and TLS infrastructure.

## Important Patterns

**Adapter implementation:**
- Inherit from base protocol in `tts_base.py`
- Implement streaming synthesis with repacketization to 20 ms frames
- Respect PAUSE/RESUME/STOP immediately (< 50 ms)
- Normalize loudness (~−16 LUFS target or RMS)

**Worker process separation:**
- Orchestrator and TTS workers run as separate processes
- Enables single-GPU (two processes) and multi-GPU (N+1 processes) deployments
- Use `CUDA_VISIBLE_DEVICES` to pin workers to specific GPUs

**State machine (orchestrator):**
- LISTENING: waiting for user speech
- SPEAKING: playing TTS audio
- BARGED_IN: user interrupted, PAUSE sent to worker

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
- Routing policy logic (M9+)
- TTS control semantics (PAUSE/RESUME/STOP)
- Model manager lifecycle (M4+): load/unload/TTL/evict/LRU
- Audio framing (exact 20 ms cadence, 48 kHz)

**Integration tests (`tests/integration/`):**
- M1 Worker Integration: 16/16 tests passing with `--forked` mode
- M3 VAD Integration: 8/8 tests passing
- M3 Barge-in Integration: 37/37 tests passing ✅ Complete
- Full pipeline WebSocket tests: 6/8 passing (2 timeout - under investigation)
- Loopback WebSocket test (FAL + frame timing)
- Barge-in timing validation (< 50 ms) ✅ Validated
- Preload defaults honored - M4+

**CI (`just ci`):**
- Runs ruff + mypy + pytest on all PRs
- GPU integration tests can be tagged/skipped on non-GPU runners

**gRPC Testing in WSL2:**
- **Issue**: grpc-python has segfault issues in WSL2 during test teardown
- **Solution**: Use `just test-integration` which runs tests with `--forked` flag (process isolation)
- **Documentation**: See [GRPC_SEGFAULT_WORKAROUND.md](GRPC_SEGFAULT_WORKAROUND.md) for details
- **Status**: 100% mitigated with pytest-forked, tests reliable
- **Alternative**: Skip gRPC tests in WSL2 (automatic detection), run in Docker or native Linux

## Docker & Deployment

**Single-GPU quickstart:**
```bash
docker compose up --build
```

This starts:
- Redis (service discovery)
- LiveKit (WebRTC server)
- Caddy (HTTPS reverse proxy for WebRTC)
- Orchestrator (WebRTC/WS server with VAD)
- TTS worker (mock adapter, pinned to GPU 0)

**Multi-GPU (same host):**
```bash
CUDA_VISIBLE_DEVICES=0 just run-tts-mock  # Worker 0
CUDA_VISIBLE_DEVICES=1 just run-tts-mock  # Worker 1 (when available)
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
- [GRPC_SEGFAULT_WORKAROUND.md](GRPC_SEGFAULT_WORKAROUND.md): WSL2 gRPC testing workaround
