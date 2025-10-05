# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Realtime Duplex Voice Demo** system enabling low-latency speech↔speech conversations with barge-in support. The system supports hot-swapping across multiple open TTS models (Sesame/Unsloth with LoRA, CosyVoice 2, XTTS-v2, Piper, etc.) and runs on single-GPU and multi-GPU setups.

**Key capabilities:**
- Realtime duplex conversation with barge-in (pause/resume < 50 ms)
- Streaming TTS with 20 ms, 48 kHz PCM frames
- Model modularity: swap among multiple TTS models via unified streaming ABI
- Dynamic model lifecycle: default preload, runtime load/unload, TTL-based eviction
- Scale: single-GPU (two-process), multi-GPU (same host), multi-host (LAN)

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
# Run TTS worker with Sesame adapter
just run-tts-sesame DEFAULT="cosyvoice2-en-base" PRELOAD=""

# Run TTS worker with CosyVoice adapter
just run-tts-cosy DEFAULT="cosyvoice2-en-base" PRELOAD=""

# Run orchestrator
just run-orch

# Run CLI client
just cli HOST="ws://localhost:8080"
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
docker compose up --build    # Start full stack (redis + orchestrator + tts workers)
```

## Architecture Overview

**Two-tier streaming architecture:**

1. **Orchestrator** (LiveKit agent or equivalent):
   - WebRTC transport for browser clients, WS fallback for CLI
   - VAD (Voice Activity Detection) for interruption detection
   - ASR (Whisper small/distil) for speech-to-text
   - Session management and state machine (LISTENING → SPEAKING → BARGED_IN)
   - Routing logic: capability-aware, prefers resident models, Redis-based discovery
   - Lives in `src/orchestrator/`

2. **TTS Workers** (one per GPU/adapter):
   - gRPC server implementing unified streaming ABI
   - Model Manager: handles load/unload, TTL eviction, warmup, LRU caching
   - Adapters: implement model-specific logic while conforming to shared interface
   - Emit 20 ms, 48 kHz mono PCM frames
   - Lives in `src/tts/`

**Key flow:**
- Client speaks → Orchestrator (VAD + ASR) → (optional LLM) → TTS Worker → Audio frames → Client
- Barge-in: VAD detects speech → sends PAUSE to worker (< 50 ms) → worker stops emitting frames
- Resume: VAD detects silence → sends RESUME → worker continues

## Code Structure

```
src/
├─ orchestrator/
│  ├─ server.py          # LiveKit Agent + WS fallback, session management
│  ├─ vad.py             # Voice Activity Detection
│  ├─ asr.py             # Automatic Speech Recognition (Whisper)
│  ├─ routing.py         # Worker selection logic (capability + load aware)
│  ├─ registry.py        # Redis-based worker discovery
│  └─ config.py          # Configuration loading
│
├─ tts/
│  ├─ worker.py          # gRPC server, adapter host, ModelManager integration
│  ├─ model_manager.py   # Model lifecycle: load/unload, TTL eviction, warmup, LRU
│  ├─ tts_base.py        # Protocol/interface for all adapters
│  ├─ adapters/          # Model-specific implementations
│  │  ├─ adapter_sesame.py
│  │  ├─ adapter_unsloth_sesame.py    # LoRA variant
│  │  ├─ adapter_xtts.py
│  │  ├─ adapter_cosyvoice2.py
│  │  └─ adapter_piper.py             # CPU-only baseline
│  ├─ audio/
│  │  ├─ framing.py      # 20ms framing, resample to 48kHz
│  │  └─ loudness.py     # RMS/LUFS normalization
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
- `StartSession` / `EndSession`: session lifecycle
- `Synthesize(stream TextChunk) → stream AudioFrame`: main streaming path
- `Control(PAUSE|RESUME|STOP|RELOAD)`: runtime control

**Model lifecycle:**
- `ListModels`: query available models
- `LoadModel(model_id)`: dynamically load a model
- `UnloadModel(model_id)`: unload when idle
- `GetCapabilities`: report worker capabilities

**Audio format:**
- Output: 20 ms frames, 48 kHz, mono PCM (Opus optional later)
- Adapters must repacketize internal chunk sizes to 20 ms

## Model Manager

`src/tts/model_manager.py` handles all model lifecycle:

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

## Voice Packs

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

## Routing & Worker Discovery

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

**Selection logic:**
1. Filter by language, capabilities, and sample rate
2. **Prefer resident models** (already loaded in VRAM)
3. Pick lowest queue_depth, then best p50 latency
4. If requested model not resident, optionally trigger async LoadModel and route to fallback

## Performance Targets

**Latency SLAs:**
- Barge-in pause latency: p95 < 50 ms
- First Audio Latency (FAL): p95 < 300 ms for GPU adapters, < 500 ms for Piper CPU
- Frame jitter: p95 < 10 ms under 3 concurrent sessions

**Metrics tracked:**
- FAL, RTF (real-time factor), frame jitter, queue depth
- Barge-in events, active sessions
- Model load/unload durations, eviction counts

## Implementation Milestones

The project follows a phased implementation plan (see `project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md`):

1. **M0**: Repo scaffold + CI skeleton
2. **M1**: gRPC ABI + Mock worker
3. **M2**: Orchestrator transport + WS fallback
4. **M3**: Barge-in end-to-end
5. **M4**: Model Manager v1 (default/preload/TTL)
6. **M5**: Piper adapter (CPU baseline)
7. **M6**: CosyVoice 2 adapter (GPU)
8. **M7**: XTTS-v2 adapter (GPU + cloning)
9. **M8**: Sesame / Unsloth (+LoRA) adapter
10. **M9**: Routing v1 (capabilities + prefer resident)
11. **M10**: ASR integration; full speech↔speech
12. **M11**: Observability & profiling
13. **M12**: Docker/Compose smoke; docs polish
14. **M13**: Multi-GPU & multi-host scale-out

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

## Testing Strategy

**Unit tests (`tests/unit/`):**
- VAD edge detection
- Routing policy logic
- TTS control semantics (PAUSE/RESUME/STOP)
- Model manager lifecycle (load/unload/TTL/evict/LRU)
- Audio framing (exact 20 ms cadence, 48 kHz)

**Integration tests (`tests/integration/`):**
- Loopback WebSocket test (FAL + frame timing)
- Barge-in timing validation (< 50 ms)
- Preload defaults honored

**CI (`just ci`):**
- Runs ruff + mypy + pytest on all PRs
- GPU integration tests can be tagged/skipped on non-GPU runners

## Docker & Deployment

**Single-GPU quickstart:**
```bash
docker compose up --build
```

This starts:
- Redis (service discovery)
- Orchestrator (WebRTC/WS server)
- TTS worker (pinned to GPU 0)

**Multi-GPU (same host):**
```bash
CUDA_VISIBLE_DEVICES=0 just run-tts-cosy
CUDA_VISIBLE_DEVICES=1 just run-tts-xtts
just run-orch
```

**Multi-host (LAN):**
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
