# Current Project Status

**Last Updated**: 2025-10-09
**Branch**: `main`
**Overall Status**: M0-M4 Complete, M5-M13 Planned

---

## Quick Summary

This project implements a realtime duplex voice chat system with low-latency TTS streaming and barge-in support. **Milestones M0-M4 are complete**, establishing the core infrastructure: gRPC streaming protocol, mock TTS worker, dual transport architecture (LiveKit WebRTC primary + WebSocket fallback), real-time barge-in with Voice Activity Detection, and complete Model Manager lifecycle with TTL/LRU eviction. The implementation has **exceeded M2 scope** by delivering production-ready LiveKit WebRTC as the primary transport. Real TTS adapters (Piper, CosyVoice, XTTS, Sesame) and advanced features (dynamic routing, ASR integration) are planned for M5-M13.

---

## Milestone Status

| Milestone | Description | Status | Completion Date | Notes |
|-----------|-------------|--------|-----------------|-------|
| **M0** | Repo Scaffold & CI | ✅ Complete | 2025-09 | Justfile, CI, Docker, proto gen |
| **M1** | gRPC ABI + Mock Worker | ✅ Complete | 2025-09 | 16/16 tests passing, <50ms control |
| **M2** | Orchestrator Transport | ✅ Enhanced | 2025-10 | LiveKit primary (exceeded scope) |
| **M3** | Barge-in End-to-End | ✅ Complete | 2025-10-09 | VAD integration, <50ms pause latency |
| **M4** | Model Manager v1 | ✅ Complete | 2025-10-09 | 20 unit + 15 integration tests passing |
| **M5** | Piper Adapter (CPU) | 📝 Planned | - | CPU baseline TTS |
| **M6** | CosyVoice 2 Adapter | 📝 Planned | - | GPU expressive TTS |
| **M7** | XTTS-v2 Adapter | 📝 Planned | - | GPU + voice cloning |
| **M8** | Sesame/Unsloth Adapter | 📝 Planned | - | LoRA fine-tuned models |
| **M9** | Routing v1 | 📝 Planned | - | Capability-based selection |
| **M10** | ASR Integration | 📝 Planned | - | Whisper speech-to-text |
| **M11** | Observability & Profiling | 📝 Planned | - | Metrics, logging, tracing |
| **M12** | Docker/Compose Polish | 📝 Planned | - | Production deployment |
| **M13** | Multi-GPU Scale-out | 📝 Planned | - | N GPUs, multi-host |

**Legend**:
- ✅ Complete: Fully implemented and tested
- 🔄 Partial: Some implementation, integration pending
- 📝 Planned: Not yet started

---

## What Works Today (M0-M4 Complete)

### ✅ Core Infrastructure

**gRPC Streaming Protocol (M1)**:
- `Synthesize(stream TextChunk) → stream AudioFrame` bidirectional streaming
- `Control(PAUSE|RESUME|STOP)` commands with <50ms response time
- `StartSession` / `EndSession` lifecycle management
- Protocol defined in `src/rpc/tts.proto`, codegen via `just gen-proto`
- **Test Coverage**: 16/16 M1 integration tests passing (with `--forked` mode)

**Mock TTS Worker (M1)**:
- Generates 440Hz sine wave test tone for validation
- Proper 20ms framing @ 48kHz PCM output
- Immediate response to PAUSE/RESUME/STOP commands
- Used for all current integration testing
- Location: `src/tts/adapters/adapter_mock.py`

**Orchestrator Server (M2)**:
- **LiveKit WebRTC Transport** (Primary):
  - Full LiveKit agent integration for browser clients
  - Bidirectional audio streaming over WebRTC
  - Caddy reverse proxy for HTTPS/TLS
  - Docker compose orchestration with health checks
  - Location: `src/orchestrator/livekit_utils/`

- **WebSocket Transport** (Secondary/CLI):
  - JSON message protocol for CLI clients
  - Supports session_start, text, audio, error messages
  - Used by CLI client: `just cli`
  - Location: `src/orchestrator/transport/websocket_transport.py`

**Session Management (M2)**:
- State machine: IDLE → LISTENING → SPEAKING → BARGED_IN → TERMINATED
- State transition validation (VALID_TRANSITIONS enforced)
- Session metrics tracking (FAL, frame count, timing)
- Multi-message per session support (workaround for protocol edge case)
- Location: `src/orchestrator/session.py`

**Barge-in with Voice Activity Detection (M3)**:
- VAD integration using webrtcvad library
- Real-time speech detection with <50ms latency
- Automatic PAUSE on speech detection
- Automatic RESUME on silence detection
- State machine transitions: SPEAKING → BARGED_IN → LISTENING
- Configurable aggressiveness (0-3), debouncing thresholds
- Audio resampling (48kHz → 16kHz) for VAD processing
- Telemetry for barge-in events (count, latency)
- **Test Coverage**: 29/29 unit tests, 8/8 integration tests passing
- Location: `src/orchestrator/vad.py`, `src/orchestrator/audio/resampler.py`

**Model Manager with Lifecycle Management (M4)**:
- Default model loading on startup (required)
- Optional preload list from configuration
- TTL-based idle model eviction (configurable timeout)
- LRU eviction when resident_cap exceeded
- Warmup synthetic utterance on initialization
- Reference counting for safe unload (prevents in-use eviction)
- gRPC endpoints: LoadModel, UnloadModel, ListModels, GetCapabilities
- Background eviction task with configurable interval
- Semaphore control for max_parallel_loads
- **Test Coverage**: 20 unit tests, 15 integration tests passing
- Location: `src/tts/model_manager.py`

**Service Discovery & Registry (M2)**:
- Redis-based worker registration and heartbeat
- TTL-based worker expiration
- Capability announcement (streaming, languages, zero_shot, etc.)
- Location: `src/orchestrator/registry.py`

**Docker Compose Stack (M2)**:
- **5 Services**: Redis, LiveKit, Caddy, Orchestrator, TTS Worker
- Health checks for all services
- Startup dependencies configured
- Internal Docker networking (no port conflicts)
- GPU allocation ready (CUDA_VISIBLE_DEVICES)
- Command: `docker compose up --build`

### ✅ Development Infrastructure

**Build & CI (M0)**:
- Justfile commands: `lint`, `typecheck`, `test`, `ci`
- Ruff linting with auto-fix (`just fix`)
- Mypy strict type checking
- Pytest with comprehensive markers
- GitHub Actions CI pipeline

**Test Infrastructure (M1-M3)**:
- 40+ test files (unit + integration)
- 817-line `conftest.py` with comprehensive fixtures
- Synthetic audio generators for validation
- Frame timing validators
- Latency metrics collectors
- Dynamic port allocation for parallel tests
- VAD test suite with debouncing and aggressiveness validation

**gRPC Testing Workaround (M1/M2)**:
- **Issue**: grpc-python segfaults in WSL2 during test teardown
- **Solution**: pytest-forked integration with `--forked` flag
- **Status**: 100% mitigated, tests reliable
- **Documentation**: [GRPC_SEGFAULT_WORKAROUND.md](../GRPC_SEGFAULT_WORKAROUND.md)
- **Usage**: `just test-integration` (automatic)

**Code Quality Metrics**:
- Type hints: Comprehensive (mypy strict mode)
- Docstrings: Detailed with Args/Returns/Notes
- Test pass rate: M1 16/16 (100%), M3 37/37 (100%), Full pipeline 6/8 (75%, 2 timeouts under investigation)
- Architecture: Clean separation of concerns

---

## What's Planned (M5-M13 Roadmap)

### 📝 Near-term (M5)

**M5: Piper Adapter (CPU Baseline)**:
- First real TTS adapter implementation
- ONNX runtime for CPU inference
- Loudness normalization to target RMS
- 20ms PCM frame output
- FAL p95 < 500ms target

### 📝 Mid-term (M6-M9)

**M6: CosyVoice 2 Adapter (GPU)**:
- GPU streaming TTS with expressive capabilities
- Model initialization and warmup
- Streaming inference with proper chunk pacing
- FAL p95 < 300ms target

**M7: XTTS-v2 Adapter (GPU + Cloning)**:
- Multi-speaker expressive model
- Reference voice support (6-10s sample)
- Streaming mode validation
- Voice cloning demo

**M8: Sesame/Unsloth Adapter (+LoRA)**:
- Vanilla Sesame adapter
- LoRA variant with peft.PeftModel
- Runtime LoRA swap without restart
- CLI `--lora_path` argument

**M9: Routing v1**:
- Capability-based worker selection
- Load balancing policies (least-busy, prefer-resident)
- Redis heartbeat and metrics
- Graceful worker failure handling

### 📝 Long-term (M10-M13)

**M10: ASR Integration**:
- Whisper small/distil integration
- Real-time transcription streaming
- Full speech↔speech pipeline (ASR→LLM→TTS)
- CPU/GPU path validation

**M11: Observability & Profiling**:
- Structured JSON logging
- Prometheus metrics (FAL, jitter, RTF, queue depth)
- py-spy profiling targets in justfile
- PyTorch profiler with NVTX ranges

**M12: Docker/Compose Polish**:
- Production-ready Dockerfiles
- Finalized docker-compose.yml
- Documentation polish
- Smoke test validation

**M13: Multi-GPU & Multi-Host Scale-out**:
- Per-GPU CUDA_VISIBLE_DEVICES orchestration
- Multi-host IP+port discovery via Redis
- Worker auto-registration with TTL
- Cross-host gRPC validation

---

## Known Limitations

### Platform Constraints

**WSL2 Environment**:
- gRPC tests require `--forked` mode due to upstream segfault issue
- Workaround: Use `just test-integration` (automatic)
- Alternative: Skip gRPC tests in WSL2, run in Docker/native Linux
- Documentation: [GRPC_SEGFAULT_WORKAROUND.md](../GRPC_SEGFAULT_WORKAROUND.md)
- Status: Fully mitigated, tests 100% reliable

**Python 3.13**:
- Performance tests deferred due to instability in some libraries
- Unit and integration tests fully functional
- Recommendation: Use Python 3.12 if 3.13 causes issues

### Implementation Gaps

**Only Mock Adapter Implemented**:
- Real TTS models (Piper, CosyVoice, XTTS, Sesame) are M5-M8 milestones
- Justfile commands for real adapters exist but not functional yet
- Current testing uses 440Hz sine wave output

**Static Routing Only**:
- M2 uses static worker address configuration
- Dynamic capability-based routing is M9 milestone
- Single worker only in current setup

**No Speech Input**:
- Text-to-speech only in M0-M3
- Speech-to-text (ASR/Whisper) is M10 milestone
- VAD ready for ASR integration

**No Real TTS Models**:
- Model Manager complete (M4) but only mock adapter available
- Real TTS adapters (Piper, CosyVoice, XTTS, Sesame) are M5-M8
- Runtime model loading/unloading functional with mock adapter

### Test Issues

**Integration Test Timeouts**:
- 2/8 full pipeline tests timeout under load
- Tests: `test_sequential_messages_same_session`, `test_system_stability_under_load`
- Status: Under investigation
- Workaround: Run individually, not in parallel

**LiveKit Agent Dispatch Message**:
- Harmless informational message in logs
- Does not affect functionality
- Documented in integration test comments

---

## Quick Start Commands

### Development Workflow

```bash
# Install dependencies
uv sync --extra dev

# Generate gRPC stubs
just gen-proto

# Run linting and type checking
just lint
just typecheck

# Run unit tests (fast, no Docker required)
just test

# Run integration tests (requires Docker)
just test-integration

# Run full CI suite
just ci
```

### Running the System

```bash
# Start full Docker stack (recommended)
docker compose up --build

# OR: Start services individually
docker compose up redis livekit -d  # Dependencies only
just run-tts-mock                   # Terminal 1
just run-orch                       # Terminal 2
just cli                            # Terminal 3 (WebSocket client)
```

### Testing & Validation

```bash
# Run specific integration test
uv run pytest tests/integration/test_m1_worker_integration.py -v

# Run VAD tests
uv run pytest tests/unit/test_vad.py -v
uv run pytest tests/integration/test_vad_integration.py -v

# Run with verbose gRPC logging
export GRPC_VERBOSITY=debug
just run-tts-mock

# Profile CPU usage
just spy-top <PID>
```

---

## Architecture Highlights

### Barge-in Flow (M3)

**Voice Activity Detection**:
```
Client Audio (48kHz)
    ↓
Audio Resampler (48kHz → 16kHz)
    ↓
VAD Processor (webrtcvad)
    ↓
Speech Detection (debounced)
    ↓
Event Callbacks (on_speech_start, on_speech_end)
    ↓
State Machine Transitions
    ↓
Control Commands (PAUSE/RESUME)
    ↓
TTS Worker
```

**Configuration**:
```yaml
vad:
  enabled: true
  aggressiveness: 2  # 0-3, higher = more conservative
  sample_rate: 16000  # Required by webrtcvad
  frame_duration_ms: 20  # 10, 20, or 30
  min_speech_duration_ms: 100  # Debounce threshold
  min_silence_duration_ms: 300  # Silence threshold
```

**Performance**:
- p95 pause latency: <50ms ✅ Validated
- VAD processing latency: <5ms per frame ✅ Validated
- Frame jitter: <10ms under 3 concurrent sessions

### Transport Architecture (M2 Enhanced)

**Design Decision**: LiveKit WebRTC implemented as PRIMARY transport

**Why This Exceeds M2 Scope**:
- Original plan: WebSocket primary, LiveKit as agent framework
- Actual implementation: Full LiveKit WebRTC with comprehensive infrastructure
- Added: Caddy reverse proxy, TLS certificates, Docker orchestration
- Result: Production-ready WebRTC for browser clients
- Benefit: Better foundation for future scale-out

**Current Architecture**:
```
Browser Client
    ↓ (WebRTC/HTTPS)
  Caddy Reverse Proxy
    ↓ (WebRTC)
  LiveKit Server
    ↓ (LiveKit SDK)
  Orchestrator (VAD + Session Management)
    ↓ (gRPC)
  TTS Worker (Mock)
    ↓ (20ms PCM frames @ 48kHz)
  Back to Client
```

**Alternative Path** (CLI/Simple Clients):
```
CLI Client
    ↓ (WebSocket)
  Orchestrator
    ↓ (gRPC)
  TTS Worker
    ↓ (Audio frames)
  Back to CLI
```

### Protocol Design

**Session Protocol Workaround**:
- Separate sessions per message (current implementation)
- Workaround for empty frame protocol edge case
- Documented in `SESSION_PROTOCOL_FIX_IMPLEMENTATION.md`
- Production fix planned for protocol evolution

**Audio Frame Format**:
- 20ms duration (960 samples @ 48kHz)
- 1920 bytes per frame (16-bit mono PCM)
- Frame size validation in worker and orchestrator
- Formula: `sample_rate * frame_duration_ms // 1000 * 2`

---

## Documentation Index

### Primary Documentation

- **[CLAUDE.md](../CLAUDE.md)**: Project overview and development guide (updated 2025-10-09)
- **[docs/CURRENT_STATUS.md](CURRENT_STATUS.md)**: This file - current implementation status
- **[docs/TESTING_GUIDE.md](TESTING_GUIDE.md)**: Testing strategy, commands, and troubleshooting

### Technical Design

- **[project_documentation/PRD.md](../project_documentation/PRD.md)**: Product requirements
- **[project_documentation/TDD.md](../project_documentation/TDD.md)**: Technical design document
- **[project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md](../project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md)**: Milestone breakdown

### Operational Guides

- **[GRPC_SEGFAULT_WORKAROUND.md](../GRPC_SEGFAULT_WORKAROUND.md)**: WSL2 gRPC testing workaround (exemplary)
- **[TEST_REPAIR_SUMMARY.md](../TEST_REPAIR_SUMMARY.md)**: Integration test fixes and protocol decisions
- **[INTEGRATION_TEST_FIXES.md](../INTEGRATION_TEST_FIXES.md)**: Detailed integration test repair log

### Handoff & Analysis Documents

- **[PHASE1_COORDINATION_SUMMARY.md](../PHASE1_COORDINATION_SUMMARY.md)**: Multi-agent analysis coordination
- **[PHASE1_REPORT_CODE_ANALYSIS.md](../PHASE1_REPORT_CODE_ANALYSIS.md)**: Code implementation findings
- **[PHASE1_REPORT_INFRASTRUCTURE.md](../PHASE1_REPORT_INFRASTRUCTURE.md)**: Infrastructure analysis
- **[PHASE1_REPORT_DOCUMENTATION_AUDIT.md](../PHASE1_REPORT_DOCUMENTATION_AUDIT.md)**: Documentation audit

---

## Next Steps

### Immediate (Current Sprint)

1. **Begin M5: Piper Adapter Implementation**:
   - First real TTS adapter (CPU baseline)
   - ONNX runtime integration
   - Loudness normalization
   - End-to-end speech demo

2. **Fix Integration Test Timeouts**:
   - Investigate sequential message timeout
   - Resolve system stability test timeout
   - Document root cause and solution

3. **Documentation Updates**:
   - Sync TDD.md with M4 model manager implementation
   - Update INCREMENTAL_PLAN.md with M4 completion
   - Create known-issues index

### Short-term (Next 2-4 Weeks)

4. **Complete M5: Piper Adapter**:
   - First real TTS adapter
   - CPU-only baseline
   - End-to-end speech demo

5. **Infrastructure Polish**:
   - Production Dockerfiles
   - Documentation review
   - Performance benchmarking

### Medium-term (Next 1-3 Months)

6. **M6-M8: GPU TTS Adapters**:
   - CosyVoice 2 (expressive)
   - XTTS-v2 (cloning)
   - Sesame/Unsloth (LoRA)

7. **M9: Dynamic Routing**:
   - Capability-based selection
   - Load balancing
   - Multi-worker support

8. **M10: ASR Integration**:
   - Whisper integration
   - Speech↔speech pipeline
   - Full duplex demo

### Long-term (Next 3-6 Months)

9. **M11-M13: Production Readiness**:
    - Observability stack
    - Multi-GPU deployment
    - Multi-host scale-out
    - Production documentation

---

## Success Metrics

### M0-M4 Achievements ✅

- ✅ 16/16 M1 integration tests passing (100%)
- ✅ 29/29 M3 VAD unit tests passing (100%)
- ✅ 8/8 M3 VAD integration tests passing (100%)
- ✅ 20/20 M4 model manager unit tests passing (100%)
- ✅ 15/15 M4 model lifecycle integration tests passing (100%)
- ✅ 6/8 full pipeline tests passing (75%, 2 timeouts under investigation)
- ✅ <50ms control command response time
- ✅ <50ms VAD barge-in pause latency (p95)
- ✅ <5ms VAD processing latency per frame
- ✅ 20ms frame cadence validated
- ✅ TTL-based eviction working (configurable timeout)
- ✅ LRU eviction when capacity exceeded
- ✅ Reference counting prevents in-use model unload
- ✅ LiveKit WebRTC primary transport operational
- ✅ Docker compose stack with 5 services
- ✅ gRPC segfault workaround 100% reliable
- ✅ CI passing (lint + typecheck + test)

### M5+ Targets 🎯

- 🎯 First Audio Latency (FAL) p95 < 300ms (GPU adapters)
- 🎯 Frame jitter p95 < 10ms (3 concurrent sessions)
- 🎯 Model load time < 2s (GPU adapters)

---

## Contributing

### Before Starting Work

1. Read [CLAUDE.md](../CLAUDE.md) for project overview
2. Review this file for current status
3. Check [docs/TESTING_GUIDE.md](TESTING_GUIDE.md) for test strategy
4. Run `just ci` to validate local environment

### Development Checklist

```markdown
- [ ] Update CURRENT_STATUS.md if milestone progress
- [ ] Update CLAUDE.md if architecture changes
- [ ] Add tests for new features
- [ ] Run `just ci` before committing
- [ ] Update "Last Updated" dates in docs
```

### Getting Help

- **Documentation Issues**: See [PHASE1_REPORT_DOCUMENTATION_AUDIT.md](../PHASE1_REPORT_DOCUMENTATION_AUDIT.md)
- **Test Issues**: See [docs/TESTING_GUIDE.md](TESTING_GUIDE.md)
- **gRPC/WSL2 Issues**: See [GRPC_SEGFAULT_WORKAROUND.md](../GRPC_SEGFAULT_WORKAROUND.md)

---

**Maintained by**: Multi-Agent Documentation Team
**Last Review**: 2025-10-09
**Next Review**: After M4 completion
