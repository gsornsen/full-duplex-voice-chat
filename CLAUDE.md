# CLAUDE.md

**Last Updated**: 2025-10-17

This file provides essential guidance to Claude Code when working with this repository. For detailed documentation, see the [.claude/modules/](.claude/modules/) directory.

## Project Overview

This is a **Realtime Duplex Voice Demo** system enabling low-latency speech↔speech conversations with barge-in support. The system supports hot-swapping across multiple open TTS models and runs on single-GPU and multi-GPU setups.

**Current Status**: Milestones M0-M10 complete (including M10 Polish), M11-M13 planned.

> 📖 **Detailed Status**: See [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md) and [.claude/modules/milestones.md](.claude/modules/milestones.md)

**Key Capabilities:**
- Realtime duplex conversation with barge-in (pause/resume < 50 ms) ✅
- Streaming TTS with 20 ms, 48 kHz PCM frames ✅
- Model modularity with unified streaming ABI ✅
- Dynamic model lifecycle (load/unload, TTL-based eviction) ✅
- ASR integration (Whisper/WhisperX) with multi-turn conversations ✅
- Adaptive noise gating for reduced false barge-ins ✅

## Quick Start

### Essential Commands

```bash
# Quality & CI
just lint          # Run ruff linting
just fix           # Auto-fix linting issues
just typecheck     # Run mypy type checking
just test          # Run pytest tests
just ci            # Run all checks (lint + typecheck + test)

# Infrastructure
just redis         # Start Redis container

# Runtime
just run-tts-piper # Run Piper TTS worker (CPU)
just run-orch      # Run orchestrator (WebRTC/WS)
just cli           # Run CLI client
```

> 🔧 **Full Command Reference**: See [.claude/modules/development.md](.claude/modules/development.md)

### Development Workflow

1. **Start a session**: Load context from CLAUDE.md (this file) - minimal tokens
2. **Working on specific feature?** Claude Code will suggest loading relevant module:
   - VAD/Barge-in → [.claude/modules/features/vad.md](.claude/modules/features/vad.md)
   - ASR/Whisper → [.claude/modules/features/asr.md](.claude/modules/features/asr.md)
   - TTS Adapters → [.claude/modules/adapters/](.claude/modules/adapters/)
   - CI/CD → [.claude/modules/testing.md#ci-cd-pipeline](.claude/modules/testing.md#ci-cd-pipeline)
3. **Need specialized agent?** See [.claude/agents/README.md](.claude/agents/README.md)

## Architecture Summary

**Two-tier streaming architecture:**

1. **Orchestrator** (`src/orchestrator/`)
   - Primary: LiveKit WebRTC for browser clients
   - Secondary: WebSocket fallback for CLI
   - VAD (M3): Voice Activity Detection for barge-in (<50ms)
   - ASR (M10): Whisper/WhisperX for speech-to-text
   - Session Management: Multi-turn conversations with idle timeout

2. **TTS Workers** (`src/tts/`)
   - gRPC server implementing unified streaming ABI
   - Model Manager: handles load/unload, TTL eviction, LRU caching
   - Adapters: Piper (CPU baseline) ✅, CosyVoice2/XTTS/Sesame (planned)
   - Output: 20 ms frames, 48 kHz mono PCM

**Flow (M10 with ASR):**
```
Client speaks → Orchestrator (VAD + ASR) → Text transcript → TTS Worker → Audio frames → Client
                                                   ↓
                                             (optional LLM)
```

> 📖 **Detailed Architecture**: See [.claude/modules/architecture.md](.claude/modules/architecture.md)

## Code Structure

```
src/
├─ orchestrator/        # LiveKit Agent + WebSocket server
│  ├─ server.py         # Main orchestrator with session management
│  ├─ vad.py            # Voice Activity Detection (M3)
│  ├─ vad_processor.py  # VAD with adaptive noise gate (M10 Polish)
│  └─ session.py        # Session state machine
├─ asr/                 # ASR adapters (M10)
│  └─ adapters/         # Whisper, WhisperX implementations
├─ tts/                 # TTS workers
│  ├─ worker.py         # gRPC server + ModelManager
│  ├─ model_manager.py  # Model lifecycle (M4)
│  ├─ tts_base.py       # Adapter protocol
│  └─ adapters/         # Mock, Piper, CosyVoice2, XTTS, Sesame
└─ rpc/                 # gRPC protocol definitions
   └─ tts.proto         # Streaming ABI
```

> 📖 **Detailed Code Map**: See [.claude/modules/architecture.md#code-structure](.claude/modules/architecture.md#code-structure)

## Mandatory Acceptance Criteria

**IMPORTANT**: Every completed feature, milestone, or code change MUST satisfy ALL criteria before being considered complete:

### Code Quality (REQUIRED)
1. ✅ `just test` passes (all tests)
2. ✅ `just lint` passes (ruff clean)
3. ✅ `just typecheck` passes (mypy strict)
4. ✅ `just ci` passes (combined check)

### Documentation (REQUIRED)
5. ✅ Update CLAUDE.md (if core changes)
6. ✅ Update [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md) (milestone status)
7. ✅ Update [project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md](project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md) (exit criteria)
8. ✅ Complete docstrings for public APIs
9. ✅ Configuration file comments

### Documentation Audit (REQUIRED for milestones)
10. ✅ Coordinate specialized agents:
    - `@documentation-engineer` - Cross-document consistency
    - `@devops-engineer` - CI/CD and deployment docs
    - `@python-pro` - Code quality and testing

> 📖 **Agent Coordination**: See [.claude/agents/README.md](.claude/agents/README.md)

### Commit & PR (REQUIRED)
11. ✅ Generate `/tmp/M{N}_commit.msg` (conventional commits format)
12. ✅ Generate `/tmp/M{N}_pr.description` (comprehensive PR description)

**Enforcement**: No milestone should be marked complete unless ALL criteria are met.

> 📖 **Detailed Acceptance Criteria**: See [.claude/modules/milestones.md#acceptance-criteria](.claude/modules/milestones.md#acceptance-criteria)

## Testing Strategy

**Quick Summary:**
- **Total**: 649 tests (500 unit + 139 integration + 10 performance)
- **M0-M5**: 113 tests (core, VAD, Model Manager, Piper)
- **M10 ASR**: 128 tests (Whisper adapters, audio buffer)
- **M10 Polish**: 65 tests (RMS buffer, session timeout, multi-turn)

**Run Tests:**
```bash
just test              # All tests
just test-integration  # Integration only (with --forked for gRPC)
```

**CI Strategy:**
- Feature Branch CI: 3-5 min (smart test selection)
- PR CI: 10-15 min (full suite + coverage)
- Main Branch: no CI (quality guaranteed by PR gates)

> 🧪 **Detailed Testing Guide**: See [.claude/modules/testing.md](.claude/modules/testing.md)
>
> 🚀 **CI/CD Deep Dive**: See [.claude/modules/testing.md#ci-cd-pipeline](.claude/modules/testing.md#ci-cd-pipeline)

## Development Environment

**Python & Tooling:**
- Python 3.13.x managed with **uv**
- `ruff` for linting, `mypy` for type checking (strict mode)
- `pytest` for tests, `justfile` for tasks

**Platform:**
- CUDA 12.8 + PyTorch 2.7.0 for main workers (Orchestrator, Whisper, future XTTS/Sesame)
- ⚠️ **M6 CosyVoice 2**: Requires PyTorch 2.3.1 + CUDA 12.1 (isolated Docker container)
  - See [docs/COSYVOICE_PYTORCH_CONFLICT.md](docs/COSYVOICE_PYTORCH_CONFLICT.md) for details
- Docker 28.x with NVIDIA container runtime
- Redis for worker service discovery
- **WSL2 Note**: gRPC tests require `--forked` flag (see [GRPC_SEGFAULT_WORKAROUND.md](GRPC_SEGFAULT_WORKAROUND.md))

> 🔧 **Full Environment Setup**: See [.claude/modules/development.md](.claude/modules/development.md)

## Performance Targets

**Latency SLAs:**
- Barge-in pause: p95 < 50 ms ✅
- VAD processing: p95 < 5 ms per frame ✅
- VAD with noise gate: p95 < 1 ms overhead ✅
- First Audio Latency (FAL): p95 < 300 ms (GPU), < 500 ms (Piper CPU) ✅
- Frame jitter: p95 < 10 ms ✅
- ASR transcription: p95 < 1.5s (CPU), < 1.0s (GPU) ✅

> 📊 **Performance Metrics**: See [docs/PERFORMANCE.md](docs/PERFORMANCE.md)

## Implementation Milestones

| Milestone | Status | Description |
|-----------|--------|-------------|
| M0 | ✅ Complete | Repo scaffold + CI |
| M1 | ✅ Complete | gRPC ABI + Mock worker |
| M2 | ✅ Enhanced | Orchestrator transport (LiveKit primary) |
| M3 | ✅ Complete | Barge-in end-to-end (VAD integration) |
| M4 | ✅ Complete | Model Manager v1 (TTL/LRU eviction) |
| M5 | ✅ Complete | Piper adapter (CPU baseline) |
| M6-M8 | 📝 Planned | GPU TTS adapters (CosyVoice2, XTTS, Sesame) |
| M9 | 📝 Planned | Routing v1 (capability-aware) |
| M10 | ✅ Complete | ASR integration (Whisper + WhisperX) |
| M10 Polish | ✅ Complete | Session timeout, adaptive noise gate |
| M11 | 📝 Planned | Observability & profiling |
| M12 | 📝 Planned | Docker/Compose smoke; docs polish |
| M13 | 📝 Planned | Multi-GPU & multi-host scale-out |

> 📖 **Detailed Milestone Plan**: See [.claude/modules/milestones.md](.claude/modules/milestones.md)

## Important Patterns

**Adapter Implementation:**
- Inherit from `tts_base.py` protocol
- Repacketize to 20 ms frames at 48 kHz
- Respect PAUSE/RESUME/STOP (<50 ms)
- Handle native sample rate → 48kHz resampling

> 🔧 **Adapter Template**: See [.claude/modules/adapters/template.md](.claude/modules/adapters/template.md)

**Voicepack Structure:**
- Standard directory layout for TTS model files
- Metadata format for Model Manager registration
- See [docs/VOICEPACK_COSYVOICE2.md](docs/VOICEPACK_COSYVOICE2.md) for CosyVoice 2 voicepacks

**State Machine (Orchestrator):**
```
IDLE → LISTENING → SPEAKING → BARGED_IN
         ↑           ↓            ↓
         ← WAITING_FOR_INPUT ←────┘
                    ↓
               TERMINATED
```

**Session Lifecycle:**
- Multi-turn conversations (persist between interactions)
- Idle timeout: 5 minutes (configurable)
- Session limits: 1 hour duration, 100 messages

> 📖 **Session Management**: See [.claude/modules/features/session-management.md](.claude/modules/features/session-management.md)

## Docker & Deployment

**Quick Start:**
```bash
docker compose up --build
```

Starts: Redis + LiveKit + Caddy + Orchestrator + TTS worker

**Multi-GPU (same host):**
```bash
CUDA_VISIBLE_DEVICES=0 just run-tts-piper  # Worker 0 (CPU)
CUDA_VISIBLE_DEVICES=1 just run-tts-mock   # Worker 1 (GPU, when available)
just run-orch
```

> 🚀 **Deployment Guide**: See [.claude/modules/deployment.md](.claude/modules/deployment.md)

## Security Notes

- **Audio retention**: off by default
- **Auth**: API key for demo; mTLS optional for production
- **TLS**: terminate at nginx/traefik for WAN exposure
- **Isolation**: workers pinned to devices; crashes don't affect other sessions

## References

**Core Documentation:**
- [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md) - Implementation status
- [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) - Testing commands and strategy
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) - Development workflows
- [docs/VOICEPACK_COSYVOICE2.md](docs/VOICEPACK_COSYVOICE2.md) - CosyVoice 2 voicepack structure

**Project Documentation:**
- [project_documentation/PRD.md](project_documentation/PRD.md) - Product requirements
- [project_documentation/TDD.md](project_documentation/TDD.md) - Technical design (v2.1)
- [project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md](project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md) - Milestone breakdown

**Modular Documentation (`.claude/modules/`):**
- [development.md](.claude/modules/development.md) - Development environment and tooling
- [architecture.md](.claude/modules/architecture.md) - Detailed architecture and flows
- [testing.md](.claude/modules/testing.md) - Testing strategy and CI/CD
- [milestones.md](.claude/modules/milestones.md) - Implementation milestones and tasks
- [features/vad.md](.claude/modules/features/vad.md) - VAD implementation details
- [features/asr.md](.claude/modules/features/asr.md) - ASR integration details
- [adapters/piper.md](.claude/modules/adapters/piper.md) - Piper TTS adapter reference

**Agent Guides:**
- [.claude/agents/README.md](.claude/agents/README.md) - Agent coordination and roles
- [.claude/agents/documentation-engineer.md](.claude/agents/documentation-engineer.md) - Documentation audit guide
- [.claude/agents/devops-engineer.md](.claude/agents/devops-engineer.md) - CI/CD and deployment guide
- [.claude/agents/python-pro.md](.claude/agents/python-pro.md) - Code quality guide

---

**Token Optimization Note**: This file is intentionally streamlined (~360 lines) to minimize context load. Detailed documentation is split into focused modules that can be loaded on-demand based on the task at hand. Original CLAUDE.md backed up as `CLAUDE.md.backup`.
