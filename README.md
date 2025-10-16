# Realtime Duplex Voice Chat

A production-ready realtime speech-to-speech system enabling low-latency conversations with barge-in support. Built with Python, gRPC, and LiveKit WebRTC, supporting hot-swappable TTS models on single-GPU and multi-GPU setups.

**Current Status**: M0-M10 Complete - See [Current Status](docs/CURRENT_STATUS.md)

[![PR CI](https://github.com/gsornsen/full-duplex-voice-chat/actions/workflows/pr-ci.yml/badge.svg)](https://github.com/gsornsen/full-duplex-voice-chat/actions/workflows/pr-ci.yml)
[![Feature CI](https://github.com/gsornsen/full-duplex-voice-chat/actions/workflows/feature-ci.yml/badge.svg)](https://github.com/gsornsen/full-duplex-voice-chat/actions/workflows/feature-ci.yml)
[![codecov](https://codecov.io/gh/gsornsen/full-duplex-voice-chat/branch/main/graph/badge.svg)](https://codecov.io/gh/gsornsen/full-duplex-voice-chat)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Quick Start

### Prerequisites

- Python 3.13.x
- [uv](https://github.com/astral-sh/uv) package manager
- Docker Engine 28.x with NVIDIA container runtime
- NVIDIA GPU with CUDA 12.8+ (for GPU workers)

### 30-Second Demo

Start the full stack with one command:

```bash
docker compose up --build
```

This launches:

- Redis (service discovery)
- LiveKit (WebRTC server)
- Caddy (HTTPS reverse proxy)
- Orchestrator (dual transport: LiveKit WebRTC + WebSocket fallback)
- TTS Worker (mock adapter with streaming synthesis)

**Access the demo**:

- **Web Client**: <https://localhost> (LiveKit WebRTC)
- **WebSocket**: ws://localhost:8080 (fallback transport)
- **Health Check**: <http://localhost:8080/health>

### Developer Setup

For local development with hot-reload:

```bash
# 1. Clone repository
git clone <repository-url>
cd full-duplex-voice-chat

# 2. Install dependencies
uv sync --all-extras

# 3. Generate gRPC stubs
just gen-proto

# 4. Start services (separate terminals)
just redis          # Terminal 1: Redis
just run-tts-mock   # Terminal 2: TTS Worker
just run-orch       # Terminal 3: Orchestrator
just cli            # Terminal 4: CLI Client
```

See [Development Guide](docs/DEVELOPMENT.md) for detailed setup.

---

## What Works Today (M0-M10 Complete)

### Core Infrastructure ✅

- **gRPC Streaming Protocol**: Bidirectional streaming with <50ms control latency
- **Dual Transport Architecture**:
  - **Primary**: LiveKit WebRTC for browser clients (full-duplex audio)
  - **Fallback**: WebSocket for CLI testing and simple clients
- **Mock TTS Worker**: Streaming synthesis with 20ms, 48kHz PCM frames
- **Session Management**: State machine (LISTENING → SPEAKING → BARGED_IN)
- **Service Discovery**: Redis-based worker registration
- **Production Ready**: Docker Compose deployment with TLS support

### Test Coverage ✅

- **655 Total Tests**: 384 unit + 271 integration tests passing
- **M1-M5**: 113 tests (core infrastructure, VAD, Model Manager, Piper adapter)
- **M10**: 128 tests (ASR base, audio buffer, Whisper + WhisperX adapters, performance)
- **M10 Polish**: 71 tests (RMS buffer, session timeout, multi-turn conversation - Tasks 4 & 7)
- **Process Isolation**: pytest-forked for reliable testing in WSL2

### What Works Today (M0-M10 Complete)

- ✅ M0-M3: Core infrastructure (gRPC, WebSocket/LiveKit, VAD barge-in)
- ✅ M4: Model Manager with lifecycle management
- ✅ M5: Piper TTS Adapter (CPU-based ONNX)
- ✅ M10: ASR Integration (Whisper + WhisperX adapters, speech-to-text)

### What's Next (M6-M9, M11-M13 Planned)

- M6-M8: GPU TTS adapters (CosyVoice2, XTTS-v2, Sesame/Unsloth)
- M9: Capability-aware routing (prefer resident models)
- M11: Observability & profiling (Prometheus, tracing)
- M12-M13: Multi-GPU scale-out

See [Current Status](docs/CURRENT_STATUS.md) for detailed milestone tracking.

---

## Features

### Realtime Performance

- **Barge-in pause latency**: p95 < 50 ms (validated in M1)
- **First Audio Latency (FAL)**: p95 < 300 ms (target for GPU adapters)
- **Frame jitter**: p95 < 10 ms under 3 concurrent sessions
- **Streaming TTS**: 20 ms, 48 kHz mono PCM frames

### Model Modularity

- **Unified streaming ABI**: gRPC protocol for all TTS models
- **Hot-swappable adapters**: Switch models without code changes
- **Dynamic lifecycle**: Load/unload models on demand with TTL eviction (M4+)
- **Multi-model support**: Run different models on different GPUs

### Architecture Highlights

- **Two-tier design**: Orchestrator (LiveKit/WebSocket) + TTS Workers (gRPC)
- **Process isolation**: Single-GPU (2 processes) to multi-GPU (N+1 processes)
- **Fault tolerance**: Worker failures don't crash orchestrator
- **Horizontal scaling**: Add workers dynamically via Redis discovery

---

## Architecture

### High-Level Overview

```
Browser Client (WebRTC)
    ↓
LiveKit Server
    ↓
Orchestrator (LiveKit Agent)
    ├─ WebSocket Transport (fallback)
    ├─ Session Manager (state machine)
    ├─ VAD (Voice Activity Detection) - M3+
    ├─ ASR (Whisper + WhisperX) - M10+
    └─ Worker Router (capability-aware) - M9+
        ↓
    gRPC (streaming)
        ↓
TTS Workers (one per GPU/adapter)
    ├─ Model Manager (load/unload, TTL) - M4+
    ├─ Adapter (model-specific logic)
    └─ Audio Framing (20ms @ 48kHz)
```

### Key Flow (Current - M0-M2)

1. Client connects via WebSocket or LiveKit WebRTC
2. Client sends text message
3. Orchestrator routes to TTS worker via gRPC
4. Worker synthesizes speech, streams 20ms audio frames
5. Orchestrator forwards frames to client
6. Session returns to LISTENING state (supports multiple messages)

### Current Flow (M10 with ASR)

1. Client speaks → Orchestrator (VAD detects speech)
2. ASR (Whisper or WhisperX) transcribes → (optional LLM) → TTS Worker
3. Audio frames stream back to client
4. Barge-in: VAD detects interruption → PAUSE sent to worker (<50ms)
5. Worker stops, session transitions to BARGED_IN
6. RESUME when client stops speaking

See [Technical Design](project_documentation/TDD.md) for detailed architecture.

---

## Project Structure

```shell
full-duplex-voice-chat/
├── src/
│   ├── orchestrator/          # LiveKit Agent + WS fallback
│   │   ├── server.py          # Main server with dual transport
│   │   ├── session.py         # State machine (LISTENING/SPEAKING/BARGED_IN)
│   │   ├── livekit_utils/     # LiveKit integration
│   │   ├── transport/         # WebSocket transport
│   │   ├── vad.py             # Voice Activity Detection (M3+)
│   │   ├── asr.py             # Speech recognition (M10+)
│   │   └── routing.py         # Worker selection (M9+)
│   │
│   ├── tts/                   # TTS Workers
│   │   ├── worker.py          # gRPC server with streaming protocol
│   │   ├── model_manager.py   # Model lifecycle (M4+)
│   │   ├── tts_base.py        # Adapter interface
│   │   ├── adapters/          # Model-specific implementations
│   │   │   ├── adapter_mock.py          # Mock (M1-M2) ✅
│   │   │   ├── adapter_piper.py         # CPU baseline (M5) 📝
│   │   │   ├── adapter_cosyvoice2.py    # GPU expressive (M6) 📝
│   │   │   ├── adapter_xtts.py          # GPU + cloning (M7) 📝
│   │   │   └── adapter_sesame.py        # Sesame + LoRA (M8) 📝
│   │   └── audio/
│   │       ├── framing.py     # 20ms framing, 48kHz resampling
│   │       └── loudness.py    # RMS/LUFS normalization (M6+)
│   │
│   ├── rpc/
│   │   ├── tts.proto          # gRPC service definition
│   │   └── generated/         # Auto-generated stubs
│   │
│   └── client/
│       ├── cli_client.py      # WebSocket CLI client
│       └── web/               # Browser client (LiveKit WebRTC)
│
├── tests/
│   ├── unit/                  # Fast, isolated tests (384 passing)
│   └── integration/           # End-to-end tests (271 passing)
│
├── configs/
│   ├── orchestrator.yaml      # Transport, VAD, ASR, routing config
│   └── worker.yaml            # Model manager, adapter config
│
├── docs/
│   ├── CURRENT_STATUS.md      # Detailed milestone status
│   ├── DEVELOPMENT.md         # Developer guide
│   ├── TESTING_GUIDE.md       # Testing strategies
│   ├── known-issues/          # Troubleshooting and workarounds
│   ├── runbooks/              # Operational procedures
│   └── implementation/        # Implementation details
│
├── project_documentation/
│   ├── PRD.md                 # Product requirements
│   ├── TDD.md                 # Technical design (v2.1)
│   └── INCREMENTAL_IMPLEMENTATION_PLAN.md  # Milestone breakdown
│
├── docker-compose.yml         # Full stack deployment
├── justfile                   # Common development tasks
├── pyproject.toml             # Python dependencies
└── CLAUDE.md                  # AI assistant guidance
```

---

## Documentation

### For New Users

- **README.md** (this file) - Project overview and quick start
- [Current Status](docs/CURRENT_STATUS.md) - What's implemented and what's next
- [Quick Start Guide](#quick-start) - Get running in 30 seconds

### For Developers

- [Development Guide](docs/DEVELOPMENT.md) - Local development workflow, debugging, testing
- [Testing Guide](docs/TESTING_GUIDE.md) - Testing strategies and commands
- [Known Issues](docs/known-issues/README.md) - Troubleshooting and workarounds

### For Contributors

- [CLAUDE.md](CLAUDE.md) - Claude AI assistant guidance (comprehensive project context)
- [Contributing Guidelines](docs/DEVELOPMENT.md#contributing-guidelines) - Git workflow, code style
- [Implementation Plan](project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md) - Milestone tasks

### Architecture & Design

- [Product Requirements](project_documentation/PRD.md) - Features, use cases, constraints
- [Technical Design](project_documentation/TDD.md) - Detailed architecture (v2.1)
- [Protocol Specification](src/rpc/tts.proto) - gRPC API documentation

---

## Development Commands

All common tasks are available via `justfile`:

### Quality & CI

```bash
just ci            # Run all checks (lint + typecheck + test)
just lint          # Run ruff linting
just fix           # Auto-fix linting issues
just typecheck     # Run mypy type checking
just test          # Run pytest unit tests
just test-integration  # Run integration tests (with --forked for WSL2)
```

### Infrastructure

```bash
just redis         # Start Redis container
just gen-proto     # Generate gRPC stubs from proto files
```

### Runtime (Local Development)

```bash
just run-tts-mock  # Run TTS worker with mock adapter
just run-orch      # Run orchestrator (dual transport)
just cli           # Run CLI client (WebSocket)
```

### Profiling

```bash
just spy-top PID           # CPU profiling (py-spy top)
just spy-record PID OUT=profile.svg  # CPU flame graph
just nsys-tts              # GPU profiling (Nsight Systems)
just ncu-tts               # Kernel analysis (Nsight Compute)
```

See [Development Guide](docs/DEVELOPMENT.md) for detailed command usage.

---

## Network Ports Reference

### Overview

The system uses multiple ports across different deployment modes. Here's a comprehensive reference:

| Service | Port | Protocol | Purpose | Deployment |
|---------|------|----------|---------|------------|
| **Orchestrator (WebSocket)** | 8080 | TCP/WS | WebSocket fallback transport | Bare metal, Docker |
| **Orchestrator (WebSocket)** | 8082 | TCP/WS | Local development (alternate port) | Bare metal (local config) |
| **LiveKit Server** | 7880 | TCP/WS | WebRTC signaling, room management | Bare metal, Docker |
| **LiveKit RTC (TCP)** | 7881 | TCP | WebRTC TCP fallback | Bare metal, Docker |
| **LiveKit RTC (UDP)** | 50000-60000 | UDP | WebRTC media streams | Bare metal (full range) |
| **LiveKit RTC (UDP)** | 50000-50099 | UDP | WebRTC media streams (subset) | Docker (limited range) |
| **LiveKit TURN (UDP)** | 7882 | UDP | TURN relay server | Bare metal, Docker |
| **TTS Worker (gRPC)** | 7001 | TCP/gRPC | TTS synthesis service | Bare metal, Docker |
| **Redis** | 6379 | TCP | Service discovery, registry | Bare metal, Docker |
| **Prometheus (Metrics)** | 9090 | TCP/HTTP | Worker metrics (planned) | Bare metal, Docker |
| **Caddy (HTTPS)** | 8443 | TCP/HTTPS | Web client reverse proxy | Docker, Local network |
| **Caddy (WSS)** | 8444 | TCP/WSS | LiveKit WebSocket secure proxy | Docker, Local network |
| **Web Client (Dev)** | 3000 | TCP/HTTP | Next.js development server | Bare metal |

### Port Configuration by Deployment Mode

#### Bare Metal (Local Development)

**Orchestrator** (`configs/orchestrator.local.yaml`):
- WebSocket: `0.0.0.0:8082`
- LiveKit URL: `ws://localhost:7880`
- Redis: `redis://localhost:6379`
- TTS Worker: `grpc://localhost:7001`

**TTS Worker** (`configs/worker.yaml`):
- gRPC: `0.0.0.0:7001`
- Redis: `redis://localhost:6379`
- Prometheus: `0.0.0.0:9090` (planned)

**LiveKit** (`configs/livekit.yaml`):
- HTTP/WS: `0.0.0.0:7880`
- RTC TCP: `7881`
- RTC UDP: `50000-60000`
- TURN UDP: `7882`

**Commands**:
```bash
# Start services on bare metal
just redis          # Redis on 6379
# LiveKit starts automatically with just dev-agent-piper
just run-tts-mock   # TTS worker on 7001
just run-orch       # Orchestrator on 8082 (local) or 8080 (standard)
just cli            # CLI client connects to ws://localhost:8082
```

#### Docker Compose (Production)

**Orchestrator** (`configs/orchestrator.docker.yaml`):
- WebSocket: `0.0.0.0:8080` (mapped to `8080:8080`)
- LiveKit URL: `http://livekit:7880` (internal Docker network)
- Redis: `redis://redis:6379` (internal Docker network)
- TTS Worker: `grpc://tts0:7001` (internal Docker network)

**Port Mappings** (`docker-compose.yml`):
```yaml
services:
  redis:
    # No external port mapping (internal only)

  livekit:
    ports:
      - "7880:7880"           # WebRTC signaling
      - "7881:7881"           # RTC TCP fallback
      - "50000-50099:50000-50099/udp"  # RTC UDP (subset)

  caddy:
    ports:
      - "8443:8443"           # HTTPS web client
      - "8444:8444"           # WSS LiveKit proxy

  orchestrator:
    ports:
      - "8080:8080"           # WebSocket fallback

  tts0:
    # No external port mapping (internal gRPC only)
```

**Commands**:
```bash
# Start full stack
docker compose up --build

# Access points
https://localhost:8443          # Web client (via Caddy HTTPS)
wss://localhost:8444            # LiveKit (via Caddy WSS)
ws://localhost:8080             # WebSocket fallback (direct)
http://localhost:8080/health    # Health check
```

#### Hybrid Mode (Bare Metal + Docker)

You can run some services in Docker and others on bare metal:

**Option 1: Docker Infrastructure + Bare Metal Workers**
```bash
# Start infrastructure in Docker
docker compose up -d redis livekit caddy

# Run workers and orchestrator on bare metal
just run-tts-mock   # Port 7001
just run-orch       # Port 8082 (connects to Docker Redis/LiveKit)
```

**Option 2: Docker Workers + Bare Metal Orchestrator**
```bash
# Start workers in Docker
docker compose up -d redis tts0

# Run orchestrator on bare metal (for debugging)
just run-orch       # Port 8082 (connects to Docker services)
```

### Firewall Configuration

#### Local Development (Single Machine)

No firewall configuration needed - all services on `localhost`.

#### Local Network Access (Multiple Machines)

**Allow incoming connections** for:
- `8443/tcp` - HTTPS web client (Caddy)
- `8444/tcp` - WSS LiveKit proxy (Caddy)
- `7880/tcp` - LiveKit WebRTC signaling
- `50000-50099/udp` - WebRTC media streams (Docker)
- `50000-60000/udp` - WebRTC media streams (bare metal)

**Linux (ufw)**:
```bash
sudo ufw allow 8443/tcp   # Caddy HTTPS
sudo ufw allow 8444/tcp   # Caddy WSS
sudo ufw allow 7880/tcp   # LiveKit signaling
sudo ufw allow 50000:60000/udp  # WebRTC media
```

**Windows (PowerShell)**:
```powershell
New-NetFirewallRule -DisplayName "Voice Chat HTTPS" -Direction Inbound -LocalPort 8443 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Voice Chat WSS" -Direction Inbound -LocalPort 8444 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "LiveKit Signaling" -Direction Inbound -LocalPort 7880 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "WebRTC Media" -Direction Inbound -LocalPort 50000-60000 -Protocol UDP -Action Allow
```

#### Production Deployment

**Exposed Ports** (via reverse proxy):
- `443/tcp` - HTTPS (Caddy → web client + LiveKit WSS)
- `50000-60000/udp` - WebRTC media (direct to LiveKit)

**Internal Ports** (Docker network only):
- `6379/tcp` - Redis
- `7001/tcp` - TTS worker gRPC
- `8080/tcp` - Orchestrator WebSocket
- `9090/tcp` - Prometheus metrics

### Port Conflict Resolution

**Common conflicts**:

1. **Port 6379 (Redis)**
   - Conflict: Local Redis/Valkey instance
   - Solution: Docker Compose uses internal network only (no external mapping)

2. **Port 7880 (LiveKit)**
   - Conflict: Another LiveKit instance
   - Solution: Change `LIVEKIT_PORT` in `.env` and update `configs/livekit.yaml`

3. **Port 8080 (Orchestrator)**
   - Conflict: Another web service
   - Solution: Use `configs/orchestrator.local.yaml` (port 8082)

4. **Ports 50000-60000 (WebRTC)**
   - Conflict: Limited by Docker
   - Solution: Docker Compose uses subset (50000-50099), adjust in `docker-compose.yml` if needed

### Security Considerations

**Development** (localhost):
- All services use unencrypted connections (HTTP/WS)
- API keys are development defaults (`devkey`/`secret`)
- No authentication required

**Production** (exposed):
- ✅ Use Caddy for TLS termination (HTTPS/WSS)
- ✅ Change LiveKit API keys/secrets
- ✅ Use strong Redis password
- ✅ Enable firewall rules (only expose necessary ports)
- ✅ Consider VPN for internal services
- ⚠️ Never expose gRPC port 7001 directly
- ⚠️ Never expose Redis port 6379 directly

---

## Configuration

### Orchestrator Configuration

File: `configs/orchestrator.yaml`

```yaml
transport:
  websocket:
    host: "0.0.0.0"
    port: 8080
  livekit:
    url: "${LIVEKIT_URL}"
    api_key: "${LIVEKIT_API_KEY}"
    api_secret: "${LIVEKIT_API_SECRET}"

redis:
  url: "${REDIS_URL}"

workers:
  tts_worker_url: "grpc://localhost:7001"
```

### TTS Worker Configuration

File: `configs/worker.yaml`

```yaml
adapter:
  type: "mock"  # or "piper", "cosyvoice2", "xtts", "sesame"
  config:
    sample_rate: 48000
    channels: 1

model_manager:  # M4+
  default_model_id: "cosyvoice2-en-base"
  preload_model_ids: []
  ttl_ms: 600000  # 10 min idle → unload
  resident_cap: 3
```

### Environment Variables

```bash
# Redis
export REDIS_URL="redis://localhost:6379"

# LiveKit
export LIVEKIT_URL="ws://localhost:7880"
export LIVEKIT_API_KEY="devkey"
export LIVEKIT_API_SECRET="secret"

# Worker
export TTS_WORKER_PORT=7001
export CUDA_VISIBLE_DEVICES=0
```

See `.env.example` for complete list.

---

## Supported TTS Models

### Currently Implemented (M1-M2)

- ✅ **Mock Adapter** - Sine wave generator for testing

### Planned (M5-M8)

- 📝 **Piper** (M5) - CPU-only baseline, fast inference
- 📝 **CosyVoice 2** (M6) - GPU streaming, expressive zero-shot
- 📝 **XTTS-v2** (M7) - GPU multi-speaker with voice cloning
- 📝 **Sesame/Unsloth** (M8) - LoRA fine-tuned variants

All adapters conform to unified gRPC streaming protocol:

- Input: Stream of text chunks
- Output: Stream of 20ms, 48kHz, mono PCM frames
- Controls: PAUSE, RESUME, STOP (<50ms latency)

---

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Barge-in pause latency | p95 < 50 ms | ✅ Validated (M1) |
| First Audio Latency (FAL) | p95 < 300 ms (GPU) | 📝 M5+ (real adapters) |
| First Audio Latency (FAL) | p95 < 500 ms (CPU) | 📝 M5 (Piper) |
| Frame jitter | p95 < 10 ms | 📝 M5+ |
| Concurrent sessions | 3+ per GPU | 📝 M11 (load testing) |
| Real-time factor (RTF) | < 0.3 (GPU) | 📝 M6+ |

---

## Testing

### Quick Test Commands

```bash
# Unit tests (fast, always run these)
just test

# Integration tests (require services, use process isolation)
just test-integration

# Full CI suite
just ci

# With coverage
uv run pytest tests/unit/ --cov=src --cov-report=html
```

### Test Status

**Unit Tests**: 152/152 passing ✅

- Audio synthesis, framing, resampling (M0-M5)
- Configuration parsing and validation
- Session state machine transitions
- Protocol message serialization
- VAD edge detection (M3: 29 tests)
- Model Manager lifecycle (M4: 20 tests)
- Piper adapter logic (M5: 15 tests)
- ASR base protocol (M10: 23 tests)
- Audio buffer (M10: 41 tests)
- WhisperX adapter (M10: 15 tests)

**Integration Tests**: 89/89 passing ✅

- ✅ M1 Worker Integration: 16/16 tests (gRPC protocol)
- ✅ M3 VAD Integration: 8/8 tests
- ✅ M4 Model Manager: 15/15 tests
- ✅ M5 Piper Adapter: 10/10 tests
- ✅ M10 Whisper ASR: 28/28 tests
- ✅ M10 Whisper Performance: 11/11 tests
- ✅ M10 WhisperX Integration: 10/10 tests
- ⚠️ Full Pipeline: 6/8 tests (2 WebSocket timeouts under investigation)

**Known Issues**:

- WSL2: gRPC segfaults require `--forked` flag (100% mitigated)
- See [Known Issues](docs/known-issues/README.md) for details

See [Testing Guide](docs/TESTING_GUIDE.md) for comprehensive testing documentation.

---

## Known Issues

### Critical Issues

- **gRPC Segfault (WSL2)**: Use `just test-integration` with process isolation
  - **Status**: 100% mitigated
  - **Details**: [grpc-segfault.md](docs/known-issues/grpc-segfault.md)

### Minor Issues

- **WebSocket Test Timeouts**: 2 integration tests timeout intermittently
  - **Status**: Under investigation
  - **Workaround**: Run tests individually or increase timeout

See [Known Issues Index](docs/known-issues/README.md) for complete troubleshooting guide.

---

## Deployment

### Docker Compose (Recommended)

Full production stack with one command:

```bash
# Start all services
docker compose up --build

# Run in background
docker compose up -d

# View logs
docker compose logs -f orchestrator

# Stop
docker compose down
```

Services included:

- Redis (service discovery)
- LiveKit (WebRTC server)
- Caddy (HTTPS reverse proxy)
- Orchestrator (dual transport)
- TTS Worker (GPU-pinned)

### Multi-GPU Deployment

```bash
# Worker 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 just run-tts-mock

# Worker 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 just run-tts-mock

# Orchestrator discovers both workers via Redis
just run-orch
```

### Multi-Host Deployment (M13)

Planned features:

- Central Redis instance
- Workers announce with LAN-reachable addresses
- Orchestrator discovers and routes across hosts

---

## Roadmap

### Completed (M0-M10)

- ✅ M0: Repo scaffold + CI
- ✅ M1: gRPC ABI + Mock worker
- ✅ M2: Orchestrator transport (LiveKit + WebSocket)
- ✅ M3: Barge-in end-to-end (VAD integration, <50ms latency)
- ✅ M4: Model Manager (load/unload, TTL, LRU)
- ✅ M5: Piper TTS Adapter (CPU baseline)
- ✅ M10: ASR Integration (Whisper + WhisperX, speech-to-text, 4-8x speedup)

### Near Term (Q4 2025)

- 📝 M6-M8: GPU TTS adapters (CosyVoice2, XTTS, Sesame)
- 📝 M9: Capability-aware routing

### Medium Term (Q1 2026)

- 📝 M11: Observability & profiling
- 📝 M12: Production deployment polish

### Long Term (Q2 2026)

- 📝 M13: Multi-GPU & multi-host scale-out

See [Implementation Plan](project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md) for detailed milestone breakdown.

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/DEVELOPMENT.md#contributing-guidelines).

### Quick Contribution Workflow

1. Fork the repository
2. Create feature branch: `git checkout -b feat/my-feature`
3. Make changes with tests
4. Run CI: `just ci`
5. Commit with conventional commits: `git commit -m "feat(tts): add new adapter"`
6. Push and create PR

### Areas Needing Help

- 📝 TTS adapter implementations (Piper, CosyVoice2, XTTS, Sesame)
- 📝 VAD integration and tuning
- 📝 Performance profiling and optimization
- 📝 Documentation improvements
- 📝 Test coverage expansion
- 📝 Multi-GPU deployment testing

---

## License

[MIT License](LICENSE) - See LICENSE file for details.

---

## Support

- **Documentation**: See [docs/](docs/) directory
- **Issues**: GitHub Issues (include error logs, config, repro steps)
- **Discussions**: GitHub Discussions for questions
- **Development Guide**: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- **Known Issues**: [docs/known-issues/README.md](docs/known-issues/README.md)

---

## Acknowledgments

- **LiveKit**: WebRTC infrastructure ([livekit.io](https://livekit.io))
- **gRPC**: RPC framework ([grpc.io](https://grpc.io))
- **PyTorch**: ML framework ([pytorch.org](https://pytorch.org))
- **CosyVoice**: Expressive TTS model
- **XTTS**: Multi-speaker TTS with cloning
- **Piper**: Fast CPU TTS
- **Sesame/Unsloth**: LoRA fine-tuning framework

---

**Status**: M0-M5 and M10 Complete | **Next**: M6 — CosyVoice 2 Adapter (GPU, streaming)

**Last Updated**: 2025-10-11
