# Realtime Duplex Voice Demo

A realtime speech↔speech system enabling low-latency conversations with barge-in support. This system supports hot-swapping across multiple open TTS models and runs on single-GPU and multi-GPU setups.

## Features

- **Realtime duplex conversation** with barge-in (pause/resume < 50 ms)
- **Streaming TTS** with 20 ms, 48 kHz PCM frames
- **Model modularity**: swap among multiple TTS models via unified streaming ABI
- **Dynamic model lifecycle**: default preload, runtime load/unload, TTL-based eviction
- **Scale**: single-GPU (two-process), multi-GPU (same host), multi-host (LAN)

## Architecture

### Two-tier streaming architecture

1. **Orchestrator** (LiveKit agent or equivalent):
   - WebRTC transport for browser clients, WS fallback for CLI
   - VAD (Voice Activity Detection) for interruption detection
   - ASR (Whisper small/distil) for speech-to-text
   - Session management and state machine (LISTENING → SPEAKING → BARGED_IN)
   - Routing logic: capability-aware, prefers resident models, Redis-based discovery

2. **TTS Workers** (one per GPU/adapter):
   - gRPC server implementing unified streaming ABI
   - Model Manager: handles load/unload, TTL eviction, warmup, LRU caching
   - Adapters: implement model-specific logic while conforming to shared interface
   - Emit 20 ms, 48 kHz mono PCM frames

## Quick Start

### Prerequisites

- Python 3.13.x
- [uv](https://github.com/astral-sh/uv) package manager
- Docker Engine 28.x with NVIDIA container runtime (for GPU workers)
- Redis (for service discovery)
- CUDA 12.8+ (for GPU workers)

### Installation

1. Clone the repository:
```bash
cd /home/gerald/git/full-duplex-voice-chat
```

2. Install dependencies:
```bash
uv sync --all-extras
```

3. Lock dependencies:
```bash
uv lock
```

4. Generate gRPC stubs:
```bash
just gen-proto
```

### Running Locally (Development)

1. Start Redis:
```bash
just redis
```

2. Run TTS worker:
```bash
just run-tts-cosy DEFAULT="cosyvoice2-en-base"
```

3. Run orchestrator:
```bash
just run-orch
```

4. Run CLI client:
```bash
just cli HOST="ws://localhost:8080"
```

### Running with Docker

Single command to start the full stack:
```bash
docker compose up --build
```

This starts:
- Redis (service discovery)
- Orchestrator (WebRTC/WS server)
- TTS worker (pinned to GPU 0)

## Development

### Quality Checks

```bash
# Run all CI checks
just ci

# Individual checks
just lint        # Ruff linting
just fix         # Auto-fix linting issues
just typecheck   # Mypy type checking
just test        # Pytest tests
```

### Code Generation

```bash
# Generate gRPC stubs from proto files
just gen-proto
```

### Profiling

```bash
# CPU profiling
just spy-top PID
just spy-record PID OUT="profile.svg"

# GPU profiling
just nsys-tts    # Nsight Systems trace
just ncu-tts     # Nsight Compute kernel analysis
```

## Project Structure

```
src/
├── orchestrator/      # WebRTC/WS transport, VAD, ASR, routing
├── tts/              # TTS workers with model management
│   ├── adapters/     # Model-specific implementations
│   ├── audio/        # Audio processing utilities
│   └── utils/        # Logging and timing utilities
├── rpc/              # gRPC service definitions
│   └── generated/    # Auto-generated stubs
└── client/           # CLI and web clients
    └── web/          # Browser client

tests/
├── unit/             # Unit tests
└── integration/      # Integration tests

configs/              # Configuration files
docs/                 # Additional documentation
```

## Configuration

- `configs/orchestrator.yaml` - Orchestrator settings (transport, VAD, ASR, routing)
- `configs/worker.yaml` - TTS worker settings (model manager, capabilities)
- `.env.example` - Environment variables template

## Documentation

- [PRD](project_documentation/PRD.md) - Product requirements
- [TDD](project_documentation/TDD.md) - Technical design document
- [Implementation Plan](project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md) - Milestone breakdown
- [CLAUDE.md](CLAUDE.md) - Claude Code guidance

## Supported TTS Models

- **CosyVoice 2** - GPU streaming TTS
- **XTTS-v2** - Expressive multi-speaker with voice cloning
- **Sesame** - With optional LoRA fine-tuning support
- **Piper** - CPU-only baseline

## Performance Targets

- Barge-in pause latency: p95 < 50 ms
- First Audio Latency (FAL): p95 < 300 ms (GPU), < 500 ms (CPU)
- Frame jitter: p95 < 10 ms under 3 concurrent sessions

## License

[Your License Here]

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
