# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### In Progress
- Browser web client implementation (M3)
- Real TTS model adapters (M4-M8)
- Multi-worker routing with capability awareness (M9)
- ASR integration for full speech-to-speech (M10)

---

## [0.2.0] - M2 - 2025-10-05

Milestone 2 introduces WebSocket transport, CLI client, mock TTS worker, and comprehensive documentation.

### Added

**Transport & Clients:**
- WebSocket transport for browser and CLI clients
- CLI WebSocket client (`src/client/cli_client.py`) with real-time audio playback
- Session management with unique session IDs
- Control commands (PAUSE, RESUME, STOP) for barge-in support
- Audio frame streaming (20ms @ 48kHz PCM)

**TTS Worker:**
- Mock TTS worker with 440Hz sine wave adapter for testing
- gRPC server implementation with streaming synthesis
- Worker registration and heartbeat to Redis
- Configurable worker capabilities advertisement

**Service Discovery:**
- Redis-based worker registry and discovery
- Worker heartbeat mechanism with TTL-based expiration
- Static routing configuration (single worker)

**VAD (Voice Activity Detection):**
- VAD stub implementation for barge-in detection
- Configurable aggressiveness levels (0-3)
- WebRTC VAD integration with 16kHz sample rate

**Integration Tests:**
- Redis service discovery tests
- Worker registration and TTL validation
- VAD speech/silence detection tests
- WebSocket connection tests
- End-to-end synthesis flow tests

**Documentation:**
- Quick Start Guide (`docs/QUICKSTART.md`)
- WebSocket Protocol Specification (`docs/WEBSOCKET_PROTOCOL.md`)
- CLI Client Usage Guide (`docs/CLI_CLIENT_GUIDE.md`)
- Configuration Reference (`docs/CONFIGURATION_REFERENCE.md`)
- Docker Setup & Troubleshooting Guide (`docs/setup/DOCKER_SETUP.md`)
- Comprehensive code examples (Python and JavaScript)
- Mermaid sequence diagrams for protocol flows

**Observability:**
- Health check endpoints (HTTP for orchestrator, gRPC for worker)
- Structured JSON logging with session IDs
- Log level configuration (DEBUG, INFO, WARNING, ERROR, CRITICAL)

**Configuration:**
- Value range comments in YAML configs
- M4+/M9+ feature markers for future milestones
- Environment variable override support
- Pydantic validation for orchestrator config

**Development Tools:**
- Pre-flight check script for system validation
- Docker Compose setup for single-command deployment
- Just commands for common tasks (run, test, lint, etc.)
- CI pipeline with ruff, mypy, pytest

### Changed
- Port consistency: Standardized on 8080 for WebSocket orchestrator
- Configuration documentation: Added valid ranges and defaults for all fields
- Error messages: Improved clarity with context and resolution steps
- README: Updated Quick Start section with correct WebSocket port

### Fixed
- Port number inconsistency: README line 81 (8080 vs 8081)
- Type annotations in integration tests for mypy strict mode
- Linting errors across test files
- VAD example code clarity (sync vs async iteration)

### Security
- API key/secret placeholder values in configs (use env vars for production)
- Documentation warnings about secret management
- Redis auth support for production deployments

---

## [0.1.0] - M1 - 2025-09-XX

Milestone 1 establishes the gRPC ABI, transport abstraction, and configuration framework.

### Added

**gRPC Protocol:**
- `src/rpc/tts.proto`: Complete TTS service definition
- Streaming synthesis API (TextChunk → AudioFrame)
- Session lifecycle management (StartSession, EndSession)
- Control commands (PAUSE, RESUME, STOP, RELOAD)
- Model management API (ListModels, LoadModel, UnloadModel)
- Capabilities reporting API
- Auto-generated gRPC stubs

**Transport Abstraction:**
- Transport layer interface for WebSocket and LiveKit
- Session management framework with state machine
- Message routing between transport and TTS worker

**Configuration System:**
- Pydantic models for type-safe configuration
- YAML-based config files (`orchestrator.yaml`)
- Modular config structure (transport, redis, routing, vad)
- Environment variable override support

**Worker Routing:**
- Basic routing logic framework
- Static worker address configuration
- gRPC client connection management

**Development Environment:**
- Python 3.13 project setup with uv
- Code quality tooling (ruff, mypy, pytest)
- CI pipeline configuration
- Just commands for automation

**Documentation:**
- Product Requirements Document (PRD)
- Technical Design Document (TDD v2.1)
- Incremental Implementation Plan
- CLAUDE.md for AI-assisted development

### Changed
- Project structure organized into logical modules
- Dependency management via uv and pyproject.toml

---

## [0.0.1] - M0 - 2025-09-XX

Milestone 0 creates the repository scaffold and project foundation.

### Added

**Repository Setup:**
- Git repository initialization
- `.gitignore` for Python, Docker, IDE files
- License file
- README.md with project overview

**Project Documentation:**
- Product Requirements Document (PRD.md)
- Technical Design Document (TDD.md)
- Implementation Plan (INCREMENTAL_IMPLEMENTATION_PLAN.md)
- Milestone task checklists

**Development Environment:**
- Python 3.13 environment setup
- uv package manager integration
- pyproject.toml for dependencies
- .python-version file

**Code Quality Tooling:**
- ruff for linting and formatting
- mypy for static type checking
- pytest for testing framework
- pre-commit hooks configuration

**Directory Structure:**
```
src/
├── orchestrator/  # Orchestrator module placeholder
├── tts/          # TTS worker module placeholder
├── rpc/          # gRPC definitions
└── client/       # Client implementations

tests/
├── unit/         # Unit tests
└── integration/  # Integration tests

configs/          # Configuration files
docs/            # Documentation
```

**Build & Deployment:**
- Dockerfile templates
- docker-compose.yml scaffold
- Justfile for task automation

---

## Template for Future Milestones

### M3: LiveKit Transport & Browser Client
**Target:** Full WebRTC support with browser client

**Planned Features:**
- LiveKit transport implementation
- Browser WebSocket client (HTML + JavaScript)
- WebRTC audio streaming
- Interactive web UI

---

### M4: Model Manager v1
**Target:** Dynamic model lifecycle management

**Planned Features:**
- Model Manager implementation with TTL eviction
- Default model preloading
- Warmup on model load
- LRU eviction when resident_cap exceeded
- LoadModel/UnloadModel gRPC endpoints

---

### M5: Piper Adapter (CPU Baseline)
**Target:** First real TTS adapter (CPU-only)

**Planned Features:**
- Piper TTS adapter implementation
- CPU-based synthesis
- Voice pack format implementation
- Model metadata support

---

### M6: CosyVoice 2 Adapter (GPU Streaming)
**Target:** High-quality GPU-based streaming TTS

**Planned Features:**
- CosyVoice 2 adapter implementation
- GPU acceleration
- Streaming synthesis
- Multi-language support (en, zh)

---

### M7: XTTS-v2 Adapter (Voice Cloning)
**Target:** Expressive TTS with zero-shot voice cloning

**Planned Features:**
- XTTS-v2 adapter implementation
- Zero-shot voice cloning from reference audio
- Expressive synthesis
- Multi-speaker support

---

### M8: Sesame/Unsloth Adapter (LoRA)
**Target:** Fine-tunable TTS with LoRA support

**Planned Features:**
- Sesame base adapter
- Unsloth LoRA variant
- LoRA model loading
- Fine-tuning workflow

---

### M9: Routing v1 (Capabilities + Prefer Resident)
**Target:** Multi-worker dynamic routing

**Planned Features:**
- Capability-aware routing
- Prefer resident models
- Load balancing strategies (queue_depth, round_robin, latency)
- Worker health tracking
- Automatic failover

---

### M10: ASR Integration (Full Speech↔Speech)
**Target:** Complete voice-to-voice pipeline

**Planned Features:**
- Whisper ASR integration (small/distil)
- Full duplex audio flow
- VAD-triggered ASR
- Optional LLM layer
- Speech-to-speech latency < 1s

---

### M11: Observability & Profiling
**Target:** Production monitoring and debugging

**Planned Features:**
- Prometheus metrics export
- Grafana dashboards
- OpenTelemetry tracing
- Profiling tools (py-spy, nsight)
- Performance benchmarks

---

### M12: Docker/Compose Smoke Tests & Docs Polish
**Target:** Production-ready packaging

**Planned Features:**
- Smoke test suite for Docker deployment
- Multi-GPU Docker Compose examples
- Production deployment guide
- Security hardening
- Documentation completeness review

---

### M13: Multi-GPU & Multi-Host Scale-Out
**Target:** Horizontal scaling

**Planned Features:**
- Multi-GPU worker deployment (same host)
- Multi-host deployment (LAN)
- Load balancer configuration
- High availability setup
- Capacity planning guide

---

## Version History Summary

| Version | Milestone | Date       | Key Features |
|---------|-----------|------------|--------------|
| 0.0.1   | M0        | 2025-09-XX | Repository scaffold, tooling, documentation |
| 0.1.0   | M1        | 2025-09-XX | gRPC ABI, transport abstraction, config framework |
| 0.2.0   | M2        | 2025-10-05 | WebSocket transport, CLI client, mock worker, docs |
| 0.3.0   | M3        | TBD        | LiveKit transport, browser client |
| 0.4.0   | M4        | TBD        | Model Manager, dynamic loading |
| 0.5.0   | M5        | TBD        | Piper adapter (CPU) |
| 0.6.0   | M6        | TBD        | CosyVoice 2 adapter (GPU) |
| 0.7.0   | M7        | TBD        | XTTS-v2 adapter (cloning) |
| 0.8.0   | M8        | TBD        | Sesame/Unsloth adapter (LoRA) |
| 0.9.0   | M9        | TBD        | Multi-worker routing |
| 1.0.0   | M10       | TBD        | ASR integration, full speech↔speech |
| 1.1.0   | M11       | TBD        | Observability, profiling |
| 1.2.0   | M12       | TBD        | Production deployment |
| 2.0.0   | M13       | TBD        | Multi-GPU/host scaling |

---

## Contributing

When adding entries to this changelog:

1. **Use semantic versioning:** MAJOR.MINOR.PATCH
2. **Group changes:** Added, Changed, Deprecated, Removed, Fixed, Security
3. **Be specific:** Include file paths, feature names, and context
4. **Link issues:** Reference GitHub issues when applicable
5. **Date format:** YYYY-MM-DD
6. **Keep unreleased:** Maintain [Unreleased] section at top

**Example Entry:**
```markdown
### Added
- New feature X in `src/module/file.py` (#123)
- Configuration option Y for Z behavior

### Fixed
- Bug causing issue A under condition B (#456)
```

---

## References

- [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
- [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
- [Project README](README.md)
- [Implementation Plan](project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md)
