# Development Environment

**Last Updated**: 2025-10-27

This document provides detailed guidance on setting up and working with the development environment.

> 📖 **Working with git worktrees?** See [git-worktrees.md](git-worktrees.md) for critical setup steps

## Python & Tooling

**Python Version:**
- Python 3.13.x managed with **uv** (fast Python package installer and resolver)
- No Python version pinned in `pyproject.toml` - let uv resolve automatically
- Project dependencies in `pyproject.toml`, locked in `uv.lock`

**Code Quality Tools:**
- **ruff**: Linting (fast Python linter)
- **mypy**: Type checking in strict mode
- **pytest**: Testing framework with async support
- **justfile**: Task automation (like Makefile but better)

**Installation:**
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (respects uv.lock)
uv sync

# Generate gRPC stubs
just gen-proto
```

## Platform Requirements

**CUDA & PyTorch:**
- **CUDA Toolkit**: 13.0.1 available on system
- **PyTorch**: 2.7.0 with **CUDA 12.8** prebuilt wheels (for stability)
- Reason: CUDA 12.8 wheels are more stable than 13.x for PyTorch 2.7.0

**Docker:**
- **Docker Engine**: 28.x with NVIDIA container runtime
- **Base Container**: `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`
- Required for GPU workers in containerized deployments

**WSL2 Considerations:**
- **gRPC Tests**: Require `--forked` flag due to segfault issues in pytest teardown
- **Solution**: Use `just test-integration` which automatically adds `--forked`
- **Details**: See [GRPC_SEGFAULT_WORKAROUND.md](../../GRPC_SEGFAULT_WORKAROUND.md)
- **Status**: 100% mitigated with pytest-forked, tests reliable

**Redis:**
- Required for worker service discovery and registry
- Run with `just redis` or `docker compose up redis`

## Essential Commands

### Quality & CI

```bash
# Linting
just lint              # Run ruff linting
just fix               # Auto-fix linting issues (safe)

# Type Checking
just typecheck         # Run mypy type checking (strict mode)

# Testing
just test              # Run all tests (unit + integration)
just test-unit         # Run unit tests only
just test-integration  # Run integration tests (with --forked for gRPC)

# Combined CI
just ci                # Run all checks: lint + typecheck + test
```

**CI Requirements:**
- All checks must pass before PR approval
- See [testing.md](testing.md#ci-cd-pipeline) for CI/CD details

### Code Generation

```bash
# Generate gRPC stubs from src/rpc/tts.proto
just gen-proto

# Output: src/rpc/generated/tts_pb2.py, tts_pb2_grpc.py
```

**When to regenerate:**
- After modifying `src/rpc/tts.proto`
- After cloning the repo
- If mypy complains about missing protobuf stubs

### Infrastructure

```bash
# Start Redis (required for worker discovery)
just redis

# Stop Redis
docker stop redis && docker rm redis
```

### Runtime Commands

**TTS Workers:**
```bash
# Run mock TTS worker (M1/M2/M3 - sine wave generator)
just run-tts-mock

# Run Piper TTS worker (M5 - CPU baseline, ONNX)
just run-tts-piper DEFAULT="piper-en-us-lessac-medium" PRELOAD=""

# Override default model
just run-tts-piper DEFAULT="piper-en-us-amy-low"

# Preload multiple models
just run-tts-piper PRELOAD="piper-en-us-lessac-medium,piper-en-us-amy-low"

# Future: GPU TTS workers (M6-M8, not yet implemented)
# just run-tts-cosy DEFAULT="cosyvoice2-en-base"
# just run-tts-xtts DEFAULT="xtts-v2-en-demo"
# just run-tts-sesame DEFAULT="sesame-en-base"
```

**Orchestrator:**
```bash
# Run orchestrator (LiveKit WebRTC + WebSocket fallback)
just run-orch

# Features: VAD, ASR, session management
# Listens on: ws://localhost:8080 (WebSocket), livekit://localhost:7880 (WebRTC)
```

**CLI Client:**
```bash
# Connect to orchestrator (WebSocket)
just cli HOST="ws://localhost:8080"

# Connect to remote orchestrator
just cli HOST="ws://10.0.0.5:8080"
```

### Docker Commands

```bash
# Start full stack (redis + livekit + caddy + orchestrator + tts workers)
docker compose up --build

# Start in background
docker compose up -d --build

# Stop all services
docker compose down

# View logs
docker compose logs -f orchestrator
docker compose logs -f tts-piper

# Rebuild specific service
docker compose up --build orchestrator
```

**Docker Environment Variables:**
```bash
# Set TTS adapter type
export TTS_ADAPTER=piper  # or mock, cosy, xtts, sesame
export TTS_MODEL_ID=piper-en-us-lessac-medium

# Rebuild and start
docker compose up --build
```

### Profiling Commands

**CPU Profiling (py-spy):**
```bash
# Real-time top-like view
just spy-top PID

# Record flame graph
just spy-record PID OUT="profile.svg"

# Example workflow
just run-tts-piper &  # Start worker in background
PID=$!                # Capture PID
just spy-record $PID OUT="/tmp/tts_profile.svg"
# ... run workload ...
kill $PID
open /tmp/tts_profile.svg  # View flame graph
```

**GPU Profiling (NVIDIA):**
```bash
# Nsight Systems (timeline trace)
just nsys-tts

# Nsight Compute (kernel analysis)
just ncu-tts

# PyTorch Profiler (with NVTX ranges)
# Use torch.profiler.profile() in code with record_function() for phases
```

**ONNX Runtime Profiling (Piper):**
```bash
# Enable ORT profiling in worker config (configs/worker_piper.yaml)
adapter:
  config:
    ort_profiling: true
    ort_profile_file: "/tmp/piper_ort_profile.json"

# Run workload, then analyze /tmp/piper_ort_profile.json
```

**Whisper ASR Profiling:**
```bash
# CPU profiling
just spy-record $(pgrep -f orchestrator) OUT="/tmp/asr_cpu_profile.svg"

# GPU profiling (if using GPU inference)
just nsys-tts  # Captures both TTS and ASR if in same process
```

## Development Workflow

### 1. Initial Setup

```bash
# Clone repo
git clone <repo-url>
cd full-duplex-voice-chat

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Generate protobuf stubs
just gen-proto

# Start Redis
just redis

# Run CI to verify setup
just ci
```

### 2. Feature Development

**Typical workflow:**
```bash
# 1. Create feature branch
git checkout -b feat/my-new-feature

# 2. Make changes
vim src/orchestrator/my_feature.py

# 3. Run relevant tests frequently
just test-unit  # Fast feedback

# 4. Fix linting issues
just fix

# 5. Run full CI before commit
just ci

# 6. Commit changes
git add .
git commit -m "feat: add my new feature"

# 7. Push and create PR
git push origin feat/my-new-feature
```

**When working on specific components:**
```bash
# Orchestrator changes
just test tests/unit/orchestrator/  # Fast unit tests
just test tests/integration/test_orchestrator_integration.py

# TTS adapter changes
just test tests/unit/tts/adapters/
just test tests/integration/test_m5_piper_integration.py

# ASR changes
just test tests/unit/asr/
just test tests/integration/test_m10_whisper_integration.py
```

### 3. Testing Cycle

**Red-Green-Refactor:**
```bash
# 1. Write failing test
vim tests/unit/orchestrator/test_my_feature.py
just test tests/unit/orchestrator/test_my_feature.py  # Should fail

# 2. Implement feature
vim src/orchestrator/my_feature.py
just test tests/unit/orchestrator/test_my_feature.py  # Should pass

# 3. Refactor (keep tests passing)
vim src/orchestrator/my_feature.py
just test tests/unit/orchestrator/test_my_feature.py  # Still pass

# 4. Run full suite
just test
```

**Test-Driven Development for Adapters:**
```bash
# 1. Define adapter interface (already exists in tts_base.py)
# 2. Write adapter tests (see tests/unit/tts/adapters/test_piper.py for template)
# 3. Implement adapter (see src/tts/adapters/adapter_piper.py for example)
# 4. Run integration tests to verify end-to-end flow
```

### 4. Pre-Commit Checklist

Before committing, ensure:
- ✅ `just ci` passes (lint + typecheck + test)
- ✅ No `type: ignore` comments without justification
- ✅ Docstrings for public APIs
- ✅ Configuration files have inline comments
- ✅ Tests added for new functionality (aim for >80% coverage)

### 5. Multi-Agent Coordination

**When to invoke specialized agents:**
- **@documentation-engineer**: Documentation audit before milestone completion
- **@devops-engineer**: CI/CD issues, deployment problems, infrastructure changes
- **@python-pro**: Code quality reviews, type checking issues, test strategy

**See [.claude/agents/README.md](../agents/README.md) for agent coordination patterns.**

## IDE & Editor Setup

### VS Code

**Recommended extensions:**
```json
{
  "recommendations": [
    "ms-python.python",
    "charliermarsh.ruff",
    "ms-python.mypy-type-checker",
    "ms-python.vscode-pylance",
    "tamasfe.even-better-toml",
    "njpwerner.autodocstring"
  ]
}
```

**Settings (`.vscode/settings.json`):**
```json
{
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "python.linting.mypyEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false
}
```

### PyCharm

**Configuration:**
1. **Project Interpreter**: Select uv-managed Python 3.13.x
2. **Type Checker**: Enable mypy in Preferences → Tools → Python Integrated Tools
3. **External Tools**: Add justfile commands as external tools
4. **Test Runner**: Configure pytest as default test runner

**Run Configurations:**
```
Name: Run TTS Worker (Piper)
Script: uv run python -m src.tts.worker
Working Directory: /path/to/repo
Environment: WORKER_CONFIG=configs/worker_piper.yaml
```

## Environment Variables

**Common variables:**
```bash
# Worker configuration
export WORKER_CONFIG=configs/worker_piper.yaml
export CUDA_VISIBLE_DEVICES=0  # Pin to specific GPU

# Orchestrator configuration
export ORCH_CONFIG=configs/orchestrator.yaml

# TTS adapter selection (for Docker)
export TTS_ADAPTER=piper
export TTS_MODEL_ID=piper-en-us-lessac-medium

# Redis connection
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Logging
export LOG_LEVEL=DEBUG  # or INFO, WARNING, ERROR
export LOG_FORMAT=json   # or text

# Testing
export PYTEST_FORKED=1  # Force --forked mode for all tests
```

**Per-session overrides:**
```bash
# Run worker with custom config
WORKER_CONFIG=configs/my_custom_config.yaml just run-tts-piper

# Run tests with verbose logging
LOG_LEVEL=DEBUG just test
```

## Dependency Management

**Adding dependencies:**
```bash
# Add runtime dependency
uv add <package-name>

# Add development dependency
uv add --dev <package-name>

# Lock dependencies
uv lock

# Commit changes
git add pyproject.toml uv.lock
git commit -m "chore(deps): add <package-name>"
```

**Upgrading dependencies:**
```bash
# Upgrade all dependencies
uv lock --upgrade

# Upgrade specific package
uv lock --upgrade-package <package-name>

# Sync environment
uv sync
```

**Resolving conflicts:**
```bash
# If uv.lock conflicts in merge
git checkout --ours uv.lock   # Or --theirs
uv sync                        # Verify it works
uv lock                        # Regenerate if needed
```

## Troubleshooting

### Common Issues

**gRPC tests failing in WSL2:**
```bash
# Solution: Use --forked flag
just test-integration  # Automatically uses --forked

# Manual
uv run pytest tests/integration/ --forked
```

**Protobuf import errors:**
```bash
# Regenerate stubs
just gen-proto

# Verify stubs exist
ls -la src/rpc/generated/

# Expected: tts_pb2.py, tts_pb2_grpc.py, tts_pb2.pyi, tts_pb2_grpc.pyi
```

**CUDA out of memory:**
```bash
# Reduce batch size in worker config
# Or use smaller model
# Or kill other GPU processes

# Check GPU memory usage
nvidia-smi

# Find GPU processes
fuser -v /dev/nvidia*
```

**Redis connection refused:**
```bash
# Start Redis
just redis

# Verify Redis is running
docker ps | grep redis

# Test connection
redis-cli ping  # Should return "PONG"
```

**uv sync fails:**
```bash
# Clear cache and retry
uv cache clean
uv sync

# If still fails, check uv.lock consistency
uv lock --check
```

**Mypy errors after dependency update:**
```bash
# Regenerate type stubs
just gen-proto

# Clear mypy cache
rm -rf .mypy_cache/

# Re-run
just typecheck
```

## Performance Optimization

**Development mode (fast iteration):**
```bash
# Skip integration tests
just test-unit

# Skip slow tests
uv run pytest tests/ -m "not slow"

# Run specific test file
just test tests/unit/orchestrator/test_vad.py
```

**Profiling during development:**
```bash
# Enable profiling in code
import cProfile
cProfile.run('my_function()', 'profile.stats')

# Analyze with snakeviz
uv run snakeviz profile.stats
```

**Memory profiling:**
```bash
# Use memory_profiler
uv add --dev memory-profiler

# Decorate function with @profile
from memory_profiler import profile

@profile
def my_function():
    pass

# Run with memory profiling
uv run python -m memory_profiler my_script.py
```

## Git Worktrees for Parallel Development

When working on multiple features simultaneously, git worktrees allow independent working directories from a single repository.

### Quick Setup

```bash
# Create worktree for new feature
git worktree add ../project-feature -b feature/feature-name

# Navigate and setup environment
cd ../project-feature

# ⚠️ CRITICAL: Install dev dependencies
uv sync --all-extras

# Generate proto files (if needed)
uv run just gen-proto

# Verify pre-push hook works
.git/hooks/pre-push origin refs/heads/feature/feature-name
```

### Why `uv sync --all-extras` is Required

Git worktrees share `.git` but have **independent Python environments**. Without running `uv sync --all-extras`:
- Pre-push hooks fail (mypy, ruff not found)
- CI catches errors that local checks miss
- Type stubs missing (types-redis, types-pyyaml)

### Common Issues

**"mypy: command not found" in worktree**
```bash
cd /path/to/worktree
uv sync --all-extras
```

**Tests pass locally, fail in CI**
- Forgot `uv sync --all-extras`
- Used `--no-verify` to bypass checks
- See [git-worktrees.md](git-worktrees.md) for full troubleshooting

### Never Use `--no-verify`

The pre-push hook now strongly discourages `--no-verify`:
- Fix issues locally instead
- Run `just ci` to catch problems early
- CI failures waste team time

📖 **Full Guide**: See [git-worktrees.md](git-worktrees.md) for complete worktree best practices

## References

- **Git Worktrees**: [git-worktrees.md](git-worktrees.md) ⭐ **New**
- **Testing Guide**: [testing.md](testing.md)
- **CI/CD Pipeline**: [testing.md#ci-cd-pipeline](testing.md#ci-cd-pipeline)
- **Architecture Details**: [architecture.md](architecture.md)
- **gRPC Workaround**: [../../GRPC_SEGFAULT_WORKAROUND.md](../../GRPC_SEGFAULT_WORKAROUND.md)
