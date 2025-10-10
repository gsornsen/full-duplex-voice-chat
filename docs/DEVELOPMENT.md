# Development Guide

**Last Updated**: 2025-10-09

This guide covers local development workflows, debugging techniques, testing strategies, and troubleshooting for the full-duplex voice chat system.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Daily Development Workflow](#daily-development-workflow)
3. [Testing Strategy](#testing-strategy)
4. [Debugging Techniques](#debugging-techniques)
5. [Common Development Tasks](#common-development-tasks)
6. [Troubleshooting](#troubleshooting)
7. [Code Quality Standards](#code-quality-standards)
8. [Contributing Guidelines](#contributing-guidelines)

---

## Development Setup

### Prerequisites

**Required:**
- Python 3.13.x
- [uv](https://github.com/astral-sh/uv) package manager (0.1.0+)
- Git
- Docker Engine 28.x with NVIDIA container runtime
- Redis (via Docker or local install)

**For GPU Development:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.8+ (or 11.8+)
- cuDNN libraries

**Platform Notes:**
- **Linux**: Native support, best performance
- **WSL2**: Full support with GPU passthrough (see [WSL2 GPU Setup](#wsl2-gpu-setup))
- **macOS**: CPU-only mode supported

### Initial Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd full-duplex-voice-chat
```

2. **Install dependencies**:
```bash
# Install all dependencies including dev tools
uv sync --all-extras

# Or minimal install
uv sync
```

3. **Install pre-commit hooks** (optional but recommended):
```bash
uv run pre-commit install
```

4. **Generate gRPC stubs**:
```bash
just gen-proto
```

5. **Verify setup**:
```bash
# Run linting and type checking
just lint
just typecheck

# Run unit tests
just test
```

### WSL2 GPU Setup

If developing in WSL2 with GPU support:

1. **Install NVIDIA driver on Windows host** (525.60+)
2. **Verify GPU access in WSL2**:
```bash
nvidia-smi
```

3. **No CUDA installation needed in WSL2** (uses Windows driver)
4. **Install PyTorch with CUDA support**:
```bash
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Known Limitation**: WSL2 GPU performance is ~10-20% lower than native Linux.

---

## Daily Development Workflow

### Quick Start (Development Mode)

Run the full stack locally for development:

```bash
# Terminal 1: Start Redis
just redis

# Terminal 2: Start TTS worker (mock adapter for M0-M2)
just run-tts-mock

# Terminal 3: Start orchestrator
just run-orch

# Terminal 4: Run CLI client
just cli HOST="ws://localhost:8080"
```

**Alternative - Docker Compose**:
```bash
# Start everything with one command
docker compose up --build

# Run in background
docker compose up -d

# View logs
docker compose logs -f orchestrator

# Stop everything
docker compose down
```

### Code-Edit-Test Cycle

1. **Make code changes** in your editor
2. **Run relevant tests**:
```bash
# Quick: run tests for modified component
uv run pytest tests/unit/test_your_module.py -v

# Comprehensive: run all unit tests
just test

# With coverage
uv run pytest tests/unit/ --cov=src --cov-report=html
```

3. **Check code quality**:
```bash
# Auto-fix linting issues
just fix

# Run full CI checks
just ci
```

4. **Commit changes**:
```bash
git add .
git commit -m "feat(component): description of changes"
```

### Hot Reloading

Most components support auto-reload during development:

- **Orchestrator**: Restart required for code changes
- **TTS Workers**: Restart required for adapter changes
- **Configuration**: Reload on file change (config.yaml)

**Tip**: Use tmux or screen to manage multiple terminals.

---

## Testing Strategy

### Test Organization

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_audio/
│   ├── test_orchestrator/
│   ├── test_tts/
│   └── test_utils/
└── integration/             # End-to-end tests
    ├── test_m1_worker_integration.py
    ├── test_full_pipeline.py
    └── test_livekit_e2e.py
```

### Running Tests

**Unit Tests** (fast, always run these):
```bash
# All unit tests
just test

# Specific module
uv run pytest tests/unit/test_orchestrator/ -v

# Single test
uv run pytest tests/unit/test_orchestrator/test_session.py::test_state_transitions -v

# With coverage
uv run pytest tests/unit/ --cov=src --cov-report=term-missing
```

**Integration Tests** (slower, require services):
```bash
# All integration tests (with process isolation)
just test-integration

# Specific integration test
uv run pytest tests/integration/test_m1_worker_integration.py --forked -v

# Fast mode (no process isolation, may segfault on WSL2)
just test-integration-fast
```

**Full CI Suite**:
```bash
# Run everything: lint + typecheck + test
just ci
```

### Test Markers

Tests are marked by type:

```python
@pytest.mark.unit
def test_something(): ...

@pytest.mark.integration
def test_full_pipeline(): ...

@pytest.mark.slow
def test_load_test(): ...
```

Run specific markers:
```bash
# Only unit tests
uv run pytest -m unit

# Only integration tests
uv run pytest -m integration

# Skip slow tests
uv run pytest -m "not slow"
```

### WSL2 Testing Considerations

**gRPC Segfault Issue**:
Integration tests using gRPC require process isolation in WSL2:

```bash
# ALWAYS use --forked for integration tests in WSL2
just test-integration

# Or manually
uv run pytest tests/integration/ --forked -v
```

See [known-issues/grpc-segfault.md](known-issues/grpc-segfault.md) for details.

**Environment Detection**:
Tests automatically detect WSL2 and apply workarounds.

### Writing Tests

**Unit Test Example**:
```python
import pytest
from src.orchestrator.session import SessionManager, SessionState

@pytest.mark.unit
def test_session_state_transitions():
    """Test valid state transitions."""
    session = SessionManager(session_id="test-123", transport=mock_transport)

    # Initial state
    assert session.state == SessionState.IDLE

    # Valid transition
    session.transition_state(SessionState.LISTENING)
    assert session.state == SessionState.LISTENING

    # Invalid transition should raise
    with pytest.raises(ValueError):
        session.transition_state(SessionState.BARGED_IN)
```

**Integration Test Example**:
```python
import pytest
import asyncio

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_synthesis_pipeline(orchestrator_server, tts_worker):
    """Test complete text-to-speech pipeline."""
    async with websockets.connect("ws://localhost:8080") as ws:
        # Send text
        await ws.send(json.dumps({"text": "Hello world"}))

        # Receive audio frames
        frames = []
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(msg)
            if data["type"] == "audio":
                frames.append(data["data"])
            if data.get("is_final"):
                break

        # Verify audio received
        assert len(frames) > 0
```

---

## Debugging Techniques

### Logging

**Enable debug logging**:
```bash
# Environment variable
export LOG_LEVEL=DEBUG

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Structured logging** (JSON format):
```python
logger.info(
    "Session started",
    extra={
        "session_id": session.session_id,
        "transport": "websocket",
        "client_addr": client_addr,
    }
)
```

**Log locations**:
- Console output (stdout/stderr)
- Docker logs: `docker compose logs -f orchestrator`
- Test logs: pytest captures logs, use `-s` to see them

### Interactive Debugging

**Python Debugger (pdb)**:
```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or Python 3.7+
breakpoint()
```

**Run tests with debugging**:
```bash
# pytest drops into pdb on failure
uv run pytest tests/unit/test_session.py --pdb

# Or on first failure
uv run pytest --pdb -x
```

**VS Code Debugging**:
Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        },
        {
            "name": "Pytest: Current File",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}", "-v"],
            "console": "integratedTerminal"
        }
    ]
}
```

### CPU Profiling

**py-spy** (recommended for production profiling):

```bash
# Top-like interface
just spy-top <PID>

# Or manually
uv run py-spy top --pid <PID>

# Record flame graph
just spy-record <PID> OUT="profile.svg"

# Or manually
uv run py-spy record -o profile.svg --pid <PID> --duration 30
```

**cProfile** (standard library):
```bash
# Profile script
python -m cProfile -o profile.stats src/orchestrator/server.py

# Analyze results
python -m pstats profile.stats
```

### GPU Profiling

**NVIDIA Nsight Systems** (timeline trace):
```bash
# Profile TTS worker
just nsys-tts

# Or manually
nsys profile --trace=cuda,nvtx --output=tts-profile.qdrep \
    uv run python src/tts/worker.py

# View in nsys-ui (GUI)
nsys-ui tts-profile.qdrep
```

**NVIDIA Nsight Compute** (kernel analysis):
```bash
# Profile specific kernels
just ncu-tts

# Or manually
ncu --set full --export=kernel-profile \
    uv run python src/tts/worker.py
```

**PyTorch Profiler**:
```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    # Your code here
    model.generate()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for Chrome trace viewer
prof.export_chrome_trace("trace.json")
```

### Network Debugging

**gRPC debugging**:
```bash
# Enable gRPC debug logs
export GRPC_VERBOSITY=DEBUG
export GRPC_TRACE=all

# Run worker/orchestrator
just run-tts-mock
```

**WebSocket debugging**:
```bash
# Use wscat for manual testing
npm install -g wscat
wscat -c ws://localhost:8080

# Send JSON message
> {"text": "Hello world"}
```

**Traffic inspection**:
```bash
# tcpdump (root required)
sudo tcpdump -i lo -A 'port 8080'

# Wireshark (GUI)
wireshark -i lo -f "port 8080"
```

### Redis Debugging

**Inspect worker registry**:
```bash
# Connect to Redis
redis-cli

# List all keys
KEYS *

# Get worker info
GET worker:tts-mock-0

# Monitor live commands
MONITOR
```

**Redis GUI tools**:
- Redis Insight (official)
- RedisDesktopManager
- redis-cli (built-in)

---

## Common Development Tasks

### Adding a New TTS Adapter

1. **Create adapter file**:
```bash
touch src/tts/adapters/adapter_mymodel.py
```

2. **Implement base interface**:
```python
from src.tts.tts_base import TTSAdapter

class MyModelAdapter(TTSAdapter):
    def __init__(self, config: dict):
        super().__init__(config)
        # Initialize model

    async def synthesize_stream(
        self, text_stream: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """Yield 20ms PCM frames at 48kHz."""
        # Implement synthesis
        async for text_chunk in text_stream:
            # Generate audio
            audio_data = self.model.synthesize(text_chunk)
            # Repacketize to 20ms frames
            async for frame in self.repacketize(audio_data):
                yield frame
```

3. **Add tests**:
```bash
# Unit tests
tests/unit/test_tts/test_adapters/test_mymodel.py

# Integration tests
tests/integration/test_mymodel_integration.py
```

4. **Register in worker**:
```python
# src/tts/worker.py
ADAPTERS = {
    "mock": MockAdapter,
    "mymodel": MyModelAdapter,
}
```

5. **Update configuration**:
```yaml
# configs/worker.yaml
adapter:
  type: "mymodel"
  config:
    model_path: "/path/to/model"
```

### Modifying the Protocol

1. **Edit proto file**:
```bash
vim src/rpc/tts.proto
```

2. **Regenerate stubs**:
```bash
just gen-proto
```

3. **Update implementations**:
- Worker: `src/tts/worker.py`
- Orchestrator: `src/orchestrator/server.py`
- Client: `src/client/cli_client.py`

4. **Add tests** for new protocol features

5. **Update documentation**:
- Protocol changes in `src/rpc/tts.proto` comments
- Architecture updates in `CLAUDE.md`
- Migration guide if breaking changes

### Adding Configuration Options

1. **Update config schema**:
```python
# src/orchestrator/config.py
class OrchestratorConfig(BaseModel):
    new_option: str = Field(
        default="default_value",
        description="Description of new option"
    )
```

2. **Add to YAML template**:
```yaml
# configs/orchestrator.yaml
new_option: "custom_value"
```

3. **Support environment variable**:
```python
# src/orchestrator/config.py
@classmethod
def from_env(cls):
    return cls(
        new_option=os.getenv("NEW_OPTION", "default_value")
    )
```

4. **Document in**:
- `CLAUDE.md` (configuration section)
- `configs/orchestrator.yaml` (inline comments)

### Performance Optimization

**Measure first**:
```bash
# Profile before optimizing
just spy-record <PID> OUT="before.svg"

# Make changes

# Profile again
just spy-record <PID> OUT="after.svg"

# Compare
diff before.svg after.svg
```

**Common optimizations**:
- Use async/await properly (no blocking calls)
- Batch operations where possible
- Cache expensive computations
- Use connection pooling (gRPC, Redis)
- Profile GPU kernels with Nsight
- Optimize memory allocations (reuse buffers)

**Performance targets**:
- First Audio Latency (FAL): p95 < 300ms (GPU), < 500ms (CPU)
- Barge-in latency: p95 < 50ms
- Frame jitter: p95 < 10ms
- CPU usage: < 50% per session
- Memory: < 2GB per worker (loaded model)

---

## Troubleshooting

### Common Issues

See [known-issues/README.md](known-issues/README.md) for comprehensive troubleshooting guide.

**Quick diagnostics**:

1. **Check service health**:
```bash
# All services
docker compose ps

# Health check
curl http://localhost:8080/health
```

2. **Check ports**:
```bash
sudo lsof -i :8080  # Orchestrator WS
sudo lsof -i :7001  # TTS worker
sudo lsof -i :6379  # Redis
```

3. **Check logs**:
```bash
# Docker logs
docker compose logs orchestrator
docker compose logs tts-worker

# Local logs
tail -f /var/log/orchestrator.log
```

4. **Verify CUDA**:
```bash
nvidia-smi                                           # GPU visible?
python -c "import torch; print(torch.cuda.is_available())"  # PyTorch sees GPU?
```

### Known Issues Quick Reference

| Issue | Solution | Documentation |
|-------|----------|---------------|
| gRPC segfault in tests | Use `--forked` flag | [grpc-segfault.md](known-issues/grpc-segfault.md) |
| Port already in use | `sudo lsof -i :<port>` then kill | [known-issues](known-issues/README.md#port-conflicts) |
| CUDA not available | Reinstall PyTorch with CUDA | [known-issues](known-issues/README.md#cuda-version-mismatches) |
| Redis connection refused | Start Redis: `just redis` | [known-issues](known-issues/README.md#redis-connection-failures) |
| Tests timing out | Increase timeout or check logs | [known-issues](known-issues/README.md#websocket-test-timeouts) |

### Getting Help

1. **Check documentation**:
   - This guide (DEVELOPMENT.md)
   - [Known Issues](known-issues/README.md)
   - [Testing Guide](TESTING_GUIDE.md)
   - [CLAUDE.md](../CLAUDE.md)

2. **Search existing issues** in the repository

3. **Ask for help**:
   - Include error messages and stack traces
   - Provide steps to reproduce
   - Share relevant configuration
   - Mention OS, Python version, CUDA version

---

## Code Quality Standards

### Linting (Ruff)

**Run linting**:
```bash
# Check only
just lint

# Auto-fix
just fix

# Manual
uv run ruff check src/ tests/
uv run ruff check --fix src/ tests/
```

**Configuration**: `pyproject.toml`
```toml
[tool.ruff]
line-length = 100
target-version = "py313"
```

**Common violations**:
- Line too long (> 100 chars)
- Unused imports
- Missing docstrings
- Incorrect import order

### Type Checking (mypy)

**Run type checking**:
```bash
just typecheck

# Or manually
uv run mypy src/ tests/
```

**Configuration**: `pyproject.toml`
```toml
[tool.mypy]
strict = true
warn_unused_configs = true
```

**Type hints required**:
```python
# Good
def process_audio(data: bytes, sample_rate: int) -> list[bytes]:
    """Process audio data."""
    ...

# Bad
def process_audio(data, sample_rate):
    """Process audio data."""
    ...
```

### Code Style

**Formatting**:
- Use Black-compatible style (Ruff auto-formats)
- 100 character line limit
- 4 spaces for indentation

**Naming conventions**:
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- `_private` for internal methods

**Docstrings**:
```python
def synthesize_stream(text: str, config: dict) -> AsyncIterator[bytes]:
    """Synthesize speech from text with streaming output.

    Args:
        text: Input text to synthesize
        config: Adapter-specific configuration

    Yields:
        Audio frames as bytes (20ms, 48kHz, mono PCM)

    Raises:
        ValueError: If text is empty
        RuntimeError: If synthesis fails
    """
    ...
```

### Testing Standards

**Coverage targets**:
- Overall: > 80%
- Critical paths: 100%
- New code: > 90%

**Test naming**:
```python
def test_<component>_<scenario>_<expected_result>():
    """Test that <component> <does what> when <scenario>."""
    ...

# Examples
def test_session_transitions_to_speaking_on_text():
    """Test that session transitions to SPEAKING when text is received."""
    ...

def test_worker_validates_empty_frames():
    """Test that worker rejects invalid empty frames."""
    ...
```

**Test organization**:
- One test class per component
- Setup in `setUp()` or fixtures
- Teardown in `tearDown()` or fixture cleanup
- Use parametrize for multiple cases

---

## Contributing Guidelines

### Git Workflow

1. **Create feature branch**:
```bash
git checkout -b feat/my-feature
# or
git checkout -b fix/bug-description
```

2. **Make changes with clear commits**:
```bash
git add src/tts/adapters/adapter_new.py
git commit -m "feat(tts): add new TTS adapter"

git add tests/unit/test_new_adapter.py
git commit -m "test(tts): add tests for new adapter"
```

3. **Keep branch updated**:
```bash
git fetch origin
git rebase origin/main
```

4. **Push and create PR**:
```bash
git push origin feat/my-feature
# Create PR on GitHub/GitLab
```

### Commit Message Format

Follow conventional commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code change that neither fixes bug nor adds feature
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Updating build tasks, package manager configs, etc.

**Scopes**:
- `tts`: TTS workers and adapters
- `orchestrator`: Orchestrator server
- `transport`: WebSocket/LiveKit transport
- `protocol`: gRPC protocol changes
- `tests`: Test infrastructure
- `docs`: Documentation
- `config`: Configuration changes

**Examples**:
```bash
git commit -m "feat(tts): add CosyVoice2 adapter"
git commit -m "fix(orchestrator): handle empty audio frames correctly"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(integration): add barge-in latency tests"
```

### Pull Request Guidelines

**Before submitting PR**:
1. Run full CI: `just ci`
2. Update documentation if needed
3. Add tests for new features
4. Update CHANGELOG.md
5. Rebase on latest main

**PR description should include**:
- What changed and why
- Related issue numbers
- Testing performed
- Screenshots/videos for UI changes
- Breaking changes (if any)
- Migration guide (if needed)

### Code Review

**As author**:
- Respond to all comments
- Make requested changes or explain reasoning
- Keep PR focused and small
- Update PR description if scope changes

**As reviewer**:
- Be constructive and specific
- Focus on logic, not style (Ruff handles style)
- Test the changes locally
- Approve when ready to merge

---

## Additional Resources

- **Architecture**: [../CLAUDE.md](../CLAUDE.md)
- **Current Status**: [CURRENT_STATUS.md](CURRENT_STATUS.md)
- **Testing Guide**: [TESTING_GUIDE.md](TESTING_GUIDE.md)
- **Known Issues**: [known-issues/README.md](known-issues/README.md)
- **Implementation Plan**: [../project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md](../project_documentation/INCREMENTAL_IMPLEMENTATION_PLAN.md)

---

**Happy coding!** If you have questions or suggestions for improving this guide, please open an issue or PR.
