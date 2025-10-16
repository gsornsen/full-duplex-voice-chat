# Development Guide

**Last Updated**: 2025-10-13

This guide covers local development workflows, debugging techniques, testing strategies, and troubleshooting for the full-duplex voice chat system.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Daily Development Workflow](#daily-development-workflow)
3. [Log Management](#log-management)
4. [Testing Strategy](#testing-strategy)
5. [Debugging Techniques](#debugging-techniques)
6. [Common Development Tasks](#common-development-tasks)
7. [Troubleshooting](#troubleshooting)
8. [Code Quality Standards](#code-quality-standards)
9. [Contributing Guidelines](#contributing-guidelines)

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
# Install all dependencies including dev tools (includes honcho process manager)
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

### Quick Start - Parallel Service Startup (Recommended)

**Single command starts all services in parallel** - no 5-minute Docker build wait!

```bash
# Start all services with mock TTS adapter (fast startup)
just dev
```

**What it does**:
- Starts Redis (Docker container on port 6379)
- Starts LiveKit (Docker container on ports 7880-7882)
- Starts TTS Worker with mock adapter (Python process on port 7001)
- Starts Orchestrator with VAD and ASR (Python process on port 8082)
- **Automatically saves logs** to `logs/dev-sessions/dev-YYYYMMDD-HHMMSS.log`

**Features**:
- **Parallel startup**: ~10 seconds total (vs 5+ minutes for Docker build)
- **Color-coded logs**: Each service has its own color in terminal output
- **Graceful shutdown**: Single Ctrl+C stops all processes cleanly
- **Hot-reload friendly**: Restart `just dev` quickly after code changes
- **Automatic logging**: Every session creates a timestamped log file

**Variants**:

```bash
# Start with Piper TTS adapter (CPU-based, real TTS)
just dev-piper

# Start with web client included (adds Next.js dev server on port 3000)
just dev-web
```

**Access the demo**:
- WebSocket endpoint: `ws://localhost:8082`
- LiveKit WebRTC: `ws://localhost:7880`
- Web client (if dev-web): `http://localhost:3000`
- CLI client: `just cli`

**How it works**:
- Uses **Procfile** format (industry standard, Heroku-compatible)
- Powered by **Honcho** process manager (Python-based, cross-platform)
- Services defined in `Procfile.dev` at repository root
- Single Ctrl+C sends SIGINT to all processes
- Logs captured via `tee` (console output + file simultaneously)

**Customizing services**:

Edit `Procfile.dev` to customize service startup:
```bash
# Procfile.dev format:
<service-name>: <shell-command>

# Example: Change TTS adapter
tts: uv run python -m src.tts.worker --adapter piper --default-model piper-en-us-lessac-medium
```

**Troubleshooting parallel startup**:

1. **Port already in use**:
```bash
# Find process using port
lsof -i :6379  # Redis
lsof -i :7001  # TTS Worker
lsof -i :7880  # LiveKit
lsof -i :8082  # Orchestrator

# Kill process
kill -9 <PID>
```

2. **Docker containers conflict**:
```bash
# Remove existing containers
docker stop redis-tts-dev livekit-dev
docker rm redis-tts-dev livekit-dev
```

3. **Honcho not found**:
```bash
# Install honcho
uv sync --all-extras
# Or manually: uv pip install honcho
```

### Alternative - Individual Services (for debugging)

Run services in separate terminals for fine-grained debugging:

```bash
# Terminal 1: Start Redis
just redis

# Terminal 2: Start TTS worker (mock adapter)
just run-tts-mock

# Terminal 3: Start orchestrator
just run-orch

# Terminal 4: Run CLI client
just cli HOST="ws://localhost:8082"
```

**Stopping services**:
```bash
# Stop individual service: Ctrl+C in terminal
# Stop Redis container: just redis-stop
```

**When to use this approach**:
- Debugging specific service in isolation
- Running profiling tools on single service (py-spy, nsys, etc.)
- Need to restart single service frequently
- Want separate log files per service

### Alternative - Docker Compose (production-like testing)

Use Docker Compose for full-stack testing with all production features:

```bash
# Start full stack (includes Caddy reverse proxy, TLS, etc.)
docker compose up --build

# Run in background
docker compose up -d

# View logs
docker compose logs -f orchestrator

# Stop everything
docker compose down
```

**Note**: Docker build takes ~5 minutes. Use `just dev` for faster iteration during development.

**When to use Docker Compose**:
- Testing production deployment configuration
- Verifying TLS/HTTPS setup
- Testing with production-like resource constraints
- Multi-service integration testing
- CI/CD pipeline validation

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

- **Orchestrator**: Restart required for code changes (use `Ctrl+C` then `just dev`)
- **TTS Workers**: Restart required for adapter changes
- **Configuration**: Reload on file change (config.yaml)
- **Web Client**: Hot-reload enabled (Next.js dev mode)

**Tip**: Use tmux or screen to manage multiple terminals when not using `just dev`.

---

## Log Management

All `just dev`, `just dev-piper`, and `just dev-web` commands automatically save rotating logs to timestamped files. This allows you to review past sessions, debug issues, and track system behavior over time.

### Log Storage

**Location**: `logs/dev-sessions/`

**Filename format**:
- `dev-YYYYMMDD-HHMMSS.log` (from `just dev`)
- `dev-piper-YYYYMMDD-HHMMSS.log` (from `just dev-piper`)
- `dev-web-YYYYMMDD-HHMMSS.log` (from `just dev-web`)

**Example**:
```
logs/dev-sessions/
├── dev-20251013-093045.log
├── dev-20251013-101522.log
├── dev-piper-20251013-143018.log
└── dev-web-20251013-151203.log
```

### Log Features

**Console + File Output**: Logs are written to both:
- **Terminal** (live, color-coded output for real-time monitoring)
- **Log file** (persistent, includes all ANSI color codes for full fidelity)

**What's logged**:
- Service startup messages (Redis, LiveKit, TTS Worker, Orchestrator)
- Application logs (DEBUG, INFO, WARNING, ERROR levels)
- Error messages and stack traces
- Performance metrics and timing information
- Network activity (WebSocket connections, gRPC calls)
- VAD events (speech start/end detection)
- ASR transcriptions

**Color preservation**: ANSI color codes are preserved in log files. Use `less -R` or `cat` to view with colors.

### Log Management Commands

#### List Recent Logs

View the 10 most recent log files with timestamps:

```bash
just logs-list
```

**Example output**:
```
Recent development session logs:
--------------------------------
logs/dev-sessions/dev-20251013-151203.log (Oct 13 15:12)
logs/dev-sessions/dev-piper-20251013-143018.log (Oct 13 14:30)
logs/dev-sessions/dev-20251013-101522.log (Oct 13 10:15)
...
```

#### Tail Most Recent Log

Follow the latest log file in real-time (useful for debugging after a session ends):

```bash
just logs-tail
```

This command:
- Finds the most recent log file
- Runs `tail -f` to follow it in real-time
- Press `Ctrl+C` to stop

#### View Specific Log

Open a specific log file with pager (supports search, scrolling):

```bash
just logs-view <filename>
```

**Example**:
```bash
# View specific log
just logs-view dev-20251013-093045.log

# Inside less:
# - Press '/' to search
# - Press 'n' for next match
# - Press 'q' to quit
# - Arrow keys to scroll
```

If the file doesn't exist, the command shows available logs automatically.

#### Clean Old Logs

Remove old log files based on retention policy:

```bash
just logs-clean
```

**Retention policy**:
- **Keep last 20 sessions** (most recent files)
- **Delete files older than 7 days**
- Whichever keeps MORE files

**Example output**:
```
Cleaning old development session logs...
Total log files: 35
Keeping last 20 files, removing 15 old files...
Deleted 3 log files older than 7 days.
Remaining log files: 20
```

**When to clean**:
- Weekly maintenance
- Before long development sessions (to avoid clutter)
- When disk space is limited

### Log Analysis Tips

**Search across all logs**:
```bash
# Find all ERROR messages
grep -r "ERROR" logs/dev-sessions/

# Find specific session ID
grep -r "session_id=abc123" logs/dev-sessions/

# Count occurrences
grep -rc "VAD detected speech" logs/dev-sessions/ | sort -t: -k2 -nr
```

**Extract timing information**:
```bash
# Find slow requests
grep "duration_ms" logs/dev-sessions/dev-20251013-*.log | awk '{if ($NF > 1000) print}'

# Analyze FAL (First Audio Latency)
grep "first_audio_latency" logs/dev-sessions/*.log | awk '{print $NF}'
```

**Debug specific service**:
```bash
# Filter by service prefix (honcho adds service name)
grep "orchestrator" logs/dev-sessions/dev-latest.log
grep "tts" logs/dev-sessions/dev-latest.log
```

**View logs with color**:
```bash
# Using less with color support
less -R logs/dev-sessions/dev-20251013-093045.log

# Or cat (no paging)
cat logs/dev-sessions/dev-20251013-093045.log
```

### Log Retention Best Practices

**For active development**:
- Keep logs for at least 7 days
- Run `just logs-clean` weekly
- Review recent logs before filing bug reports

**For long-term debugging**:
- Archive important logs outside `logs/dev-sessions/`
- Compress old logs: `gzip logs/dev-sessions/*.log`
- Keep logs related to production incidents indefinitely

**Disk space management**:
- Average log size: 5-50 MB per session (depends on duration and verbosity)
- 20 sessions ≈ 100-1000 MB
- Monitor with: `du -sh logs/dev-sessions/`

### Troubleshooting Logs

**No logs directory**:
```bash
# Created automatically on first 'just dev' run
# Or create manually:
mkdir -p logs/dev-sessions
```

**Logs not being created**:
- Check if `tee` command is available: `which tee`
- Verify write permissions: `ls -ld logs/dev-sessions`
- Check disk space: `df -h`

**Log file too large**:
- Reduce log verbosity: `export LOG_LEVEL=INFO` (instead of DEBUG)
- Clean old logs: `just logs-clean`
- Compress old logs: `gzip logs/dev-sessions/*.log`

**Can't view logs with color**:
- Use `less -R` instead of plain `less`
- Or use `cat` for direct output
- Terminal must support ANSI colors

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
- Development session logs: `logs/dev-sessions/dev-YYYYMMDD-HHMMSS.log`
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
# Development session logs
just logs-tail

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

2. **Review recent logs**:
```bash
# List recent sessions
just logs-list

# View specific log
just logs-view <filename>

# Search for errors
grep -r "ERROR" logs/dev-sessions/
```

3. **Search existing issues** in the repository

4. **Ask for help**:
   - Include error messages and stack traces
   - Provide steps to reproduce
   - Share relevant configuration
   - Mention OS, Python version, CUDA version
   - Attach relevant log files from `logs/dev-sessions/`

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

## CI/CD Workflows

The project uses a modern three-tier CI/CD strategy for fast feedback and comprehensive quality assurance.

### CI Architecture

**Three Workflow Types:**

1. **Feature Branch CI** - Fast feedback during development (3-5 min)
2. **Pull Request CI** - Full validation before merge (10-15 min, REQUIRED)
3. **Main Branch** - No CI (quality guaranteed by PR gates)

### Running CI Locally

**Before pushing code:**

```bash
# Run all quality checks locally (recommended)
just ci  # Runs: lint + typecheck + test

# Or run individually
just lint       # Ruff linting
just typecheck  # mypy type checking
just test       # pytest unit tests
```

**Full PR CI simulation (with coverage):**

```bash
# Run full test suite with coverage
uv run pytest tests/ \
  -v \
  --cov=src \
  --cov-report=xml \
  --cov-report=term \
  --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Feature Branch CI

**Workflow**: `.github/workflows/feature-ci.yml`

**Triggers**: Push to feature/feat/fix/* branches

**Smart Test Selection:**

The Feature CI automatically detects which files changed and runs only relevant tests:

| Changed Files | Tests Run |
|--------------|-----------|
| `src/orchestrator/**` | Orchestrator unit + integration |
| `src/tts/**` | TTS unit + integration |
| `src/asr/**` | ASR unit + integration |
| `src/rpc/**` | All integration tests |
| `pyproject.toml`, `uv.lock` | Full test suite |
| `*.md`, `docs/**` | Skip all (docs-only) |

**Performance:**
- Dependency install: ~30 sec (with cache)
- Test execution: 1-3 min (selective)
- Total: 3-5 minutes (vs 10-15 min full suite)

**Status**: Informational (failures don't block pushes)

### Pull Request CI

**Workflow**: `.github/workflows/pr-ci.yml`

**Triggers**: PR creation/updates to main

**Quality Gates (ALL must pass):**

1. ✅ **Lint** (ruff) - Code style and conventions
2. ✅ **Type Check** (mypy) - Strict type checking
3. ✅ **Full Test Suite** (pytest) - All 649 tests
4. ✅ **Code Coverage** (codecov) - ≥80% overall, ≥60% patch
5. ✅ **Security Scan** (bandit) - Security vulnerabilities
6. ✅ **Dependency Check** (pip-audit) - Known CVEs
7. ✅ **Build Check** - Verify uv.lock integrity

**Status**: REQUIRED (failures block merge)

**Coverage Requirements:**
- Overall project: ≥80% coverage
- New code (patch): ≥60% coverage
- Threshold: 2% drop allowed from base branch

### Codecov Integration

The PR CI automatically uploads coverage to [Codecov](https://codecov.io), which:
- Comments on PRs with coverage diff
- Shows exactly which lines need tests
- Tracks coverage trends over time
- Provides coverage badges for README

**Setup required:**
1. Sign up at codecov.io with your GitHub account
2. Enable Codecov for your repository
3. Add `CODECOV_TOKEN` secret to GitHub Actions

**Viewing coverage:**
- PR comments show coverage diff automatically
- Visit codecov.io for detailed reports
- Check `htmlcov/` folder after local test runs

### CI Caching

**Dependency Caching:**

Both workflows use aggressive caching for fast runs:

```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v3
  with:
    enable-cache: true
    cache-dependency-glob: "uv.lock"
```

**Cache performance:**
- Cache hit (same uv.lock): 30 sec install (90% faster)
- Partial hit (some changes): 2-3 min (50% faster)
- Cache miss (all new): 5-6 min (baseline)

**Protobuf Caching:**

Generated protobuf stubs are cached based on `.proto` file hash:

```yaml
- name: Cache protobuf stubs
  uses: actions/cache@v4
  with:
    path: src/rpc/generated
    key: protobuf-${{ hashFiles('src/rpc/tts.proto') }}
```

Saves 10-15 seconds per job when `.proto` files haven't changed.

### Troubleshooting CI Failures

**Lint failures:**

```bash
# Auto-fix most issues
just fix

# Check what remains
just lint
```

**Type check failures:**

```bash
# Regenerate protobuf stubs
just gen-proto

# Run type check
just typecheck

# Common issues:
# - Missing protobuf stubs (run just gen-proto)
# - Missing type annotations (add hints)
# - Import errors (check module paths)
```

**Test failures:**

```bash
# Run specific failing test
uv run pytest tests/unit/orchestrator/test_session.py -v

# Run with full traceback
uv run pytest tests/ -vv --tb=long

# Debug with pdb
uv run pytest tests/ --pdb
```

**Coverage failures:**

```bash
# Generate coverage report
uv run pytest tests/ --cov=src --cov-report=term-missing

# View HTML report
uv run pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Tips:
# - Add tests for untested code
# - Remove dead/unreachable code
# - Use # pragma: no cover for test-only code
```

**Cache issues:**

If CI is unexpectedly slow:

```bash
# Check cache status in CI logs:
# "Cache restored from key: Linux-uv-abc123..."

# Force cache refresh (if needed):
uv lock --upgrade  # Updates all dependencies

# Clear GitHub Actions cache:
# Go to Settings → Actions → Caches → Delete caches
```

### CI Performance Metrics

**Expected performance:**
- Feature CI: 3-5 minutes (60-70% faster than old CI)
- PR CI: 10-15 minutes (same duration, added coverage)
- Cache hit rate: >90% for feature branches

**Monitoring:**
- View run history: Actions tab → Workflow → All runs
- Check duration trends over time
- Investigate jobs that take >15 minutes
- Report flaky tests immediately

### CI Best Practices

**For developers:**

1. **Run `just ci` before pushing** - Catch issues locally
2. **Keep feature branches small** - Faster CI, easier reviews
3. **Fix failures immediately** - Don't let them accumulate
4. **Monitor coverage** - Aim for >80% on new code
5. **Test dependency updates** - Run `uv lock` locally first

**For code reviewers:**

1. **Check CI status** - All checks must pass before approval
2. **Review coverage report** - Ensure new code is tested
3. **Check security scan** - Review bandit/pip-audit results
4. **Verify test quality** - Not just quantity
5. **Request changes** - If coverage drops or tests are missing

### Branch Protection

**Required for main branch:**

The repository should have branch protection enabled with these settings:

1. ✅ Require pull request reviews (1 reviewer)
2. ✅ Require status checks to pass:
   - `PR CI / lint`
   - `PR CI / typecheck`
   - `PR CI / test`
   - `PR CI / build`
3. ✅ Require branches to be up to date
4. ✅ Do not allow force pushes
5. ✅ Do not allow deletions

**Setup instructions:**

See `/tmp/branch-protection-setup.md` for detailed step-by-step setup guide.

### CI Cost and Efficiency

**GitHub Actions usage:**

- Free tier: 2,000 minutes/month
- Estimated usage with new CI: ~3,000 min/month
- Cost: ~$8/month overage (vs $27/month with old CI)
- **Savings: 70% cost reduction**

**Efficiency gains:**

- 44% fewer CI minutes (no runs on main)
- 60-70% faster feature branch feedback
- 80-90% faster dependency installation (caching)

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
