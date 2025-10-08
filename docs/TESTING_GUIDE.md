# Testing Guide for Contributors

## Overview

This guide helps contributors run tests, write new tests, and understand the testing strategy for the Realtime Duplex Voice Demo system.

**Test Framework:** pytest with asyncio support

**Test Types:**
- **Unit Tests:** Fast, isolated, no external dependencies
- **Integration Tests:** Require Docker (Redis), test component interactions
- **End-to-End Tests:** Full system tests

---

## Table of Contents

- [Running Tests](#running-tests)
- [Test Markers](#test-markers)
- [Writing Tests](#writing-tests)
- [Test Fixtures](#test-fixtures)
- [Mocking Strategies](#mocking-strategies)
- [Coverage Expectations](#coverage-expectations)
- [Testing Best Practices](#testing-best-practices)
- [Continuous Integration](#continuous-integration)

---

## Running Tests

### Quick Start

**Run all tests:**
```bash
just test
```

**Run with verbose output:**
```bash
pytest -vv
```

**Run specific test categories:**
```bash
# Unit tests only (fast, no Docker)
pytest tests/unit/

# Integration tests (requires Docker)
pytest tests/integration/

# Specific test file
pytest tests/unit/test_vad.py

# Specific test function
pytest tests/unit/test_vad.py::test_speech_detection
```

---

### Common Test Commands

```bash
# Run all tests with coverage
pytest --cov=src tests/

# Coverage report in terminal
pytest --cov=src --cov-report=term-missing tests/

# Coverage HTML report
pytest --cov=src --cov-report=html tests/
# Open: htmlcov/index.html

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run with live logging
pytest -o log_cli=true -o log_cli_level=DEBUG

# Fail fast (stop on first failure)
pytest -x

# Run last failed tests only
pytest --lf

# Run tests matching pattern
pytest -k "test_speech"
```

---

## Performance Tests (Manual Only)

Performance tests are isolated in `tests/performance/` and are **NOT run automatically** in CI or with `just test`.

**Why separate?**
- Performance tests can take 10-15 minutes to complete
- Some tests may segfault due to Python 3.13.6 issues with concurrent websockets
- Tests require significant system resources (4+ GB RAM)
- Results are more meaningful on isolated systems without background load

**To run performance tests:**
```bash
just test-performance
```

Or with pytest directly:
```bash
uv run pytest tests/performance/ -v -m performance
```

**Run specific performance test:**
```bash
uv run pytest tests/performance/test_performance.py::test_fal_single_session -v
```

**Important notes:**
- Performance tests require Docker containers (Redis, TTS worker)
- Some concurrent tests may be unstable (known Python 3.13.6 websockets issue)
- Run on a system with minimal background processes for reliable benchmarks

See [`tests/performance/README.md`](../tests/performance/README.md) for detailed information on:
- Performance targets (FAL, frame jitter, throughput)
- Individual test descriptions
- Troubleshooting guidance
- Result interpretation

---

## Test Markers

Tests are marked with pytest markers to categorize and selectively run them.

### Available Markers

#### @pytest.mark.unit

**Purpose:** Fast tests with no external dependencies.

**Characteristics:**
- No Docker required
- No Redis required
- No network calls
- Mock all external services
- Run in < 1 second each

**Example:**
```python
import pytest

@pytest.mark.unit
def test_vad_speech_detection():
    """Test VAD detects speech correctly."""
    vad = VADDetector(aggressiveness=2)
    speech_frame = generate_speech_audio()
    assert vad.is_speech(speech_frame) is True
```

**Run unit tests:**
```bash
pytest -m unit
```

---

#### @pytest.mark.integration

**Purpose:** Tests requiring external services (Redis, Docker).

**Characteristics:**
- Requires Docker running
- May require Redis container
- Tests component interactions
- Run in 1-10 seconds each

**Example:**
```python
import pytest

@pytest.mark.integration
async def test_worker_registration(redis_container):
    """Test worker registers with Redis."""
    registry = WorkerRegistry(redis_url="redis://localhost:6379")
    worker_info = {"name": "test-worker", "addr": "grpc://localhost:7001"}

    await registry.register_worker(worker_info)

    workers = await registry.get_workers()
    assert len(workers) == 1
    assert workers[0]["name"] == "test-worker"
```

**Run integration tests:**
```bash
pytest -m integration
```

---

#### @pytest.mark.docker

**Purpose:** Tests specifically requiring Docker.

**Example:**
```python
@pytest.mark.docker
@pytest.mark.integration
async def test_redis_container_healthy(redis_container):
    """Test Redis container is accessible."""
    # redis_container fixture ensures container running
    assert await ping_redis() is True
```

---

#### @pytest.mark.redis

**Purpose:** Tests requiring Redis specifically.

**Example:**
```python
@pytest.mark.redis
@pytest.mark.integration
async def test_worker_heartbeat(redis_container):
    """Test worker heartbeat updates TTL."""
    # Test logic
```

---

#### @pytest.mark.slow

**Purpose:** Tests taking > 5 seconds.

**Example:**
```python
@pytest.mark.slow
@pytest.mark.integration
async def test_model_loading_performance():
    """Test model loads within time budget."""
    start = time.time()
    await model_manager.load("cosyvoice2-en-base")
    duration = time.time() - start
    assert duration < 10.0  # 10 second budget
```

**Skip slow tests:**
```bash
pytest -m "not slow"
```

---

### Using Multiple Markers

```python
@pytest.mark.integration
@pytest.mark.redis
@pytest.mark.slow
async def test_worker_registration_stress():
    """Stress test worker registration."""
    # Test many concurrent registrations
```

**Run specific combination:**
```bash
pytest -m "integration and not slow"
pytest -m "unit or integration"
```

---

## Writing Tests

### Test Structure: Arrange-Act-Assert (AAA)

```python
import pytest

@pytest.mark.unit
def test_vad_silence_detection():
    # Arrange: Set up test conditions
    vad = VADDetector(aggressiveness=2, sample_rate=16000)
    silence_frame = generate_silence(duration_ms=20)

    # Act: Execute the behavior under test
    result = vad.is_speech(silence_frame)

    # Assert: Verify the outcome
    assert result is False
```

---

### Unit Test Patterns

#### Testing Sync Functions

```python
@pytest.mark.unit
def test_frame_repacketization():
    """Test audio repacketization to 20ms frames."""
    # Arrange
    audio = np.random.randn(48000)  # 1 second @ 48kHz
    frame_size = 960  # 20ms @ 48kHz

    # Act
    frames = list(frame_audio(audio, frame_size))

    # Assert
    assert len(frames) == 50  # 1 second / 20ms
    assert all(len(f) == frame_size for f in frames)
```

---

#### Testing Async Functions

```python
import pytest

@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_synthesis():
    """Test async synthesis generates frames."""
    # Arrange
    adapter = MockAdapter()
    text = "Hello world"

    # Act
    frames = []
    async for frame in adapter.synthesize_streaming(text, "session-1", {}):
        frames.append(frame)

    # Assert
    assert len(frames) > 0
    assert all(len(f) == 1920 for f in frames)  # 20ms @ 48kHz
```

**Note:** `@pytest.mark.asyncio` required for async tests.

---

#### Testing Exceptions

```python
@pytest.mark.unit
def test_invalid_config_raises():
    """Test invalid config raises ValidationError."""
    invalid_config = {"port": 99999}  # Invalid port

    with pytest.raises(ValueError, match="port must be"):
        Config(**invalid_config)
```

---

#### Parametrized Tests

```python
@pytest.mark.unit
@pytest.mark.parametrize("aggressiveness,expected", [
    (0, True),   # Least aggressive, detects soft speech
    (1, True),
    (2, True),
    (3, False),  # Most aggressive, may miss soft speech
])
def test_vad_aggressiveness_levels(aggressiveness, expected):
    """Test VAD with different aggressiveness levels."""
    vad = VADDetector(aggressiveness=aggressiveness)
    soft_speech = generate_soft_speech()

    result = vad.is_speech(soft_speech)

    # Result may vary, this is illustrative
    # In practice, use deterministic test data
```

---

### Integration Test Patterns

#### Using Docker Fixtures

```python
@pytest.mark.integration
@pytest.mark.redis
async def test_worker_registry_integration(redis_container):
    """Test worker registry against real Redis."""
    # redis_container fixture ensures Redis running

    # Arrange
    registry = WorkerRegistry(redis_url="redis://localhost:6379")
    worker = {
        "name": "test-worker-0",
        "addr": "grpc://localhost:7001",
        "capabilities": {"streaming": True}
    }

    # Act
    await registry.register_worker(worker)
    workers = await registry.get_workers()

    # Assert
    assert len(workers) == 1
    assert workers[0]["name"] == "test-worker-0"
```

---

#### Testing gRPC Communication

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_grpc_synthesis(registered_mock_worker):
    """Test gRPC synthesis against mock worker."""
    # registered_mock_worker fixture starts worker and registers

    # Arrange
    channel = grpc.aio.insecure_channel("localhost:7001")
    stub = tts_pb2_grpc.TTSServiceStub(channel)

    # Act
    request = tts_pb2.StartSessionRequest(
        session_id="test-session",
        model_id="mock"
    )
    response = await stub.StartSession(request)

    # Assert
    assert response.success is True
```

---

#### Testing WebSocket

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_websocket_connection(running_orchestrator):
    """Test WebSocket connection to orchestrator."""
    # running_orchestrator fixture starts orchestrator

    # Arrange
    import websockets

    # Act
    async with websockets.connect("ws://localhost:8080") as ws:
        # Receive session start
        msg = await ws.recv()
        data = json.loads(msg)

        # Assert
        assert data["type"] == "session_start"
        assert "session_id" in data
```

---

## Test Fixtures

### Built-in Fixtures

#### redis_container

**Purpose:** Start Redis container for tests.

**Usage:**
```python
@pytest.mark.integration
@pytest.mark.redis
async def test_with_redis(redis_container):
    """Test requiring Redis."""
    # Redis available at localhost:6379
    # Automatically cleaned up after test
```

**Defined in:** `tests/conftest.py`

**Implementation:**
```python
@pytest.fixture(scope="session")
async def redis_container():
    """Start Redis container for integration tests."""
    container = docker_client.containers.run(
        "redis:7-alpine",
        ports={"6379/tcp": 6379},
        detach=True,
        remove=True
    )
    yield container
    container.stop()
```

---

#### registered_mock_worker

**Purpose:** Start mock TTS worker and register with Redis.

**Usage:**
```python
@pytest.mark.integration
async def test_with_worker(registered_mock_worker):
    """Test requiring TTS worker."""
    # Worker running at localhost:7001
    # Registered in Redis
```

---

### Custom Fixtures

#### Creating Fixtures

**Define in `conftest.py`:**
```python
import pytest

@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    sample_rate = 48000
    duration_ms = 20
    samples = int(sample_rate * duration_ms / 1000)
    audio = np.random.randn(samples).astype(np.float32)
    return audio

@pytest.fixture
async def test_session():
    """Create test session with cleanup."""
    session = Session(session_id="test-session-id")
    yield session
    # Cleanup
    await session.cleanup()
```

**Use in tests:**
```python
@pytest.mark.unit
def test_audio_processing(sample_audio):
    """Test audio processing with sample audio."""
    processed = process_audio(sample_audio)
    assert len(processed) == len(sample_audio)
```

---

### Fixture Scopes

```python
@pytest.fixture(scope="function")  # Default, new instance per test
def per_test_fixture():
    pass

@pytest.fixture(scope="class")  # Shared across test class
def per_class_fixture():
    pass

@pytest.fixture(scope="module")  # Shared across test module
def per_module_fixture():
    pass

@pytest.fixture(scope="session")  # Shared across entire test session
def per_session_fixture():
    pass
```

---

## Mocking Strategies

### Mocking External Services

#### Mock Redis

**Using unittest.mock:**
```python
from unittest.mock import AsyncMock, MagicMock
import pytest

@pytest.mark.unit
async def test_registry_without_redis():
    """Test registry with mocked Redis."""
    # Arrange
    mock_redis = AsyncMock()
    mock_redis.get.return_value = json.dumps({
        "name": "worker-0",
        "addr": "grpc://localhost:7001"
    })

    registry = WorkerRegistry(redis_client=mock_redis)

    # Act
    worker = await registry.get_worker("worker-0")

    # Assert
    assert worker["name"] == "worker-0"
    mock_redis.get.assert_called_once_with("worker:worker-0")
```

---

#### Mock gRPC Stubs

```python
@pytest.mark.unit
async def test_orchestrator_grpc_call():
    """Test orchestrator with mocked gRPC stub."""
    # Arrange
    mock_stub = AsyncMock()
    mock_stub.StartSession.return_value = tts_pb2.StartSessionResponse(
        success=True,
        message=""
    )

    orchestrator = Orchestrator(grpc_stub=mock_stub)

    # Act
    result = await orchestrator.start_session("session-1", "model-1")

    # Assert
    assert result is True
    mock_stub.StartSession.assert_called_once()
```

---

### Mocking File I/O

```python
from unittest.mock import patch, mock_open

@pytest.mark.unit
def test_config_loading():
    """Test config loading with mocked file."""
    mock_yaml = """
    port: 8080
    host: "0.0.0.0"
    """

    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        config = load_config("config.yaml")

    assert config["port"] == 8080
```

---

### Mocking Time

```python
from unittest.mock import patch
import time

@pytest.mark.unit
def test_ttl_expiration():
    """Test TTL expiration with mocked time."""
    with patch("time.time", side_effect=[0, 100, 700]):
        # Initial time: 0
        manager = ModelManager(ttl_ms=600)
        manager.load("model-1")  # time=100

        # Check expiration at time=700 (600ms later)
        manager.check_evictions()  # time=700

        # Model should be evicted
        assert not manager.is_loaded("model-1")
```

---

## Coverage Expectations

### Target Coverage

**Overall:** 80%+ code coverage

**Critical Paths:** 95%+ coverage
- VAD detection logic
- Worker routing
- Transport message handling
- Session state machine

**Generated Code:** Excluded
- `src/rpc/generated/` - Auto-generated gRPC stubs

---

### Measuring Coverage

**Run tests with coverage:**
```bash
pytest --cov=src --cov-report=term-missing tests/
```

**Output:**
```
---------- coverage: platform linux, python 3.13.0 -----------
Name                                Stmts   Miss  Cover   Missing
-----------------------------------------------------------------
src/orchestrator/server.py            120      5    96%   45-48, 102
src/orchestrator/vad.py                50      2    96%   67, 89
src/orchestrator/routing.py           80      8    90%   23-25, 67-70
src/tts/worker.py                     150     12    92%   89-95, 145-150
-----------------------------------------------------------------
TOTAL                                 400     27    93%
```

---

### Coverage Reports

**HTML Report:**
```bash
pytest --cov=src --cov-report=html tests/
open htmlcov/index.html
```

**XML Report (for CI):**
```bash
pytest --cov=src --cov-report=xml tests/
# Generates coverage.xml
```

---

### Coverage Configuration

**In `pyproject.toml`:**
```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "src/rpc/generated/*",
    "*/tests/*",
    "*/__pycache__/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

---

## Testing Best Practices

### 1. Test Naming

**Convention:** `test_<feature>_<condition>_<expected_result>`

**Examples:**
```python
def test_vad_speech_detection_returns_true()
def test_vad_silence_detection_returns_false()
def test_worker_registration_with_invalid_data_raises_error()
```

---

### 2. One Assert Per Test (Guideline)

**Prefer:**
```python
@pytest.mark.unit
def test_vad_detects_speech():
    vad = VADDetector()
    assert vad.is_speech(speech_frame) is True

@pytest.mark.unit
def test_vad_detects_silence():
    vad = VADDetector()
    assert vad.is_speech(silence_frame) is False
```

**Over:**
```python
@pytest.mark.unit
def test_vad():
    vad = VADDetector()
    assert vad.is_speech(speech_frame) is True  # Which assertion failed?
    assert vad.is_speech(silence_frame) is False
```

---

### 3. Deterministic Tests

**Avoid randomness:**
```python
# Bad: Non-deterministic
def test_audio_generation():
    audio = generate_random_audio()  # Different each run
    assert len(audio) > 0

# Good: Deterministic
def test_audio_generation():
    seed = 42
    audio = generate_audio(seed=seed)
    assert len(audio) == 960  # Exact expectation
```

---

### 4. Fast Unit Tests

**Keep unit tests fast (<1s each):**
- Mock expensive operations
- Use small data samples
- Avoid I/O

```python
# Bad: Slow unit test
@pytest.mark.unit
def test_model_loading():
    model = load_large_model()  # 10 seconds!
    assert model is not None

# Good: Fast unit test with mock
@pytest.mark.unit
def test_model_loading():
    with patch("load_large_model", return_value=MockModel()):
        model = load_large_model()
        assert model is not None
```

---

### 5. Isolated Tests

**Tests should not depend on each other:**
```python
# Bad: Tests depend on order
def test_step_1():
    global data
    data = process_data()

def test_step_2():
    # Assumes test_step_1 ran first
    assert data is not None

# Good: Tests are independent
@pytest.fixture
def processed_data():
    return process_data()

def test_step_1(processed_data):
    assert processed_data is not None

def test_step_2(processed_data):
    assert len(processed_data) > 0
```

---

## Audio Test Helpers

### Generate Test Audio

**In `tests/conftest.py`:**
```python
import numpy as np

def generate_speech_audio(duration_ms: int = 20, sample_rate: int = 16000) -> bytes:
    """Generate synthetic speech-like audio for testing."""
    samples = int(sample_rate * duration_ms / 1000)
    # Generate audio with speech characteristics (frequency modulation)
    t = np.linspace(0, duration_ms / 1000, samples)
    freq = 200 + 50 * np.sin(2 * np.pi * 5 * t)  # Voice-like frequency
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return (audio * 32767).astype(np.int16).tobytes()

def generate_silence(duration_ms: int = 20, sample_rate: int = 16000) -> bytes:
    """Generate silence for testing."""
    samples = int(sample_rate * duration_ms / 1000)
    audio = np.zeros(samples, dtype=np.int16)
    return audio.tobytes()
```

**Usage:**
```python
@pytest.mark.unit
def test_vad_with_speech():
    vad = VADDetector()
    speech = generate_speech_audio()
    assert vad.is_speech(speech) is True
```

---

## Continuous Integration

### GitHub Actions Workflow

**File:** `.github/workflows/ci.yml`

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.13'

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run linting
        run: uv run ruff check src tests

      - name: Run type checking
        run: uv run mypy src

      - name: Run tests
        run: uv run pytest --cov=src --cov-report=xml tests/

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

---

### Local CI Check

**Run all CI checks locally:**
```bash
just ci
```

**Equivalent to:**
```bash
just lint       # Ruff linting
just typecheck  # Mypy type checking
just test       # Pytest tests
```

---

## Troubleshooting Tests

### Redis Container Issues

**Symptom:** Integration tests fail with "Connection refused"

**Solution:**
```bash
# Check Redis running
docker ps | grep redis

# Start Redis manually
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Test connection
docker exec redis redis-cli ping
# Expected: PONG
```

---

### Async Test Issues

**Symptom:** `RuntimeError: Event loop is closed`

**Solution:** Add `@pytest.mark.asyncio` to async tests:
```python
import pytest

@pytest.mark.asyncio  # Required!
async def test_async_function():
    result = await async_function()
    assert result is not None
```

---

### Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Ensure running tests from project root
cd /path/to/full-duplex-voice-chat

# Run tests
pytest tests/
```

---

## Further Reading

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Plugin](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov Plugin](https://pytest-cov.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Quick Start Guide](QUICKSTART.md) - Running the system for testing
- [Architecture](architecture/ARCHITECTURE.md) - Understanding components
