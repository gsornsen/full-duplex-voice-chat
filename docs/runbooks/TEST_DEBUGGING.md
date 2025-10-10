# Runbook: Test Debugging Guide

**Time to Resolution:** 5-20 minutes
**Severity:** Medium (development productivity)
**Related:** [Environment Setup](ENVIRONMENT.md), [Log Debugging](LOG_DEBUGGING.md)

---

## Overview

This runbook covers common test failures, debugging techniques, and resolution strategies for the M2 TTS system test suite.

**Test Categories:**
- Unit tests (`tests/unit/`)
- Integration tests (`tests/integration/`)
- End-to-end tests (`tests/e2e/`)

---

## Quick Reference

```bash
# Run all tests
just test

# Run with verbose output
just test -v

# Run specific test file
uv run pytest tests/unit/test_config.py -v

# Run specific test function
uv run pytest tests/unit/test_config.py::test_orchestrator_config_validation -v

# Run with debugging
uv run pytest tests/unit/test_config.py -vv --pdb

# Show print statements
uv run pytest tests/unit/test_config.py -v -s

# Stop on first failure
uv run pytest tests/unit/test_config.py -x

# Run only failed tests from last run
uv run pytest --lf

# Run tests matching pattern
uv run pytest -k "test_config" -v
```

---

## Common Test Failures

### 1. Import Errors

**Symptom:**
```
E   ModuleNotFoundError: No module named 'rpc.generated'
```

**Cause:** gRPC stubs not generated

**Resolution:**
```bash
# Generate proto stubs
just gen-proto

# Verify
ls -la src/rpc/generated/
# Should contain: tts_pb2.py, tts_pb2_grpc.py

# Re-run tests
just test
```

---

### 2. Fixture Errors

**Symptom:**
```
E   fixture 'redis_client' not found
```

**Cause:** Missing or misconfigured pytest fixture

**Check fixtures:**
```bash
# List available fixtures
uv run pytest --fixtures tests/integration/

# Show fixture definition
grep -r "def redis_client" tests/
```

**Common fixture issues:**

**a) Redis not running:**
```bash
# Check Redis
docker ps | grep redis

# Start if needed
just redis

# Verify connection
redis-cli -u redis://localhost:6379 ping
```

**b) Fixture scope issue:**
```python
# conftest.py
@pytest.fixture(scope="session")  # Wrong scope
async def redis_client():
    ...

# Fix: Use "function" scope for isolation
@pytest.fixture(scope="function")
async def redis_client():
    ...
```

---

### 3. Async Test Failures

**Symptom:**
```
E   RuntimeError: Event loop is closed
```

**Cause:** Incorrect async test setup

**Solution:**

**Install pytest-asyncio:**
```bash
# Should already be in dependencies
uv sync

# Verify
uv run pytest --version
# Should show: pytest-asyncio plugin
```

**Mark async tests correctly:**
```python
import pytest

# Correct
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_call()
    assert result is not None

# Wrong: Missing decorator
async def test_async_function():  # Will fail
    ...
```

**Configure pytest-asyncio in `pyproject.toml`:**
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # Auto-detect async tests
```

---

### 4. Docker Container Tests

**Symptom:**
```
E   docker.errors.APIError: 500 Server Error: ... container already exists
```

**Cause:** Leftover containers from previous test run

**Resolution:**
```bash
# Clean up test containers
docker ps -a | grep test_ | awk '{print $1}' | xargs docker rm -f

# Or clean all stopped containers
docker container prune -f

# Re-run tests
just test
```

**Prevent:**
```python
# tests/integration/conftest.py
import pytest
import docker

@pytest.fixture
async def test_container():
    client = docker.from_env()
    container = client.containers.run(
        "redis:7",
        name="test-redis-" + str(uuid.uuid4()),  # Unique name
        detach=True,
        remove=True  # Auto-remove on stop
    )
    yield container
    container.stop()  # Cleanup
```

---

### 5. Port Conflicts in Tests

**Symptom:**
```
E   OSError: [Errno 98] Address already in use
```

**Cause:** Test trying to bind to already-used port

**Find conflicting process:**
```bash
lsof -i :8080
# Or
netstat -tuln | grep 8080
```

**Fix in tests:**
```python
# Bad: Hard-coded port
async def test_server():
    app.listen(8080)  # May conflict

# Good: Dynamic port
import socket

def get_free_port():
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]

async def test_server():
    port = get_free_port()
    app.listen(port)
```

**Or use pytest-asyncio ports:**
```python
@pytest.fixture
def unused_tcp_port():
    import socket
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]

async def test_server(unused_tcp_port):
    app.listen(unused_tcp_port)
```

---

### 6. Timeout Errors

**Symptom:**
```
E   asyncio.exceptions.TimeoutError
```

**Cause:** Test operation taking longer than expected

**Increase timeout:**
```python
# Bad: No timeout
result = await slow_operation()

# Good: Explicit timeout
import asyncio

try:
    result = await asyncio.wait_for(slow_operation(), timeout=10.0)
except asyncio.TimeoutError:
    pytest.fail("Operation timed out after 10s")
```

**Or adjust pytest timeout:**
```bash
# Install pytest-timeout
uv add --dev pytest-timeout

# Run with timeout
uv run pytest --timeout=30 tests/
```

**In pyproject.toml:**
```toml
[tool.pytest.ini_options]
timeout = 30  # 30 seconds per test
```

---

### 7. Mock/Patch Issues

**Symptom:**
```
E   AttributeError: <Mock> object has no attribute 'xyz'
```

**Cause:** Incorrect mock setup

**Common patterns:**

**Mock async functions:**
```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_with_async_mock():
    mock_grpc = AsyncMock()
    mock_grpc.Synthesize.return_value = AsyncMock()

    with patch('orchestrator.grpc_client.TTSClient', return_value=mock_grpc):
        result = await synthesize_text("Hello")
        assert result is not None
```

**Mock context managers:**
```python
from unittest.mock import MagicMock, patch

async def test_redis_mock():
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True

    with patch('redis.asyncio.from_url', return_value=mock_redis):
        client = await get_redis_client()
        assert await client.ping() is True
```

---

### 8. Assertion Failures

**Symptom:**
```
E   AssertionError: assert 0.35 > 0.5
E    +  where 0.35 = result.rtf
```

**Debug:**

**Use pytest's detailed assertion rewriting:**
```python
# Pytest automatically rewrites assertions for better output
assert result.rtf < 0.5  # Good error message

# Avoid assertEqual for better output
# self.assertEqual(result.rtf, 0.3)  # Less detailed
```

**Add context:**
```python
assert result.rtf < 0.5, f"RTF too high: {result.rtf} (expected < 0.5)"
```

**Use approx for floats:**
```python
from pytest import approx

assert result.rtf == approx(0.3, abs=0.1)  # Within ±0.1
assert result.duration == approx(1.5, rel=0.05)  # Within 5%
```

---

### 9. Fixture Cleanup Failures

**Symptom:**
```
E   RuntimeError: Session closed
```

**Cause:** Fixture cleanup order issue

**Solution:**

**Use yield fixtures:**
```python
@pytest.fixture
async def redis_client():
    # Setup
    client = await aioredis.from_url("redis://localhost:6379")

    # Provide to test
    yield client

    # Cleanup (always runs)
    await client.close()
```

**Handle cleanup errors:**
```python
@pytest.fixture
async def redis_client():
    client = await aioredis.from_url("redis://localhost:6379")
    yield client

    try:
        await client.close()
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")
```

---

### 10. Configuration Errors in Tests

**Symptom:**
```
E   ValidationError: field required
```

**Cause:** Invalid test configuration

**Use valid configs in tests:**
```python
# Bad: Incomplete config
config = TTSWorkerConfig(worker={"name": "test"})

# Good: Complete config or use defaults
config = TTSWorkerConfig(
    worker=WorkerConfig(name="test-worker"),
    model_manager=ModelManagerConfig(default_model_id="test-model"),
    audio=AudioConfig(),
    redis=RedisConfig()
)

# Or load from fixture
@pytest.fixture
def valid_worker_config():
    return TTSWorkerConfig.from_yaml("tests/fixtures/worker.yaml")
```

---

## Debugging Techniques

### 1. Interactive Debugging (pdb)

**Drop into debugger on failure:**
```bash
uv run pytest tests/unit/test_config.py --pdb
```

**Set breakpoint in code:**
```python
def test_something():
    result = calculate()
    breakpoint()  # Debugger will stop here
    assert result > 0
```

**pdb commands:**
```
(Pdb) l          # List code around current line
(Pdb) n          # Next line
(Pdb) s          # Step into function
(Pdb) c          # Continue execution
(Pdb) p variable # Print variable
(Pdb) pp dict    # Pretty-print
(Pdb) w          # Where am I (stack trace)
(Pdb) q          # Quit
```

---

### 2. Verbose Output

**Show all output:**
```bash
# -v: Verbose test names
# -s: Show print statements
# -vv: Very verbose
uv run pytest tests/ -vv -s
```

**Capture modes:**
```bash
# Capture nothing (show all output)
uv run pytest tests/ -s

# Capture only failed test output
uv run pytest tests/ --capture=no

# Show logs
uv run pytest tests/ --log-cli-level=DEBUG
```

---

### 3. Test Isolation

**Run single test:**
```bash
uv run pytest tests/unit/test_config.py::test_worker_config_validation -v
```

**Run only failed tests:**
```bash
uv run pytest --lf  # Last failed
uv run pytest --ff  # Failed first, then others
```

**Mark tests for selective running:**
```python
@pytest.mark.slow
def test_slow_operation():
    ...

@pytest.mark.integration
def test_redis_integration():
    ...
```

**Run by marker:**
```bash
# Run only slow tests
uv run pytest -m slow

# Skip slow tests
uv run pytest -m "not slow"

# Run integration tests
uv run pytest -m integration
```

---

### 4. Logging in Tests

**Enable logging:**
```python
import logging

def test_with_logging(caplog):
    caplog.set_level(logging.DEBUG)

    # Run test
    result = my_function()

    # Check logs
    assert "Expected log message" in caplog.text
    assert any("error" in record.message.lower() for record in caplog.records)
```

**View logs:**
```bash
uv run pytest tests/ --log-cli-level=DEBUG
```

---

### 5. Test Coverage

**Run with coverage:**
```bash
# Install coverage
uv add --dev pytest-cov

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# View report
open htmlcov/index.html
```

**Coverage configuration (`pyproject.toml`):**
```toml
[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/generated/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

---

## Integration Test Debugging

### Redis Integration Tests

**Check Redis connection:**
```python
@pytest.fixture
async def redis_client():
    # Ensure Redis is running
    import subprocess
    result = subprocess.run(["docker", "ps", "|", "grep", "redis"], capture_output=True)
    if result.returncode != 0:
        pytest.skip("Redis not running")

    client = await aioredis.from_url("redis://localhost:6379")
    yield client
    await client.close()
```

**Clean Redis between tests:**
```python
@pytest.fixture(autouse=True)
async def clean_redis(redis_client):
    # Clean before test
    await redis_client.flushdb()
    yield
    # Clean after test
    await redis_client.flushdb()
```

---

### gRPC Integration Tests

**Mock gRPC server:**
```python
import grpc
from grpc.aio import server as aio_server
from concurrent import futures

@pytest.fixture
async def mock_grpc_server():
    server = aio_server()
    port = server.add_insecure_port('[::]:0')  # Random port

    # Add servicer
    from rpc.generated import tts_pb2_grpc
    tts_pb2_grpc.add_TTSServicer_to_server(MockTTSServicer(), server)

    await server.start()
    yield f"localhost:{port}"
    await server.stop(grace=1.0)
```

---

### Docker Container Tests

**Wait for container ready:**
```python
import time

@pytest.fixture
def redis_container():
    client = docker.from_env()
    container = client.containers.run(
        "redis:7",
        detach=True,
        ports={'6379/tcp': 6379},
        remove=True
    )

    # Wait for Redis to be ready
    for _ in range(30):
        try:
            r = redis.Redis(host='localhost', port=6379)
            r.ping()
            break
        except redis.ConnectionError:
            time.sleep(0.1)
    else:
        pytest.fail("Redis container did not become ready")

    yield container
    container.stop()
```

---

## CI/CD Test Failures

### GitHub Actions Debugging

**View test logs:**
```bash
# In GitHub Actions workflow
- name: Run tests
  run: |
    uv run pytest tests/ -v --log-cli-level=DEBUG
```

**Reproduce CI environment locally:**
```bash
# Use same Python version
python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
uv sync

# Run tests
just ci
```

---

### Docker Test Issues

**Network isolation:**
```yaml
# docker-compose.test.yml
services:
  test-redis:
    image: redis:7
    networks:
      - test-network

  test-runner:
    build: .
    environment:
      - REDIS_URL=redis://test-redis:6379
    networks:
      - test-network
    depends_on:
      - test-redis

networks:
  test-network:
```

**Run tests in Docker:**
```bash
docker compose -f docker-compose.test.yml run test-runner pytest tests/
```

---

## Test Organization Best Practices

### Directory Structure

```
tests/
├── unit/                    # Fast, no external dependencies
│   ├── test_config.py
│   ├── test_audio.py
│   └── test_utils.py
├── integration/             # External services (Redis, gRPC)
│   ├── test_redis.py
│   ├── test_grpc.py
│   └── test_worker.py
├── e2e/                     # Full system tests
│   ├── test_websocket.py
│   └── test_synthesis.py
├── fixtures/                # Test data
│   ├── worker.yaml
│   └── sample_audio.wav
└── conftest.py              # Shared fixtures
```

---

### Test Naming

**Convention:** `test_<functionality>_<condition>_<expected_result>`

Examples:
```python
def test_config_validation_missing_field_raises_error():
    ...

def test_synthesis_valid_text_returns_audio():
    ...

def test_worker_connection_timeout_raises_exception():
    ...
```

---

### Fixture Organization

**conftest.py hierarchy:**
```python
# tests/conftest.py - Shared across all tests
@pytest.fixture
def temp_dir():
    ...

# tests/integration/conftest.py - Integration-specific
@pytest.fixture
async def redis_client():
    ...

# tests/e2e/conftest.py - E2E-specific
@pytest.fixture
async def full_system():
    ...
```

---

## Performance Testing

### Slow Test Detection

**Mark slow tests:**
```python
@pytest.mark.slow
def test_model_loading():
    ...
```

**Find slowest tests:**
```bash
uv run pytest --durations=10
```

**Profile tests:**
```bash
uv add --dev pytest-profiling
uv run pytest --profile tests/
```

---

## Troubleshooting Checklist

**Before running tests:**
```bash
# 1. Environment validation
./scripts/validate-environment.sh

# 2. Generate protos
just gen-proto

# 3. Start required services
just redis

# 4. Validate configs
uv run python scripts/validate-config.py

# 5. Run linter
just lint

# 6. Run type checker
just typecheck

# 7. Run tests
just test
```

**Test debugging steps:**
```bash
# 1. Run specific failing test with verbose output
uv run pytest tests/path/to/test.py::test_name -vv -s

# 2. Use debugger
uv run pytest tests/path/to/test.py::test_name --pdb

# 3. Check fixtures
uv run pytest --fixtures tests/

# 4. Clean environment
docker compose down -v
docker container prune -f
rm -rf .pytest_cache

# 5. Re-run tests
just test
```

---

## Related Runbooks

- **[Environment Setup](ENVIRONMENT.md)** - Setup validation
- **[Log Debugging](LOG_DEBUGGING.md)** - Log analysis
- **[Advanced Troubleshooting](ADVANCED_TROUBLESHOOTING.md)** - Deep diagnostics

---

## Further Help

**Quick test commands:**
```bash
# Run all tests
just test

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Debug single test
uv run pytest tests/unit/test_config.py::test_name -vv --pdb

# Show available fixtures
uv run pytest --fixtures

# Find slow tests
uv run pytest --durations=10
```

**Useful pytest plugins:**
- `pytest-asyncio` - Async test support
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel test execution
- `pytest-timeout` - Test timeouts
- `pytest-mock` - Mocking utilities

**Still stuck?**
1. Check test logs: `uv run pytest -vv -s`
2. Verify environment: `./scripts/validate-environment.sh`
3. Clean state: `docker compose down -v && just redis`
4. Run single test: `uv run pytest path/to/test.py::test_name --pdb`
5. Review test fixtures: `uv run pytest --fixtures`
