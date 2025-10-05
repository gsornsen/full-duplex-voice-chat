# Integration Tests - Quick Start

## Prerequisites

```bash
# Install dependencies
uv sync --dev

# Ensure Docker is running
docker info
```

## Run Tests

### All Integration Tests

```bash
uv run pytest tests/integration/ -v
```

### Skip Docker Tests (if Docker unavailable)

```bash
uv run pytest tests/integration/ -v -m "not docker"
```

### By Category

```bash
# WebSocket E2E
uv run pytest tests/integration/test_websocket_e2e.py -v

# VAD Integration
uv run pytest tests/integration/test_vad_integration.py -v

# Redis Discovery
uv run pytest tests/integration/test_redis_discovery.py -v

# Full Pipeline
uv run pytest tests/integration/test_full_pipeline.py -v

# Performance Benchmarks
uv run pytest tests/integration/test_performance.py -v
```

### By Marker

```bash
# Performance tests only
uv run pytest -m performance -v

# Skip slow tests
uv run pytest -m "not slow" -v

# Redis-dependent tests
uv run pytest -m redis -v
```

## Expected Results

### Performance Targets

- **FAL (single session)**: < 300ms (p95) ✅
- **FAL (3 concurrent)**: < 400ms (p95) ✅
- **FAL (10 concurrent)**: < 600ms (p95) ✅
- **Frame jitter**: < 5ms (p95) ✅
- **VAD latency**: < 5ms (mean) ✅

### Test Duration

- **With Docker**: ~3-4 minutes (includes container setup)
- **Without Docker**: ~30 seconds (VAD and utility tests only)

## Troubleshooting

### Docker Issues

```bash
# Check Docker is running
docker info

# Check for port conflicts
lsof -i :6379  # Redis
lsof -i :7880  # LiveKit
lsof -i :8080  # Orchestrator
```

### Test Failures

```bash
# Run with verbose output
uv run pytest tests/integration/ -vv --tb=long

# Run specific failing test
uv run pytest tests/integration/test_websocket_e2e.py::test_name -vv

# Check logs
cat tests/logs/uv run pytest.log
```

### Timeout Issues

```bash
# Increase timeout
uv run pytest tests/integration/ -v --timeout=600
```

## CI/CD

### GitHub Actions

```bash
# Run tests like CI does
uv run pytest tests/integration/ -v -m "not livekit" --tb=short
```

### Coverage

```bash
# Generate coverage report
uv run pytest tests/integration/ --cov=src --cov-report=html
open htmlcov/index.html
```

## Common Commands

```bash
# Full test suite with coverage
uv run pytest tests/integration/ -v --cov=src --cov-report=term-missing

# Performance benchmarks only
uv run pytest tests/integration/test_performance.py -v

# Quick smoke test (no Docker)
uv run pytest tests/integration/ -v -m "not docker" -k "not slow"

# Verbose debugging
uv run pytest tests/integration/ -vv --tb=long --log-cli-level=DEBUG
```

## Test Markers

- `integration`: All integration tests
- `docker`: Requires Docker containers
- `redis`: Requires Redis
- `livekit`: Requires LiveKit (mostly skipped in M2)
- `performance`: Performance benchmark tests
- `slow`: Tests taking > 5 seconds

## Files

- `conftest.py`: Shared fixtures and utilities
- `test_websocket_e2e.py`: WebSocket flow tests
- `test_vad_integration.py`: VAD tests
- `test_redis_discovery.py`: Redis tests
- `test_full_pipeline.py`: Full system tests
- `test_performance.py`: Performance benchmarks
- `test_livekit_e2e.py`: LiveKit tests (placeholders)

## Help

For detailed documentation, see:

- `README_INTEGRATION.md`: Full integration test guide
