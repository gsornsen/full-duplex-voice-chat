# Integration Tests

Comprehensive integration tests for the Realtime Duplex Voice system.

## Overview

The integration test suite validates end-to-end functionality across all system components:

- **WebSocket Transport**: Full text → audio flow via WebSocket
- **VAD Processing**: Voice Activity Detection with synthetic audio
- **Redis Discovery**: Worker registration and service discovery
- **Full Pipeline**: Complete system integration with all components
- **LiveKit Transport**: WebRTC-based transport (placeholder for future)
- **Performance Benchmarks**: Latency, throughput, and stability metrics

## Prerequisites

### Required

- **Docker**: For running Redis and LiveKit containers
- **Python 3.13+**: For running tests
- **uv**: Python package manager

### Optional

- **LiveKit server**: For WebRTC transport tests (currently skipped in M2)

## Running Tests

### All Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run with detailed output
pytest tests/integration/ -vv --tb=long
```

### By Category

```bash
# WebSocket tests only
pytest tests/integration/test_websocket_e2e.py -v

# VAD tests only
pytest tests/integration/test_vad_integration.py -v

# Redis discovery tests
pytest tests/integration/test_redis_discovery.py -v

# Full pipeline tests
pytest tests/integration/test_full_pipeline.py -v

# Performance benchmarks
pytest tests/integration/test_performance.py -v

# LiveKit tests (currently skipped)
pytest tests/integration/test_livekit_e2e.py -v
```

### By Marker

```bash
# All integration tests
pytest -m integration

# Docker-dependent tests
pytest -m docker

# Redis-dependent tests
pytest -m redis

# Performance benchmarks
pytest -m performance

# Skip slow tests
pytest -m "not slow"
```

### Skip Docker Tests

```bash
# Run tests that don't require Docker
pytest tests/integration/ -v -m "not docker"
```

## Test Files

### `conftest.py`

Central fixture file providing:

- **Docker Container Management**: Redis and LiveKit containers
- **Mock TTS Worker**: Spawns and manages test TTS worker
- **Orchestrator Server**: Starts orchestrator for testing
- **WebSocket Clients**: Pre-configured WS client connections
- **Audio Generation**: Synthetic audio for VAD testing
- **Validation Utilities**: Frame timing and latency measurement
- **Metric Collectors**: Performance metric aggregation

### `test_websocket_e2e.py`

End-to-end WebSocket flow tests covering full text → audio pipeline.

### `test_vad_integration.py`

Voice Activity Detection integration tests with synthetic audio.

### `test_redis_discovery.py`

Redis service discovery tests for worker registration.

### `test_full_pipeline.py`

Full system pipeline integration tests with all components.

### `test_performance.py`

Performance benchmark tests measuring latency and throughput.

### `test_livekit_e2e.py`

LiveKit transport tests (M2 placeholders, mostly skipped).

## Performance Targets

### First Audio Latency (FAL)

- **Single session**: < 300ms (p95)
- **3 concurrent**: < 400ms (p95)
- **10 concurrent**: < 600ms (p95)

### Frame Timing

- **Cadence**: 20ms ± 5ms
- **p95 jitter**: < 5ms
- **Mean interval**: 18-22ms

### VAD Processing

- **Per-frame latency**: < 5ms (mean)
- **p95 latency**: < 10ms
- **p99 latency**: < 20ms

## Troubleshooting

### Docker Not Available

Tests will skip automatically if Docker is not installed.

### Redis Connection Issues

Check Docker logs: `docker logs test-redis-integration`

### Port Conflicts

Find and kill processes using ports 6379, 7880, or 8080.

### Test Timeouts

Increase timeout in pytest.ini or use `@pytest.mark.timeout(600)`.

## Contributing

When adding new integration tests:

1. Follow existing structure and naming conventions
2. Add appropriate markers for categorization
3. Include fixtures in `conftest.py` if reusable
4. Document new utilities
5. Ensure tests pass in CI before merging
