# Performance Tests

This directory contains performance benchmark tests for the full-duplex voice chat system.

## Overview

Performance tests measure system behavior under load, including:
- **First Audio Latency (FAL)**: Time from text submission to first audio frame
- **Frame Jitter**: Consistency of audio frame delivery timing
- **Barge-in Latency**: Time to respond to pause/resume commands
- **Throughput**: Messages per second under concurrent load
- **Resource Usage**: Baseline memory and CPU consumption
- **Cold Start**: Initialization latency for new sessions
- **Sustained Load**: System stability under continuous operation

## Docker Requirements

**IMPORTANT:** All performance and integration tests require Docker containers to be running.

### Required Services

The tests depend on the following Docker services:
- **Redis** (port 6380 for tests, 6379 for dev) - Service discovery and worker registration
- **Mock TTS Worker** (port 7001) - gRPC TTS worker for testing
- **Orchestrator** (port 8080) - WebSocket server

### Starting Services

**Option 1: Using Docker Compose (Recommended)**
```bash
# Start all required services in background
docker compose up -d

# Verify services are running
docker compose ps

# View logs if needed
docker compose logs -f
```

**Option 2: Using justfile**
```bash
# Start Redis only (minimal setup)
just redis

# Or start full stack
just docker-up
```

### Stopping Services

```bash
# Stop all services
docker compose down

# Or stop Redis only
just redis-stop
```

## Running Performance Tests

**IMPORTANT:** Performance tests MUST be run with the `--forked` flag to prevent segmentation faults.

### Step 1: Start Docker Services

```bash
docker compose up -d
```

Wait 5-10 seconds for services to be ready.

### Step 2: Run Tests

**Recommended: Using justfile**
```bash
just test-performance
```

**Or directly with pytest:**
```bash
uv run pytest tests/performance/test_performance.py --forked -v -m performance
```

**Run specific test:**
```bash
uv run pytest tests/performance/test_performance.py::test_fal_single_session --forked -v
```

### Why --forked is Required

The concurrent WebSocket tests (`test_fal_concurrent_3_sessions` and `test_fal_concurrent_10_sessions`) trigger a garbage collection segfault in Python 3.13.6 during concurrent `websocket.connect()` operations. This is a known issue with Python 3.13.x's garbage collector when combined with the websockets library.

The crash occurs during GC while creating concurrent WebSocket connections:
```
Fatal Python error: Segmentation fault
File "/lib/python3.13/urllib/parse.py", line 401 in urlparse
File "websockets/uri.py", line 75 in parse_uri
File "websockets/asyncio/client.py", line 378 in create_connection
```

**Process isolation via pytest-forked prevents crashes in one test from affecting other tests**, allowing the full test suite to complete and report results correctly.

### Fixture Architecture

The module-scoped fixtures (`orchestrator_server`, `redis_container`, `mock_tts_worker`) are managed as follows:

1. **Fixture Setup:** Fixtures start Docker containers once at module import time
2. **Test Execution:** Each test forks a subprocess that inherits fixture connections
3. **Process Isolation:** If a test crashes (segfault), only that subprocess dies
4. **Result Reporting:** The parent process collects results and continues to next test
5. **Fixture Teardown:** Parent process handles cleanup after all tests complete

This ensures:
- Full test suite completes even if individual tests crash
- All test results are properly reported
- No contamination between tests
- Performance metrics remain accurate

## Test Markers

All tests in this directory are marked with:
- `@pytest.mark.integration` - Tests requiring full system integration
- `@pytest.mark.docker` - Tests requiring Docker containers
- `@pytest.mark.redis` - Tests requiring Redis container
- `@pytest.mark.performance` - Performance benchmark tests
- `@pytest.mark.asyncio` - Async execution

These markers allow selective test execution:
```bash
# Run only Docker-dependent tests
uv run pytest -m docker

# Run only Redis-dependent tests
uv run pytest -m redis

# Run only integration tests (not performance)
uv run pytest -m "integration and not performance"

# Skip Docker tests (useful for quick local runs)
uv run pytest -m "not docker"
```

## Performance Targets

- **FAL (single session, p95):** < 300ms
- **FAL (concurrent 3 sessions, p95):** < 400ms
- **FAL (concurrent 10 sessions, p95):** < 600ms
- **Frame jitter (p95):** < 5ms
- **Barge-in latency (p95):** < 50ms
- **Throughput:** > 20 messages/second

## Test Suite

### 1. test_fal_single_session
Benchmarks First Audio Latency with a single session under no load.
- **Target:** FAL p95 < 300ms
- **Measurements:** 20 iterations
- **Status:** Stable with --forked
- **Markers:** `integration`, `docker`, `redis`, `performance`, `asyncio`

### 2. test_fal_concurrent_3_sessions
Measures FAL with 3 concurrent sessions to test system behavior under moderate load.
- **Target:** FAL p95 < 400ms
- **Status:** Requires --forked to prevent segfault
- **Markers:** `integration`, `docker`, `redis`, `performance`, `asyncio`

### 3. test_fal_concurrent_10_sessions
Stress test with 10 concurrent sessions to measure system scalability.
- **Target:** FAL p95 < 600ms
- **Duration:** ~2-3 minutes
- **Status:** Requires --forked to prevent segfault
- **Markers:** `integration`, `docker`, `redis`, `performance`, `asyncio`

### 4. test_frame_jitter_measurement
Measures consistency of audio frame delivery timing.
- **Target:** Frame jitter p95 < 5ms
- **Measurements:** Up to 50 frames analyzed
- **Status:** Stable with --forked
- **Markers:** `integration`, `docker`, `redis`, `performance`, `asyncio`

### 5. test_throughput_benchmark
Measures maximum messages per second the system can handle.
- **Target:** > 20 messages/second
- **Duration:** 10 seconds
- **Status:** Stable with --forked
- **Markers:** `integration`, `docker`, `redis`, `performance`, `asyncio`

### 6. test_latency_percentiles_distribution
Analyzes full latency distribution (p10, p25, p50, p75, p90, p95, p99).
- **Purpose:** Characterize tail latency behavior
- **Measurements:** 100 samples
- **Status:** Stable with --forked
- **Markers:** `integration`, `docker`, `redis`, `performance`, `asyncio`

### 7. test_cold_start_vs_warm_latency
Compares first-connection latency vs subsequent requests.
- **Purpose:** Measure connection/warmup overhead
- **Measurements:** Cold start + 10 warm requests
- **Status:** Stable with --forked
- **Markers:** `integration`, `docker`, `redis`, `performance`, `asyncio`

### 8. test_frame_delivery_consistency
Validates that all frames are delivered in correct sequence without drops.
- **Purpose:** Frame integrity validation
- **Measurements:** 10 iterations of consistent text
- **Status:** Stable with --forked
- **Markers:** `integration`, `docker`, `redis`, `performance`, `asyncio`

### 9. test_stress_test_rapid_messages
Sends messages as fast as possible to find breaking point.
- **Purpose:** Identify system limits
- **Measurements:** 50 messages with minimal delay
- **Status:** Stable with --forked
- **Markers:** `integration`, `docker`, `redis`, `performance`, `asyncio`

### 10. test_memory_stability_long_session
Long-running session to detect memory leaks or degradation.
- **Measurements:** 50 messages in single session
- **Purpose:** Memory leak detection and latency degradation validation
- **Status:** Stable with --forked
- **Markers:** `integration`, `docker`, `redis`, `performance`, `asyncio`

## Troubleshooting

### Issue: Docker not available
**Error:** `Docker not available`
**Solution:** Install Docker and start the Docker daemon

### Issue: Tests timeout waiting for containers
**Error:** `RuntimeError: Redis failed to start`
**Solution:**
1. Ensure Docker containers are running: `docker compose ps`
2. Check container health: `docker compose logs redis`
3. Restart services: `docker compose restart`

### Issue: Connection refused errors
**Error:** `ConnectionRefusedError: [Errno 111] Connection refused`
**Solution:**
1. Verify services are listening on expected ports:
   ```bash
   docker compose ps
   netstat -tlnp | grep -E "(6380|7001|8080)"
   ```
2. Check firewall rules
3. Ensure no port conflicts with other services

### Issue: Segmentation fault during tests
**Error:** `Fatal Python error: Segmentation fault`
**Solution:** Always run tests with `--forked` flag:
```bash
uv run pytest tests/performance/test_performance.py --forked -v
```

### Issue: Inconsistent latency results
**Solution:**
1. Ensure no other processes consuming CPU/GPU
2. Run on isolated system for reliable benchmarks
3. Close unnecessary applications
4. Check system load: `top` or `htop`

### Issue: Services fail to start
**Solution:**
1. Check Docker logs: `docker compose logs`
2. Verify port availability: `sudo lsof -i :6379 -i :7001 -i :8080`
3. Clean up old containers: `docker compose down -v`
4. Rebuild images: `docker compose build --no-cache`

## How pytest-forked Works

With `--forked`, each test runs in a separate subprocess:

1. **Fixture Setup:** Module-scoped fixtures (Redis, orchestrator, mock TTS worker) are set up once in the parent process
2. **Test Execution:** Each test forks a subprocess that inherits fixture connections
3. **Process Isolation:** If a test crashes (segfault), only that subprocess dies
4. **Result Reporting:** The parent process collects results and continues to next test
5. **Fixture Teardown:** Parent process handles cleanup after all tests complete

The overhead is minimal (process forking adds ~100-200ms per test), which is negligible compared to the test execution time.

## Interpreting Results

Performance test results are logged with summary statistics:
- **mean**: Average latency across all measurements
- **p50**: Median latency (50th percentile)
- **p95**: 95th percentile (only 5% of requests slower)
- **p99**: 99th percentile (worst-case typical)
- **min**: Best observed latency
- **max**: Worst observed latency

Focus on p95/p99 for production SLA targets, as mean can be misleading with outliers.

## CI/CD Integration

Performance tests are excluded from standard CI runs due to:
- Long execution time (10-15 minutes)
- Docker infrastructure requirements
- Resource intensive workloads

To enable performance tests in CI, create a dedicated workflow:
```yaml
name: Performance Tests
on:
  workflow_dispatch:
  schedule:
    - cron: '0 2 * * *'  # Run nightly

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Start Docker Services
        run: docker compose up -d
      - name: Wait for Services
        run: sleep 10
      - name: Run Performance Tests
        run: |
          uv run pytest tests/performance/test_performance.py --forked -v -m performance
      - name: Stop Docker Services
        run: docker compose down
```

## Contributing

When adding new performance tests:
1. Add `@pytest.mark.integration`, `@pytest.mark.docker`, `@pytest.mark.redis`, and `@pytest.mark.performance` decorators
2. Include clear docstring with target metrics
3. Log summary statistics for easy result interpretation
4. Keep test duration reasonable (< 5 minutes per test)
5. Document any Docker/infrastructure requirements
6. Test with `--forked` flag to ensure compatibility
7. Add status note in this README

## Development Workflow

**Quick iteration (no Docker):**
```bash
# Run only unit tests (no Docker required)
just test
```

**Full integration testing:**
```bash
# Start services
docker compose up -d

# Run integration tests
just test-integration

# Run performance tests
just test-performance

# Stop services
docker compose down
```

**Complete validation:**
```bash
# Run all quality checks + unit tests (no Docker)
just ci

# Then run Docker-dependent tests separately
docker compose up -d
just test-integration
just test-performance
docker compose down
```
