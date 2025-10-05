# Integration Tests - Quick Reference

## Quick Start

```bash
# Run all integration tests
uv run pytest tests/integration/

# Run worker tests only
uv run pytest tests/integration/test_m1_worker_integration.py

# Run with verbose output
uv run pytest tests/integration/test_m1_worker_integration.py -v

# Run with debug logging
uv run pytest tests/integration/test_m1_worker_integration.py -v -s --log-cli-level=DEBUG
```

## Run Specific Tests

```bash
# Connection tests
uv run pytest tests/integration/test_m1_worker_integration.py::test_worker_connection -v

# Session tests
uv run pytest tests/integration/test_m1_worker_integration.py::test_session_lifecycle -v
uv run pytest tests/integration/test_m1_worker_integration.py::test_session_isolation -v

# Streaming tests
uv run pytest tests/integration/test_m1_worker_integration.py::test_streaming_synthesis -v
uv run pytest tests/integration/test_m1_worker_integration.py::test_frame_size_validation -v

# Control command tests
uv run pytest tests/integration/test_m1_worker_integration.py::test_pause_command -v
uv run pytest tests/integration/test_m1_worker_integration.py::test_resume_command -v
uv run pytest tests/integration/test_m1_worker_integration.py::test_stop_command -v

# Timing/performance tests
uv run pytest tests/integration/test_m1_worker_integration.py::test_pause_response_timing -v

# Error handling tests
uv run pytest tests/integration/test_m1_worker_integration.py::test_invalid_session_error -v
uv run pytest tests/integration/test_m1_worker_integration.py::test_invalid_command_error -v
```

## Filter Tests by Pattern

```bash
# Run all tests with "pause" in name
uv run pytest tests/integration/test_m1_worker_integration.py -k "pause" -v

# Run all tests with "session" in name
uv run pytest tests/integration/test_m1_worker_integration.py -k "session" -v

# Run all tests with "timing" in name
uv run pytest tests/integration/test_m1_worker_integration.py -k "timing" -v

# Run all control command tests
uv run pytest tests/integration/test_m1_worker_integration.py -k "command" -v
```

## Debugging Options

```bash
# Show print statements
uv run pytest tests/integration/test_m1_worker_integration.py -s

# Stop on first failure
uv run pytest tests/integration/test_m1_worker_integration.py -x

# Show local variables on failure
uv run pytest tests/integration/test_m1_worker_integration.py -l

# Full traceback
uv run pytest tests/integration/test_m1_worker_integration.py --tb=long

# Show captured output even for passing tests
uv run pytest tests/integration/test_m1_worker_integration.py -v -s

# Combination: verbose + stop on fail + show locals
uv run pytest tests/integration/test_m1_worker_integration.py -vxl
```

## Test Coverage

```bash
# Run with coverage report
uv run pytest tests/integration/test_m1_worker_integration.py --cov=src.tts --cov=src.orchestrator

# Generate HTML coverage report
uv run pytest tests/integration/test_m1_worker_integration.py --cov=src.tts --cov-report=html

# Show missing lines
uv run pytest tests/integration/test_m1_worker_integration.py --cov=src.tts --cov-report=term-missing
```

## Performance Profiling

```bash
# Run with duration report
uv run pytest tests/integration/test_m1_worker_integration.py --durations=10

# Run with detailed duration report
uv run pytest tests/integration/test_m1_worker_integration.py --durations=0
```

## Test Selection

```bash
# Run first 5 tests only
uv run pytest tests/integration/test_m1_worker_integration.py -k "test_worker_connection or test_list_models or test_session_lifecycle or test_streaming_synthesis or test_pause_command" -v

# Skip specific tests
uv run pytest tests/integration/test_m1_worker_integration.py -k "not session_isolation" -v
```

## Environment Variables

```bash
# Set log level via environment
uv run pytest_LOG_CLI_LEVEL=DEBUG uv run pytest tests/integration/test_m1_worker_integration.py -v

# Disable log output
uv run pytest_LOG_DISABLE=1 uv run pytest tests/integration/test_m1_worker_integration.py
```

## CI/CD Integration

```bash
# Run with JUnit XML output for CI
uv run pytest tests/integration/test_m1_worker_integration.py --junitxml=test-results.xml

# Run with both console and XML output
uv run pytest tests/integration/test_m1_worker_integration.py -v --junitxml=test-results.xml

# Run all quality checks (via justfile)
just ci
```

## Troubleshooting

### Worker won't start

```bash
# Check if port is in use
lsof -i :7001

# Kill process on port 7001
kill -9 $(lsof -t -i:7001)

# Check worker can start manually
uv run python -m src.tts.worker_main
```

### Tests hanging

```bash
# Add timeout (10 seconds per test)
uv run pytest tests/integration/test_m1_worker_integration.py --timeout=10
```

### Memory issues

```bash
# Run with memory profiling
uv run pytest tests/integration/test_m1_worker_integration.py --memprof
```

## Expected Results

```shell
tests/integration/test_m1_worker_integration.py::test_worker_connection PASSED           [  5%]
tests/integration/test_m1_worker_integration.py::test_list_models PASSED                 [ 11%]
tests/integration/test_m1_worker_integration.py::test_session_lifecycle PASSED           [ 16%]
tests/integration/test_m1_worker_integration.py::test_streaming_synthesis PASSED         [ 22%]
tests/integration/test_m1_worker_integration.py::test_pause_command PASSED               [ 27%]
tests/integration/test_m1_worker_integration.py::test_resume_command PASSED              [ 33%]
tests/integration/test_m1_worker_integration.py::test_stop_command PASSED                [ 38%]
tests/integration/test_m1_worker_integration.py::test_load_unload_model PASSED           [ 44%]
tests/integration/test_m1_worker_integration.py::test_multiple_sessions_sequential PASSED [ 50%]
tests/integration/test_m1_worker_integration.py::test_invalid_session_error PASSED       [ 55%]
tests/integration/test_m1_worker_integration.py::test_invalid_command_error PASSED       [ 61%]
tests/integration/test_m1_worker_integration.py::test_frame_size_validation PASSED       [ 66%]
tests/integration/test_m1_worker_integration.py::test_pause_response_timing PASSED       [ 72%]
tests/integration/test_m1_worker_integration.py::test_session_isolation PASSED           [ 77%]
tests/integration/test_m1_worker_integration.py::test_empty_text_chunks PASSED           [ 83%]
tests/integration/test_m1_worker_integration.py::test_capabilities_consistency PASSED    [ 88%]

================== 18 passed in 15.23s ==================
```

## Performance Benchmarks

- **Worker Startup**: < 2s
- **PAUSE Response**: < 50ms (SLA)
- **Frame Generation**: ~5ms/frame
- **Session Start**: < 50ms
- **Total Suite**: ~15-20s

## Test Categories

| Category | Test Count | Focus |
|----------|-----------|-------|
| Connection | 3 | Connection, discovery, capabilities |
| Sessions | 3 | Lifecycle, isolation, multiple sessions |
| Streaming | 3 | Synthesis, frames, empty text |
| Control | 4 | PAUSE, RESUME, STOP, timing |
| Models | 1 | Load, unload |
| Errors | 2 | Invalid session, invalid command |
| **Total** | **18** | **Complete coverage** |
