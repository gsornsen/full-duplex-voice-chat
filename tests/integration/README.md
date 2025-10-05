# Integration Tests

This directory contains integration tests for the TTS worker and client system.

## Test Coverage

### M1 Worker Integration Tests (`test_m1_worker_integration.py`)

Comprehensive integration tests covering:

1. **Connection & Discovery**
   - `test_worker_connection`: Basic connection and capabilities retrieval
   - `test_list_models`: Model listing and metadata validation
   - `test_capabilities_consistency`: Capabilities consistency across calls

2. **Session Management**
   - `test_session_lifecycle`: Session start/end lifecycle
   - `test_multiple_sessions_sequential`: Multiple sequential sessions
   - `test_session_isolation`: Concurrent session isolation

3. **Streaming Synthesis**
   - `test_streaming_synthesis`: Text-to-audio streaming flow
   - `test_frame_size_validation`: Exact frame size verification (1920 bytes)
   - `test_empty_text_chunks`: Empty text handling

4. **Control Commands**
   - `test_pause_command`: PAUSE command and timing
   - `test_resume_command`: RESUME command after pause
   - `test_stop_command`: STOP command and termination
   - `test_pause_response_timing`: PAUSE response time under load (< 50ms SLA)

5. **Model Lifecycle**
   - `test_load_unload_model`: Dynamic model loading/unloading

6. **Error Handling**
   - `test_invalid_session_error`: Session validation
   - `test_invalid_command_error`: Command validation

## Running Tests

### Run All Integration Tests

```bash
uv run pytest tests/integration/
```

### Run Worker Integration Tests Only

```bash
uv run pytest tests/integration/test_m1_worker_integration.py
```

### Run Specific Test

```bash
uv run pytest tests/integration/test_m1_worker_integration.py::test_pause_command -v
```

### Run with Debug Logging

```bash
uv run pytest tests/integration/test_m1_worker_integration.py -v -s --log-cli-level=DEBUG
```

## Test Architecture

### Fixtures

**`worker_process` (module-scoped)**

- Starts TTS worker in subprocess
- Waits 2 seconds for worker to initialize
- Validates worker is running
- Terminates worker on teardown (5s timeout, then kill)

**`client` (function-scoped)**

- Creates `TTSWorkerClient` instance
- Connects to worker at `localhost:7001`
- Disconnects and cleans up after test

### Test Strategy

1. **Process Isolation**: Worker runs in subprocess for realistic testing
2. **Connection Management**: Each test gets fresh client connection
3. **Session Cleanup**: Sessions are properly ended to avoid resource leaks
4. **Timing Validation**: Control commands measured for < 50ms SLA
5. **Frame Validation**: Audio frames checked for exact size (1920 bytes)

## Performance SLAs

The tests verify these performance requirements:

- **Pause Response Time**: < 50ms (critical for barge-in)
- **Frame Duration**: Exactly 20ms
- **Sample Rate**: 48kHz
- **Frame Size**: 1920 bytes (960 samples × 2 bytes/sample)

## Dependencies

- `pytest >= 8.0.0`
- `pytest-asyncio >= 1.2.0`
- Worker must be runnable via `uv run python -m src.tts.worker_main`

## Troubleshooting

### Worker Fails to Start

- Check that port 7001 is available
- Verify `uv` is installed and configured
- Check worker logs in subprocess stderr

### Tests Timeout

- Increase worker startup wait time in `worker_process` fixture
- Check system resources (CPU, memory)
- Verify no firewall blocking localhost:7001

### Frame Count Mismatches

- Mock adapter generates 25 frames per 500ms chunk
- Tests allow ±10 frames variation for timing jitter
- Check `INTER_FRAME_DELAY_MS` in `adapter_mock.py`

## Future Tests

Planned integration tests:

- Multi-GPU worker coordination
- Model manager TTL eviction
- LRU cache behavior
- Concurrent session stress testing
- Network error handling and retries
- Redis service discovery integration
