# M2 Code Examples

Standalone code examples for common M2 usage patterns. All examples are self-contained and runnable.

## Requirements

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Running the Examples

### Prerequisites

Make sure the required services are running:

```bash
# Start Redis (required for worker discovery)
just redis

# Start TTS worker (in separate terminal)
just run-tts-mock

# Start orchestrator (in separate terminal)
just run-orch
```

## Examples

### 1. Basic WebSocket Client
**File:** `basic_websocket_client.py`

Simple WebSocket client that connects to the orchestrator, starts a session, sends text for synthesis, and receives audio frames.

```bash
python examples/basic_websocket_client.py
```

**Demonstrates:**
- WebSocket connection to orchestrator
- Session lifecycle (start/end)
- Text chunk streaming
- Audio frame reception
- Proper error handling

### 2. gRPC Worker Client
**File:** `grpc_worker_client.py`

Direct gRPC client that communicates with a TTS worker without going through the orchestrator. Useful for testing worker functionality in isolation.

```bash
python examples/grpc_worker_client.py
```

**Demonstrates:**
- Direct gRPC connection to worker
- Bidirectional streaming synthesis
- Session management
- Control commands (PAUSE/RESUME/STOP)
- Capability querying

### 3. Audio Frame Handler
**File:** `audio_frame_handler.py`

Helper utilities for processing audio frames, including validation, file writing, and format conversion.

```bash
python examples/audio_frame_handler.py
```

**Demonstrates:**
- Audio frame validation (size, format)
- Writing frames to WAV file
- RMS level calculation
- Frame statistics collection
- Buffer management

### 4. Worker Registration
**File:** `worker_registration.py`

Example of worker registration and discovery using Redis. Shows how workers announce themselves and how clients discover available workers.

```bash
python examples/worker_registration.py
```

**Demonstrates:**
- Worker registration in Redis
- Capability advertisement
- Heartbeat mechanism
- Worker discovery queries
- TTL and expiration handling

## Common Patterns

### Error Handling

All examples include proper error handling:

```python
try:
    await run_client()
except KeyboardInterrupt:
    print("\nInterrupted by user")
    sys.exit(0)
except ConnectionError as e:
    print(f"Connection failed: {e}")
    sys.exit(1)
```

### Async Context Managers

Examples use async context managers for resource cleanup:

```python
async with websockets.connect(url) as ws:
    # Connection automatically closed on exit
    await process_messages(ws)
```

### Type Hints

All examples include complete type hints:

```python
from src.common.types import AudioFrame, SessionID

async def process_frame(frame: AudioFrame, session_id: SessionID) -> None:
    # Type-safe processing
    pass
```

## Troubleshooting

### Connection Refused

If you see connection errors:
1. Verify services are running: `docker ps` or check process list
2. Check port availability: `netstat -tuln | grep 8080`
3. Verify Redis is accessible: `redis-cli ping`

### Import Errors

Make sure you've activated the virtual environment:
```bash
source .venv/bin/activate
which python  # Should show .venv/bin/python
```

### Audio Issues

If audio frames are invalid:
1. Check frame size: Should be exactly 1920 bytes (20ms at 48kHz)
2. Verify sample rate: Must be 48000 Hz
3. Confirm format: 16-bit PCM, mono, little-endian

## Additional Resources

- [Testing Guide](../project_documentation/testing_guide.md) - Integration testing patterns
- [API Reference](../project_documentation/api_reference.md) - Complete API documentation
- [TDD](../project_documentation/TDD.md) - Technical design details
- [CLAUDE.md](../CLAUDE.md) - Project overview and architecture

## Contributing

When adding new examples:

1. Follow the existing code style (ruff format)
2. Include complete type hints
3. Add comprehensive docstrings
4. Test the example end-to-end
5. Update this README with usage instructions
6. Keep examples self-contained and runnable
