# Orchestrator

## Overview

The orchestrator is the client-facing service that manages WebSocket connections, session state, and routes synthesis requests to TTS workers.

**Responsibilities:**
- Accept WebSocket and LiveKit client connections
- Manage session lifecycle and state machine
- Detect voice activity for barge-in (VAD)
- Convert speech to text (ASR, M10+)
- Route requests to available TTS workers
- Stream audio frames back to clients

---

## Architecture

```
orchestrator/
├── server.py              # Main entry point, starts transport servers
├── config.py              # Pydantic configuration models
├── session.py             # Session management and state machine
├── vad.py                 # Voice Activity Detection
├── asr.py                 # Automatic Speech Recognition (M10+)
├── routing.py             # Worker selection and load balancing
├── registry.py            # Redis-based worker discovery
├── transport/
│   ├── base.py            # Transport abstraction interface
│   ├── websocket_transport.py    # WebSocket transport implementation
│   ├── websocket_protocol.py     # WebSocket message definitions
│   └── livekit/           # LiveKit transport (M3+)
├── audio/
│   └── __init__.py        # Audio utilities
└── livekit/
    └── __init__.py        # LiveKit integration (M3+)
```

---

## Key Modules

### server.py

**Purpose:** Application entry point and server lifecycle management.

**Key Functions:**
- `main()`: Parse config, initialize transports, start server
- `shutdown_handler()`: Graceful shutdown on SIGTERM/SIGINT

**Usage:**
```bash
# Run orchestrator
python -m src.orchestrator.server

# With custom config
CONFIG_PATH=configs/custom.yaml python -m src.orchestrator.server

# With environment overrides
LOG_LEVEL=DEBUG PORT=8080 python -m src.orchestrator.server
```

---

### config.py

**Purpose:** Type-safe configuration using Pydantic models.

**Key Classes:**
- `WebSocketConfig`: WebSocket transport settings
- `LiveKitConfig`: LiveKit transport settings (M3+)
- `RedisConfig`: Redis connection and discovery settings
- `RoutingConfig`: Worker routing configuration
- `VADConfig`: Voice activity detection settings
- `OrchestratorConfig`: Top-level config container

**Example:**
```python
from src.orchestrator.config import OrchestratorConfig
import yaml

# Load config
with open("configs/orchestrator.yaml") as f:
    config_dict = yaml.safe_load(f)

config = OrchestratorConfig(**config_dict)

# Access config values (validated and typed)
port = config.transport.websocket.port  # int
host = config.transport.websocket.host  # str
```

**Validation:**
- Port ranges (1024-65535)
- Required fields
- Type checking
- Default values

---

### session.py

**Purpose:** Session state management and lifecycle.

**Key Classes:**
- `SessionState`: Enum of session states (LISTENING, SPEAKING, BARGED_IN)
- `Session`: Session container with metadata and state

**State Machine:**
```
INITIALIZING → LISTENING → SPEAKING → LISTENING
                    ↓          ↓
                 CLOSED ← BARGED_IN
```

**Session Fields:**
- `session_id`: Unique UUID
- `state`: Current session state
- `created_at`: Timestamp
- `worker_addr`: Assigned worker gRPC address
- `metadata`: Client metadata (language, model preferences, etc.)

**Example:**
```python
from src.orchestrator.session import Session, SessionState

session = Session(
    session_id="550e8400-e29b-41d4-a716-446655440000",
    worker_addr="grpc://localhost:7001"
)

# State transitions
session.transition_to(SessionState.LISTENING)
session.transition_to(SessionState.SPEAKING)
session.transition_to(SessionState.BARGED_IN)
```

---

### vad.py

**Purpose:** Voice Activity Detection for barge-in support.

**Key Classes:**
- `VADDetector`: WebRTC VAD wrapper with configurable aggressiveness

**Features:**
- Speech/silence detection
- Configurable aggressiveness (0-3)
- Frame-based processing (10/20/30ms)

**Example:**
```python
from src.orchestrator.vad import VADDetector

vad = VADDetector(
    aggressiveness=2,  # Moderate
    sample_rate=16000,
    frame_duration_ms=20
)

# Detect speech in audio frames
for frame in audio_frames:
    is_speech = vad.is_speech(frame)
    if is_speech:
        print("Speech detected - trigger barge-in")
```

**Usage Patterns:**
```python
# Synchronous (pre-loaded frames)
frames = [frame1, frame2, frame3]
for frame in frames:
    if vad.is_speech(frame):
        handle_speech(frame)

# Asynchronous (streaming)
async for frame in audio_stream:
    if vad.is_speech(frame):
        await handle_speech(frame)
```

---

### asr.py

**Purpose:** Automatic Speech Recognition (M10+ feature).

**Key Classes:**
- `ASREngine`: Whisper-based speech-to-text (placeholder in M2)

**Planned Features (M10+):**
- Real-time transcription
- Streaming mode
- Model variants (small, distil)
- Language detection

---

### routing.py

**Purpose:** Worker selection and load balancing.

**Key Functions:**
- `select_worker()`: Choose best worker for request
- `get_available_workers()`: Query Redis registry
- `filter_by_capabilities()`: Match request requirements

**Routing Strategies:**

**M2 (Current): Static Routing**
```python
from src.orchestrator.routing import Router

router = Router(config)
worker_addr = router.get_static_worker()
# Returns: grpc://localhost:7001
```

**M9+ (Future): Dynamic Routing**
```python
# Capability-based routing
worker = router.select_worker(
    language="en",
    model_id="cosyvoice2-en-base",
    prefer_resident=True
)

# Load balancing
worker = router.select_worker(
    strategy="queue_depth"  # or "round_robin", "latency"
)
```

**Selection Criteria (M9+):**
1. Filter by capabilities (language, streaming, etc.)
2. Prefer workers with model already loaded
3. Choose by load balancing strategy:
   - `queue_depth`: Fewest active sessions
   - `round_robin`: Even distribution
   - `latency`: Lowest p50 latency

---

### registry.py

**Purpose:** Redis-based worker discovery and health tracking.

**Key Classes:**
- `WorkerRegistry`: Interface to Redis worker registry

**Operations:**
- `get_workers()`: List all registered workers
- `get_worker(name)`: Get specific worker metadata
- `wait_for_workers()`: Block until workers available

**Worker Metadata:**
```json
{
  "name": "tts-worker-0",
  "addr": "grpc://tts-worker:7002",
  "capabilities": {
    "streaming": true,
    "zero_shot": true,
    "languages": ["en", "zh"],
    "max_concurrent_sessions": 3
  },
  "resident_models": ["cosyvoice2-en-base"],
  "metrics": {
    "rtf": 0.2,
    "queue_depth": 1,
    "p50_latency_ms": 250
  }
}
```

**Example:**
```python
from src.orchestrator.registry import WorkerRegistry

registry = WorkerRegistry(redis_url="redis://localhost:6379")

# Get all workers
workers = await registry.get_workers()

# Filter by capability
en_workers = [w for w in workers if "en" in w["capabilities"]["languages"]]

# Check worker health
worker = await registry.get_worker("tts-worker-0")
if worker:
    ttl = await registry.get_worker_ttl("tts-worker-0")
    print(f"Worker TTL: {ttl}s")
```

---

### transport/

**Purpose:** Pluggable transport layer supporting multiple protocols.

#### websocket_transport.py

**WebSocket transport implementation.**

**Key Classes:**
- `WebSocketTransport`: Handles WebSocket connections and message routing

**Features:**
- Connection management (max_connections limit)
- JSON message serialization/deserialization
- Frame queuing and backpressure
- Graceful shutdown

**Example:**
```python
from src.orchestrator.transport.websocket_transport import WebSocketTransport

transport = WebSocketTransport(
    host="0.0.0.0",
    port=8080,
    max_connections=100,
    frame_queue_size=50
)

await transport.start()
```

#### websocket_protocol.py

**Message type definitions.**

**Message Types:**
- `TextMessage`: Client → Server text input
- `ControlMessage`: Client → Server control commands
- `AudioMessage`: Server → Client audio frames
- `SessionStartMessage`: Server → Client session started
- `SessionEndMessage`: Server → Client session ended
- `ErrorMessage`: Server → Client error notification

**Example:**
```python
from src.orchestrator.transport.websocket_protocol import (
    TextMessage,
    AudioMessage
)

# Parse incoming message
text_msg = TextMessage.model_validate_json(raw_json)
print(text_msg.text, text_msg.is_final)

# Create outgoing message
audio_msg = AudioMessage(
    pcm=base64_encoded_pcm,
    sequence=1,
    sample_rate=48000,
    frame_ms=20
)
json_output = audio_msg.model_dump_json()
```

---

## Configuration

**File:** `configs/orchestrator.yaml`

**Key Settings:**

```yaml
transport:
  websocket:
    enabled: true
    host: "0.0.0.0"
    port: 8080
    max_connections: 100
    frame_queue_size: 50

redis:
  url: "redis://localhost:6379"
  db: 0
  worker_key_prefix: "worker:"
  worker_ttl_seconds: 30

routing:
  static_worker_addr: "grpc://localhost:7001"  # M2
  prefer_resident_models: true                 # M9+
  load_balance_strategy: "queue_depth"         # M9+

vad:
  enabled: true
  aggressiveness: 2
  sample_rate: 16000
  frame_duration_ms: 20

log_level: "INFO"
graceful_shutdown_timeout_s: 10
```

**See:** [Configuration Reference](../../docs/CONFIGURATION_REFERENCE.md)

---

## Usage

### Running the Orchestrator

**Docker Compose (Recommended):**
```bash
docker compose up orchestrator
```

**Local Development:**
```bash
# With just
just run-orch

# Or directly
python -m src.orchestrator.server

# With environment overrides
LOG_LEVEL=DEBUG PORT=8080 python -m src.orchestrator.server
```

**Verify Running:**
```bash
# Check logs
docker logs orchestrator

# Test WebSocket connection
just cli
# Or: python -m src.client.cli_client --host ws://localhost:8080
```

---

## Testing

### Unit Tests

```bash
# Run all orchestrator unit tests
pytest tests/unit/test_vad.py
pytest tests/unit/test_session.py
pytest tests/unit/test_routing.py

# With coverage
pytest --cov=src.orchestrator tests/unit/
```

### Integration Tests

```bash
# Requires Docker (Redis)
pytest tests/integration/test_redis_registry.py
pytest tests/integration/test_websocket_transport.py

# Full integration test
pytest tests/integration/
```

---

## Development Workflows

### Adding a New Transport

1. Create `transport/new_transport.py`
2. Implement `BaseTransport` interface
3. Add config to `config.py`
4. Register in `server.py`

**Example:**
```python
from src.orchestrator.transport.base import BaseTransport

class NewTransport(BaseTransport):
    async def start(self) -> None:
        # Initialize transport
        pass

    async def handle_client(self, client):
        # Handle client connection
        pass
```

### Adding ASR Support (M10+)

1. Implement `ASREngine` in `asr.py`
2. Load Whisper model at startup
3. Integrate with VAD (trigger ASR on speech)
4. Route transcription to LLM or TTS

---

## Performance Considerations

### Connection Limits

```yaml
# Max concurrent WebSocket connections
transport:
  websocket:
    max_connections: 100  # Tune based on resources
```

**Resource Impact:**
- Each connection: ~1-2 MB memory
- CPU: ~1% per active session (VAD + routing)

### Frame Queue Size

```yaml
# Audio frame buffer per connection
transport:
  websocket:
    frame_queue_size: 50  # 1 second @ 20ms frames
```

**Tradeoffs:**
- Smaller (10-30): Lower latency, risk of frame drops
- Larger (100-200): Higher latency, resilient to jitter

---

## Troubleshooting

### Worker Not Found

**Symptom:** "No available workers" error

**Solutions:**
1. Check Redis connectivity:
   ```bash
   docker exec -it redis redis-cli ping
   ```

2. Verify worker registration:
   ```bash
   docker exec -it redis redis-cli
   > KEYS worker:*
   > GET worker:tts-worker-0
   ```

3. Check worker logs:
   ```bash
   docker logs tts-worker
   ```

### WebSocket Connection Failed

**Symptom:** Client cannot connect

**Solutions:**
1. Verify port not in use:
   ```bash
   sudo lsof -i :8080
   ```

2. Check orchestrator logs:
   ```bash
   docker logs orchestrator
   ```

3. Test with curl:
   ```bash
   curl -i -N \
     -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     http://localhost:8080
   ```

---

## Further Reading

- [Architecture Diagrams](../../docs/architecture/ARCHITECTURE.md) - System architecture
- [WebSocket Protocol Specification](../../docs/WEBSOCKET_PROTOCOL.md) - Message protocol
- [Configuration Reference](../../docs/CONFIGURATION_REFERENCE.md) - Config options
- [Quick Start Guide](../../docs/QUICKSTART.md) - Getting started
