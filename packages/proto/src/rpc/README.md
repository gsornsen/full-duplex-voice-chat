# RPC (Remote Procedure Call)

## Overview

Protocol Buffer (protobuf) definitions and auto-generated gRPC stubs for the TTS service API. Defines the unified streaming ABI between orchestrator and TTS workers.

**Key Files:**
- `tts.proto` - Service and message definitions
- `generated/` - Auto-generated Python stubs (do not edit)

---

## Directory Structure

```
rpc/
├── tts.proto              # Protobuf service definition
└── generated/             # Auto-generated stubs
    ├── __init__.py
    ├── tts_pb2.py         # Message definitions
    └── tts_pb2_grpc.py    # Service stubs
```

---

## tts.proto

**Purpose:** Define the TTS service API contract.

**Language:** Protocol Buffers v3

**Service:** `TTSService`

---

## Service Definition

### TTSService

```protobuf
service TTSService {
  // Session lifecycle
  rpc StartSession(StartSessionRequest) returns (StartSessionResponse);
  rpc EndSession(EndSessionRequest) returns (EndSessionResponse);

  // Main streaming synthesis
  rpc Synthesize(stream TextChunk) returns (stream AudioFrame);

  // Runtime control
  rpc Control(ControlRequest) returns (ControlResponse);

  // Model lifecycle (M4+)
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
  rpc LoadModel(LoadModelRequest) returns (LoadModelResponse);
  rpc UnloadModel(UnloadModelRequest) returns (UnloadModelResponse);
  rpc GetCapabilities(GetCapabilitiesRequest) returns (GetCapabilitiesResponse);
}
```

---

## Message Types

### Session Management

#### StartSessionRequest
```protobuf
message StartSessionRequest {
  string session_id = 1;      // Unique session identifier
  string model_id = 2;        // Model to use for synthesis
  map<string, string> options = 3;  // Optional parameters
}
```

**Example:**
```python
request = StartSessionRequest(
    session_id="550e8400-e29b-41d4-a716-446655440000",
    model_id="cosyvoice2-en-base",
    options={"language": "en", "speed": "1.0"}
)
```

---

#### StartSessionResponse
```protobuf
message StartSessionResponse {
  bool success = 1;          // True if session started successfully
  string message = 2;        // Error message if success=false
}
```

---

#### EndSessionRequest
```protobuf
message EndSessionRequest {
  string session_id = 1;
}
```

---

#### EndSessionResponse
```protobuf
message EndSessionResponse {
  bool success = 1;
}
```

---

### Streaming Synthesis

#### TextChunk
```protobuf
message TextChunk {
  string session_id = 1;       // Session this chunk belongs to
  string text = 2;             // Text to synthesize
  bool is_final = 3;           // Whether this is the last chunk
  int64 sequence_number = 4;   // Chunk sequence number
}
```

**Example:**
```python
chunk = TextChunk(
    session_id="550e8400...",
    text="Hello world",
    is_final=True,
    sequence_number=1
)
```

---

#### AudioFrame
```protobuf
message AudioFrame {
  string session_id = 1;         // Session this frame belongs to
  bytes audio_data = 2;          // PCM audio data (1920 bytes @ 48kHz)
  int32 sample_rate = 3;         // Sample rate (48000)
  int32 frame_duration_ms = 4;   // Frame duration (20)
  int64 sequence_number = 5;     // Frame sequence number
  bool is_final = 6;             // Last frame in synthesis
}
```

**Frame Specifications:**
- `audio_data`: 16-bit signed PCM, little-endian, mono
- `sample_rate`: 48000 Hz (fixed)
- `frame_duration_ms`: 20 ms (fixed)
- Frame size: 960 samples = 1920 bytes

**Example:**
```python
frame = AudioFrame(
    session_id="550e8400...",
    audio_data=pcm_bytes,  # 1920 bytes
    sample_rate=48000,
    frame_duration_ms=20,
    sequence_number=1,
    is_final=False
)
```

---

### Control Commands

#### ControlCommand (Enum)
```protobuf
enum ControlCommand {
  PAUSE = 0;    // Pause synthesis
  RESUME = 1;   // Resume synthesis
  STOP = 2;     // Stop synthesis
  RELOAD = 3;   // Reload model
}
```

---

#### ControlRequest
```protobuf
message ControlRequest {
  string session_id = 1;
  ControlCommand command = 2;
}
```

**Example:**
```python
request = ControlRequest(
    session_id="550e8400...",
    command=ControlCommand.PAUSE
)
```

---

#### ControlResponse
```protobuf
message ControlResponse {
  bool success = 1;           // Command executed successfully
  string message = 2;         // Error/status message
  int64 timestamp_ms = 3;     // Timestamp when command executed
}
```

---

### Model Management (M4+)

#### ListModelsRequest
```protobuf
message ListModelsRequest {}
```

---

#### ModelInfo
```protobuf
message ModelInfo {
  string model_id = 1;                    // Model identifier
  string family = 2;                      // Model family (e.g., "cosyvoice2")
  bool is_loaded = 3;                     // Currently loaded in VRAM
  repeated string languages = 4;          // Supported languages
  map<string, string> metadata = 5;       // Additional metadata
}
```

---

#### ListModelsResponse
```protobuf
message ListModelsResponse {
  repeated ModelInfo models = 1;
}
```

**Example:**
```python
response = ListModelsResponse(
    models=[
        ModelInfo(
            model_id="cosyvoice2-en-base",
            family="cosyvoice2",
            is_loaded=True,
            languages=["en"],
            metadata={"quality": "high", "vram_mb": "4096"}
        ),
        ModelInfo(
            model_id="xtts-v2-en-demo",
            family="xtts-v2",
            is_loaded=False,
            languages=["en"],
            metadata={"quality": "high", "vram_mb": "6144"}
        )
    ]
)
```

---

#### LoadModelRequest
```protobuf
message LoadModelRequest {
  string model_id = 1;          // Model to load
  bool preload_only = 2;        // Load but don't start using it
}
```

---

#### LoadModelResponse
```protobuf
message LoadModelResponse {
  bool success = 1;
  string message = 2;
  int64 load_duration_ms = 3;   // Time taken to load model
}
```

---

#### UnloadModelRequest
```protobuf
message UnloadModelRequest {
  string model_id = 1;
}
```

---

#### UnloadModelResponse
```protobuf
message UnloadModelResponse {
  bool success = 1;
  string message = 2;
}
```

---

### Capabilities

#### GetCapabilitiesRequest
```protobuf
message GetCapabilitiesRequest {}
```

---

#### Capabilities
```protobuf
message Capabilities {
  bool streaming = 1;                    // Supports streaming synthesis
  bool zero_shot = 2;                    // Supports zero-shot (no reference)
  bool lora = 3;                         // Supports LoRA models
  bool cpu_ok = 4;                       // Can run on CPU
  repeated string languages = 5;          // Supported languages
  bool emotive_zero_prompt = 6;          // Emotional control without reference
  int32 max_concurrent_sessions = 7;     // Max concurrent sessions
}
```

---

#### GetCapabilitiesResponse
```protobuf
message GetCapabilitiesResponse {
  Capabilities capabilities = 1;
  repeated string resident_models = 2;   // Currently loaded models
  map<string, double> metrics = 3;       // Performance metrics (RTF, etc.)
}
```

**Example:**
```python
response = GetCapabilitiesResponse(
    capabilities=Capabilities(
        streaming=True,
        zero_shot=True,
        lora=False,
        cpu_ok=False,
        languages=["en", "zh"],
        emotive_zero_prompt=True,
        max_concurrent_sessions=3
    ),
    resident_models=["cosyvoice2-en-base", "cosyvoice2-zh-base"],
    metrics={"rtf": 0.2, "queue_depth": 1}
)
```

---

## Code Generation

### Generate gRPC Stubs

**Automatic (via just):**
```bash
just gen-proto
```

**Manual:**
```bash
python -m grpc_tools.protoc \
  --proto_path=src/rpc \
  --python_out=src/rpc/generated \
  --grpc_python_out=src/rpc/generated \
  --pyi_out=src/rpc/generated \
  src/rpc/tts.proto
```

**Output:**
- `generated/tts_pb2.py` - Message classes
- `generated/tts_pb2_grpc.py` - Service stubs
- `generated/tts_pb2.pyi` - Type stubs (for mypy)

**Important:** Do not edit generated files. Regenerate after modifying `tts.proto`.

---

## Usage Examples

### Client (Orchestrator)

#### StartSession
```python
import grpc
from src.rpc.generated import tts_pb2, tts_pb2_grpc

# Create channel
channel = grpc.aio.insecure_channel("localhost:7001")
stub = tts_pb2_grpc.TTSServiceStub(channel)

# Start session
request = tts_pb2.StartSessionRequest(
    session_id="550e8400-e29b-41d4-a716-446655440000",
    model_id="cosyvoice2-en-base"
)
response = await stub.StartSession(request)

if response.success:
    print("Session started")
else:
    print(f"Error: {response.message}")
```

---

#### Streaming Synthesis
```python
async def synthesize(stub, session_id: str, text: str):
    # Create text chunk stream
    async def text_chunks():
        yield tts_pb2.TextChunk(
            session_id=session_id,
            text=text,
            is_final=True,
            sequence_number=1
        )

    # Stream synthesis
    async for audio_frame in stub.Synthesize(text_chunks()):
        print(f"Received frame {audio_frame.sequence_number}: "
              f"{len(audio_frame.audio_data)} bytes")

        # Send to client
        send_audio_to_client(audio_frame.audio_data)
```

---

#### Control Commands
```python
# Pause synthesis
request = tts_pb2.ControlRequest(
    session_id=session_id,
    command=tts_pb2.ControlCommand.PAUSE
)
response = await stub.Control(request)
print(f"Paused at: {response.timestamp_ms}ms")

# Resume synthesis
request = tts_pb2.ControlRequest(
    session_id=session_id,
    command=tts_pb2.ControlCommand.RESUME
)
response = await stub.Control(request)
```

---

#### Get Capabilities
```python
request = tts_pb2.GetCapabilitiesRequest()
response = await stub.GetCapabilities(request)

print(f"Streaming: {response.capabilities.streaming}")
print(f"Languages: {response.capabilities.languages}")
print(f"Resident models: {response.resident_models}")
print(f"RTF: {response.metrics['rtf']}")
```

---

### Server (TTS Worker)

#### Implement TTSService
```python
from src.rpc.generated import tts_pb2, tts_pb2_grpc

class TTSServicer(tts_pb2_grpc.TTSServiceServicer):
    async def StartSession(self, request, context):
        session_id = request.session_id
        model_id = request.model_id

        # Initialize session
        success = await self.session_manager.start(session_id, model_id)

        return tts_pb2.StartSessionResponse(
            success=success,
            message="" if success else "Failed to start session"
        )

    async def Synthesize(self, request_iterator, context):
        async for text_chunk in request_iterator:
            # Synthesize audio for text chunk
            async for audio_data in self.adapter.synthesize(text_chunk.text):
                yield tts_pb2.AudioFrame(
                    session_id=text_chunk.session_id,
                    audio_data=audio_data,
                    sample_rate=48000,
                    frame_duration_ms=20,
                    sequence_number=self.frame_counter,
                    is_final=False
                )
                self.frame_counter += 1

            # Final frame
            yield tts_pb2.AudioFrame(
                session_id=text_chunk.session_id,
                audio_data=b"",
                sample_rate=48000,
                frame_duration_ms=20,
                sequence_number=self.frame_counter,
                is_final=True
            )

    async def Control(self, request, context):
        session_id = request.session_id
        command = request.command

        if command == tts_pb2.ControlCommand.PAUSE:
            await self.adapter.pause()
        elif command == tts_pb2.ControlCommand.RESUME:
            await self.adapter.resume()
        elif command == tts_pb2.ControlCommand.STOP:
            await self.adapter.stop()

        return tts_pb2.ControlResponse(
            success=True,
            message="",
            timestamp_ms=int(time.time() * 1000)
        )
```

---

#### Start gRPC Server
```python
import grpc
from src.rpc.generated import tts_pb2_grpc

async def serve():
    server = grpc.aio.server()
    tts_pb2_grpc.add_TTSServiceServicer_to_server(
        TTSServicer(), server
    )
    server.add_insecure_port("[::]:7001")
    await server.start()
    await server.wait_for_termination()

asyncio.run(serve())
```

---

## Protocol Design Principles

### Streaming First

All primary operations use bidirectional streaming for low latency:
- Text → Audio: Incremental synthesis
- Control messages: Real-time interruption

### Explicit Session Management

- `StartSession`: Initialize resources
- `EndSession`: Cleanup resources
- Prevents resource leaks

### Idempotent Operations

- `LoadModel`: Safe to call multiple times
- `UnloadModel`: No-op if already unloaded
- Resilient to retries

### Binary Efficiency

- Protobuf binary encoding (smaller than JSON)
- PCM audio as bytes (no base64 overhead)
- gRPC HTTP/2 multiplexing

---

## Testing

### Test Generated Stubs

```bash
# Verify stubs generated
ls src/rpc/generated/
# Expected: __init__.py, tts_pb2.py, tts_pb2_grpc.py, tts_pb2.pyi

# Test import
python -c "from src.rpc.generated import tts_pb2, tts_pb2_grpc; print('✓ Stubs OK')"
```

### Test gRPC Service

```bash
# Integration tests
pytest tests/integration/test_grpc_synthesis.py

# Using grpcurl (manual testing)
grpcurl -plaintext localhost:7001 list
# Expected: tts.TTSService, grpc.health.v1.Health

grpcurl -plaintext localhost:7001 tts.TTSService/GetCapabilities
```

---

## Troubleshooting

### Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'src.rpc.generated'`

**Solution:**
```bash
# Regenerate stubs
just gen-proto

# Verify files exist
ls src/rpc/generated/

# Check __init__.py exists
touch src/rpc/generated/__init__.py
```

---

### Type Checking Errors

**Symptom:** `mypy` errors about missing types

**Solution:**
```bash
# Regenerate with .pyi stubs
just gen-proto

# Verify .pyi files
ls src/rpc/generated/*.pyi
```

---

### gRPC Connection Failed

**Symptom:** `grpc._channel._InactiveRpcError: StatusCode.UNAVAILABLE`

**Solution:**
1. Verify worker running:
   ```bash
   docker ps | grep tts-worker
   ```

2. Check port:
   ```bash
   sudo lsof -i :7001
   ```

3. Test connection:
   ```bash
   grpcurl -plaintext localhost:7001 list
   ```

---

## Advanced Topics

### Adding New RPC Methods

1. Edit `tts.proto`
2. Add new message types
3. Add method to `TTSService`
4. Regenerate stubs: `just gen-proto`
5. Implement in worker
6. Update orchestrator client

**Example:**
```protobuf
// Add to tts.proto
message GetMetricsRequest {}
message GetMetricsResponse {
  double rtf = 1;
  int32 queue_depth = 2;
}

service TTSService {
  // ... existing methods
  rpc GetMetrics(GetMetricsRequest) returns (GetMetricsResponse);
}
```

---

### Error Handling

**gRPC Status Codes:**
```python
import grpc

try:
    response = await stub.StartSession(request)
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.UNAVAILABLE:
        print("Worker unavailable")
    elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
        print("Invalid request")
    elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        print("Request timeout")
```

---

## Further Reading

- [gRPC Python Documentation](https://grpc.io/docs/languages/python/)
- [Protocol Buffers Guide](https://protobuf.dev/programming-guides/proto3/)
- [Architecture Diagrams](../../docs/architecture/ARCHITECTURE.md) - gRPC flows
- [TTS Worker README](../tts/README.md) - Server implementation
- [Orchestrator README](../orchestrator/README.md) - Client implementation
