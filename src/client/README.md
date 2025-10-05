# Client

## Overview

Client implementations for interacting with the orchestrator. Supports CLI and browser-based clients for testing and demonstration.

**Current Status (M2):**
- ✅ CLI WebSocket client (functional)
- 🚧 Browser client (planned for M3+)

---

## Directory Structure

```
client/
├── cli_client.py          # WebSocket CLI client (M2)
└── web/                   # Browser client (M3+)
    ├── index.html
    ├── app.js
    └── styles.css
```

---

## CLI Client

**File:** `cli_client.py`

### Purpose

Command-line WebSocket client for testing the orchestrator.

**Use Cases:**
- Quick synthesis testing
- Barge-in demonstration
- Latency measurement
- Protocol debugging
- Automation and scripting

### Features

- **Real-time audio playback** via sounddevice
- **File output fallback** if no audio device
- **Control commands** (/pause, /resume, /stop)
- **Interactive mode** with command history
- **Verbose logging** for debugging
- **Graceful shutdown** (Ctrl+C, Ctrl+D)

### Usage

**Basic:**
```bash
# Using just
just cli

# Or directly
python -m src.client.cli_client --host ws://localhost:8080
```

**Options:**
```bash
# Specify host
python -m src.client.cli_client --host ws://192.168.1.100:8080

# Select audio device
python -m src.client.cli_client --device "USB Audio"

# Enable verbose logging
python -m src.client.cli_client --verbose
```

**See:** [CLI Client Usage Guide](../../docs/CLI_CLIENT_GUIDE.md)

---

### Key Classes

#### AudioPlayer

**Purpose:** Manage audio playback with proper buffering.

**Features:**
- Cross-platform audio output via sounddevice
- Fallback to file output if device unavailable
- Frame counting and tracking

**Example:**
```python
from src.client.cli_client import AudioPlayer

player = AudioPlayer(sample_rate=48000, device=None)

# Play PCM frame (1920 bytes @ 48kHz)
player.play_frame(pcm_bytes)

# Wait for completion
player.wait_for_completion()
```

---

#### CLIClient

**Purpose:** WebSocket client for orchestrator communication.

**Key Methods:**
- `connect()`: Establish WebSocket connection
- `send_text()`: Send text for synthesis
- `send_control()`: Send control command (PAUSE/RESUME/STOP)
- `handle_message()`: Process incoming messages
- `run()`: Main event loop

**Example:**
```python
from src.client.cli_client import CLIClient

client = CLIClient(
    server_url="ws://localhost:8080",
    verbose=True
)

await client.run()
```

---

### Message Handling

**Client → Server:**
```python
# Text message
text_msg = TextMessage(text="Hello world", is_final=True)
await websocket.send(text_msg.model_dump_json())

# Control message
control_msg = ControlMessage(command="PAUSE")
await websocket.send(control_msg.model_dump_json())
```

**Server → Client:**
```python
async for message in websocket:
    data = json.loads(message)

    if data["type"] == "session_start":
        # Handle session start
        session_id = data["session_id"]

    elif data["type"] == "audio":
        # Decode and play audio
        pcm_data = base64.b64decode(data["pcm"])
        audio_player.play_frame(pcm_data)

    elif data["type"] == "session_end":
        # Handle session end
        reason = data["reason"]

    elif data["type"] == "error":
        # Handle error
        error_code = data["code"]
        error_message = data["message"]
```

---

### Interactive Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/pause` | Pause synthesis | `You: /pause` |
| `/resume` | Resume synthesis | `You: /resume` |
| `/stop` | Stop synthesis | `You: /stop` |
| `/quit` | Exit client | `You: /quit` |
| `/help` | Show help | `You: /help` |

**Text Input:**
```
You: This is text to synthesize.
```

**Keyboard Shortcuts:**
- `Ctrl+C`: Interrupt and exit
- `Ctrl+D`: Exit gracefully (same as /quit)

---

### Audio Output

**Real-Time Playback (Default):**
- Requires `sounddevice` installed
- Audio plays through speakers
- Low-latency output

**File Output (Fallback):**
- Used when sounddevice unavailable
- Saves frames to `audio_output_XXXX.wav`
- Numbered sequentially

**Configure Device:**
```python
# List available devices
import sounddevice as sd
print(sd.query_devices())

# Use specific device
client = CLIClient(device="USB Audio Device")
```

---

### Testing Workflows

#### Simple Synthesis Test
```bash
just cli
You: Hello world
.......
✓ Session ended: completed
You: /quit
```

#### Barge-In Test
```bash
just cli
You: A very long sentence to synthesize...
......
You: /pause
[Audio stops]
You: /resume
......
✓ Session ended: completed
```

#### Latency Measurement
```bash
python -m src.client.cli_client --verbose
You: Test message
# Observe timestamps in logs for FAL calculation
```

---

### Programmatic Usage

**Automated Testing:**
```python
import asyncio
from src.client.cli_client import CLIClient

async def automated_test():
    client = CLIClient("ws://localhost:8080", verbose=True)

    async with client.connect() as websocket:
        # Send test messages
        await client.send_text(websocket, "First test")
        await asyncio.sleep(5)

        await client.send_text(websocket, "Second test")
        await asyncio.sleep(5)

        print("Tests complete")

asyncio.run(automated_test())
```

**Batch Processing:**
```python
import asyncio
from src.client.cli_client import run_client

async def batch_process(texts: list[str]):
    for text in texts:
        # Process each text through CLI client
        # (Would need to modify cli_client for non-interactive mode)
        pass
```

---

## Browser Client (M3+)

**Status:** 🚧 Planned for M3

**Planned Features:**
- WebRTC audio streaming via LiveKit
- WebSocket fallback
- Browser-based microphone input
- Real-time audio playback
- Interactive UI
- Visual audio waveforms
- Session controls (pause/resume/stop)

**Directory:** `src/client/web/`

**Files:**
- `index.html`: Main UI
- `app.js`: WebSocket/WebRTC client logic
- `styles.css`: Styling
- `audio.js`: Web Audio API integration

**Technology Stack:**
- Vanilla JavaScript (no framework dependencies)
- Web Audio API for playback
- WebSocket API for JSON messages
- LiveKit SDK for WebRTC (M3+)

**Example Usage (Planned):**
```javascript
// Connect to orchestrator
const client = new TTSClient('ws://localhost:8080');
await client.connect();

// Send text for synthesis
client.sendText('Hello from the browser!');

// Handle audio frames
client.onAudio = (audioFrame) => {
    audioContext.decodeAudioData(audioFrame, (buffer) => {
        playAudio(buffer);
    });
};

// Control commands
client.pause();
client.resume();
client.stop();
```

---

## Testing

### Unit Tests

```bash
# Test client components
pytest tests/unit/test_cli_client.py

# Test audio player
pytest tests/unit/test_audio_player.py
```

### Integration Tests

```bash
# Requires running orchestrator
pytest tests/integration/test_cli_client_integration.py

# End-to-end test
just run-orch  # Terminal 1
just run-tts-mock  # Terminal 2
pytest tests/integration/test_e2e_synthesis.py  # Terminal 3
```

---

## Troubleshooting

### Connection Failed

**Symptom:** `Connection refused: [Errno 111]`

**Solutions:**
1. Verify orchestrator running:
   ```bash
   docker ps | grep orchestrator
   ```

2. Check correct port:
   ```bash
   just cli HOST="ws://localhost:8080"  # Verify port
   ```

3. Test with curl:
   ```bash
   curl -i http://localhost:8080
   ```

---

### No Audio Output

**Symptom:** No sound from speakers

**Solutions:**
1. Install sounddevice:
   ```bash
   pip install sounddevice
   ```

2. List audio devices:
   ```python
   import sounddevice as sd
   print(sd.query_devices())
   ```

3. Check system volume

4. Use file output mode:
   ```bash
   # Audio saved to .wav files
   python -m src.client.cli_client
   # Check for audio_output_*.wav files
   ```

---

### Audio Playback Errors

**Symptom:** `Audio playback failed: [Errno -9996]`

**Solutions:**
1. Verify device exists:
   ```bash
   python -c "import sounddevice as sd; print(sd.query_devices())"
   ```

2. Try default device:
   ```bash
   python -m src.client.cli_client  # Don't specify --device
   ```

3. Check sample rate support:
   ```bash
   speaker-test -c 1 -r 48000
   ```

---

## Development

### Adding Features

**New Message Type:**
1. Add to `websocket_protocol.py`
2. Implement handler in `handle_message()`
3. Update UI to display/send new message

**Example:**
```python
# Add to websocket_protocol.py
class MetricsMessage(BaseModel):
    type: Literal["metrics"] = "metrics"
    latency_ms: float
    rtf: float

# Handle in cli_client.py
elif msg_type == "metrics":
    metrics = MetricsMessage(**data)
    print(f"Latency: {metrics.latency_ms}ms, RTF: {metrics.rtf}")
```

---

### Custom Client

**Building Your Own Client:**

1. **Connect to WebSocket:**
   ```python
   import websockets
   async with websockets.connect("ws://localhost:8080") as ws:
       # ...
   ```

2. **Receive session start:**
   ```python
   msg = await ws.recv()
   session_data = json.loads(msg)
   session_id = session_data["session_id"]
   ```

3. **Send text:**
   ```python
   text_msg = {
       "type": "text",
       "text": "Hello world",
       "is_final": True
   }
   await ws.send(json.dumps(text_msg))
   ```

4. **Receive audio:**
   ```python
   async for message in ws:
       data = json.loads(message)
       if data["type"] == "audio":
           pcm_base64 = data["pcm"]
           pcm_bytes = base64.b64decode(pcm_base64)
           # Play or save pcm_bytes
   ```

**See:** [WebSocket Protocol Specification](../../docs/WEBSOCKET_PROTOCOL.md)

---

## Performance Tips

### Reduce Latency

- Use wired network (avoid WiFi)
- Run client on same host as orchestrator
- Use verbose mode to identify bottlenecks

### Optimize Playback

```python
# Increase buffer size for stability
player = AudioPlayer(buffer_size=2048)
```

### Monitor Performance

```bash
# CPU usage
top -p $(pgrep -f cli_client)

# Network usage
iftop -i lo  # for localhost
```

---

## Further Reading

- [WebSocket Protocol Specification](../../docs/WEBSOCKET_PROTOCOL.md) - Protocol details
- [CLI Client Usage Guide](../../docs/CLI_CLIENT_GUIDE.md) - Complete usage guide
- [Quick Start Guide](../../docs/QUICKSTART.md) - Getting started
- [Architecture](../../docs/architecture/ARCHITECTURE.md) - System architecture
