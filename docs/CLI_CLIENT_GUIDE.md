# CLI Client Usage Guide

## Overview

The CLI client provides a command-line interface for testing the WebSocket orchestrator. It's useful for:

- Quick testing of TTS synthesis
- Demonstrating barge-in functionality
- Measuring latency and performance
- Debugging WebSocket protocol issues
- Validating system setup

---

## Quick Start

### Basic Usage

```bash
# Connect to local orchestrator (default)
just cli

# Connect to specific host
just cli HOST="ws://localhost:8080"

# Or run directly with Python
python -m src.client.cli_client --host ws://localhost:8080
```

### Prerequisites

**Required:**
- Python 3.13+ with dependencies installed (`uv sync`)
- Running orchestrator (see [Quick Start Guide](QUICKSTART.md))

**Optional (for audio playback):**
- `sounddevice` library for real-time audio output
- Audio output device configured

**Without sounddevice:**
Audio will be saved to `.wav` files in the current directory instead of playing in real-time.

---

## Command-Line Options

### `--host` (HOST)

WebSocket server URL to connect to.

**Default:** `ws://localhost:8080`

**Examples:**
```bash
# Local development
just cli HOST="ws://localhost:8080"

# Remote server
just cli HOST="ws://192.168.1.100:8080"

# Custom port
just cli HOST="ws://localhost:9000"
```

### `--device` (DEVICE)

Audio output device name or index (optional).

**Default:** System default audio device

**Examples:**
```bash
# Use specific device by name
python -m src.client.cli_client --device "USB Audio Device"

# Use device by index
python -m src.client.cli_client --device 1

# List available devices (run in Python)
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### `-v, --verbose`

Enable verbose logging for debugging.

**Default:** INFO level logging

**Example:**
```bash
# Enable verbose output
python -m src.client.cli_client --host ws://localhost:8080 --verbose
```

**Verbose Output Includes:**
- Detailed frame-by-frame information
- WebSocket message debugging
- Audio playback events
- Connection state changes

---

## Interactive Mode

Once connected, the CLI client enters interactive mode.

### Session Start

```
Connected to ws://localhost:8080
12:34:56 [INFO] Session started: 550e8400-e29b-41d4-a716-446655440000

================================================================
WebSocket CLI Client
================================================================

Commands:
  /pause  - Pause audio playback
  /resume - Resume audio playback
  /stop   - Stop current synthesis
  /quit   - Exit client
  /help   - Show this help

Enter text to synthesize, or a command (starting with /):

You:
```

### Sending Text for Synthesis

Simply type text and press Enter:

```
You: Hello, this is a test of the text-to-speech system.
```

**What happens:**
1. Text is sent to the server as a `TextMessage`
2. Server processes through TTS worker
3. Audio frames stream back to client
4. Frames are played in real-time (or saved to files)
5. Progress dots (`.`) shown for each frame received
6. Session ends when synthesis completes

**Example Output:**
```
You: Hello world!
.......
✓ Session ended: completed
```

### Commands

All commands start with `/` to distinguish them from text input.

#### `/pause` - Pause Synthesis

Pauses the current TTS synthesis with < 50ms latency.

**Usage:**
```
You: /pause
12:34:56 [INFO] Sent control: PAUSE
```

**Effect:**
- Synthesis stops immediately (< 50ms)
- Audio frames stop streaming
- Worker maintains synthesis state
- Use `/resume` to continue

**Use Cases:**
- Testing barge-in functionality
- Demonstrating interruption handling
- Measuring pause latency

#### `/resume` - Resume Synthesis

Resumes paused synthesis.

**Usage:**
```
You: /resume
12:34:56 [INFO] Sent control: RESUME
```

**Effect:**
- Synthesis continues from pause point
- Audio frames resume streaming
- Playback continues seamlessly

#### `/stop` - Stop Synthesis

Stops current synthesis and clears queue.

**Usage:**
```
You: /stop
12:34:56 [INFO] Sent control: STOP
```

**Effect:**
- Synthesis stops immediately
- Queue cleared
- Ready for new text input
- More aggressive than `/pause`

**Difference from `/pause`:**
- `/pause`: Temporary interruption, can resume
- `/stop`: Permanent termination, cannot resume

#### `/quit` - Exit Client

Gracefully exits the CLI client.

**Usage:**
```
You: /quit

Goodbye!
```

**Cleanup:**
- Waits for audio playback to complete
- Closes WebSocket connection
- Exits program

**Keyboard Shortcuts:**
- `Ctrl+D` - Same as `/quit`
- `Ctrl+C` - Interrupt and exit immediately

#### `/help` - Show Help

Displays available commands.

**Usage:**
```
You: /help

Commands:
  /pause  - Pause audio playback
  /resume - Resume audio playback
  /stop   - Stop current synthesis
  /quit   - Exit client
  /help   - Show this help
```

---

## Common Workflows

### 1. Simple Synthesis Test

**Goal:** Verify basic text-to-speech functionality

**Steps:**
```bash
# 1. Start orchestrator and worker (see Quick Start)
just run-orch  # Terminal 1
just run-tts-cosy DEFAULT="cosyvoice2-en-base"  # Terminal 2

# 2. Start CLI client
just cli  # Terminal 3

# 3. Enter text
You: This is a test.

# 4. Wait for synthesis
.......
✓ Session ended: completed

# 5. Exit
You: /quit
```

**Expected Result:**
- Audio plays through speakers (or saved to files)
- No errors in logs
- Clean session termination

---

### 2. Barge-In Demonstration

**Goal:** Demonstrate real-time interruption with < 50ms latency

**Steps:**
```bash
# 1. Start client
just cli

# 2. Send long text
You: This is a very long sentence that will take many seconds to synthesize completely, allowing us to demonstrate the barge-in functionality by pausing and resuming the audio playback mid-sentence.

# 3. After hearing a few words, pause
You: /pause
12:34:56 [INFO] Sent control: PAUSE

# 4. Wait a few seconds (silence)

# 5. Resume
You: /resume
12:34:56 [INFO] Sent control: RESUME

# 6. Synthesis continues from pause point
```

**Measuring Latency:**

With verbose mode:
```bash
python -m src.client.cli_client --host ws://localhost:8080 --verbose
```

Look for timestamps:
```
12:34:56.123 [DEBUG] Received audio frame 15
12:34:56.140 [INFO] Sent control: PAUSE
12:34:56.142 [DEBUG] Last audio frame before pause: 15
# Latency = 142 - 140 = 2ms (< 50ms target ✓)
```

---

### 3. Latency Measurement

**Goal:** Measure First Audio Latency (FAL) and frame jitter

**Setup:**
```bash
# Run with verbose logging and timing
python -m src.client.cli_client --host ws://localhost:8080 --verbose 2>&1 | tee client.log
```

**Workflow:**
```
You: Test message for latency measurement.
```

**Analysis:**

Look in `client.log` for timestamps:

```bash
# First Audio Latency (FAL)
grep "Sent:" client.log          # Text sent timestamp
grep "Received audio frame 1" client.log  # First frame timestamp
# FAL = difference between these timestamps

# Frame jitter
grep "Received audio frame" client.log | awk '{print $1}' > frame_times.txt
# Analyze inter-frame intervals (should be ~20ms)
```

**Target Metrics:**
- **FAL:** < 300ms for GPU workers (p95)
- **Frame jitter:** < 10ms variance (p95)

---

### 4. Multi-Message Session

**Goal:** Test multiple synthesis requests in one session

**Workflow:**
```
You: First message.
.......
✓ Session ended: completed

You: Second message.
.......
✓ Session ended: completed

You: Third message.
.......
✓ Session ended: completed

You: /quit
```

**Note:** Each text input creates a new synthesis session within the same WebSocket connection.

---

### 5. Error Scenario Testing

**Goal:** Test error handling and recovery

**Scenarios:**

**A. Worker Unavailable:**
```bash
# 1. Start only orchestrator (no worker)
just run-orch

# 2. Start client and send text
just cli
You: Test message

# Expected error:
❌ Error: TTS worker is not available. Check worker connectivity.
```

**B. Invalid Commands:**
```
You: /invalid_command
Unknown command: invalid_command
Type /help for available commands
```

**C. Connection Loss:**
```bash
# 1. Start client
just cli

# 2. Stop orchestrator (in another terminal)
docker stop orchestrator

# Expected:
12:34:56 [INFO] Connection closed by server
```

---

## Audio Output Options

### 1. Real-Time Playback (Default)

**Requirements:**
- `sounddevice` installed: `pip install sounddevice`
- Working audio output device

**Behavior:**
- Audio plays through speakers in real-time
- 20ms frames queued and played sequentially
- Low-latency playback

**Verify Setup:**
```bash
# Test audio device
python -c "import sounddevice as sd; sd.play(sd.rec(48000), 48000); sd.wait()"
```

### 2. File Output (Fallback)

**When Used:**
- `sounddevice` not installed
- No audio device available
- Audio device error

**Behavior:**
- Each frame saved to `audio_output_XXXX.wav`
- Files numbered sequentially: `0001.wav`, `0002.wav`, etc.
- Located in current working directory

**Example:**
```
12:34:56 [WARNING] sounddevice not available, audio will be saved to file
12:34:56 [DEBUG] Saved audio to audio_output_0001.wav
12:34:56 [DEBUG] Saved audio to audio_output_0002.wav
...
```

**Concatenating Files:**
```bash
# Combine all frames into single file (requires ffmpeg)
ffmpeg -f concat -safe 0 -i <(ls -1 audio_output_*.wav | sed 's/^/file /') -c copy output.wav

# Or use SoX
sox audio_output_*.wav output.wav
```

### 3. No Output (Testing Only)

**Purpose:** Protocol testing without audio concerns

**Method:** Redirect audio player output

```python
# Modify cli_client.py temporarily
class AudioPlayer:
    def play_frame(self, pcm_data: bytes) -> None:
        # Just count frames, don't play
        self.frame_count += 1
        print(f"Frame {self.frame_count}: {len(pcm_data)} bytes")
```

---

## Troubleshooting

### Connection Issues

#### Problem: "Connection failed: [Errno 111] Connection refused"

**Cause:** Orchestrator not running or wrong port

**Solutions:**
1. Verify orchestrator running:
   ```bash
   docker ps | grep orchestrator
   # OR
   ps aux | grep orchestrator
   ```

2. Check correct port:
   ```bash
   # Verify port in config
   grep "port:" configs/orchestrator.yaml
   ```

3. Start orchestrator:
   ```bash
   just run-orch
   # OR
   docker compose up orchestrator
   ```

#### Problem: "Connection failed: Invalid URI"

**Cause:** Malformed WebSocket URL

**Solution:** Verify URL format:
```bash
# Correct formats
ws://localhost:8080
ws://192.168.1.100:8080

# Incorrect formats (missing ws://)
localhost:8080  # ❌
http://localhost:8080  # ❌
```

---

### Audio Issues

#### Problem: "sounddevice not available, audio will be saved to file"

**Cause:** `sounddevice` library not installed

**Solution:**
```bash
# Install with uv
uv pip install sounddevice

# Or with pip
pip install sounddevice
```

#### Problem: "Audio playback failed: [Errno -9996] Invalid output device"

**Cause:** Specified audio device doesn't exist

**Solution:**
```bash
# List available devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Use correct device name or index
python -m src.client.cli_client --device 1
```

#### Problem: No sound from speakers

**Causes & Solutions:**

1. **Volume muted:**
   - Check system volume
   - Check application volume

2. **Wrong device selected:**
   - List devices and select correct one
   - Try default device (omit `--device` flag)

3. **Audio format issue:**
   - Verify 48kHz support: `speaker-test -c 1 -r 48000`

---

### Message/Protocol Issues

#### Problem: No audio frames received

**Diagnostic Steps:**

1. **Check session start:**
   ```
   # Should see:
   🔗 Session started: <session-id>
   ```

2. **Verify text sent:**
   ```bash
   # Run with verbose
   python -m src.client.cli_client --verbose
   # Look for: [DEBUG] Sent: <your text>
   ```

3. **Check for errors:**
   ```
   # Look for:
   ❌ Error: <message>
   ```

4. **Check worker status:**
   ```bash
   docker logs tts-worker
   # Look for synthesis activity
   ```

#### Problem: "Server error [WORKER_UNAVAILABLE]"

**Cause:** TTS worker not connected

**Solutions:**
1. Start worker:
   ```bash
   just run-tts-cosy
   ```

2. Check Redis:
   ```bash
   docker ps | grep redis
   just redis  # if not running
   ```

3. Verify worker registration:
   ```bash
   docker exec -it redis redis-cli
   > KEYS worker:*
   # Should show registered workers
   ```

---

### Signal Handling

#### Ctrl+C Behavior

**Expected:**
- Graceful shutdown
- "Interrupted!" message
- Audio completion attempted
- Clean exit

**If Hanging:**
```bash
# Force kill (last resort)
Ctrl+\  # SIGQUIT

# Or from another terminal
pkill -9 -f cli_client
```

#### Ctrl+D Behavior

**Expected:**
- Same as `/quit` command
- Graceful shutdown
- Clean exit

**Usage:**
```
You: Hello world
.......
✓ Session ended: completed
You: <Ctrl+D>

Goodbye!
```

---

## Advanced Usage

### Custom Scripts

**Automated Testing:**

```bash
#!/bin/bash
# test_synthesis.sh - Automated TTS testing

echo "Starting automated TTS test..."

# Test cases
test_texts=(
    "Hello world"
    "This is a longer sentence to test synthesis quality"
    "Numbers: one two three four five"
)

for text in "${test_texts[@]}"; do
    echo "Testing: $text"
    echo "$text" | timeout 30 python -m src.client.cli_client --host ws://localhost:8080
    sleep 2
done

echo "Tests complete"
```

**Batch Processing:**

```bash
# Process file of text inputs
while IFS= read -r line; do
    echo "$line" | python -m src.client.cli_client --host ws://localhost:8080
done < input_texts.txt
```

### Programmatic Usage

**Import in Python:**

```python
import asyncio
from src.client.cli_client import CLIClient

async def automated_test():
    """Automated testing example."""
    client = CLIClient(
        server_url="ws://localhost:8080",
        verbose=True
    )

    async with client.connect() as websocket:
        # Send multiple test messages
        test_messages = [
            "First test message",
            "Second test message",
            "Third test message"
        ]

        for msg in test_messages:
            await client.send_text(websocket, msg)
            await asyncio.sleep(5)  # Wait for synthesis

        print("All tests complete")

asyncio.run(automated_test())
```

---

## Performance Tips

### 1. Reduce Latency

- Use wired network connection (avoid WiFi)
- Run client and server on same machine
- Use `--verbose` to identify bottlenecks

### 2. Optimize Audio Playback

```python
# Increase buffer size for stability (modify client code)
class AudioPlayer:
    def __init__(self, buffer_size: int = 2048):
        self.buffer_size = buffer_size
```

### 3. Monitor Resource Usage

```bash
# CPU usage
top -p $(pgrep -f cli_client)

# Network usage
iftop -i lo  # for localhost
nethogs      # per-process network monitoring
```

---

## Related Documentation

- [WebSocket Protocol Specification](WEBSOCKET_PROTOCOL.md) - Protocol details and message formats
- [Quick Start Guide](QUICKSTART.md) - System setup and installation
- [Configuration Reference](CONFIGURATION_REFERENCE.md) - Orchestrator configuration
- [Troubleshooting Guide](setup/DOCKER_SETUP.md) - Common issues and solutions

---

## Changelog

**v0.2.0 (M2):**
- Initial CLI client implementation
- WebSocket protocol support
- Real-time audio playback
- Control commands (pause/resume/stop)
- File output fallback

**Future Enhancements:**
- Model selection option (M4+)
- Save session to file option
- Record and replay sessions
- Batch processing mode
