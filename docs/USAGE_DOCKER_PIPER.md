# Docker Compose Quick Start with Piper TTS

**Last Updated**: 2025-10-10
**Target Audience**: New users, developers, testers
**Prerequisites**: Docker, Docker Compose, ~2GB disk space

---

## Overview

This guide will help you spin up the complete full-duplex voice chat system in **under 5 minutes** using Docker Compose and the Piper TTS model (CPU-based neural text-to-speech). This is the fastest way to experience realtime speech synthesis with barge-in support.

### What You'll Get

- **Complete working system**: All services orchestrated with Docker Compose
- **Piper TTS adapter**: First real TTS model (CPU-based, no GPU required)
- **Dual client access**: WebSocket CLI client + Browser WebRTC client
- **Barge-in support**: Real-time voice activity detection with <50ms pause latency
- **Production-ready stack**: Redis, LiveKit, Caddy, Orchestrator, TTS Worker

### Architecture Overview

```
┌─────────────────┐
│  Browser Client │
│   (WebRTC)      │
└────────┬────────┘
         │ HTTPS
         ▼
┌─────────────────┐
│  Caddy Proxy    │
│  (Reverse Proxy)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐      ┌────────────────┐
│  LiveKit Server │◄────►│ Orchestrator │◄────►│  TTS Worker    │
│    (WebRTC)     │      │  (VAD + SM)  │ gRPC │  (Piper CPU)   │
└─────────────────┘      └──────┬───────┘      └────────────────┘
                                 │
                                 ▼
                         ┌──────────────┐
                         │    Redis     │
                         │  (Discovery) │
                         └──────────────┘

Alternative CLI Path:
┌─────────────────┐      ┌──────────────┐      ┌────────────────┐
│   CLI Client    │◄────►│ Orchestrator │◄────►│  TTS Worker    │
│  (WebSocket)    │  WS  │              │ gRPC │  (Piper CPU)   │
└─────────────────┘      └──────────────┘      └────────────────┘
```

**Legend**:
- **Caddy**: HTTPS reverse proxy for WebRTC
- **LiveKit**: WebRTC server for browser clients
- **Orchestrator**: Session management, VAD (Voice Activity Detection), routing
- **TTS Worker**: Piper adapter for neural TTS synthesis
- **Redis**: Service discovery and worker registration

---

## Quick Start (5 Minutes)

### Prerequisites Check

Before starting, verify you have:

1. **Docker Engine 28.x** with NVIDIA container runtime
   ```bash
   docker --version
   # Expected: Docker version 28.x.x or later
   ```

2. **Docker Compose**
   ```bash
   docker compose version
   # Expected: Docker Compose version v2.x.x or later
   ```

3. **Available ports**: 80, 8080, 8081, 7001, 7880, 7881, 7882, 8443, 8444, 9090
   ```bash
   # Check if any ports are in use
   netstat -tuln | grep -E ':(80|8080|8081|7001|7880|7881|7882|8443|8444|9090) '
   ```

4. **Disk space**: At least 2GB free
   ```bash
   df -h .
   ```

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd full-duplex-voice-chat
```

### Step 2: Verify Voicepack Exists

The Piper TTS model should already be included in the repository:

```bash
ls -la voicepacks/piper/en-us-lessac-medium/
```

**Expected output**:
```
total 28M
drwxr-xr-x 3 user user 4.0K Oct 10 17:18 .
drwxr-xr-x 3 user user 4.0K Oct 10 17:18 ..
-rw-r--r-- 1 user user  28M Oct 10 17:18 en_US-lessac-medium.onnx
-rw-r--r-- 1 user user 1.2K Oct 10 17:18 en_US-lessac-medium.onnx.json
-rw-r--r-- 1 user user  300 Oct 10 17:18 metadata.yaml
```

> **Note**: If voicepack is missing, see [Adding Custom Piper Voices](#adding-custom-piper-voices) section.

### Step 3: Start the Full Stack

```bash
docker compose up --build
```

**What happens** (takes ~30-60 seconds):
1. **Redis** starts (service discovery)
2. **LiveKit** starts (WebRTC server)
3. **Caddy** starts (HTTPS reverse proxy)
4. **Orchestrator** builds and starts (VAD + session management)
5. **TTS Worker** builds and starts, loads Piper model (~10s warmup)

**Expected final output**:
```
[+] Running 5/5
 ✔ Container redis-tts       Healthy   10s
 ✔ Container livekit-server  Healthy   15s
 ✔ Container caddy-proxy     Healthy   5s
 ✔ Container orchestrator    Started   20s
 ✔ Container tts-worker-0    Healthy   30s
```

### Step 4: Verify Services Are Running

Open a new terminal and check service health:

```bash
# Check all services are up
docker compose ps

# Expected: All services should show "Up" or "healthy"
```

**Health check endpoints**:
```bash
# Orchestrator health
curl http://localhost:8081/health
# Expected: {"status": "healthy"}

# TTS Worker (TCP check, no HTTP endpoint)
timeout 1 bash -c '</dev/tcp/localhost/7001' && echo "TTS Worker: OK"

# LiveKit health
curl http://localhost:7880/
# Expected: 200 OK
```

### Step 5: Test with CLI Client

Test the system using the WebSocket CLI client:

```bash
# From project root
uv sync  # Install dependencies (first time only)
uv run python -m src.client.cli_client --host ws://localhost:8080
```

**Usage**:
1. Type a message (e.g., "Hello world")
2. Press Enter
3. Hear synthesized speech from Piper TTS
4. Type another message to test multiple utterances
5. Press Ctrl+C to exit

**Expected behavior**:
- First synthesis: ~300ms latency (warmup)
- Subsequent synthesis: <200ms latency
- Audio plays through your default speakers
- Clean session termination on exit

---

## Detailed Service Breakdown

### Service 1: Redis (Service Discovery)

**Purpose**: Worker registration and service discovery
**Port**: Internal only (6379 inside Docker network)
**Health check**: `redis-cli ping`

**Why no external port?**
To prevent conflicts with host Redis/Valkey instances. Services use internal DNS: `redis:6379`

**Debug access**:
```bash
docker compose exec redis redis-cli
> ping
PONG
> keys *
(shows worker registrations)
```

### Service 2: LiveKit (WebRTC Server)

**Purpose**: WebRTC transport for browser clients
**Ports**:
- 7880: WebSocket/WebRTC signaling
- 7881: RTC TCP port
- 7882/udp: TURN/UDP
- 50000-50099/udp: RTC port range

**Configuration**: `configs/livekit.yaml`
```yaml
port: 7880
keys:
  devkey: devsecret1234567890abcdefghijklmn  # Must match orchestrator env
room:
  auto_create: true
  empty_timeout: 300  # 5 minutes
  max_participants: 10
```

**Logs**:
```bash
docker compose logs -f livekit
```

### Service 3: Caddy (HTTPS Reverse Proxy)

**Purpose**: TLS termination for WebRTC (required for browser security)
**Ports**:
- 80: HTTP (redirects to HTTPS)
- 8443: HTTPS for web client
- 8444: HTTPS for LiveKit WebSocket

**Configuration**: `Caddyfile`
**Certificates**: Self-signed (voicechat.local+3.pem)

**Access logs**:
```bash
docker compose logs -f caddy
```

### Service 4: Orchestrator

**Purpose**: Session management, VAD (Voice Activity Detection), routing
**Ports**:
- 8080: WebSocket server (CLI clients)
- 8081: Health check HTTP endpoint

**Configuration**: `configs/orchestrator.docker.yaml`
```yaml
transport:
  websocket:
    host: "0.0.0.0"
    port: 8080
  livekit:
    url: "${LIVEKIT_URL}"
    api_key: "${LIVEKIT_API_KEY}"
    api_secret: "${LIVEKIT_API_SECRET}"

vad:
  enabled: true
  aggressiveness: 2  # 0-3, higher = more conservative
  sample_rate: 16000
  min_speech_duration_ms: 100
  min_silence_duration_ms: 300

workers:
  tts_worker_url: "grpc://tts0:7001"
```

**Key features**:
- **VAD barge-in**: Detects speech, sends PAUSE to TTS worker (<50ms)
- **State machine**: LISTENING → SPEAKING → BARGED_IN → LISTENING
- **Dual transport**: LiveKit WebRTC (primary) + WebSocket (fallback)

**Logs**:
```bash
docker compose logs -f orchestrator | grep -E "(session|vad|barge)"
```

### Service 5: TTS Worker (Piper Adapter)

**Purpose**: Neural TTS synthesis using Piper (CPU-based ONNX)
**Ports**:
- 7001: gRPC server
- 9090: Prometheus metrics (future)

**Configuration**: `configs/worker.yaml`
```yaml
worker:
  name: "tts-worker-0"
  grpc_port: 7001
  capabilities:
    streaming: true
    languages: ["en"]
    cpu_ok: true  # Piper runs on CPU

model_manager:
  default_model_id: "piper-en-us-lessac-medium"
  preload_model_ids: []  # Optional additional models
  ttl_ms: 600000  # 10 min idle → unload
  warmup_enabled: true
  warmup_text: "This is a warmup test."

audio:
  output_sample_rate: 48000  # Target output rate
  frame_duration_ms: 20      # 960 samples per frame
```

**Model warmup** (first startup):
```bash
docker compose logs tts0 | grep warmup
# Expected: "Warmup synthesis complete" (~300ms)
```

**Frame format**:
- **Sample rate**: 48kHz (resampled from Piper's native 22.05kHz)
- **Frame size**: 20ms = 960 samples = 1920 bytes (int16 PCM)
- **Channels**: Mono

**Logs**:
```bash
docker compose logs -f tts0 | grep -E "(Piper|synthesis|frame)"
```

---

## Using the Piper TTS Model

### What is Piper?

**Piper** is a fast, high-quality, CPU-only neural TTS system:
- **Technology**: ONNX Runtime with optimized inference
- **Sample rate**: 22.05kHz native (resampled to 48kHz for output)
- **Latency**: ~100-300ms first audio latency (CPU)
- **Quality**: Natural-sounding neural voices
- **Footprint**: ~28MB per voice model
- **Repository**: https://github.com/rhasspy/piper

### Available Voice: en-us-lessac-medium

**Voicepack location**: `voicepacks/piper/en-us-lessac-medium/`

**Metadata** (`metadata.yaml`):
```yaml
model_id: "piper-en-us-lessac-medium"
family: "piper"
language: "en-US"
voice_name: "Lessac"
variant: "medium"
tags:
  - cpu_ok
  - streaming
  - neural
sample_rate: 22050  # Will be resampled to 48000
description: "High-quality English (US) voice from Piper TTS"
```

**Voice characteristics**:
- **Gender**: Female
- **Accent**: General American English
- **Style**: Clear, professional, neutral
- **Use cases**: Assistants, narration, announcements

### Performance Expectations

**Warmup** (first synthesis):
- **Duration**: ~300ms (CPU model load + first inference)
- **When**: TTS worker startup
- **Log message**: `"Warmup synthesis complete"`

**Synthesis** (subsequent):
- **Latency**: 100-200ms first audio latency
- **RTF (Real-time factor)**: ~0.3-0.5 (faster than realtime)
- **Streaming**: Frames emitted every 20ms
- **Control latency**: <50ms for PAUSE/RESUME/STOP

**Resource usage**:
- **CPU**: 1-2 cores during synthesis
- **Memory**: ~200MB (model + runtime)
- **Disk**: 28MB per voice model

---

## Testing the System

### WebSocket CLI Client Testing

The CLI client provides the simplest way to test TTS synthesis:

**Basic usage**:
```bash
cd /path/to/full-duplex-voice-chat
uv run python -m src.client.cli_client --host ws://localhost:8080
```

**Test scenarios**:

1. **Single utterance**:
   ```
   > Hello, this is a test of the Piper text-to-speech system.
   [Hear synthesized speech]
   ```

2. **Multiple messages**:
   ```
   > First message.
   [Audio plays]
   > Second message.
   [Audio plays]
   > Third message with more text to test longer synthesis.
   [Audio plays]
   ```

3. **Special characters**:
   ```
   > Testing numbers: 1, 2, 3, 4, 5.
   > Testing punctuation! Question? Exclamation!
   ```

4. **Timing validation**:
   - First message: Expect ~300ms latency (includes warmup)
   - Subsequent messages: Expect <200ms latency
   - Clean audio playback with no clicks/pops
   - Proper session termination on Ctrl+C

### Browser Client Testing

The web client provides a full-featured WebRTC interface:

**Setup** (first time):
```bash
cd src/client/web
pnpm install  # Install dependencies
cp .env.example .env.local  # Copy environment config
```

**Start the web client**:
```bash
cd src/client/web
pnpm dev
```

**Access**: Open browser to http://localhost:3000

**Usage**:
1. Click "Start voice chat" button
2. Grant microphone permissions (for VAD)
3. Type text message in chat input
4. Press Enter to synthesize
5. Hear audio through speakers/headphones

**WebRTC features**:
- **Full-duplex audio**: Bidirectional audio streaming
- **VAD visualization**: See speech detection in real-time
- **Chat history**: See all messages and responses
- **Connection status**: Monitor WebRTC connection health

**Troubleshooting browser client**:
- See [Web Client Setup Guide](../src/client/web/SETUP.md)
- Check browser console for errors (F12)
- Verify microphone permissions granted
- Ensure LiveKit server is accessible

### Testing Barge-in Functionality

The system supports real-time barge-in (interrupting TTS during speech):

**Note**: Barge-in requires browser client (WebRTC) for microphone access. CLI client supports text-only interaction.

**Test with browser client**:
1. Start web client: http://localhost:3000
2. Type a long message: "This is a very long message that will take several seconds to synthesize so I can test the barge-in functionality."
3. Press Enter to start synthesis
4. **Speak into microphone** while TTS is playing
5. VAD detects speech → sends PAUSE to worker (<50ms)
6. TTS stops immediately
7. Stop speaking → VAD sends RESUME
8. TTS resumes from pause point

**Expected behavior**:
- **Pause latency**: <50ms from speech detection to TTS stop
- **VAD sensitivity**: Configurable (aggressiveness 0-3)
- **Debouncing**: Min 100ms speech, 300ms silence
- **State transitions**: SPEAKING → BARGED_IN → LISTENING

**Barge-in configuration** (`configs/orchestrator.docker.yaml`):
```yaml
vad:
  enabled: true
  aggressiveness: 2  # 0=least aggressive, 3=most aggressive
  sample_rate: 16000  # Required by webrtcvad
  frame_duration_ms: 20
  min_speech_duration_ms: 100  # Debounce threshold for speech start
  min_silence_duration_ms: 300  # Debounce threshold for speech end
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Port Conflicts

**Symptom**: Docker Compose fails with "port already allocated"

**Solution**:
```bash
# Find process using the port (example: port 8080)
sudo lsof -i :8080

# Kill the process or stop the service
sudo kill <PID>

# Or modify docker-compose.yml to use different ports
```

**Alternative ports** (edit `docker-compose.yml`):
```yaml
orchestrator:
  ports:
    - "8082:8080"  # Change host port to 8082
```

#### 2. Voicepack Not Found

**Symptom**: TTS worker logs show "No ONNX model found in voicepacks/piper/en-us-lessac-medium"

**Solution 1**: Verify voicepack exists
```bash
ls -la voicepacks/piper/en-us-lessac-medium/
# Should show: *.onnx, *.onnx.json, metadata.yaml
```

**Solution 2**: Download Piper voice (if missing)
```bash
# Download Lessac medium voice
mkdir -p voicepacks/piper/en-us-lessac-medium
cd voicepacks/piper/en-us-lessac-medium

# Download from Piper releases
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en-us-lessac-medium.tar.gz
tar -xzf voice-en-us-lessac-medium.tar.gz
rm voice-en-us-lessac-medium.tar.gz

# Create metadata.yaml (see "Adding Custom Piper Voices" section)
```

**Solution 3**: Check volume mount in docker-compose.yml
```yaml
tts0:
  volumes:
    - ./voicepacks:/app/voicepacks:ro  # Ensure this line exists
```

#### 3. Service Health Checks Failing

**Symptom**: `docker compose ps` shows services as "unhealthy"

**Diagnosis**:
```bash
# Check logs for specific service
docker compose logs tts0
docker compose logs orchestrator
docker compose logs livekit

# Manual health check
docker compose exec tts0 timeout 1 bash -c '</dev/tcp/localhost/7001' && echo OK
docker compose exec orchestrator curl -f http://localhost:8081/health
```

**Solutions**:
- **TTS Worker unhealthy**: Model loading timeout (increase `start_period` in docker-compose.yml)
- **Orchestrator unhealthy**: Can't reach TTS worker (check gRPC connectivity)
- **LiveKit unhealthy**: Port 7880 conflict (check `lsof -i :7880`)

#### 4. No Audio Output (CLI Client)

**Symptom**: CLI client runs but no sound plays

**Solutions**:

**Linux**:
```bash
# Check ALSA/PulseAudio
aplay -l  # List playback devices

# Test audio
speaker-test -t wav -c 2

# Run CLI with specific device
AUDIODEV=hw:0,0 uv run python -m src.client.cli_client
```

**macOS**:
```bash
# Check system audio preferences
# Ensure default output device is set
```

**WSL2**:
```bash
# Install PulseAudio for WSL2
sudo apt install pulseaudio
pulseaudio --start

# Configure PULSE_SERVER
export PULSE_SERVER=tcp:localhost:4713
```

#### 5. gRPC Connection Refused

**Symptom**: Orchestrator logs show "failed to connect to TTS worker"

**Diagnosis**:
```bash
# Check TTS worker is listening
docker compose exec tts0 netstat -tuln | grep 7001

# Check DNS resolution inside orchestrator
docker compose exec orchestrator ping tts0

# Test gRPC connectivity (requires grpcurl)
docker compose exec orchestrator grpcurl -plaintext tts0:7001 list
```

**Solutions**:
- **TTS worker not started**: Check `docker compose ps tts0`
- **DNS issue**: Restart Docker network (`docker compose down && docker compose up`)
- **Firewall**: Check Docker network isolation

#### 6. Slow Synthesis / High Latency

**Symptom**: First audio latency >1 second

**Diagnosis**:
```bash
# Check TTS worker logs for timing
docker compose logs tts0 | grep -E "(warmup|synthesis|duration)"

# Check CPU usage
docker stats tts-worker-0
```

**Solutions**:
- **CPU throttling**: Check system load (`top`, `htop`)
- **I/O bottleneck**: Ensure SSD storage (not network mount)
- **Memory pressure**: Check available RAM (`free -h`)
- **Docker resource limits**: Increase in docker-compose.yml:
  ```yaml
  tts0:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 2G
  ```

#### 7. WebRTC Connection Failed (Browser Client)

**Symptom**: Browser client can't connect to LiveKit

**Diagnosis**:
```bash
# Check LiveKit is running
docker compose ps livekit

# Check browser console (F12) for errors
# Look for: "Failed to connect to ws://localhost:7880"
```

**Solutions**:
- **Port 7880 blocked**: Check firewall (`sudo ufw status`)
- **HTTPS required**: Use Caddy proxy (https://localhost:8444)
- **API key mismatch**: Verify `.env.local` matches `configs/livekit.yaml`
- **CORS issue**: Check Caddy logs for CORS errors

---

## Advanced Configuration

### Changing TTS Worker Configuration

Edit `configs/worker.yaml` to customize behavior:

**Example: Increase TTL for model caching**
```yaml
model_manager:
  ttl_ms: 1800000  # 30 minutes instead of 10
  min_residency_ms: 300000  # Keep at least 5 minutes
```

**Example: Adjust audio output**
```yaml
audio:
  output_sample_rate: 48000  # Keep at 48kHz for compatibility
  frame_duration_ms: 20      # Keep at 20ms for low latency
  normalization_enabled: true
  loudness_target_lufs: -16.0  # Adjust volume (-23 to -16 LUFS)
```

**Apply changes**:
```bash
# Rebuild TTS worker with new config
docker compose up -d --build tts0

# Verify config loaded
docker compose logs tts0 | grep config
```

### Adding Custom Piper Voices

Piper supports 50+ languages and voices. Add new voices to `voicepacks/piper/`:

**Step 1: Download voice files**
```bash
# Example: Add Spanish voice
mkdir -p voicepacks/piper/es-es-sharvard-medium
cd voicepacks/piper/es-es-sharvard-medium

# Download from Piper releases
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-es-es-sharvard-medium.tar.gz
tar -xzf voice-es-es-sharvard-medium.tar.gz
```

**Step 2: Create metadata.yaml**
```yaml
model_id: "piper-es-es-sharvard-medium"
family: "piper"
language: "es-ES"
voice_name: "Sharvard"
variant: "medium"
tags:
  - cpu_ok
  - streaming
  - neural
sample_rate: 22050
description: "Spanish (Spain) voice from Piper TTS"
onnx_model: "es_ES-sharvard-medium.onnx"
config_file: "es_ES-sharvard-medium.onnx.json"
```

**Step 3: Update worker config**
```yaml
# configs/worker.yaml
model_manager:
  default_model_id: "piper-es-es-sharvard-medium"  # Use new voice
  preload_model_ids:
    - "piper-en-us-lessac-medium"  # Keep English available
```

**Step 4: Restart TTS worker**
```bash
docker compose restart tts0

# Verify model loaded
docker compose logs tts0 | grep "es-es-sharvard"
```

**Available voices**: Browse all Piper voices at https://rhasspy.github.io/piper-samples/

### Multi-GPU Setup (Future)

**Note**: Piper is CPU-only. Multi-GPU setup applies to future GPU adapters (M6-M8).

**Preview** (not functional with Piper):
```yaml
# docker-compose.yml (future)
services:
  tts0:  # GPU 0
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - device_ids: ['0']
              capabilities: [gpu]

  tts1:  # GPU 1
    environment:
      - CUDA_VISIBLE_DEVICES=1
    deploy:
      resources:
        reservations:
          devices:
            - device_ids: ['1']
              capabilities: [gpu]
```

### Development Mode vs Production Mode

**Development** (current docker-compose.yml):
- Uses `devkey` and simple secrets
- Self-signed TLS certificates
- Debug logging enabled
- Auto-rebuild on code changes (volumes mounted)

**Production** (future):
- Strong API keys and secrets
- Valid TLS certificates (Let's Encrypt)
- Error-level logging only
- Optimized Docker images (no source mounts)
- Resource limits enforced
- Health monitoring (Prometheus + Grafana)

**To enable production mode** (future):
```bash
# Use production compose file
docker compose -f docker-compose.prod.yml up -d

# Or set production environment
export ENVIRONMENT=production
docker compose up -d
```

---

## Log Analysis

### Viewing Logs

**All services**:
```bash
docker compose logs -f
```

**Specific service**:
```bash
docker compose logs -f tts0
docker compose logs -f orchestrator
docker compose logs -f livekit
```

**Filter by keyword**:
```bash
# Show only synthesis-related logs
docker compose logs tts0 | grep synthesis

# Show only errors
docker compose logs --tail=100 | grep -i error

# Show timing metrics
docker compose logs orchestrator | grep -E "(latency|duration|timing)"
```

### Understanding TTS Worker Logs

**Warmup phase**:
```
INFO  Loading Piper voice  model_id="piper-en-us-lessac-medium"
INFO  PiperTTSAdapter initialized  native_sample_rate=22050 target_sample_rate=48000
INFO  Starting warmup synthesis  model_id="piper-en-us-lessac-medium"
INFO  Warmup synthesis complete  warmup_duration_ms=287 audio_duration_ms=2100
```

**Synthesis logs**:
```
INFO  Starting synthesis stream  state=synthesizing model_id="piper-en-us-lessac-medium"
DEBUG Processing text chunk  chunk_id=1 text_length=42 text_preview="Hello, this is a test of the Piper system."
DEBUG Generated frames for chunk  chunk_id=1 frame_count=87 audio_duration_ms=1740
INFO  Synthesis stream completed  total_chunks=1 model_id="piper-en-us-lessac-medium"
```

**Control commands**:
```
INFO  Adapter paused  command=PAUSE previous_state=synthesizing
INFO  Adapter resumed  command=RESUME previous_state=paused
INFO  Adapter stopped  command=STOP previous_state=synthesizing
```

### Understanding Orchestrator Logs

**Session lifecycle**:
```
INFO  Session started  session_id="abc123" transport=websocket
INFO  State transition  session_id="abc123" from=LISTENING to=SPEAKING
INFO  State transition  session_id="abc123" from=SPEAKING to=LISTENING
INFO  Session terminated  session_id="abc123" duration_ms=5420
```

**VAD events** (when using browser client):
```
DEBUG VAD frame processed  is_speech=false frames_total=142 speech_ratio=0.12
INFO  Speech detected  timestamp_ms=1234567890 debounce_ms=100
INFO  Barge-in triggered  session_id="abc123" pause_latency_ms=42
INFO  Silence detected  timestamp_ms=1234568200 debounce_ms=300
```

**Error patterns**:
```
ERROR Failed to connect to TTS worker  url="grpc://tts0:7001" error="connection refused"
ERROR Synthesis failed  session_id="abc123" error="worker unavailable"
ERROR Invalid state transition  session_id="abc123" from=SPEAKING to=SPEAKING
```

---

## Next Steps

### Explore GPU TTS Adapters (M6-M8)

After mastering Piper (CPU baseline), explore GPU-accelerated models:

1. **M6: CosyVoice 2** - Expressive GPU TTS
   - Zero-shot voice synthesis
   - Emotional control
   - Multilingual support
   - Target FAL: <300ms (p95)

2. **M7: XTTS-v2** - Voice cloning
   - Multi-speaker synthesis
   - 6-10s voice cloning
   - High-quality output
   - GPU required

3. **M8: Sesame/Unsloth** - LoRA fine-tuning
   - Custom voice training
   - LoRA adapters
   - Domain adaptation
   - Advanced users

### Integrate with Your Application

**Use the gRPC API directly**:
```python
import grpc
from src.rpc.generated import tts_pb2, tts_pb2_grpc

# Connect to TTS worker
channel = grpc.insecure_channel('localhost:7001')
stub = tts_pb2_grpc.TTSServiceStub(channel)

# Start session
response = stub.StartSession(tts_pb2.SessionRequest(session_id="my-session"))

# Stream text and receive audio
def text_generator():
    yield tts_pb2.TextChunk(text="Hello, world!")

for audio_frame in stub.Synthesize(text_generator()):
    # Process audio_frame.audio_data (bytes)
    # Each frame is 20ms, 48kHz, mono PCM
    pass
```

**Use the WebSocket API** (simpler):
```python
import asyncio
import websockets
import json

async def test_tts():
    async with websockets.connect('ws://localhost:8080') as ws:
        # Start session
        await ws.send(json.dumps({
            "type": "session_start",
            "session_id": "test-123"
        }))

        # Send text
        await ws.send(json.dumps({
            "type": "text",
            "text": "Hello from WebSocket!"
        }))

        # Receive audio frames
        while True:
            message = await ws.recv()
            data = json.loads(message)
            if data["type"] == "audio":
                # Process audio_data (base64 encoded)
                pass
            elif data["type"] == "done":
                break

asyncio.run(test_tts())
```

### Contribute

Help improve the project:

1. **Add more Piper voices**: Submit PR with new voicepacks
2. **Improve documentation**: Fix typos, add examples
3. **Report issues**: Open GitHub issue with logs and repro steps
4. **Test edge cases**: Try unusual inputs, stress testing
5. **Optimize performance**: Profile and submit optimizations

See [Contributing Guidelines](../docs/DEVELOPMENT.md#contributing-guidelines)

### Get Help

- **Documentation**: [docs/](../docs/) directory
- **Known Issues**: [docs/known-issues/README.md](../docs/known-issues/README.md)
- **Development Guide**: [docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md)
- **GitHub Issues**: Include logs, config, and repro steps
- **GitHub Discussions**: For questions and feature requests

---

## Summary

You've successfully deployed a complete realtime TTS system using:

- **Docker Compose**: Orchestrated 5 services with health checks
- **Piper TTS**: CPU-based neural TTS with 22kHz→48kHz resampling
- **Streaming protocol**: 20ms frames over gRPC
- **Dual clients**: WebSocket CLI + WebRTC browser
- **Barge-in support**: VAD with <50ms pause latency

**Key achievements**:
- ✅ Full stack running in <5 minutes
- ✅ Natural-sounding speech synthesis
- ✅ Real-time audio streaming (20ms frames)
- ✅ Production-ready infrastructure
- ✅ CPU-only deployment (no GPU required)

**What's next**:
- Explore GPU adapters (M6-M8) for lower latency
- Integrate with your application via gRPC/WebSocket
- Add custom Piper voices for multilingual support
- Contribute to the project

**Happy voice chatting! 🎙️**

---

**Last Updated**: 2025-10-10
**Maintained by**: Documentation Team
**Next Review**: After M6 completion
