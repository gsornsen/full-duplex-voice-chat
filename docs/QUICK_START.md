# Quick Start Guide

**Version**: 2.0
**Last Updated**: 2025-10-26
**Status**: Complete - M0-M10 + Parallel Synthesis

Get the Full-Duplex Voice Chat system running in **under 15 minutes** with realtime speech-to-speech, GPU TTS, and parallel synthesis for 2x throughput.

---

## Table of Contents

- [What You'll Get](#what-youll-get)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start (CPU - Piper)](#quick-start-cpu---piper)
- [Quick Start (GPU - CosyVoice)](#quick-start-gpu---cosyvoice)
- [Enable Parallel Synthesis](#enable-parallel-synthesis)
- [Verify System](#verify-system)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## What You'll Get

After completing this guide, you'll have:

- **LiveKit WebRTC** server for browser-based voice chat
- **Realtime ASR** (Whisper/WhisperX) for speech-to-text
- **TTS synthesis** (Piper CPU or CosyVoice GPU)
- **Parallel synthesis** option for 2x throughput improvement
- **Barge-in support** with <50ms latency
- **Web client** accessible at https://localhost:8443

**Performance Targets:**
- First Audio Latency: <500ms (Piper CPU), <300ms (CosyVoice GPU)
- ASR Transcription: ~3s (CPU), <1s (GPU)
- Synthesis Throughput: 2x with parallel mode enabled

---

## Prerequisites

### Required Software

- [ ] **Python 3.13+** - Check: `python3 --version`
- [ ] **uv package manager** - Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] **Docker Engine 28.x** - Check: `docker --version`
- [ ] **Docker Compose** - Check: `docker compose version`
- [ ] **Git** - Check: `git --version`

### Optional (for GPU TTS/ASR)

- [ ] **NVIDIA GPU** - 4GB+ VRAM (8GB+ recommended for parallel synthesis)
- [ ] **CUDA 12.1+** - Check: `nvidia-smi`
- [ ] **NVIDIA Container Runtime** - Test: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### Recommended System

**Minimum (CPU-only):**
- 4 CPU cores
- 8 GB RAM
- 20 GB disk space

**Recommended (GPU):**
- 8 CPU cores
- 16 GB RAM
- NVIDIA GPU with 8GB+ VRAM
- 50 GB disk space (for models)

### Port Availability

The following ports must be available:

- **7880** - LiveKit WebRTC signaling
- **8080** - Orchestrator WebSocket (optional)
- **8443** - HTTPS web client (via Caddy)
- **8444** - WSS LiveKit proxy (via Caddy)
- **50000-50099/UDP** - WebRTC media streams

Check ports:
```bash
sudo lsof -i :7880,8080,8443,8444
# No output = ports available ✓
```

---

## Installation

### Step 1: Clone Repository

```bash
cd ~/git  # Or your preferred directory
git clone https://github.com/gsornsen/full-duplex-voice-chat.git
cd full-duplex-voice-chat
```

### Step 2: Install Dependencies

```bash
# Install Python dependencies
uv sync --all-extras

# Generate gRPC protocol stubs
just gen-proto
```

**Expected output:**
```
Resolved 127 packages in 1.2s
Installed 127 packages
✓ Proto generation complete
```

### Step 3: Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration (optional - defaults work for CPU)
nano .env
```

**Default configuration uses Piper CPU TTS** - works on any system, no GPU required.

---

## Quick Start (CPU - Piper)

**Best for:** First-time users, systems without GPU, quick testing

### Configuration

Verify `.env` has these settings (default):

```bash
# TTS Configuration (CPU)
ADAPTER_TYPE=piper
DEFAULT_MODEL=piper-en-us-lessac-medium

# ASR Configuration (CPU)
ASR_DEVICE=cpu
ASR_MODEL_SIZE=small
ASR_COMPUTE_TYPE=default
```

### Start Services

```bash
# Start all services in unified development mode
just dev-agent-piper
```

**What's happening:**
- Redis starts (service discovery)
- LiveKit server starts (WebRTC)
- Caddy starts (HTTPS reverse proxy)
- Orchestrator starts (LiveKit Agent + ASR)
- TTS Worker starts (Piper CPU)
- All logs saved to `logs/dev-sessions/dev-agent-piper-YYYYMMDD-HHMMSS.log`

**Expected output:**
```
[Redis] Redis 7.0 ready on port 6379
[LiveKit] LiveKit server started on port 7880
[Caddy] HTTPS server listening on :8443
[Orchestrator] LiveKit Agent initialized
[TTS] Piper worker registered: tts-worker-0
✓ All services running (startup: ~10 seconds)
```

### Access Web Client

Open your browser to:

**https://localhost:8443**

**Note:** You'll see a certificate warning (self-signed cert). Click "Advanced" → "Proceed to localhost" to continue.

**Expected behavior:**
1. Web client loads
2. Click "Join Room"
3. Allow microphone access
4. Speak: "Hello, how are you?"
5. Hear synthesized response (~500ms latency)

---

## Quick Start (GPU - CosyVoice)

**Best for:** Production deployments, highest quality, parallel synthesis

### Prerequisites

Verify GPU is accessible:

```bash
# Check NVIDIA GPU
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Download CosyVoice Voicepack

CosyVoice models require voicepacks in `voicepacks/cosyvoice/<name>/`:

```bash
# Option 1: Use setup script (recommended)
./scripts/setup_cosyvoice_voicepack.sh en-base

# Option 2: Manual download (if script unavailable)
mkdir -p voicepacks/cosyvoice/en-base
# Download model files to voicepacks/cosyvoice/en-base/
# - model.pt (CosyVoice weights)
# - config.json (model configuration)
# - metadata.yaml (voicepack metadata)
```

**See [VOICEPACK_COSYVOICE2.md](VOICEPACK_COSYVOICE2.md) for detailed voicepack setup.**

### Configuration

Edit `.env`:

```bash
# TTS Configuration (GPU)
ADAPTER_TYPE=cosyvoice2
DEFAULT_MODEL=cosyvoice2-en-base

# ASR Configuration (GPU)
ASR_DEVICE=auto  # Auto-detect GPU
ASR_MODEL_SIZE=small
ASR_COMPUTE_TYPE=default
```

### Start Services

```bash
# Option 1: Unified development mode
just dev cosyvoice2

# Option 2: Docker Compose with CosyVoice profile
docker compose --profile cosyvoice up
```

**Expected startup:**
```
[Redis] Redis 7.0 ready on port 6379
[LiveKit] LiveKit server started on port 7880
[Caddy] HTTPS server listening on :8443
[Orchestrator] WhisperX initialized (GPU: 3-5s)
[TTS] CosyVoice worker registered: tts-worker-0
✓ All services running (startup: ~15 seconds)
```

### Performance

**CosyVoice GPU Performance:**
- First Audio Latency: <300ms (target)
- ASR Transcription: <1s (GPU WhisperX)
- Synthesis Quality: High (zero-shot voice cloning)
- VRAM Usage: ~2-4 GB (single worker)

**Access:** https://localhost:8443

---

## Enable Parallel Synthesis

**Parallel synthesis** runs multiple TTS workers concurrently, improving throughput by **2x** while maintaining strict FIFO playback order.

### When to Use Parallel Synthesis

**Recommended for:**
- Production deployments with high concurrency
- Systems with 8GB+ VRAM GPU
- Scenarios with long responses (>3 sentences)
- Reduced perceived latency requirements

**Not recommended for:**
- CPU-only systems (overhead outweighs benefits)
- Limited VRAM (<4GB)
- Single-sentence responses

### Configuration

Edit `.env`:

```bash
# Enable parallel synthesis
PARALLEL_SYNTHESIS_ENABLED=true

# Number of parallel workers (2-3 recommended)
PARALLEL_SYNTHESIS_NUM_WORKERS=2

# Maximum sentence queue depth
PARALLEL_SYNTHESIS_MAX_QUEUE_DEPTH=10

# Maximum concurrent GPU operations
PARALLEL_SYNTHESIS_GPU_LIMIT=2
```

**GPU Memory Planning:**
```
Single worker:  ~2-4 GB VRAM
2 workers:      ~4-6 GB VRAM
3 workers:      ~6-10 GB VRAM
```

### Restart Services

```bash
# Stop current services
just dev-stop

# Start with new configuration
just dev-agent-piper  # or just dev cosyvoice2
```

### Verify Parallel Mode

Check logs for confirmation:

```bash
just logs-tail
```

**Expected log messages:**
```
[TTS] gRPC TTS plugin created with PARALLEL synthesis
[TTS] ParallelTTSWrapper initialized (num_workers=2, queue_size=10, gpu_limit=2)
[TTS] Starting 2 persistent TTS workers
[TTS] Worker 0 selected for synthesis
[TTS] Worker 1 selected for synthesis
[TTS] Worker 0 selected for synthesis  # Round-robin rotation
```

### Performance Validation

**Metrics to observe:**
- **Synthesis latency:** 50% reduction (6s → 3s for long responses)
- **Worker alternation:** Logs show "Worker 0 → Worker 1 → Worker 0" pattern
- **Throughput:** ~85% sustained worker utilization
- **Memory:** GPU memory increases with worker count

**See [PARALLEL_TTS.md](PARALLEL_TTS.md) for detailed performance analysis.**

---

## Verify System

### Checkpoint 1: Services Running

```bash
# Check Docker containers (if using Docker Compose)
docker ps

# Expected: 5 containers
# - redis
# - livekit
# - caddy
# - orchestrator
# - tts0 (or tts-worker-0)
```

**For unified dev mode:**
```bash
# Check processes
ps aux | grep -E "(orchestrator|tts-worker|livekit|redis|caddy)"

# Check logs
just logs-tail
```

### Checkpoint 2: Web Client Accessible

```bash
# Test HTTPS endpoint
curl -k https://localhost:8443

# Expected: HTML page (200 OK)
```

**Browser test:**
1. Navigate to https://localhost:8443
2. Accept certificate warning
3. See "Join Room" button
4. No JavaScript errors in console (F12)

### Checkpoint 3: ASR Working

**Test speech recognition:**

1. Join room in web client
2. Allow microphone access
3. Speak: "Testing one two three"
4. Check orchestrator logs for transcription:

```bash
just logs-tail | grep "Transcription"

# Expected:
# [ASR] Transcription: "testing one two three" (confidence: 0.95)
```

### Checkpoint 4: TTS Working

**Test synthesis:**

1. After speaking, wait for response
2. Should hear synthesized audio within 500ms (Piper) or 300ms (CosyVoice)
3. Check TTS logs:

```bash
just logs-tail | grep "Synthesis"

# Expected:
# [TTS] Synthesis started: session_id=xxx, text="..."
# [TTS] Synthesis complete: 245ms elapsed, 42 frames
```

### Checkpoint 5: Parallel Synthesis (if enabled)

**Verify worker rotation:**

```bash
just logs-tail | grep "Worker.*selected"

# Expected (round-robin):
# [TTS] Worker 0 selected for synthesis
# [TTS] Worker 1 selected for synthesis
# [TTS] Worker 0 selected for synthesis
# [TTS] Worker 1 selected for synthesis
```

**Performance check:**

Send a long message (5+ sentences) and observe:
- Multiple workers synthesizing concurrently
- Frames arriving in correct order (FIFO maintained)
- Reduced total latency vs sequential mode

---

## Troubleshooting

### Issue: Certificate Warning in Browser

**Symptom:** Browser shows "Your connection is not private" warning.

**Cause:** Self-signed TLS certificate used for local development.

**Solution:**

**Chrome/Edge:**
1. Click "Advanced"
2. Click "Proceed to localhost (unsafe)"

**Firefox:**
1. Click "Advanced"
2. Click "Accept the Risk and Continue"

**Production:** Replace self-signed cert with valid certificate (Let's Encrypt).

---

### Issue: No Audio Heard After Speaking

**Symptom:** Transcription works but no synthesized speech.

**Cause:** TTS worker not running or model mismatch.

**Solution:**

```bash
# 1. Check TTS worker status
docker ps | grep tts
# OR for dev mode:
ps aux | grep tts-worker

# 2. Verify configuration match
cat .env | grep -E "(ADAPTER|DEFAULT_MODEL)"
# Should show:
# ADAPTER_TYPE=piper
# DEFAULT_MODEL=piper-en-us-lessac-medium

# 3. Check voicepack exists (for CosyVoice)
ls -la voicepacks/cosyvoice/en-base/
# Should show: model.pt, config.json, metadata.yaml

# 4. Restart services
just dev-stop
just dev-agent-piper
```

---

### Issue: WhisperX Initialization Takes >30 Seconds

**Symptom:** Long delay before first transcription, high CPU usage.

**Cause:** Running on CPU instead of GPU.

**Solution:**

```bash
# 1. Set ASR_DEVICE=auto in .env
echo "ASR_DEVICE=auto" >> .env

# 2. Verify CUDA available
nvidia-smi

# 3. Restart services
just dev-stop
just dev

# Expected: 3-5s initialization (vs 28s on CPU)
```

---

### Issue: "Port already in use" Error

**Symptom:**
```
Error: bind: address already in use (port 7880)
```

**Solution:**

```bash
# Find process using port
sudo lsof -i :7880

# Kill process
kill -9 <PID>

# OR stop all services first
just dev-stop
docker compose down

# Then restart
just dev-agent-piper
```

---

### Issue: Parallel Synthesis Not Working

**Symptom:** Only single worker logs, no "Worker X selected" alternation.

**Diagnosis:**

```bash
# Check configuration
cat .env | grep PARALLEL
# Should show:
# PARALLEL_SYNTHESIS_ENABLED=true
# PARALLEL_SYNTHESIS_NUM_WORKERS=2

# Check TTS plugin initialization
just logs-tail | grep "PARALLEL\|SEQUENTIAL"
# Should show: "gRPC TTS plugin created with PARALLEL synthesis"
```

**Solution:**

```bash
# 1. Ensure environment variable is set
export PARALLEL_SYNTHESIS_ENABLED=true

# 2. Restart services (required for config reload)
just dev-stop
just dev-agent-piper

# 3. Verify in logs
just logs-tail | grep "ParallelTTSWrapper initialized"
```

---

### Issue: GPU Out of Memory (OOM)

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Cause:** Too many parallel workers for available VRAM.

**Solution:**

```bash
# 1. Reduce parallel worker count
nano .env
# Set: PARALLEL_SYNTHESIS_NUM_WORKERS=1

# 2. OR reduce GPU limit
# Set: PARALLEL_SYNTHESIS_GPU_LIMIT=1

# 3. Restart services
just dev-stop
just dev

# 4. Monitor GPU memory
watch -n 1 nvidia-smi
```

**VRAM Guidelines:**
- 4GB GPU: 1 worker
- 8GB GPU: 2 workers
- 12GB+ GPU: 3+ workers

---

### Common Issues Links

- [Docker Setup Troubleshooting](setup/DOCKER_SETUP.md#common-errors-and-resolutions)
- [Redis Configuration Guide](REDIS_CONFIGURATION.md) - Port conflicts, connectivity
- [Configuration Guide](CONFIGURATION.md) - Model selection, adapter setup
- [CosyVoice Voicepack Setup](VOICEPACK_COSYVOICE2.md) - GPU TTS setup
- [Performance Tuning](PERFORMANCE.md) - Optimization tips

---

## Next Steps

### Test Full Pipeline

1. **Join web client:** https://localhost:8443
2. **Test conversation flow:**
   - You: "Tell me about the weather"
   - System transcribes (ASR)
   - (Optional) LLM generates response
   - System synthesizes speech (TTS)
   - You hear response
3. **Test barge-in:**
   - Interrupt system while speaking
   - System pauses within 50ms
   - Resume when you stop speaking

### Enable LLM Integration

**Add OpenAI API key** for conversational responses:

```bash
# Edit .env
echo "OPENAI_API_KEY=sk-your-api-key" >> .env

# Restart services
just dev-stop
just dev-agent-piper
```

**Now the system will:**
1. Transcribe your speech (ASR)
2. Send to OpenAI GPT-4 (LLM)
3. Synthesize response (TTS)
4. Stream audio back to you

### Performance Optimization

**For production deployments:**

1. **Enable parallel synthesis** (2x throughput)
2. **Use GPU for ASR** (4-8x faster)
3. **Use CosyVoice for TTS** (better quality)
4. **Tune worker counts** based on VRAM
5. **Monitor performance metrics**

**See [PARALLEL_TTS.md](PARALLEL_TTS.md) for optimization guide.**

### Explore Advanced Features

- **Multi-turn conversations:** Session timeout configuration
- **Custom voices:** Voice cloning with CosyVoice
- **Multiple languages:** WhisperX multi-language support
- **Custom LLM:** Replace OpenAI with local LLM
- **Deployment:** Production Docker Compose setup

### Documentation

**Core Guides:**
- [Configuration Guide](CONFIGURATION.md) - All environment variables
- [Parallel TTS Guide](PARALLEL_TTS.md) - Performance tuning
- [User Guide](USER_GUIDE.md) - Complete user journey
- [Development Guide](DEVELOPMENT.md) - Developer workflows

**Reference:**
- [Current Status](CURRENT_STATUS.md) - Implementation status
- [Performance Metrics](PERFORMANCE.md) - Benchmarks and targets
- [Architecture Overview](architecture/ARCHITECTURE.md) - System design

---

## Getting Help

### Check Logs

```bash
# Tail most recent log file
just logs-tail

# List all log files
just logs-list

# Follow specific service logs (Docker Compose)
docker compose logs -f orchestrator
docker compose logs -f tts0
```

### Run Health Checks

```bash
# Test web client
curl -k https://localhost:8443

# Test LiveKit
curl http://localhost:7880

# Test orchestrator WebSocket
curl http://localhost:8080/health
```

### Community Support

- **GitHub Issues:** https://github.com/gsornsen/full-duplex-voice-chat/issues
- **Discussions:** https://github.com/gsornsen/full-duplex-voice-chat/discussions
- **Documentation:** [docs/](../docs/) directory

---

## Performance Summary

| Configuration | FAL (p95) | ASR Latency | Throughput | VRAM Usage |
|---------------|-----------|-------------|------------|------------|
| **Piper CPU** | <500ms | ~3s (CPU) | 1x | 0 GB |
| **Piper CPU + Parallel** | <500ms | ~3s | ~1.2x | 0 GB |
| **CosyVoice GPU** | <300ms | <1s (GPU) | 1x | 2-4 GB |
| **CosyVoice GPU + Parallel** | <300ms | <1s | **2x** | 4-6 GB |

**Parallel synthesis provides maximum benefit with GPU TTS models.**

---

**Status:** Complete - M0-M10 + Parallel Synthesis
**Last Updated:** 2025-10-26
**Maintained by:** Documentation Engineering Team
