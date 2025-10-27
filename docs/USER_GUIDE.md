# User Guide

**Version**: 1.0
**Last Updated**: 2025-10-26
**Status**: Complete

Comprehensive guide to using the Full-Duplex Voice Chat system from first-time setup to production deployment.

---

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Choose Your Model](#choose-your-model)
- [Enable Parallel Synthesis](#enable-parallel-synthesis)
- [Test the System](#test-the-system)
- [Monitor Performance](#monitor-performance)
- [Production Deployment](#production-deployment)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

---

## Introduction

### What is Full-Duplex Voice Chat?

A production-ready realtime speech-to-speech system that enables natural conversations with:

- **Realtime ASR:** Speech-to-text using Whisper/WhisperX
- **LLM Integration:** Optional conversational AI (OpenAI GPT-4)
- **Streaming TTS:** Text-to-speech with Piper (CPU) or CosyVoice (GPU)
- **Barge-in Support:** Interrupt and resume with <50ms latency
- **WebRTC Transport:** Browser-based voice chat via LiveKit

### System Requirements

**Minimum (CPU-only):**
- 4 CPU cores
- 8 GB RAM
- 20 GB disk space
- Python 3.13+, Docker, uv

**Recommended (GPU):**
- 8 CPU cores
- 16 GB RAM
- NVIDIA GPU with 8GB+ VRAM
- 50 GB disk space (for models)

### Architecture Overview

```
Browser Client (WebRTC)
    ↓
LiveKit Server (WebRTC signaling)
    ↓
Orchestrator (LiveKit Agent)
    ├─ ASR (Whisper/WhisperX) → Transcription
    ├─ LLM (OpenAI GPT-4) → Response generation
    └─ TTS (Piper/CosyVoice) → Audio synthesis
         ↓
    gRPC TTS Worker(s)
         ↓
    Audio Frames → Browser Client
```

---

## Getting Started

### Step 1: Clone Repository

```bash
cd ~/git  # Or your preferred directory
git clone https://github.com/gsornsen/full-duplex-voice-chat.git
cd full-duplex-voice-chat
```

### Step 2: Install Dependencies

```bash
# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
uv sync --all-extras

# Generate gRPC stubs
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

# Edit configuration (optional - defaults work)
nano .env
```

**Default configuration uses Piper CPU TTS** - works on any system.

### Step 4: Start Services

```bash
# Start all services with unified development mode
just dev-agent-piper
```

**What's happening:**
- Redis starts (service discovery)
- LiveKit server starts (WebRTC)
- Caddy starts (HTTPS reverse proxy)
- Orchestrator starts (LiveKit Agent + ASR)
- TTS Worker starts (Piper CPU)

**Expected startup time:** ~10 seconds

**Logs saved to:** `logs/dev-sessions/dev-agent-piper-YYYYMMDD-HHMMSS.log`

### Step 5: Access Web Client

Open browser to: **https://localhost:8443**

**Note:** Accept certificate warning (self-signed cert for development).

**Expected:**
1. Web client loads
2. Click "Join Room"
3. Allow microphone access
4. Ready to talk!

---

## Choose Your Model

The system supports multiple TTS engines with different trade-offs:

### Option 1: Piper (CPU Baseline)

**When to use:**
- Systems without GPU
- Quick testing and development
- Cost-optimized deployments

**Configuration:**

```bash
# Edit .env
ADAPTER_TYPE=piper
DEFAULT_MODEL=piper-en-us-lessac-medium

# ASR on CPU
ASR_DEVICE=cpu
ASR_MODEL_SIZE=small
```

**Performance:**
- First Audio Latency: <500ms
- Quality: Good (natural speech)
- VRAM: 0 GB (CPU-only)
- Speed: Real-time (RTF < 1.0)

**Start services:**
```bash
just dev-agent-piper
```

---

### Option 2: CosyVoice (GPU High Quality)

**When to use:**
- Production deployments
- Highest quality requirements
- Systems with NVIDIA GPU (8GB+ VRAM)

**Prerequisites:**

```bash
# Verify GPU access
nvidia-smi

# Test Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Setup Voicepack:**

```bash
# Download CosyVoice voicepack
./scripts/setup_cosyvoice_voicepack.sh en-base

# Verify voicepack structure
ls -la voicepacks/cosyvoice/en-base/
# Should show: model.pt, config.json, metadata.yaml
```

**See [VOICEPACK_COSYVOICE2.md](VOICEPACK_COSYVOICE2.md) for detailed setup.**

**Configuration:**

```bash
# Edit .env
ADAPTER_TYPE=cosyvoice2
DEFAULT_MODEL=cosyvoice2-en-base

# ASR on GPU (auto-detect)
ASR_DEVICE=auto
ASR_MODEL_SIZE=small
ASR_COMPUTE_TYPE=default
```

**Performance:**
- First Audio Latency: <300ms
- Quality: Excellent (expressive, zero-shot)
- VRAM: 2-4 GB (single worker)
- Speed: Real-time (RTF < 0.3)

**Start services:**
```bash
# Option 1: Unified development mode
just dev cosyvoice2

# Option 2: Docker Compose with profile
docker compose --profile cosyvoice up
```

---

### Option 3: Mock (Testing Only)

**When to use:**
- CI/CD pipelines
- Protocol validation
- Quick testing without models

**Configuration:**

```bash
# Edit .env
ADAPTER_TYPE=mock
DEFAULT_MODEL=mock
```

**Behavior:**
- Generates 440Hz sine wave (beep)
- No actual speech synthesis
- Useful for testing protocol flow

**Start services:**
```bash
just dev
```

---

## Enable Parallel Synthesis

**Parallel synthesis** runs multiple TTS workers concurrently for **2x throughput improvement**.

### When to Enable

**Recommended for:**
- ✅ Production deployments with GPU
- ✅ Systems with 8GB+ VRAM
- ✅ Long multi-sentence responses
- ✅ High concurrency requirements

**Not recommended for:**
- ❌ CPU-only systems (overhead > benefits)
- ❌ Limited VRAM (<4GB)
- ❌ Single-sentence responses

### Configuration

**Edit .env:**

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

| GPU VRAM | Recommended Workers | GPU Limit |
|----------|---------------------|-----------|
| 4 GB | 1 (or disable) | 1 |
| 8 GB | 2 | 2 |
| 12 GB | 3 | 3 |
| 16+ GB | 4+ | 4+ |

### Apply Configuration

```bash
# Restart services to load new config
just dev-stop
just dev cosyvoice2
```

### Verify Parallel Mode

**Check logs:**

```bash
just logs-tail | grep "PARALLEL"
```

**Expected:**
```
[TTS] gRPC TTS plugin created with PARALLEL synthesis
[TTS] ParallelTTSWrapper initialized (num_workers=2, queue_size=10, gpu_limit=2)
[TTS] Starting 2 persistent TTS workers
```

**Verify worker rotation:**

```bash
just logs-tail | grep "Worker.*selected"
```

**Expected:**
```
[TTS] Worker 0 selected for synthesis
[TTS] Worker 1 selected for synthesis
[TTS] Worker 0 selected for synthesis  # Round-robin
```

**See [PARALLEL_TTS.md](PARALLEL_TTS.md) for detailed guide.**

---

## Test the System

### Test 1: Basic Conversation

**Objective:** Verify end-to-end speech-to-speech flow.

**Steps:**
1. Open web client: https://localhost:8443
2. Click "Join Room"
3. Allow microphone access
4. Speak: "Hello, how are you?"
5. Wait for response

**Expected:**
- Your speech transcribed within ~1s (GPU) or ~3s (CPU)
- LLM generates response (if enabled)
- Synthesized speech plays within <500ms (Piper) or <300ms (CosyVoice)

**Success criteria:**
- ✅ Transcription appears in logs
- ✅ Audio plays back clearly
- ✅ Latency within targets

---

### Test 2: Barge-in Support

**Objective:** Verify interrupt and resume functionality.

**Steps:**
1. Join room
2. Speak a long sentence (20+ words)
3. While system is speaking, interrupt by speaking again
4. System should pause within 50ms
5. When you stop speaking, system resumes

**Expected:**
- System pauses immediately on interruption (<50ms)
- VAD detects your speech
- System sends PAUSE command to TTS worker
- Audio stops cleanly
- System resumes when you finish speaking

**Success criteria:**
- ✅ Pause latency <50ms
- ✅ No audio glitches during pause/resume
- ✅ Conversation flow feels natural

**Check logs:**
```bash
just logs-tail | grep "VAD\|PAUSE\|RESUME"
```

**Expected:**
```
[VAD] Speech detected
[Orchestrator] Sending PAUSE to TTS worker
[TTS] PAUSE received, stopping synthesis
[VAD] Silence detected
[Orchestrator] Sending RESUME to TTS worker
[TTS] RESUME received, continuing synthesis
```

---

### Test 3: Multi-turn Conversation

**Objective:** Verify session persistence across multiple exchanges.

**Steps:**
1. Join room
2. Speak: "What's the weather like?"
3. Wait for response
4. Speak: "Tell me more about that"
5. Wait for response
6. Continue conversation for 5+ turns

**Expected:**
- Session persists across turns
- No reconnection needed
- Context maintained (if using LLM)
- Consistent performance

**Success criteria:**
- ✅ No disconnections
- ✅ Latency consistent across turns
- ✅ No memory leaks (check logs)

**Monitor session:**
```bash
just logs-tail | grep "Session"
```

---

### Test 4: Parallel Synthesis Performance

**Prerequisites:** Parallel synthesis enabled (see above)

**Objective:** Verify throughput improvement.

**Steps:**
1. Join room
2. Ask for long response: "Tell me a story about space exploration in 5 sentences"
3. Observe synthesis timing in logs
4. Compare to sequential mode

**Measurement:**

```bash
# Monitor worker selection
just logs-tail | grep "Worker.*selected"

# Monitor synthesis timing
just logs-tail | grep "Synthesis.*elapsed"
```

**Expected (2 workers):**
```
[TTS] Worker 0 selected for synthesis  # Sentence 1
[TTS] Worker 1 selected for synthesis  # Sentence 2 (concurrent!)
[TTS] Worker 0 selected for synthesis  # Sentence 3
[TTS] Worker 1 selected for synthesis  # Sentence 4
[TTS] Worker 0 selected for synthesis  # Sentence 5
```

**Performance:**
- Sequential mode: ~7.5s total (5 × 1.5s)
- Parallel mode (2 workers): ~3.5s total (50% reduction)

**Success criteria:**
- ✅ Workers alternate evenly
- ✅ Total time reduced by ~50%
- ✅ Audio plays in correct order (FIFO maintained)

**See [PARALLEL_TTS.md](PARALLEL_TTS.md) for detailed metrics.**

---

## Monitor Performance

### Real-time Monitoring

**Watch logs:**

```bash
# Tail most recent log file
just logs-tail

# Follow specific service (Docker Compose)
docker compose logs -f orchestrator
docker compose logs -f tts0
```

**Monitor GPU (if using GPU TTS/ASR):**

```bash
# Real-time GPU stats
watch -n 1 nvidia-smi

# Continuous monitoring
nvidia-smi dmon -s mu -c 100
```

### Key Metrics

**Latency Metrics:**

```bash
# First Audio Latency (FAL)
just logs-tail | grep "First audio frame"

# Target: <500ms (Piper), <300ms (CosyVoice)
```

**ASR Performance:**

```bash
# Transcription latency
just logs-tail | grep "Transcription.*elapsed"

# Target: <1s (GPU), <3s (CPU)
```

**TTS Performance:**

```bash
# Synthesis latency per sentence
just logs-tail | grep "Synthesis.*elapsed"

# Target: <500ms per sentence (with parallel mode)
```

**Worker Utilization (Parallel Mode):**

```bash
# Count synthesis per worker
just logs-tail | grep "Worker 0 selected" | wc -l
just logs-tail | grep "Worker 1 selected" | wc -l

# Should be balanced (~50/50 for 2 workers)
```

### Health Checks

**Web client accessibility:**

```bash
curl -k https://localhost:8443
# Expected: 200 OK
```

**LiveKit server:**

```bash
curl http://localhost:7880
# Expected: LiveKit response
```

**Orchestrator health:**

```bash
curl http://localhost:8080/health
# Expected: {"status":"healthy"}
```

### Performance Targets

| Metric | Target (CPU) | Target (GPU) | Target (GPU + Parallel) |
|--------|--------------|--------------|-------------------------|
| **First Audio Latency** | <500ms | <300ms | <300ms |
| **ASR Latency** | <3s | <1s | <1s |
| **Synthesis Latency (per sentence)** | ~1.5s | ~1.5s | ~0.75s (2x workers) |
| **Total Response (5 sentences)** | ~7.5s | ~7.5s | ~3.5s |
| **VRAM Usage** | 0 GB | 2-4 GB | 4-6 GB (2 workers) |

---

## Production Deployment

### Docker Compose (Recommended)

**Production-ready stack with one command:**

```bash
# Start all services
docker compose up --build -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

**Services included:**
- Redis (service discovery)
- LiveKit (WebRTC server)
- Caddy (HTTPS reverse proxy)
- Orchestrator (LiveKit Agent + ASR)
- TTS Worker (GPU-pinned)

### Security Configuration

**1. Replace self-signed certificates:**

```bash
# Generate Let's Encrypt cert (production domain)
certbot certonly --standalone -d yourdomain.com

# Update Caddyfile
nano Caddyfile
# Add: tls /path/to/cert.pem /path/to/key.pem
```

**2. Set strong API credentials:**

```bash
# Edit .env
nano .env

# Update LiveKit credentials
LIVEKIT_API_KEY=$(openssl rand -base64 32)
LIVEKIT_API_SECRET=$(openssl rand -base64 48)

# Restart services
docker compose restart
```

**3. Configure firewall:**

```bash
# Allow required ports
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow 7880/tcp    # LiveKit signaling
sudo ufw allow 50000:50099/udp  # WebRTC media
```

**See [HTTPS_SETUP.md](HTTPS_SETUP.md) for detailed security guide.**

### Environment Variables for Production

```bash
# .env (production)

# TTS: GPU for quality
ADAPTER_TYPE=cosyvoice2
DEFAULT_MODEL=cosyvoice2-en-base

# ASR: GPU for performance
ASR_DEVICE=auto
ASR_MODEL_SIZE=small

# Parallel Synthesis: Enabled for throughput
PARALLEL_SYNTHESIS_ENABLED=true
PARALLEL_SYNTHESIS_NUM_WORKERS=2
PARALLEL_SYNTHESIS_GPU_LIMIT=2

# LiveKit: Strong credentials
LIVEKIT_API_KEY=<strong-random-key>
LIVEKIT_API_SECRET=<strong-random-secret>

# OpenAI: Production API key
OPENAI_API_KEY=<your-production-key>
OPENAI_MODEL=gpt-4-turbo

# Logging: JSON for structured logs
LOG_LEVEL=INFO
LOG_FORMAT=json

# Debug: Disabled
DEBUG=false
```

### Monitoring & Observability

**Set up monitoring stack:**

1. **Prometheus:** Metrics collection
   ```bash
   # Add prometheus service to docker-compose.yml
   # Configure scrape targets
   ```

2. **Grafana:** Metrics visualization
   ```bash
   # Add grafana service
   # Import TTS dashboard
   ```

3. **Logging:** Centralized log aggregation
   ```bash
   # Configure log shipping (e.g., Loki, ELK)
   # Set retention policies
   ```

**See [OBSERVABILITY.md](OBSERVABILITY.md) for detailed monitoring guide.**

### Scaling

**Single-GPU deployment:**
```bash
# Single TTS worker on GPU 0
CUDA_VISIBLE_DEVICES=0 docker compose up
```

**Multi-GPU deployment:**
```bash
# Worker 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 just run-tts-cosyvoice2 &

# Worker 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 just run-tts-cosyvoice2 &

# Orchestrator discovers both via Redis
just run-orch
```

**See [MULTI_GPU.md](MULTI_GPU.md) for scaling guide.**

---

## Advanced Features

### Custom Voice Models

**Piper voices:**

1. Download voicepack from https://github.com/rhasspy/piper
2. Extract to `voicepacks/piper/<voice-name>/`
3. Update configuration:
   ```bash
   DEFAULT_MODEL=piper-<voice-name>
   ```
4. Restart services

**CosyVoice voice cloning:**

1. Prepare reference audio (6-10s, clean speech)
2. Use CosyVoice cloning API
3. Generate custom voicepack
4. Deploy to `voicepacks/cosyvoice/<custom-name>/`

**See [VOICE_PACKS.md](VOICE_PACKS.md) for voice customization.**

### Multi-language Support

**WhisperX supports 99+ languages:**

```bash
# Edit .env
ASR_LANGUAGE=es  # Spanish
ASR_LANGUAGE=fr  # French
ASR_LANGUAGE=zh  # Chinese
ASR_LANGUAGE=ja  # Japanese
```

**Restart services to apply.**

### LLM Customization

**Use custom LLM endpoint:**

```python
# Modify src/orchestrator/agent.py
# Replace OpenAI client with custom endpoint
llm_client = CustomLLMClient(
    endpoint="http://localhost:8000/v1/chat/completions"
)
```

**Supported:**
- Local LLMs (Ollama, llama.cpp)
- Anthropic Claude
- Google Gemini
- Azure OpenAI

### Dual-LLM Strategy (Experimental)

**Reduce perceived latency with filler responses:**

```bash
# Edit .env
DUAL_LLM_ENABLED=true
```

**How it works:**
1. System immediately responds with template ("Let me think...")
2. Full LLM response generated in parallel
3. User perceives faster response

**See `.env.example` for configuration details.**

---

## Troubleshooting

### Quick Diagnostics

**Check all services running:**

```bash
# Docker Compose
docker ps

# Unified dev mode
ps aux | grep -E "(orchestrator|tts-worker|livekit)"
```

**Check logs for errors:**

```bash
just logs-tail | grep -i "error\|exception\|fail"
```

**Test connectivity:**

```bash
# Web client
curl -k https://localhost:8443

# LiveKit
curl http://localhost:7880

# Orchestrator
curl http://localhost:8080/health
```

### Common Issues

**Issue: No audio heard**

See [QUICK_START.md#troubleshooting](QUICK_START.md#troubleshooting)

**Issue: High latency**

See [PARALLEL_TTS.md#troubleshooting](PARALLEL_TTS.md#troubleshooting)

**Issue: GPU out of memory**

See [PARALLEL_TTS.md#issue-gpu-out-of-memory-oom](PARALLEL_TTS.md#issue-gpu-out-of-memory-oom)

**Issue: Configuration not applied**

```bash
# Stop all services
just dev-stop
docker compose down

# Restart with clean state
just dev-agent-piper
```

### Getting Help

**Documentation:**
- [Quick Start Guide](QUICK_START.md)
- [Configuration Guide](CONFIGURATION.md)
- [Parallel TTS Guide](PARALLEL_TTS.md)
- [Performance Guide](PERFORMANCE.md)

**Community:**
- GitHub Issues: https://github.com/gsornsen/full-duplex-voice-chat/issues
- GitHub Discussions: https://github.com/gsornsen/full-duplex-voice-chat/discussions

---

## Next Steps

**Master the system:**
1. ✅ Complete Quick Start (CPU)
2. ✅ Upgrade to GPU (CosyVoice)
3. ✅ Enable parallel synthesis
4. ✅ Deploy to production
5. ✅ Monitor performance
6. ✅ Customize for your use case

**Explore advanced topics:**
- Custom voice models
- Multi-language support
- LLM integration
- Multi-GPU scaling
- Performance tuning

**Contribute:**
- Report bugs and issues
- Submit feature requests
- Share your configuration
- Contribute code improvements

---

**Status:** Complete - M0-M10 + Parallel Synthesis
**Last Updated:** 2025-10-26
**Maintained by:** Documentation Engineering Team
