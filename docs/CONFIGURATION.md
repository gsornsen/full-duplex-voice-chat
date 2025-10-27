# Configuration Guide

**Version**: 1.1
**Last Updated**: 2025-10-26
**Status**: Production Ready

This guide explains how to configure the Full-Duplex Voice Chat system for different deployment scenarios.

---

## Table of Contents
- [Environment Variables](#environment-variables)
- [TTS Configuration](#tts-configuration)
- [ASR Configuration](#asr-configuration)
- [Parallel TTS Configuration](#parallel-tts-configuration)
- [Deployment Profiles](#deployment-profiles)
- [Configuration Validation](#configuration-validation)
- [Troubleshooting](#troubleshooting)
- [See Also](#see-also)

---


---

## Environment Variables

All configuration is managed through environment variables in `.env` file.

### Quick Start

1. Copy the example configuration:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your preferred settings:
   ```bash
   nano .env
   ```

3. Start services:
   ```bash
   just dev  # Uses .env configuration
   ```

### Critical Variables

| Variable | Description | Default | Values |
|----------|-------------|---------|--------|
| `ADAPTER_TYPE` | TTS adapter selection | `piper` | `piper`, `cosyvoice2`, `mock` |
| `DEFAULT_MODEL` | Model ID to load at startup | `piper-en-us-lessac-medium` | See [TTS Configuration](#tts-configuration) |
| `ASR_DEVICE` | WhisperX device selection | `auto` | `auto`, `cpu`, `cuda` |
| `ASR_MODEL_SIZE` | Whisper model size | `small` | `tiny`, `small`, `medium`, `large` |
| `ASR_COMPUTE_TYPE` | Whisper compute optimization | `default` | `default`, `int8`, `float16` |

### All Environment Variables

See `.env.example` for complete list with documentation.

---

## TTS Configuration

### Adapter Selection

The `ADAPTER_TYPE` variable controls which TTS engine is used:

```bash
# Piper (CPU, fast, good quality)
ADAPTER_TYPE=piper
DEFAULT_MODEL=piper-en-us-lessac-medium

# CosyVoice 2 (GPU, high quality, requires CUDA)
ADAPTER_TYPE=cosyvoice2
DEFAULT_MODEL=cosyvoice2-en-base

# Mock (testing only, generates sine waves)
ADAPTER_TYPE=mock
DEFAULT_MODEL=mock
```

### Model ID Format

Each adapter requires specific model ID format:

| Adapter | Model ID Format | Example | Hardware |
|---------|-----------------|---------|----------|
| **Piper** | `piper-<voice>` | `piper-en-us-lessac-medium` | CPU (no GPU required) |
| **CosyVoice2** | `cosyvoice2-<voicepack>` | `cosyvoice2-en-base` | GPU (CUDA 12.1) |
| **Mock** | `mock` | `mock` | CPU (testing only) |

**⚠️ Important:** Model ID must match adapter type or system will fall back to Mock adapter!

### Voicepacks

#### Piper Voicepacks

Piper models use ONNX format and are located in `voicepacks/piper/<voice>/`:

```
voicepacks/piper/
├── en-us-lessac-medium/
│   ├── en_US-lessac-medium.onnx
│   ├── en_US-lessac-medium.onnx.json
│   └── metadata.yaml
└── en-us-amy-medium/
    ├── en_US-amy-medium.onnx
    ├── en_US-amy-medium.onnx.json
    └── metadata.yaml
```

See [VOICE_PACKS.md](VOICE_PACKS.md) for Piper voicepack details.

#### CosyVoice Voicepacks

CosyVoice models require voicepacks in `voicepacks/cosyvoice/<name>/`:

```
voicepacks/cosyvoice/
├── en-base/          # cosyvoice2-en-base
│   ├── model.pt
│   ├── config.json
│   └── metadata.yaml
└── en-expressive/    # cosyvoice2-en-expressive (if available)
    ├── model.pt
    ├── config.json
    └── metadata.yaml
```

See [VOICEPACK_COSYVOICE2.md](VOICEPACK_COSYVOICE2.md) for CosyVoice voicepack structure and setup.

### Example Configurations

#### Development (Piper CPU)

```bash
# .env
ADAPTER_TYPE=piper
DEFAULT_MODEL=piper-en-us-lessac-medium
ASR_DEVICE=cpu
ASR_MODEL_SIZE=small
```

#### Production (CosyVoice GPU)

```bash
# .env
ADAPTER_TYPE=cosyvoice2
DEFAULT_MODEL=cosyvoice2-en-base
ASR_DEVICE=auto
ASR_MODEL_SIZE=small
ASR_COMPUTE_TYPE=default
```

#### Testing (Mock Adapter)

```bash
# .env
ADAPTER_TYPE=mock
DEFAULT_MODEL=mock
ASR_DEVICE=cpu
```

---

## ASR Configuration

### Device Selection

Control where WhisperX runs:

```bash
# Auto-detect (recommended - uses GPU if available)
ASR_DEVICE=auto

# Force CPU (slower but compatible)
ASR_DEVICE=cpu

# Force GPU (requires CUDA)
ASR_DEVICE=cuda
```

**Performance Impact:**

| Device | Initialization Time | Transcription RTF | Memory Usage |
|--------|---------------------|-------------------|--------------|
| **CPU** | ~28s | ~0.095 | 1470 MB |
| **GPU** (recommended) | ~3-5s | ~0.048 | 900 MB |

**Recommendation:** Use `ASR_DEVICE=auto` to automatically select GPU when available.

### Model Size

Balance accuracy vs performance:

```bash
# Fast but less accurate
ASR_MODEL_SIZE=tiny    # ~1s transcription, lower accuracy

# Balanced (recommended)
ASR_MODEL_SIZE=small   # ~3s CPU, <1s GPU, good accuracy

# High accuracy (requires more resources)
ASR_MODEL_SIZE=medium  # ~10s CPU, ~2s GPU, excellent accuracy
```

**Word Error Rate (WER):**

| Model Size | WER (LibriSpeech) | Relative Speed |
|------------|-------------------|----------------|
| tiny | ~10% | 4x faster |
| small | ~5% | 2x faster |
| medium | ~3% | baseline |

### Compute Type

Optimize for device:

```bash
# Auto-select (recommended)
ASR_COMPUTE_TYPE=default  # int8 on CPU, float16 on GPU

# Manual override
ASR_COMPUTE_TYPE=int8     # CPU-optimized (quantized)
ASR_COMPUTE_TYPE=float16  # GPU-optimized (half precision)
```

**Recommendation:** Use `default` to let WhisperX choose optimal compute type.

---

## Parallel TTS Configuration

**Feature Status:** Production Ready (2025-10-24)

Parallel TTS synthesis runs multiple TTS workers concurrently for **2x throughput improvement** while maintaining strict FIFO playback order.

### Overview

**What it does:**
- Synthesizes multiple sentences concurrently using persistent worker pool
- Maintains FIFO (first-in-first-out) ordering for correct audio playback
- Reduces synthesis latency by 50% for multi-sentence responses
- Eliminates cold-start latency with persistent workers

**Performance:**
- **2x throughput** with 2 workers (validated)
- **3x throughput** with 3 workers
- ~85% sustained worker utilization
- 50% latency reduction (6s → 3s for 5 sentences)

See [PARALLEL_TTS.md](PARALLEL_TTS.md) for detailed guide.

### Configuration

**Environment Variables:**

```bash
# Enable parallel synthesis
PARALLEL_SYNTHESIS_ENABLED=true

# Number of parallel workers (2-3 recommended)
PARALLEL_SYNTHESIS_NUM_WORKERS=2

# Maximum sentence queue depth (backpressure threshold)
PARALLEL_SYNTHESIS_MAX_QUEUE_DEPTH=10

# Maximum concurrent GPU operations
PARALLEL_SYNTHESIS_GPU_LIMIT=2
```

**Parameters:**

| Variable | Type | Default | Description | Recommended |
|----------|------|---------|-------------|-------------|
| `PARALLEL_SYNTHESIS_ENABLED` | bool | `false` | Enable parallel synthesis | `true` for GPU, `false` for CPU |
| `PARALLEL_SYNTHESIS_NUM_WORKERS` | int | `2` | Number of parallel workers | 2-3 (GPU), 1 (CPU) |
| `PARALLEL_SYNTHESIS_MAX_QUEUE_DEPTH` | int | `10` | Max buffered sentences | 5-20 |
| `PARALLEL_SYNTHESIS_GPU_LIMIT` | int | `2` | Max concurrent GPU ops | 1-3 based on VRAM |

### GPU Memory Planning

| Configuration | VRAM Required | Recommended GPU |
|---------------|---------------|-----------------|
| 1 worker (sequential) | 2-4 GB | GTX 1660 (6GB) |
| 2 workers (parallel) | 4-6 GB | RTX 3060 (12GB) |
| 3 workers (parallel) | 6-10 GB | RTX 3080 (10GB) |

### When to Enable

**Recommended for:**
- ✅ Production deployments with GPU
- ✅ Systems with 8GB+ VRAM
- ✅ Long multi-sentence responses (>3 sentences)
- ✅ High concurrency requirements

**Not recommended for:**
- ❌ CPU-only systems (overhead > benefits)
- ❌ Limited VRAM (<4GB)
- ❌ Single-sentence responses
- ❌ Mock adapter (testing)

### Example Configurations

**CPU Development (Parallel Disabled):**

```bash
# .env
ADAPTER_TYPE=piper
DEFAULT_MODEL=piper-en-us-lessac-medium
PARALLEL_SYNTHESIS_ENABLED=false
```

**GPU Production (Parallel Enabled):**

```bash
# .env
ADAPTER_TYPE=cosyvoice2
DEFAULT_MODEL=cosyvoice2-en-base
ASR_DEVICE=auto

# Parallel synthesis
PARALLEL_SYNTHESIS_ENABLED=true
PARALLEL_SYNTHESIS_NUM_WORKERS=2
PARALLEL_SYNTHESIS_MAX_QUEUE_DEPTH=10
PARALLEL_SYNTHESIS_GPU_LIMIT=2
```

**High-Performance (3 Workers):**

```bash
# .env (requires 12GB+ VRAM)
ADAPTER_TYPE=cosyvoice2
DEFAULT_MODEL=cosyvoice2-en-base
ASR_DEVICE=auto

# Parallel synthesis (maximum throughput)
PARALLEL_SYNTHESIS_ENABLED=true
PARALLEL_SYNTHESIS_NUM_WORKERS=3
PARALLEL_SYNTHESIS_MAX_QUEUE_DEPTH=15
PARALLEL_SYNTHESIS_GPU_LIMIT=3
```

### Verification

**Check logs after startup:**

```bash
just logs-tail | grep "PARALLEL\|ParallelTTSWrapper"
```

**Expected (Parallel Enabled):**
```
[TTS] gRPC TTS plugin created with PARALLEL synthesis
[TTS] ParallelTTSWrapper initialized (num_workers=2, queue_size=10, gpu_limit=2)
[TTS] Starting 2 persistent TTS workers
```

**Expected (Sequential Mode):**
```
[TTS] gRPC TTS plugin created with SEQUENTIAL synthesis
```

**Verify worker rotation:**

```bash
just logs-tail | grep "Worker.*selected"
```

**Expected (Round-robin):**
```
[TTS] Worker 0 selected for synthesis
[TTS] Worker 1 selected for synthesis
[TTS] Worker 0 selected for synthesis  # Alternating
```

### Performance Metrics

**Baseline (Sequential Mode):**
- Total latency (5 sentences): ~7.5s
- Sentences per second: 0.67 SPS
- Worker utilization: ~45%

**Parallel Mode (2 Workers):**
- Total latency (5 sentences): ~3.5s (53% faster)
- Sentences per second: 1.33 SPS (2x improvement)
- Worker utilization: ~85%

**Parallel Mode (3 Workers):**
- Total latency (5 sentences): ~2.5s (67% faster)
- Sentences per second: 2.00 SPS (3x improvement)
- Worker utilization: ~90%

### Troubleshooting

**Issue: Parallel mode not active**

Check configuration:
```bash
cat .env | grep PARALLEL_SYNTHESIS_ENABLED
# Should show: PARALLEL_SYNTHESIS_ENABLED=true
```

Restart services:
```bash
just dev-stop
just dev-agent-piper
```

**Issue: GPU out of memory**

Reduce worker count:
```bash
# Edit .env
PARALLEL_SYNTHESIS_NUM_WORKERS=1
# OR disable parallel synthesis
PARALLEL_SYNTHESIS_ENABLED=false
```

**Issue: Only one worker active**

Check worker initialization in logs:
```bash
just logs-tail | grep "Starting.*persistent TTS workers"
# Should show: "Starting 2 persistent TTS workers"
```

See [PARALLEL_TTS.md](PARALLEL_TTS.md#troubleshooting) for comprehensive troubleshooting.

---

## Deployment Profiles

### Development (Piper CPU)

Fast iteration, no GPU required:

```bash
# .env
ADAPTER_TYPE=piper
DEFAULT_MODEL=piper-en-us-lessac-medium
ASR_DEVICE=cpu

# Start
just dev-agent-piper
```

**Access Points:**
- Web client: https://localhost:8443
- LiveKit: wss://localhost:8444

**Expected Performance:**
- Startup: ~10 seconds
- First audio latency: <500ms (CPU Piper)
- ASR transcription: ~3s (CPU WhisperX)

### Production (CosyVoice GPU)

High quality, requires NVIDIA GPU:

```bash
# .env
ADAPTER_TYPE=cosyvoice2
DEFAULT_MODEL=cosyvoice2-en-base
ASR_DEVICE=auto

# Start
just dev cosyvoice2
# OR
docker compose --profile cosyvoice up
```

**Requirements:**
- NVIDIA GPU with CUDA 12.1 support
- 4GB+ VRAM
- Docker with NVIDIA runtime

**Expected Performance:**
- Startup: ~15 seconds (model loading)
- First audio latency: <300ms (GPU CosyVoice)
- ASR transcription: <1s (GPU WhisperX)

### Testing (Mock Adapter)

Fast testing with synthetic audio:

```bash
# .env
ADAPTER_TYPE=mock
DEFAULT_MODEL=mock

# Start
just dev
```

**Use Cases:**
- Unit testing
- Integration testing
- CI/CD pipelines
- Protocol validation

---

## Configuration Validation

The system validates configuration at startup and logs warnings for common mismatches.

### Validation Checks

1. **Adapter/Model Compatibility:**
   - `DEFAULT_MODEL` prefix must match `ADAPTER_TYPE`
   - Example: `cosyvoice2-en-base` requires `ADAPTER_TYPE=cosyvoice2`

2. **GPU Availability:**
   - Warns if `ASR_DEVICE=cuda` but CUDA not available
   - Falls back to CPU with performance warning

3. **Voicepack Existence:**
   - Checks if model files exist in voicepack directory
   - Warns if `metadata.yaml` missing or invalid

### Example Warning Messages

```
2025-10-19 03:30:00 - WARNING - Configuration validation found issues:
  1. Model 'cosyvoice2-en-base' requires ADAPTER_TYPE=cosyvoice2, but ADAPTER_TYPE=piper is set
  2. ASR_DEVICE=cuda but CUDA not available - will fall back to CPU
  3. Voicepack not found: voicepacks/cosyvoice/en-base/ (falling back to Mock adapter)
```

### Checking Configuration

```bash
# View current configuration
docker compose exec orchestrator env | grep -E "(ADAPTER|DEFAULT_MODEL|ASR)"

# Expected output (CosyVoice example):
# ADAPTER_TYPE=cosyvoice2
# DEFAULT_MODEL=cosyvoice2-en-base
# ASR_DEVICE=auto
# ASR_MODEL_SIZE=small

# Check startup logs for validation warnings
docker compose logs orchestrator | grep "Configuration validation"
```

---

## Troubleshooting

### Issue: Hearing Beep Tones Instead of Speech

**Symptom:** Client receives 440Hz sine wave beeps instead of synthesized speech.

**Cause:** Mock adapter is being used instead of configured model.

**Root Causes:**
1. `DEFAULT_MODEL` doesn't match `ADAPTER_TYPE`
2. Voicepack files missing or invalid
3. Configuration changes not reloaded

**Solution:**

```bash
# 1. Verify configuration
cat .env | grep -E "(ADAPTER|DEFAULT_MODEL)"

# Should show matching values:
# ADAPTER_TYPE=cosyvoice2
# DEFAULT_MODEL=cosyvoice2-en-base

# 2. Check voicepack exists
ls -la voicepacks/cosyvoice/en-base/
# Should show: model.pt, config.json, metadata.yaml

# 3. Restart services to reload configuration
just dev-stop
just dev
```

### Issue: WhisperX Initialization Takes >30 Seconds

**Symptom:** Long delay before first transcription, high CPU usage during initialization.

**Cause:** Running on CPU instead of GPU.

**Solution:**

```bash
# 1. Set ASR_DEVICE=auto in .env
echo "ASR_DEVICE=auto" >> .env

# 2. Verify CUDA available on host
nvidia-smi

# 3. Verify Docker GPU access
docker compose exec orchestrator nvidia-smi

# 4. Restart services
just dev-stop
just dev

# Expected: 3-5s initialization (vs 28s on CPU)
```

### Issue: Configuration Changes Not Taking Effect

**Symptom:** Editing `.env` file doesn't change runtime behavior.

**Cause:** Services not restarted after configuration change.

**Solution:**

```bash
# Stop all services
just dev-stop

# Verify no processes running
ps aux | grep -E "(orchestrator|tts-worker|livekit)"

# Restart with new configuration
just dev  # Reloads .env automatically
```

### Issue: "Voicepack not found" Warning

**Symptom:** Startup logs show "Voicepack not found for model X" and Mock adapter is used.

**Cause:** CosyVoice model requested but voicepack missing or incomplete.

**Solution:**

```bash
# 1. Download and setup voicepack
./scripts/setup_cosyvoice_voicepack.sh en-base

# 2. Verify directory structure
tree voicepacks/cosyvoice/en-base/
# Expected: model.pt, config.json, metadata.yaml

# 3. Verify DEFAULT_MODEL matches directory name
cat .env | grep DEFAULT_MODEL
# Should show: DEFAULT_MODEL=cosyvoice2-en-base

# 4. Restart services
just dev-stop
just dev
```

See [VOICEPACK_COSYVOICE2.md](VOICEPACK_COSYVOICE2.md) for detailed voicepack setup instructions.

### Issue: Docker Compose Profile Not Working

**Symptom:** `docker compose --profile cosyvoice up` still uses Piper adapter.

**Cause:** Environment variables in `.env` don't match profile selection.

**Solution:**

Docker Compose profiles automatically set environment variables. Verify with:

```bash
# Check what profile sets
docker compose config --profile cosyvoice | grep -A5 environment

# For CosyVoice profile, should show:
# ADAPTER_TYPE=cosyvoice2
# DEFAULT_MODEL=cosyvoice2-en-base

# If manual .env overrides exist, profile variables take precedence
# Remove conflicting variables from .env
```

### Issue: GPU Not Detected in Docker

**Symptom:** `ASR_DEVICE=auto` falls back to CPU in Docker environment.

**Cause:** NVIDIA Docker runtime not configured or GPU not allocated to container.

**Solution:**

```bash
# 1. Verify host GPU access
nvidia-smi

# 2. Check Docker NVIDIA runtime installed
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 3. Verify docker-compose.yml GPU allocation
cat docker-compose.yml | grep -A3 "deploy:"
# Should show:
#   deploy:
#     resources:
#       reservations:
#         devices:
#           - driver: nvidia
#             count: 1
#             capabilities: [gpu]

# 4. Rebuild with GPU support
docker compose down
docker compose up --build
```

---

## See Also

### Core Documentation

- [CLAUDE.md](../CLAUDE.md) - Quick reference and project overview
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development workflows and tools
- [CURRENT_STATUS.md](CURRENT_STATUS.md) - Implementation status and roadmap

### TTS Configuration

- [VOICEPACK_COSYVOICE2.md](VOICEPACK_COSYVOICE2.md) - CosyVoice voicepack structure and setup
- [VOICE_PACKS.md](VOICE_PACKS.md) - Piper voicepack documentation
- [PIPER_TECHNICAL_REFERENCE.md](PIPER_TECHNICAL_REFERENCE.md) - Piper adapter details

### Deployment

- [DOCKER_DEPLOYMENT_COSYVOICE.md](DOCKER_DEPLOYMENT_COSYVOICE.md) - CosyVoice Docker deployment
- [DOCKER_UNIFIED_WORKFLOW.md](DOCKER_UNIFIED_WORKFLOW.md) - Unified development workflow
- [USAGE_DOCKER_PIPER.md](USAGE_DOCKER_PIPER.md) - Piper Docker usage

### Reference

- [.env.example](../.env.example) - All available configuration options with comments
- [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) - Comprehensive configuration reference
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick reference card

---

**Maintained by:** Documentation Engineering Team
**Last Review:** 2025-10-19
**Next Review:** After M12 completion
