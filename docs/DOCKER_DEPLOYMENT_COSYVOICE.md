# CosyVoice 2 Docker Deployment Guide

**Last Updated**: 2025-10-17
**Status**: M6 Phase 4
**Target**: Production deployment of CosyVoice 2 TTS adapter with PyTorch 2.3.1 isolation

---

## Overview

This guide covers deploying the CosyVoice 2 TTS adapter in a Docker environment with isolated PyTorch 2.3.1 + CUDA 12.1 dependencies. This isolation is necessary because CosyVoice 2 requires PyTorch 2.3.1, while the main project uses PyTorch 2.7.0 + CUDA 12.8.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Docker Compose Stack                                         │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│ │   Redis     │  │  LiveKit     │  │  Orchestrator     │   │
│ │ (Discovery) │  │  (WebRTC)    │  │  (PyTorch 2.7.0)  │   │
│ └─────────────┘  └──────────────┘  └───────────────────┘   │
│                                              │               │
│                                              ▼               │
│ ┌───────────────────────────────────────────────────────┐   │
│ │        TTS Worker: tts-cosyvoice                      │   │
│ │  ┌─────────────────────────────────────────────────┐  │   │
│ │  │  Isolated PyTorch 2.3.1 + CUDA 12.1             │  │   │
│ │  │  CosyVoice 2 Model (6-8GB VRAM)                 │  │   │
│ │  │  Port: 7002 (gRPC)                              │  │   │
│ │  └─────────────────────────────────────────────────┘  │   │
│ └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 22.04+ recommended) or WSL2
- **GPU**: NVIDIA GPU with 8GB+ VRAM (GTX 1080 Ti, RTX 3080, RTX 4080, RTX 4090, etc.)
- **CUDA**: CUDA-capable GPU with driver version 535+ (for CUDA 12.1)
- **Docker**: Docker Engine 28.x+ with NVIDIA Container Toolkit
- **Disk Space**: ~20GB for Docker images + models

### Software Installation

1. **Install Docker Engine**:
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   newgrp docker  # Or log out and back in
   ```

2. **Install NVIDIA Container Toolkit**:
   ```bash
   # Add NVIDIA package repository
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
       sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   # Install toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

3. **Verify GPU Access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```

   Expected output should show your GPU(s).

---

## Setup

### Step 1: Clone Repository and Prepare Environment

```bash
# Clone repository
cd /path/to/your/workspace
git clone https://github.com/your-org/full-duplex-voice-chat.git
cd full-duplex-voice-chat

# Create environment configuration
cp .env.cosyvoice.example .env.cosyvoice

# Edit .env.cosyvoice if needed (optional)
# nano .env.cosyvoice
```

### Step 2: Download CosyVoice 2 Model

CosyVoice 2 models are available via ModelScope. Use the provided setup script:

```bash
# Run the voicepack setup script
chmod +x scripts/setup_cosyvoice_voicepack.sh
./scripts/setup_cosyvoice_voicepack.sh

# Follow the prompts:
# - Select model variant (e.g., CosyVoice2-0.5B)
# - Choose language (en, zh, multilingual)
# - Confirm download location

# Verify voicepack structure
ls -la voicepacks/cosyvoice/
# Expected:
# voicepacks/cosyvoice/
# └── en-base/
#     ├── model.safetensors
#     ├── config.json
#     └── metadata.yaml
```

**Manual Download** (if script fails):

```bash
# Install ModelScope CLI
pip install modelscope

# Download CosyVoice 2 model
python3 -c "
from modelscope import snapshot_download
model_dir = snapshot_download('iic/CosyVoice2-0.5B', cache_dir='./voicepacks/cosyvoice/en-base')
print(f'Model downloaded to: {model_dir}')
"

# Create metadata.yaml
cat > voicepacks/cosyvoice/en-base/metadata.yaml << EOF
model_id: cosyvoice2-en-base
family: cosyvoice
version: 2.0
language: en
sample_rate: 24000
capabilities:
  streaming: true
  zero_shot: true
  lora: false
  cpu_ok: false
tags:
  - expressive
  - multilingual
  - gpu
EOF
```

### Step 3: Build Docker Image

```bash
# Build CosyVoice Docker image
docker build -f Dockerfile.tts-cosyvoice -t tts-cosyvoice:latest .

# Build time: 10-15 minutes (downloads PyTorch, CUDA libraries)
# Image size: ~8GB
```

**Troubleshooting Build Issues**:

- **Out of disk space**: Clean up Docker: `docker system prune -a`
- **Network timeout**: Increase Docker daemon timeout in `/etc/docker/daemon.json`
- **CUDA mismatch**: Ensure NVIDIA drivers are 535+ for CUDA 12.1

---

## Deployment

### Option A: CosyVoice Only (Isolated)

Run only the CosyVoice service with Redis:

```bash
# Start Redis + CosyVoice
docker compose --profile cosyvoice up --build

# Services started:
# - redis (port 6379, internal)
# - tts-cosyvoice (port 7002)

# Verify service health
docker compose ps
# Should show both containers as "healthy"
```

### Option B: Full Stack (Orchestrator + CosyVoice)

Run the complete system with LiveKit, Orchestrator, and CosyVoice:

```bash
# Start all services
docker compose --profile cosyvoice up --build

# Services started:
# - redis
# - livekit (ports 7880-7882)
# - caddy (ports 80, 8443, 8444)
# - orchestrator (port 8080)
# - tts0 (port 7001, Piper/Mock)
# - tts-cosyvoice (port 7002)

# Check logs
docker compose logs -f tts-cosyvoice

# Expected output:
# tts-cosyvoice  | INFO: Loading model cosyvoice2-en-base...
# tts-cosyvoice  | INFO: Model loaded successfully (24kHz, GPU)
# tts-cosyvoice  | INFO: Worker registered to Redis
# tts-cosyvoice  | INFO: gRPC server listening on port 7002
```

### Managing Services

```bash
# Stop services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v

# Restart specific service
docker compose restart tts-cosyvoice

# View logs
docker compose logs tts-cosyvoice
docker compose logs -f --tail=50 tts-cosyvoice  # Follow logs

# Execute commands in container
docker compose exec tts-cosyvoice bash
```

---

## Configuration

### Environment Variables (.env.cosyvoice)

Key configuration options:

```bash
# Model settings
DEFAULT_MODEL_ID=cosyvoice2-en-base
RESIDENT_CAP=2                 # Max models in VRAM
TTL_MS=600000                  # 10 minutes idle → unload

# Performance tuning
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Reduce fragmentation
ENABLE_FP16=true               # Use FP16 for faster inference

# Logging
LOG_LEVEL=INFO                 # DEBUG for troubleshooting
ENABLE_TELEMETRY=true          # Performance metrics
```

### GPU Allocation

For multi-GPU systems, allocate specific GPUs:

```bash
# .env.cosyvoice
CUDA_VISIBLE_DEVICES=1  # Use second GPU (0-indexed)

# Or in docker-compose.yml:
services:
  tts-cosyvoice:
    environment:
      - CUDA_VISIBLE_DEVICES=1
```

### Resource Limits

Adjust memory limits in `docker-compose.yml`:

```yaml
services:
  tts-cosyvoice:
    deploy:
      resources:
        limits:
          memory: 10G  # Increase if OOM errors
        reservations:
          memory: 6G
          devices:
            - driver: nvidia
              device_ids: ['0']  # Specific GPU
              capabilities: [gpu]
```

---

## Performance Validation

### Step 1: Run Performance Tests

```bash
# With Docker service running, execute performance tests
uv run pytest tests/performance/test_cosyvoice_performance.py -v --gpu

# Tests run:
# - test_first_audio_latency_p95: FAL < 300ms ✅
# - test_frame_jitter_p95: Jitter < 10ms ✅
# - test_concurrent_sessions_jitter: 3 concurrent < 10ms ✅
# - test_model_warmup_time: Warmup < 5s ✅
# - test_throughput_estimation: Chars/s ✅

# Expected results:
# ============================================================
# First Audio Latency (FAL) Results (20 trials):
# ============================================================
#   Mean:   180.23 ms
#   Median: 175.45 ms
#   p95:    245.67 ms (target: < 300 ms) ✅
#   Min:    142.11 ms
#   Max:    298.34 ms
# ============================================================
```

### Step 2: Manual gRPC Testing

```bash
# Install grpcurl for manual testing
go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest

# List available services
grpcurl -plaintext localhost:7002 list

# Get capabilities
grpcurl -plaintext localhost:7002 TTS/GetCapabilities

# Start session
grpcurl -plaintext -d '{"session_id": "test1", "model_id": "cosyvoice2-en-base"}' \
    localhost:7002 TTS/StartSession

# Synthesize (streaming, use grpcurl with -d @stdin)
echo '{"text": "Hello world", "is_final": true}' | \
    grpcurl -plaintext -d @ localhost:7002 TTS/Synthesize
```

### Step 3: Monitoring

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Monitor container resources
docker stats tts-cosyvoice

# Check Redis for worker registration
docker compose exec redis redis-cli
> KEYS worker:*
> GET worker:tts-cosyvoice
```

---

## Troubleshooting

### Issue: Container exits immediately

**Symptoms**:
```bash
docker compose ps
# tts-cosyvoice   Exited (1)
```

**Diagnosis**:
```bash
docker compose logs tts-cosyvoice
```

**Common Causes**:

1. **Missing voicepack**:
   ```
   ERROR: Voicepack not found: /app/voicepacks/cosyvoice/en-base
   ```
   **Fix**: Run `./scripts/setup_cosyvoice_voicepack.sh`

2. **CUDA version mismatch**:
   ```
   ERROR: CUDA runtime version mismatch (host: 12.8, container: 12.1)
   ```
   **Fix**: Update NVIDIA drivers to 535+ or rebuild with matching CUDA

3. **Out of VRAM**:
   ```
   torch.cuda.OutOfMemoryError: CUDA out of memory
   ```
   **Fix**:
   - Close other GPU processes: `nvidia-smi` → kill PIDs
   - Increase `PYTORCH_CUDA_ALLOC_CONF` in .env
   - Reduce `RESIDENT_CAP` to 1

### Issue: Slow synthesis (FAL > 1s)

**Diagnosis**:
```bash
# Check if GPU is being used
docker compose exec tts-cosyvoice nvidia-smi

# Should show python process using GPU
# If not, check CUDA_VISIBLE_DEVICES
```

**Fixes**:

1. **GPU not detected**:
   - Verify Docker GPU access: `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`
   - Check `docker-compose.yml` has `deploy.resources.reservations.devices`

2. **Running on CPU**:
   - Check logs for: `WARNING: CUDA not available, using CPU`
   - Set `CUDA_VISIBLE_DEVICES=0` in .env.cosyvoice

3. **FP32 instead of FP16**:
   - Enable FP16: `ENABLE_FP16=true` in .env.cosyvoice

### Issue: High frame jitter (>20ms)

**Diagnosis**:
```bash
# Run jitter test
uv run pytest tests/performance/test_cosyvoice_performance.py::test_frame_jitter_p95 -v

# Check system load
top
```

**Fixes**:

1. **CPU throttling**: Ensure system not under heavy load
2. **I/O bottleneck**: Use SSD for Docker volumes
3. **Network latency**: Run tests locally (not over network)

### Issue: Redis connection failed

**Symptoms**:
```
ERROR: Failed to connect to Redis at redis://redis:6379
```

**Fix**:
```bash
# Check Redis is running
docker compose ps redis

# Restart Redis
docker compose restart redis

# Verify network
docker compose exec tts-cosyvoice ping redis
```

### Issue: gRPC connection refused

**Symptoms**:
```
grpc._channel._InactiveRpcError: failed to connect to all addresses
```

**Fix**:
```bash
# Check port mapping
docker compose ps tts-cosyvoice
# Should show: 0.0.0.0:7002->7002/tcp

# Test from host
telnet localhost 7002

# Check firewall
sudo ufw status
sudo ufw allow 7002/tcp
```

---

## Production Deployment

### Multi-GPU Setup

For multiple workers on different GPUs:

```yaml
# docker-compose.yml
services:
  tts-cosyvoice-0:
    <<: *cosyvoice-template
    container_name: tts-cosyvoice-0
    ports: ["7002:7002"]
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - WORKER_NAME=tts-cosyvoice-0

  tts-cosyvoice-1:
    <<: *cosyvoice-template
    container_name: tts-cosyvoice-1
    ports: ["7003:7002"]
    environment:
      - CUDA_VISIBLE_DEVICES=1
      - WORKER_NAME=tts-cosyvoice-1
```

### Load Balancing

Use Redis-based service discovery for automatic load balancing across workers. The orchestrator will select the least-busy worker.

### Monitoring & Alerts

Set up Prometheus + Grafana:

```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./configs/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
```

**Metrics to monitor**:
- FAL p95, p99
- Frame jitter p95
- GPU utilization
- VRAM usage
- Request throughput (req/s)
- Error rate

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/cosyvoice-docker.yml
name: CosyVoice Docker CI

on:
  push:
    paths:
      - 'Dockerfile.tts-cosyvoice'
      - 'src/tts/adapters/adapter_cosyvoice.py'

jobs:
  build-and-test:
    runs-on: ubuntu-latest-gpu  # Self-hosted GPU runner
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -f Dockerfile.tts-cosyvoice -t tts-cosyvoice:${{ github.sha }} .

      - name: Start services
        run: docker compose --profile cosyvoice up -d

      - name: Wait for healthy
        run: |
          timeout 120s bash -c 'until docker compose exec -T tts-cosyvoice \
            python3 -c "import grpc; grpc.insecure_channel(\"localhost:7002\").close()"; do sleep 2; done'

      - name: Run performance tests
        run: uv run pytest tests/performance/test_cosyvoice_performance.py -v --gpu

      - name: Cleanup
        if: always()
        run: docker compose down -v
```

---

## Next Steps

1. **M7-M8**: Implement XTTS-v2 and Sesame adapters using the same Docker pattern
2. **M9**: Add intelligent routing to select best worker based on load
3. **M11**: Comprehensive observability with Prometheus/Grafana
4. **M13**: Multi-host deployment with k8s or Docker Swarm

---

## References

- **CosyVoice Repository**: https://github.com/FunAudioLLM/CosyVoice
- **PyTorch Conflict Analysis**: [docs/COSYVOICE_PYTORCH_CONFLICT.md](COSYVOICE_PYTORCH_CONFLICT.md)
- **Voicepack Specification**: [docs/VOICEPACK_COSYVOICE2.md](VOICEPACK_COSYVOICE2.md)
- **NVIDIA Container Toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- **Docker Compose Documentation**: https://docs.docker.com/compose/

---

**Maintained by**: Gerald Sornsen
**Last Review**: 2025-10-17
**Next Review**: After M6 Phase 4 validation
