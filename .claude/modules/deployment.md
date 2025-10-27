# Deployment Module

**Version**: 1.0
**Last Updated**: 2025-10-19

This module provides comprehensive deployment guidance for the Full-Duplex Voice Chat system.

---

## Table of Contents

- [Configuration Management](#configuration-management)
- [Docker Deployment](#docker-deployment)
- [Development Deployment](#development-deployment)
- [Production Deployment](#production-deployment)
- [Multi-GPU Deployment](#multi-gpu-deployment)

---

## Configuration Management

### Environment Variables

All deployment configuration is managed through `.env` file:

```bash
# Create from template
cp .env.example .env

# Edit configuration
nano .env
```

**Critical Variables:**

- `ADAPTER_TYPE`: TTS adapter (piper, cosyvoice2, mock)
- `DEFAULT_MODEL`: Model ID matching adapter type
- `ASR_DEVICE`: WhisperX device (auto, cpu, cuda)
- `ASR_MODEL_SIZE`: Whisper model size (tiny, small, medium)

**Configuration Validation:**

System validates configuration at startup:
- Adapter/model compatibility
- GPU availability for ASR
- Voicepack existence

See [docs/CONFIGURATION.md](../../docs/CONFIGURATION.md) for full guide.

### Profile-to-Model Mapping

| Docker Profile | Adapter | Model ID | GPU Required |
|----------------|---------|----------|--------------|
| (default) | piper | `piper-en-us-lessac-medium` | No |
| cosyvoice | cosyvoice2 | `cosyvoice2-en-base` | Yes (CUDA 12.1) |

**Start with profile:**
```bash
# Piper (CPU)
docker compose up

# CosyVoice (GPU)
docker compose --profile cosyvoice up
```

**⚠️ Important:** Environment variables must match profile!

**Example for CosyVoice profile:**
```bash
# .env
ADAPTER_TYPE=cosyvoice2
DEFAULT_MODEL=cosyvoice2-en-base
ASR_DEVICE=auto
```

If `.env` has conflicting values (e.g., `ADAPTER_TYPE=piper`), the profile's environment variables take precedence in Docker Compose, but it's best to keep `.env` aligned with the intended profile.

---

## Docker Deployment

### Docker Compose Profiles

The system uses Docker Compose profiles to manage different TTS adapters:

```bash
# Default profile (Piper TTS - CPU)
docker compose up

# CosyVoice 2 (GPU, isolated PyTorch 2.3.1 environment)
docker compose --profile cosyvoice up

# Start specific services only
docker compose up redis livekit caddy
```

### Service Architecture

**5 Core Services:**

1. **Redis**: Worker service discovery and heartbeat
2. **LiveKit**: WebRTC media server for browser clients
3. **Caddy**: Reverse proxy with HTTPS/TLS termination
4. **Orchestrator**: Session management, VAD, ASR
5. **TTS Worker**: Model inference (Piper or CosyVoice)

**Network Topology:**

```
Browser Client (HTTPS/WebRTC)
    ↓
Caddy (8443) → LiveKit (7880)
    ↓
Orchestrator (8080/8081)
    ↓ (gRPC)
TTS Worker (7002)
    ↓ (Redis)
Service Discovery (6379)
```

### GPU Configuration

**Piper (CPU):**
- No GPU required
- Low memory footprint (~500 MB)
- Good quality, moderate latency

**CosyVoice (GPU):**
- Requires NVIDIA GPU with CUDA 12.1
- ~1.5 GB VRAM
- High quality, low latency
- Isolated Docker container (PyTorch 2.3.1)

**GPU Allocation in docker-compose.yml:**

```yaml
services:
  tts-cosyvoice:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
```

---

## Development Deployment

### Unified Development Workflow

**Fastest iteration (recommended):**

```bash
# Honcho parallel startup (~10 seconds)
just dev-agent-piper

# Access points:
# - Web client: https://localhost:8443
# - LiveKit: wss://localhost:8444
# - Logs: logs/dev-sessions/dev-agent-piper-YYYYMMDD-HHMMSS.log
```

**Features:**
- Parallel service startup
- Color-coded logs
- Auto-logging with timestamps
- Graceful shutdown (single Ctrl+C)
- Hot-reload friendly

### Docker Compose Development

**Traditional workflow:**

```bash
# Build and start all services
docker compose up --build

# View logs
docker compose logs -f orchestrator

# Stop services
docker compose down
```

**First build:** ~5 minutes
**Subsequent starts:** ~30 seconds

### Configuration Switching

**Switch between Piper and CosyVoice:**

```bash
# Option 1: Environment variables
export ADAPTER_TYPE=cosyvoice2
export DEFAULT_MODEL=cosyvoice2-en-base
just dev

# Option 2: Docker Compose profiles
docker compose --profile cosyvoice up

# Option 3: Edit .env file
nano .env  # Change ADAPTER_TYPE and DEFAULT_MODEL
just dev
```

**⚠️ Important:** Always restart services after configuration changes!

---

## Production Deployment

### Prerequisites

1. **Hardware:**
   - NVIDIA GPU (for CosyVoice/WhisperX GPU)
   - 8GB+ RAM
   - 50GB+ storage

2. **Software:**
   - Docker 20.10+
   - Docker Compose 2.0+
   - NVIDIA Container Toolkit
   - CUDA 12.1+ drivers

3. **Network:**
   - Public IP or domain
   - Ports 80, 443 open
   - TLS certificates

### Production Checklist

**Before deployment:**

- [ ] Configure `.env` with production values
- [ ] Set up TLS certificates in `configs/caddy/`
- [ ] Configure Redis persistence
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log rotation
- [ ] Test failover scenarios
- [ ] Document runbook procedures

**Security:**

- [ ] Enable authentication (API keys)
- [ ] Configure firewall rules
- [ ] Set up mTLS for gRPC (optional)
- [ ] Enable audit logging
- [ ] Disable debug endpoints
- [ ] Review audio retention policy

### Production Configuration

**Recommended .env for production:**

```bash
# TTS Configuration
ADAPTER_TYPE=cosyvoice2
DEFAULT_MODEL=cosyvoice2-en-base

# ASR Configuration
ASR_DEVICE=auto
ASR_MODEL_SIZE=small
ASR_COMPUTE_TYPE=default

# Performance
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security
DEBUG=false
```

### Monitoring

**Health Checks:**

```bash
# Orchestrator health
curl http://localhost:8081/health

# Worker health (via gRPC)
grpcurl -plaintext localhost:7002 health.v1.Health/Check

# Redis health
docker exec redis-tts redis-cli PING
```

**Metrics:**

- First Audio Latency (FAL)
- Frame jitter
- ASR transcription time
- Worker queue depth
- Memory usage
- GPU utilization

See [docs/OBSERVABILITY.md](../../docs/OBSERVABILITY.md) for full monitoring setup.

---

## Multi-GPU Deployment

### Same-Host Multi-GPU

**Use case:** Multiple TTS workers on different GPUs

```bash
# Worker 0 on GPU 0 (Piper CPU fallback)
CUDA_VISIBLE_DEVICES=0 just run-tts-piper

# Worker 1 on GPU 1 (CosyVoice)
CUDA_VISIBLE_DEVICES=1 just run-tts-cosyvoice

# Orchestrator (routes to workers via Redis)
just run-orch
```

**Docker Compose:**

```yaml
services:
  tts-worker-0:
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - WORKER_NAME=tts-worker-0
      - DEFAULT_MODEL=piper-en-us-lessac-medium
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  tts-worker-1:
    environment:
      - CUDA_VISIBLE_DEVICES=1
      - WORKER_NAME=tts-worker-1
      - DEFAULT_MODEL=cosyvoice2-en-base
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
```

### Multi-Host Deployment

**Planned for M13:**

- Cross-host service discovery via Redis
- Worker auto-registration with TTL
- Load balancing policies
- Health monitoring and failover

See [milestones.md](milestones.md) for M13 details.

---

## Troubleshooting

### Common Issues

#### Issue: Configuration Changes Not Applied

**Symptom:** Services still use old configuration after editing `.env`

**Solution:**
```bash
just dev-stop
just dev  # Reloads .env
```

#### Issue: GPU Not Detected

**Symptom:** WhisperX falls back to CPU despite `ASR_DEVICE=auto`

**Solution:**
```bash
# Verify host GPU access
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check container logs
docker compose logs orchestrator | grep "GPU"
```

#### Issue: CosyVoice Model Not Loading

**Symptom:** "Voicepack not found" warning, falls back to Mock adapter

**Solution:**
```bash
# Verify voicepack exists
ls -la voicepacks/cosyvoice/en-base/

# Should show: model.pt, config.json, metadata.yaml

# Verify configuration
cat .env | grep -E "(ADAPTER|DEFAULT_MODEL)"
# Should show:
# ADAPTER_TYPE=cosyvoice2
# DEFAULT_MODEL=cosyvoice2-en-base
```

See [docs/CONFIGURATION.md](../../docs/CONFIGURATION.md) for comprehensive troubleshooting.

---

## See Also

- [docs/CONFIGURATION.md](../../docs/CONFIGURATION.md) - Configuration guide
- [docs/DOCKER_DEPLOYMENT_COSYVOICE.md](../../docs/DOCKER_DEPLOYMENT_COSYVOICE.md) - CosyVoice Docker setup
- [docs/DOCKER_UNIFIED_WORKFLOW.md](../../docs/DOCKER_UNIFIED_WORKFLOW.md) - Unified dev workflow
- [docs/MULTI_GPU.md](../../docs/MULTI_GPU.md) - Multi-GPU deployment
- [development.md](development.md) - Development environment
- [milestones.md](milestones.md) - Implementation roadmap

---

**Maintained by:** DevOps Engineering Team
**Last Review:** 2025-10-19
**Next Review:** After M12 completion
