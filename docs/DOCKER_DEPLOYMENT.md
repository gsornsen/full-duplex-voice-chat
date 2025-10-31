# Docker Deployment Guide

**Last Updated**: 2025-10-28

This guide covers Docker-based deployment using the unified `docker-compose.yml` configuration with service profiles.

## Table of Contents

- [Quick Start](#quick-start)
- [Service Profiles](#service-profiles)
- [Docker Optimizations](#docker-optimizations)
- [Build Performance](#build-performance)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Quick Start

### Infrastructure Only

```bash
# Start Redis, LiveKit, and Caddy
docker compose up -d

# Check status
docker compose ps
```

### Development with Piper (CPU)

```bash
# Using just (recommended)
just dev piper

# Or directly with docker compose
docker compose --profile piper up -d
```

### Development with CosyVoice (GPU)

```bash
# Using just (recommended)
just dev cosyvoice

# Or directly with docker compose
docker compose --profile cosyvoice up -d
```

### Full Stack (All Services)

```bash
# Everything: infrastructure + orchestrator + TTS + web + monitoring
docker compose --profile full-stack up -d
```

## Service Profiles

The unified `docker-compose.yml` uses profiles to organize services:

| Profile | Services Included |
|---------|-------------------|
| (none) | Redis, LiveKit, Caddy (infrastructure only) |
| `piper` | Infrastructure + Orchestrator + Piper TTS Worker |
| `cosyvoice` | Infrastructure + Orchestrator + CosyVoice TTS Worker |
| `web` | + Next.js Web Frontend |
| `monitoring` | + Prometheus + Grafana |
| `full-stack` | All services |

### Combining Profiles

```bash
# Infrastructure + Piper + Web Frontend
docker compose --profile piper --profile web up -d

# Infrastructure + CosyVoice + Monitoring
docker compose --profile cosyvoice --profile monitoring up -d

# Everything
docker compose --profile full-stack up -d
```

## Docker Optimizations

### BuildKit Cache Mounts

All Dockerfiles now use BuildKit syntax with cache mounts:

```dockerfile
# syntax=docker/dockerfile:1.4

# Cache Python packages
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv sync --frozen

# Cache npm packages
RUN --mount=type=cache,target=/root/.npm \
    npm ci
```

**Benefits:**
- **Cold build**: ~5 minutes (first time)
- **Warm build**: ~2 minutes (with cache)
- **No changes**: ~10 seconds (image exists)

### Permission Fixes

All Dockerfiles now:
1. Create non-root user **before** copying files
2. Create cache directories with proper ownership
3. Use `COPY --chown` for all application files
4. Set environment variables for cache directories

**No more permission errors!**

### Layer Optimization

Dependencies copied before application code:

```dockerfile
# 1. Copy dependency files (changes rarely)
COPY pyproject.toml uv.lock ./

# 2. Install dependencies (cached layer)
RUN uv sync --frozen

# 3. Copy application code (changes frequently)
COPY src/ ./src/
```

### .dockerignore

Optimized context to exclude unnecessary files:

- Development artifacts (`.venv`, `__pycache__`, `.pytest_cache`)
- Documentation (`.claude/`, `docs/`, `*.md`)
- IDE files (`.vscode`, `.idea`)
- Logs and temporary files
- Git metadata

**Result:** Faster context transfer to Docker daemon

## Build Performance

### Smart Build Detection

The `justfile` includes smart build detection:

```bash
# Default: Build only if image doesn't exist
just dev piper

# Force rebuild (after code changes)
FORCE_BUILD=true just dev piper

# Never rebuild (fail if missing)
SKIP_BUILD=true just dev piper
```

### Build Metrics

| Operation | Cold | Warm | Cached |
|-----------|------|------|--------|
| Orchestrator | ~180s | ~60s | ~5s |
| Piper TTS | ~180s | ~60s | ~5s |
| CosyVoice TTS | ~300s | ~90s | ~5s |
| Web Frontend | ~120s | ~40s | ~5s |

**Cold**: No Docker cache, no BuildKit cache
**Warm**: Docker cache exists, BuildKit cache populated
**Cached**: Image already exists

### Build Commands

```bash
# Rebuild specific service
docker compose build orchestrator
docker compose build tts0
docker compose build tts-cosyvoice
docker compose build web

# Rebuild all services
docker compose build

# Rebuild with no cache (full rebuild)
docker compose build --no-cache orchestrator

# Rebuild and restart
docker compose up -d --build orchestrator
```

### Helper Commands (Justfile)

```bash
# Rebuild specific TTS worker
just dev-rebuild piper
just dev-rebuild cosyvoice

# Rebuild orchestrator
just dev-rebuild-orch
```

## Troubleshooting

### Permission Errors

**Symptom**: Container fails with permission denied errors

**Solution**: All Dockerfiles now create users and directories with proper permissions. If you encounter issues:

```bash
# Rebuild image from scratch
docker compose build --no-cache orchestrator

# Clean volumes and rebuild
docker compose down -v
docker compose build
docker compose up -d
```

### Build Failures

**Symptom**: Build fails with cache-related errors

**Solution**: Disable BuildKit cache mounts:

```bash
# Set environment variable
export DOCKER_BUILDKIT=0

# Or use legacy build
docker-compose build --no-cache
```

### Slow Builds

**Symptom**: Builds take longer than expected

**Check**:
1. BuildKit enabled: `docker buildx version`
2. Cache mount working: Check build output for "CACHED" steps
3. .dockerignore exists: Reduces context size

**Enable BuildKit** (if not already):

```bash
# Linux/Mac
export DOCKER_BUILDKIT=1

# Or permanently in ~/.bashrc
echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
```

### Out of Memory

**Symptom**: Container killed by OOM (Out of Memory)

**Solution**: Adjust resource limits in `docker-compose.yml`:

```yaml
services:
  orchestrator:
    deploy:
      resources:
        limits:
          memory: 8G  # Increase from 4G
          cpus: '4.0'  # Increase from 2.0
```

### GPU Not Available

**Symptom**: CUDA not found, or GPU not accessible

**Check**:
1. NVIDIA Docker runtime: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`
2. Container runtime: `docker info | grep -i runtime`
3. Compose GPU config: Check `deploy.resources.reservations.devices`

**Fix**:

```bash
# Install NVIDIA container toolkit (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Advanced Configuration

### Environment Variables

Configure services via `.env` file:

```bash
# Copy example
cp .env.example .env

# Edit configuration
nano .env
```

**Key Variables:**

```bash
# Orchestrator Mode
ORCHESTRATOR_MODE=agent  # or legacy

# TTS Configuration
ADAPTER_TYPE=piper  # or cosyvoice2
DEFAULT_MODEL=piper-en-us-lessac-medium

# ASR Configuration
ASR_ENABLED=true
ASR_ADAPTER=whisperx
ASR_DEVICE=auto  # auto, cpu, cuda

# LiveKit
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=devsecret1234567890abcdefghijklmn

# Monitoring
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
```

### Volume Management

**Persistent Volumes:**

```bash
# List volumes
docker volume ls | grep full-duplex

# Inspect volume
docker volume inspect full-duplex-voice-chat_huggingface-cache

# Backup volume
docker run --rm -v full-duplex-voice-chat_huggingface-cache:/data \
  -v $(pwd):/backup alpine tar czf /backup/hf-cache-backup.tar.gz -C /data .

# Restore volume
docker run --rm -v full-duplex-voice-chat_huggingface-cache:/data \
  -v $(pwd):/backup alpine tar xzf /backup/hf-cache-backup.tar.gz -C /data
```

**Clean Volumes:**

```bash
# Remove all volumes (WARNING: deletes model caches)
docker compose down -v

# Remove specific volume
docker volume rm full-duplex-voice-chat_huggingface-cache
```

### Multi-GPU Setup

**Run multiple TTS workers on different GPUs:**

```bash
# Worker 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 docker compose --profile piper up -d tts0

# Worker 2 on GPU 1 (requires custom compose override)
CUDA_VISIBLE_DEVICES=1 docker compose -f docker-compose.yml \
  -f docker-compose.gpu1.yml up -d tts1
```

### Network Debugging

**Access container network:**

```bash
# Execute command in running container
docker compose exec orchestrator bash

# Check connectivity
docker compose exec orchestrator ping tts
docker compose exec orchestrator ping redis

# View logs
docker compose logs -f orchestrator
docker compose logs -f tts0
```

### Health Checks

**Check service health:**

```bash
# All services
docker compose ps

# Specific service
docker compose ps orchestrator

# Health check logs
docker inspect orchestrator | jq '.[0].State.Health'
```

## Monitoring and Observability

### Prometheus Metrics

**Access Prometheus:**

```bash
# Start with monitoring profile
docker compose --profile monitoring up -d

# Access UI
open http://localhost:9090
```

**Key Metrics:**

- `tts_synthesis_latency_seconds` - TTS generation time
- `orchestrator_session_duration_seconds` - Session length
- `asr_transcription_latency_seconds` - ASR processing time

### Grafana Dashboards

**Access Grafana:**

```bash
# Start with monitoring profile
docker compose --profile monitoring up -d

# Access UI (admin/admin)
open http://localhost:3033
```

**Pre-configured Dashboards:**

- TTS Performance
- Orchestrator Sessions
- ASR Metrics
- System Resources

### Log Aggregation

**View logs:**

```bash
# All services
just dev-logs

# Specific service
just dev-logs orchestrator
docker compose logs -f orchestrator

# Follow logs with timestamps
docker compose logs -f --timestamps orchestrator
```

## Production Deployment

### Checklist

- [ ] Set strong passwords in `.env`
- [ ] Configure TLS certificates for Caddy
- [ ] Set up log rotation
- [ ] Configure backup for volumes
- [ ] Set resource limits appropriately
- [ ] Enable monitoring (Prometheus + Grafana)
- [ ] Configure alerts for failures
- [ ] Test disaster recovery procedures

### Security Hardening

```yaml
# docker-compose.prod.yml (example)
services:
  orchestrator:
    # Run as non-root (already configured)
    user: orchestrator

    # Read-only root filesystem
    read_only: true

    # Drop capabilities
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE

    # No new privileges
    security_opt:
      - no-new-privileges:true
```

## Migration from Old Compose Files

### Deprecated Files

The following files are now replaced by the unified `docker-compose.yml`:

- `docker-compose.full-stack.yml` (use `--profile full-stack`)
- `docker-compose.monitoring.yml` (use `--profile monitoring`)

**Migration:**

```bash
# Old way
docker-compose -f docker-compose.full-stack.yml up -d

# New way
docker compose --profile full-stack up -d
```

**Backup files preserved**: `*.yml.bak`

## References

- [Docker Compose Profiles](https://docs.docker.com/compose/profiles/)
- [BuildKit Documentation](https://docs.docker.com/build/buildkit/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

**Need Help?**

- Check existing issues: GitHub Issues
- Review logs: `just dev-logs`
- Clean state: `just dev-clean && just dev piper`
