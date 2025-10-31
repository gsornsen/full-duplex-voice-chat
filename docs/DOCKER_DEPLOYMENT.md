# Docker Deployment with UV Workspace

**Last Updated**: 2025-10-30
**Status**: ✅ Updated for UV Workspace Migration

## Overview

All Dockerfiles have been updated to work with the new UV workspace structure (`packages/` instead of `src/`). This guide covers Docker-based deployment using the unified `docker-compose.yml` configuration with service profiles.

## Table of Contents

- [Workspace Migration Summary](#workspace-migration-summary)
- [Quick Start](#quick-start)
- [Updated Dockerfiles](#updated-dockerfiles)
- [Build Performance](#build-performance)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Workspace Migration Summary

### Key Changes

**Old Pattern (src/)**:
```dockerfile
COPY src/ ./src/
CMD ["uv", "run", "python", "-m", "src.orchestrator.server"]
```

**New Pattern (packages/)**:
```dockerfile
# Copy workspace files
COPY pyproject.toml uv.lock ./
COPY packages/proto/ ./packages/proto/
COPY packages/orchestrator/ ./packages/orchestrator/

# Install packages (Python 3.12 required)
RUN uv pip install --system -e packages/proto -e packages/orchestrator

# Add to PYTHONPATH for editable installs
ENV PYTHONPATH="/app/packages/proto/src:/app/packages/orchestrator/src:${PYTHONPATH}"

# Run module (no src. prefix)
CMD ["python3", "-m", "orchestrator.server"]
```

### Python Version Requirement

**IMPORTANT**: All workspace packages require **Python 3.12** (was Python 3.10).

All Dockerfiles now install Python 3.12 from deadsnakes PPA:

```dockerfile
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip
```

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

## Updated Dockerfiles

### 1. Dockerfile.orchestrator

**Workspace Packages**:
- `packages/proto/` - gRPC protocol definitions
- `packages/common/` - Shared utilities
- `packages/tts-base/` - TTS base classes
- `packages/orchestrator/` - Orchestrator service

**Key Changes**:
- ✅ Uses Python 3.12 (deadsnakes PPA)
- ✅ Copies from `packages/` instead of `src/`
- ✅ Uses `uv pip install --system` for package installation
- ✅ Sets PYTHONPATH for editable installs (`-e` mode)
- ✅ Module: `orchestrator.server` (not `src.orchestrator.server`)
- ✅ Copies installed packages to runtime stage

**Build Command**:
```bash
docker compose build orchestrator
```

**Runtime Command**:
```bash
CMD ["python3", "-m", "orchestrator.server", "--config", "configs/orchestrator.yaml"]
```

### 2. Dockerfile.tts (Piper)

**Workspace Packages**:
- `packages/proto/` - gRPC protocol definitions
- `packages/common/` - Shared utilities
- `packages/tts-base/` - TTS base classes
- `packages/tts-piper/` - Piper TTS adapter

**Key Changes**:
- ✅ Uses Python 3.12 (deadsnakes PPA)
- ✅ Copies from `packages/` instead of `src/`
- ✅ Uses `uv pip install --system` for package installation
- ✅ Sets PYTHONPATH for editable installs
- ✅ Module: `tts.worker_main` (not `src.tts.worker_main`)

**Build Command**:
```bash
docker compose build tts-piper
```

**Runtime Command**:
```bash
CMD ["python3", "-m", "tts.worker_main"]
```

### 3. Dockerfile.tts-cosyvoice

**Special Considerations**:
- ⚠️ Uses Python 3.10 (CosyVoice requirement - exception to Python 3.12 rule)
- ⚠️ Uses PyTorch 2.3.1 + CUDA 12.1 (isolated from main project's PyTorch 2.7.0)
- ⚠️ Uses protobuf 4.24.4 (conflicts with proto package's protobuf 5.x)

**Solution for Protobuf Conflict**:

Instead of copying generated proto files, we generate them directly in the container:

```dockerfile
# Copy proto source file only
COPY packages/proto/src/rpc/tts.proto ./packages/proto/src/rpc/

# Generate gRPC stubs in container with protobuf 4.24.4
RUN mkdir -p packages/proto/src/rpc/generated && \
    python3 -m grpc_tools.protoc \
    -I packages/proto/src/rpc \
    --python_out=packages/proto/src/rpc/generated \
    --grpc_python_out=packages/proto/src/rpc/generated \
    packages/proto/src/rpc/tts.proto
```

**Key Changes**:
- ✅ Generates proto files in container (avoids version conflict)
- ✅ Manually installs packages with `pip install -e`
- ✅ Sets PYTHONPATH explicitly for all packages
- ✅ No workspace sync (due to protobuf conflict)

**Build Command**:
```bash
docker compose --profile cosyvoice build tts-cosyvoice
```

### 4. Dockerfile.web

**Status**: ⚠️ **Not Updated** (web client removed during migration)

The Next.js web client (`src/client/web/`) was removed during the workspace migration. The Dockerfile.web still references the old paths and will not build until the web client is restored.

**Action Required**:
- Restore web client from git history, OR
- Remove web service from docker-compose.yml temporarily

## Package Installation Pattern

### Multi-Stage Build

All Dockerfiles use a two-stage build for optimal image size:

**Stage 1: Builder** (install dependencies with dev tools)
```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS builder

# Install Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.12 python3.12-dev

# Copy workspace
COPY pyproject.toml uv.lock ./
COPY packages/proto/ ./packages/proto/
COPY packages/orchestrator/ ./packages/orchestrator/

# Install with uv
RUN uv pip install --system \
    -e packages/proto \
    -e packages/orchestrator
```

**Stage 2: Runtime** (minimal production image)
```dockerfile
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Install Python 3.12 (runtime only)
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.12

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages

# Copy source for editable installs
COPY --from=builder /app/packages /app/packages

# Set PYTHONPATH for editable mode
ENV PYTHONPATH="/app/packages/proto/src:/app/packages/orchestrator/src:${PYTHONPATH}"
```

### PYTHONPATH Configuration

For editable installs (`-e` mode), PYTHONPATH must point to package source directories:

```dockerfile
ENV PYTHONPATH="/app/packages/proto/src:/app/packages/common/src:/app/packages/tts-base/src:/app/packages/orchestrator/src:${PYTHONPATH}"
```

This allows Python to find modules like `orchestrator.server` and `rpc.generated.tts_pb2`.

## Build Performance

### BuildKit Cache Mounts

All Dockerfiles use BuildKit cache mounts for faster rebuilds:

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system -e packages/orchestrator
```

**Benefits:**
- **Cold build** (no cache): ~10-15 minutes
- **Warm build** (with cache): ~2-3 minutes
- **No changes**: ~30 seconds (layer cache hit)

### Layer Optimization

Packages copied individually for better layer caching:

```dockerfile
# Root files change rarely
COPY pyproject.toml uv.lock ./

# Proto changes rarely
COPY packages/proto/ ./packages/proto/

# Common changes occasionally
COPY packages/common/ ./packages/common/

# Orchestrator changes frequently
COPY packages/orchestrator/ ./packages/orchestrator/
```

### Build Times (Workspace)

| Service | Cold Build | Warm Build | Cached |
|---------|-----------|------------|--------|
| Orchestrator | ~10 min | ~3 min | ~30s |
| TTS Piper | ~8 min | ~2 min | ~30s |
| TTS CosyVoice | ~12 min | ~4 min | ~30s |

**Cold**: No Docker cache, no BuildKit cache, installing Python 3.12
**Warm**: Docker layers cached, BuildKit cache populated
**Cached**: Image already exists

### Build Commands

```bash
# Rebuild specific service
docker compose build orchestrator
docker compose build tts-piper
docker compose build tts-cosyvoice

# Rebuild all services
docker compose build

# Rebuild with no cache (full rebuild)
docker compose build --no-cache orchestrator

# Rebuild and restart
docker compose up -d --build orchestrator
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'orchestrator'`

**Solution**: Ensure PYTHONPATH is set correctly in Dockerfile:

```dockerfile
ENV PYTHONPATH="/app/packages/orchestrator/src:${PYTHONPATH}"
```

**Verify** in running container:
```bash
docker compose exec orchestrator python3 -c "import sys; print('\\n'.join(sys.path))"
docker compose exec orchestrator python3 -c "from orchestrator.config import Config; print('✓')"
```

### Python Version Mismatch

**Problem**: `Python>=3.12,<3.13 required but Python 3.10 found`

**Cause**: Ubuntu 22.04 ships with Python 3.10 by default

**Solution**: Install Python 3.12 from deadsnakes PPA (already in updated Dockerfiles):

```dockerfile
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev

# Set as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
```

### Package Not Found During Build

**Problem**: `error: Package proto not found`

**Cause**: Workspace package not copied to build context

**Solution**: Ensure all required packages are copied before installation:

```dockerfile
COPY packages/proto/ ./packages/proto/
COPY packages/common/ ./packages/common/
COPY packages/tts-base/ ./packages/tts-base/
```

### Protobuf Version Conflict (CosyVoice)

**Problem**: CosyVoice requires protobuf 4.24.4 but workspace uses 5.x

**Solution**: Generate proto files in container instead of copying:

```dockerfile
# Copy proto source only (not generated files)
COPY packages/proto/src/rpc/tts.proto ./packages/proto/src/rpc/

# Generate in container with correct protobuf version
RUN python3 -m grpc_tools.protoc \
    -I packages/proto/src/rpc \
    --python_out=packages/proto/src/rpc/generated \
    --grpc_python_out=packages/proto/src/rpc/generated \
    packages/proto/src/rpc/tts.proto
```

### Permission Errors

**Problem**: Container fails with permission denied errors

**Solution**: All Dockerfiles now create users and directories with proper permissions. If issues persist:

```bash
# Rebuild from scratch
docker compose build --no-cache orchestrator

# Clean volumes and rebuild
docker compose down -v
docker compose build
docker compose up -d
```

### Build Failures with uv pip install

**Problem**: `error: No virtual environment found`

**Solution**: Use `--system` flag (already in updated Dockerfiles):

```dockerfile
RUN uv pip install --system -e packages/orchestrator
```

This installs into system Python instead of requiring a venv (which is container best practice).

## Testing Builds

### Verify Workspace Migration

```bash
# Build orchestrator
docker compose build orchestrator

# Check logs for errors
docker compose up orchestrator 2>&1 | head -100

# Test imports
docker compose run --rm orchestrator python3 -c "from orchestrator.config import Config; print('✓ Config')"
docker compose run --rm orchestrator python3 -c "from rpc.generated import tts_pb2; print('✓ Proto')"
docker compose run --rm orchestrator python3 -c "from orchestrator.server import main; print('✓ Server')"
```

### End-to-End Test

```bash
# Start full stack
docker compose --profile piper up -d

# Check service health
docker compose ps

# View orchestrator logs
docker logs orchestrator | head -50

# View TTS logs
docker logs tts-piper | head -50

# Test connectivity
docker compose exec orchestrator curl http://localhost:8081/health
```

## Migration Checklist

- [x] Dockerfile.orchestrator updated for workspace
- [x] Dockerfile.tts (Piper) updated for workspace
- [x] Dockerfile.tts-cosyvoice updated for workspace
- [ ] Dockerfile.web - ⚠️ **Blocked** (web client removed during migration)
- [x] Python 3.12 installed in all images (except CosyVoice = 3.10)
- [x] PYTHONPATH configured for editable installs
- [x] Module paths updated (removed `src.` prefix)
- [x] BuildKit cache mounts preserved
- [x] Multi-stage builds maintained
- [x] CosyVoice protobuf conflict resolved (generate in container)

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
```

### Multi-GPU Setup

**Run multiple TTS workers on different GPUs:**

```bash
# Worker 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 docker compose --profile piper up -d tts0

# Worker 2 on GPU 1
CUDA_VISIBLE_DEVICES=1 docker compose -f docker-compose.yml \
  -f docker-compose.gpu1.yml up -d tts1
```

## References

- [UV Workspace Migration Guide](UV_WORKSPACE_MIGRATION_GUIDE.md)
- [UV Workspace Status](UV_WORKSPACE_STATUS.md)
- [UV Workspace Quick Start](UV_WORKSPACE_QUICK_START.md)
- [UV Workspace Summary](UV_WORKSPACE_SUMMARY.md)
- [Docker Compose Profiles](https://docs.docker.com/compose/profiles/)
- [BuildKit Documentation](https://docs.docker.com/build/buildkit/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

**Need Help?**

- Review workspace migration docs (above)
- Check build logs: `docker compose build orchestrator 2>&1 | less`
- Test imports: `docker compose run --rm orchestrator python3 -c "import orchestrator"`
- Clean state: `docker compose down -v && docker compose build --no-cache`
