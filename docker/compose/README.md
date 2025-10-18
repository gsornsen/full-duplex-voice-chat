# Docker Compose Profile-Based Architecture

This directory contains a modular Docker Compose configuration for the Realtime Duplex Voice Demo project. The architecture supports hot-swappable TTS model workers via Docker Compose profiles.

## Architecture Overview

### Design Principles

1. **Separation of Concerns**
   - Infrastructure services (Redis, LiveKit, Caddy) → Always running
   - Model workers (Piper, CosyVoice, XTTS) → Swappable by profile
   - Orchestrator → Depends on generic "tts" service alias

2. **Profile-Based Model Selection**
   - Docker Compose profiles for each model variant
   - Default profile: `piper` (CPU, no GPU required)
   - Optional profiles: `cosyvoice`, `xtts`, `sesame`, `openai`

3. **Idempotent Operations**
   - Can switch models without full restart
   - Infrastructure remains running during worker swaps
   - Orchestrator auto-reconnects via Redis service discovery

## File Structure

```
docker/compose/
├── docker-compose.yml        # Base infrastructure (Redis, LiveKit, Caddy, Orchestrator)
├── docker-compose.models.yml # Model worker variants (profile-based)
├── docker-compose.dev.yml    # Development overrides (hot reload, debug logging)
└── README.md                 # This file
```

## Quick Start

### 1. Infrastructure Only

Start just the infrastructure services (no TTS workers):

```bash
docker compose -f docker/compose/docker-compose.yml up
```

This starts:
- Redis (service discovery)
- LiveKit (WebRTC transport)
- Caddy (HTTPS proxy)
- Orchestrator (application logic)

### 2. With Piper Worker (CPU, Default)

```bash
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  --profile piper \
  up
```

Or use the shorthand (default profile):

```bash
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  up
```

### 3. With CosyVoice Worker (GPU)

```bash
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  --profile cosyvoice \
  up
```

### 4. Development Mode (with hot reload)

```bash
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  -f docker/compose/docker-compose.dev.yml \
  --profile piper \
  up
```

## Model Switching (Idempotent)

### Scenario: Switch from Piper to CosyVoice

**Step 1**: Start with Piper
```bash
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  --profile piper \
  up -d
```

**Step 2**: Stop Piper worker only
```bash
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  stop tts-piper
```

**Step 3**: Start CosyVoice worker
```bash
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  --profile cosyvoice \
  up -d tts-cosyvoice
```

**Result**: Infrastructure and orchestrator remain running, only the worker is swapped.

### Scenario: Full Restart with Different Model

```bash
# Stop everything
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  down

# Start with new model
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  --profile cosyvoice \
  up -d
```

## Service Dependencies

### Dependency Graph

```
Infrastructure Layer (Always Running)
├─ redis (no dependencies)
├─ livekit (no dependencies)
└─ caddy (depends_on: livekit)

Worker Layer (Profile-Based)
├─ tts-piper (depends_on: redis) [profile: piper, default]
└─ tts-cosyvoice (depends_on: redis) [profile: cosyvoice]

Application Layer
└─ orchestrator (depends_on: redis, livekit)
   # Note: No hard dependency on TTS worker
   # Uses Redis service discovery for worker connections
```

### Network Aliases

Both workers share the same network alias: `tts`

This allows the orchestrator to connect to whichever worker is active without code changes.

```yaml
networks:
  tts-network:
    aliases:
      - tts  # Generic alias (no version suffix)
```

## Environment Configuration

### Loading Order

1. **Base configuration**: Environment variables in `docker-compose.yml`
2. **Model-specific**: `.env.models/.env.{model}` loaded via `env_file`
3. **Development overrides**: `docker-compose.dev.yml` (if used)

### Model-Specific Env Files

```
.env.models/
├── .env.piper       # Piper worker configuration
└── .env.cosyvoice   # CosyVoice worker configuration
```

Example `.env.models/.env.piper`:
```bash
WORKER_NAME=tts-piper
WORKER_GRPC_PORT=7001
DEFAULT_MODEL_ID=piper-en-us-lessac-medium
MAX_CONCURRENT_SESSIONS=4
```

## Service Details

### Redis

- **Image**: `redis:7-alpine`
- **Container**: `redis-tts`
- **Port**: 6379 (internal only, no host mapping)
- **Purpose**: Service discovery and state management

### LiveKit

- **Image**: `livekit/livekit-server:latest`
- **Container**: `livekit-server`
- **Ports**:
  - 7880: WebRTC/WebSocket
  - 7881: RTC TCP
  - 7882: TURN/UDP
  - 50000-50099: RTC port range
- **Config**: `../../configs/livekit.yaml`

### Caddy

- **Image**: `caddy:2-alpine`
- **Container**: `caddy-proxy`
- **Ports**:
  - 8443: HTTPS for web client
  - 8444: HTTPS for LiveKit WebSocket
  - 80: HTTP redirects
- **Purpose**: Reverse proxy with automatic HTTPS

### Orchestrator

- **Build**: `Dockerfile.orchestrator`
- **Container**: `orchestrator`
- **Ports**:
  - 8080: WebSocket API
  - 8081: Health check HTTP
- **GPU**: Requires NVIDIA GPU (for ASR/WhisperX)
- **Discovery**: Finds TTS workers via Redis

### TTS Workers

#### Piper (CPU)

- **Build**: `Dockerfile.tts`
- **Container**: `tts-piper`
- **Ports**:
  - 7001: gRPC
  - 9090: Metrics
- **Profile**: `piper`, `default`
- **GPU**: Not required
- **VRAM**: N/A
- **Startup**: ~30s

#### CosyVoice 2 (GPU)

- **Build**: `Dockerfile.tts-cosyvoice`
- **Container**: `tts-cosyvoice`
- **Ports**:
  - 7002: gRPC
  - 9091: Metrics
- **Profile**: `cosyvoice`
- **GPU**: Required (NVIDIA)
- **VRAM**: ~6-8GB
- **Startup**: ~90s (model loading)
- **PyTorch**: 2.3.1 (isolated from main project's 2.7.0)

## Health Checks

All services have health checks configured:

```bash
# Check all service health
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  --profile piper \
  ps

# Example output:
# redis-tts        redis:7-alpine   "docker-entrypoint..."   Up (healthy)
# livekit-server   livekit/...      "livekit-server..."      Up (healthy)
# caddy-proxy      caddy:2-alpine   "caddy run..."           Up (healthy)
# orchestrator     ...              "uv run python..."       Up (healthy)
# tts-piper        ...              "uv run python..."       Up (healthy)
```

### Health Check Endpoints

- **Redis**: `redis-cli ping` (internal)
- **LiveKit**: `http://localhost:7880/` (internal)
- **Caddy**: `http://localhost:2019/config/` (internal)
- **Orchestrator**: `http://localhost:8081/health` (exposed)
- **TTS Workers**: TCP connection test to gRPC port (internal)

## Debugging

### View Logs

```bash
# All services
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  --profile piper \
  logs -f

# Specific service
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  logs -f orchestrator

# Just the current worker
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  logs -f tts-piper
```

### Shell Access

```bash
# Orchestrator
docker exec -it orchestrator bash

# Current worker (Piper)
docker exec -it tts-piper bash

# Redis CLI
docker exec -it redis-tts redis-cli
```

### Redis Service Discovery

Check registered workers:

```bash
docker exec -it redis-tts redis-cli
> KEYS tts:workers:*
> GET tts:workers:tts-piper
```

## Cleanup

### Stop Services (preserve volumes)

```bash
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  down
```

### Stop Services (remove volumes)

```bash
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  down -v
```

### Remove Orphaned Containers

```bash
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  down --remove-orphans
```

## Testing Idempotency

### Test 1: Infrastructure Startup

```bash
# Start infrastructure
docker compose -f docker/compose/docker-compose.yml up -d

# Verify all healthy
docker compose -f docker/compose/docker-compose.yml ps

# Expected: redis, livekit, caddy, orchestrator all healthy
```

### Test 2: Add Piper Worker

```bash
# Start Piper (infrastructure already running)
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  --profile piper \
  up -d tts-piper

# Verify Piper healthy
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  ps tts-piper
```

### Test 3: Switch to CosyVoice

```bash
# Stop Piper
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  stop tts-piper

# Start CosyVoice
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  --profile cosyvoice \
  up -d tts-cosyvoice

# Verify infrastructure still running
docker compose -f docker/compose/docker-compose.yml ps

# Verify CosyVoice healthy
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  ps tts-cosyvoice
```

### Test 4: Clean Restart

```bash
# Stop everything
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  down

# Start fresh with Piper
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  --profile piper \
  up -d

# Verify all healthy
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  ps
```

## Justfile Integration

For convenience, use the project's justfile commands:

```bash
# Start with default model (Piper)
just dev

# Start with specific model
just dev-cosyvoice

# Switch models (idempotent)
just dev-switch cosyvoice

# View logs
just dev-logs
just dev-logs-worker piper

# Stop services
just dev-stop
just dev-clean  # Also removes volumes
```

See `justfile` for full command reference.

## Troubleshooting

### Issue: Orchestrator can't find TTS worker

**Symptom**: Orchestrator logs show "No TTS workers available"

**Solution**: Verify worker is registered in Redis
```bash
docker exec -it redis-tts redis-cli KEYS "tts:workers:*"
```

If empty, restart the worker:
```bash
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  --profile piper \
  restart tts-piper
```

### Issue: Port conflict (7001 already in use)

**Symptom**: `Error starting userland proxy: listen tcp4 0.0.0.0:7001: bind: address already in use`

**Solution**: Stop the conflicting service or change the port mapping in `docker-compose.models.yml`

### Issue: GPU not available in container

**Symptom**: CosyVoice worker fails with "CUDA not available"

**Solution**: Verify NVIDIA container runtime is installed
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

If fails, install NVIDIA container toolkit:
```bash
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Issue: Worker health check fails

**Symptom**: Worker stuck in "starting (unhealthy)" state

**Solution**: Check worker logs for errors
```bash
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.models.yml \
  logs tts-piper

# Look for startup errors, missing models, or port conflicts
```

## Future Enhancements

### Planned Model Workers

- **XTTS**: GPU-based TTS with voice cloning support
- **Sesame**: High-performance streaming TTS
- **OpenAI**: Cloud-based TTS API proxy (no GPU required)

Each will follow the same profile-based pattern.

### Multi-GPU Support

Future `docker-compose.multi-gpu.yml` will enable running multiple workers simultaneously:

```yaml
services:
  tts-cosyvoice-gpu0:
    extends:
      service: tts-cosyvoice
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - WORKER_NAME=tts-cosyvoice-gpu0
    ports:
      - "7002:7002"

  tts-xtts-gpu1:
    extends:
      service: tts-xtts
    environment:
      - CUDA_VISIBLE_DEVICES=1
      - WORKER_NAME=tts-xtts-gpu1
    ports:
      - "7003:7003"
```

## References

- **Design Document**: `/tmp/unified_dev_workflow_design.md`
- **Dockerfiles**: `/home/gerald/git/full-duplex-voice-chat/Dockerfile.*`
- **Project Documentation**: `/home/gerald/git/full-duplex-voice-chat/CLAUDE.md`
- **Environment Files**: `/home/gerald/git/full-duplex-voice-chat/.env.models/`

---

**Last Updated**: 2025-10-17
**Author**: DevOps Engineer (Claude Code)
**Status**: Phase 2 Complete - Ready for Phase 3 (Python-Pro DEFAULT_MODEL support)
