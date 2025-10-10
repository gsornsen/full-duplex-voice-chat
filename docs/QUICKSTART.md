# Quick Start Guide

## Overview

Get the Realtime Duplex Voice Demo system up and running in **under 30 minutes**. This guide provides the fastest path from zero to working TTS synthesis.

**What You'll Get:**
- WebSocket server for TTS synthesis
- CLI client for testing
- Mock TTS worker (440Hz sine wave)
- Redis service discovery

**Milestone:** M2 - WebSocket Transport + Mock Worker

---

## Prerequisites Checklist

Before starting, ensure you have:

### Required
- [ ] **Python 3.13+** - Check: `python3 --version`
- [ ] **uv package manager** - Check: `uv --version` (install: https://github.com/astral-sh/uv)
- [ ] **Docker Engine** - Check: `docker --version` (≥ 24.0)
- [ ] **Docker Compose** - Check: `docker compose version`
- [ ] **Git** - Check: `git --version`

### Optional (for GPU workers)
- [ ] **NVIDIA GPU** - 8GB+ VRAM
- [ ] **NVIDIA Container Runtime** - Check: `docker run --rm --gpus all nvidia/cuda:12.8.0-base nvidia-smi`

### Port Availability

**Docker Compose (Recommended):**

Only the following ports need to be available on your host:
- [ ] **7001** - TTS Worker gRPC (external access optional)
- [ ] **8080** - Orchestrator WebSocket
- [ ] **9090** - Metrics (external access optional)

**Note:** Redis runs on Docker's internal network and does **not** require port 6379 to be available on your host. This prevents conflicts with existing Redis/Valkey instances.

Check ports:
```bash
sudo lsof -i :7001,8080
# No output = ports available ✓
```

**Local Development (Without Docker):**

If running services directly on your host, you'll also need:
- [ ] **6379** - Redis (or use existing instance)

**Need Help?**
- [Docker Setup Guide](setup/DOCKER_SETUP.md) - Complete Docker installation
- [Redis Configuration Guide](REDIS_CONFIGURATION.md) - Redis setup and troubleshooting
- [Pre-flight Check Script](../scripts/preflight_check.sh) - Automated validation (coming soon)

---

## Quick Start (Docker Compose - Recommended)

**Best for:** First-time users, M2 demonstration

### Step 1: Clone Repository

```bash
cd ~/git  # Or your preferred directory
git clone https://github.com/your-org/full-duplex-voice-chat.git
cd full-duplex-voice-chat
```

### Step 2: Start the System

```bash
docker compose up --build
```

**What's Happening:**
- Building Docker images (~2-5 minutes first time)
- Starting Redis (internal network only - no host port conflicts)
- Starting LiveKit server
- Starting orchestrator and TTS worker
- Containers starting in dependency order

**Expected Output:**
```
[+] Building 45.2s (24/24) FINISHED
[+] Running 4/4
 ✔ Container redis-tts      Started                                    0.5s
 ✔ Container livekit-server Started                                    0.8s
 ✔ Container tts-worker-0   Started                                    1.8s
 ✔ Container orchestrator   Started                                    2.2s
```

### Step 3: Verify Services Running

Open a new terminal:

```bash
docker ps
```

**Expected Output:**
```
CONTAINER ID   IMAGE                    STATUS         PORTS
abc123...      full-duplex-orch         Up 1 minute    0.0.0.0:8080->8080/tcp
def456...      full-duplex-tts-worker   Up 1 minute    0.0.0.0:7001->7001/tcp
ghi789...      livekit/livekit-server   Up 1 minute    0.0.0.0:7880->7880/tcp
jkl012...      redis:7-alpine           Up 1 minute    (internal network only)
```

**Note:** Redis container shows no external port mapping - this is correct! Services communicate via Docker's internal network.

**All 4 containers running?** ✓ Proceed to Step 4

**Containers missing or exited?** See [Troubleshooting](#troubleshooting)

### Step 4: Test with CLI Client

Install Python dependencies (one-time):

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync --all-extras
```

Run CLI client:

```bash
just cli
# Or: python -m src.client.cli_client --host ws://localhost:8080
```

**Expected Output:**
```
Connected to ws://localhost:8080

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

### Step 5: Synthesize Text

```
You: Hello world, this is a test!
.......
✓ Session ended: completed
```

**Hearing a 440Hz sine wave?** ✓ Success! Mock worker is running.

**Note:** M2 uses a mock worker that generates a sine wave instead of real speech. Real TTS adapters (CosyVoice, XTTS, etc.) are coming in M4-M8.

### Step 6: Test Barge-In (Optional)

```
You: This is a very long sentence that will take several seconds to synthesize.
......
You: /pause
[Audio stops]
You: /resume
......
✓ Session ended: completed
```

**Pause latency < 50ms?** ✓ Barge-in working!

---

## Local Development Setup

**Best for:** Development, debugging, code changes

### Step 1: Clone and Install

```bash
# Clone repository
git clone https://github.com/your-org/full-duplex-voice-chat.git
cd full-duplex-voice-chat

# Install dependencies with uv
uv sync --all-extras

# Generate gRPC stubs
just gen-proto
```

### Step 2: Start Redis

**Option A: Use existing Redis/Valkey instance**

If you already have Redis running on port 6379:
```bash
# Test connectivity
redis-cli ping
# Expected: PONG

# No further action needed - services will use redis://localhost:6379
```

**Option B: Start Redis in Docker**

```bash
# Terminal 1: Start Redis container
just redis
```

**Verify Redis:**
```bash
docker exec -it redis redis-cli ping
# Expected: PONG
```

**Note:** For detailed Redis setup and troubleshooting, see [Redis Configuration Guide](REDIS_CONFIGURATION.md).

### Step 3: Start TTS Worker

```bash
# Terminal 2: Start mock TTS worker
just run-tts-mock

# Or with environment overrides
LOG_LEVEL=DEBUG WORKER_NAME=tts-worker-0 just run-tts-mock
```

**Expected Output:**
```
[INFO] Worker tts-worker-0 starting...
[INFO] Loaded mock adapter (440Hz sine wave)
[INFO] Registered with Redis
[INFO] gRPC server listening on 0.0.0.0:7001
```

### Step 4: Start Orchestrator

```bash
# Terminal 3: Start orchestrator
just run-orch

# Or with environment overrides
LOG_LEVEL=DEBUG PORT=8080 just run-orch
```

**Expected Output:**
```
[INFO] Orchestrator starting...
[INFO] WebSocket transport enabled on 0.0.0.0:8080
[INFO] Connected to Redis: redis://localhost:6379
[INFO] Static worker configured: grpc://localhost:7001
[INFO] Server ready
```

### Step 5: Test with CLI Client

```bash
# Terminal 4: Run CLI client
just cli

# Or with verbose logging
python -m src.client.cli_client --host ws://localhost:8080 --verbose
```

**Workflow:**
```
You: Test message
.......
✓ Session ended: completed

You: /quit
```

---

## Verification Checkpoints

### Checkpoint 1: Docker Running (Step 2)

**Check:**
```bash
docker ps
```

**Success:** 4 containers running (redis-tts, livekit-server, orchestrator, tts-worker-0)

**Failure:** See [Docker Troubleshooting](setup/DOCKER_SETUP.md#common-errors-and-resolutions)

---

### Checkpoint 2: WebSocket Accessible (Step 3)

**Check:**
```bash
# Using websocat (install: cargo install websocat)
echo '{"type":"text","text":"test","is_final":true}' | websocat ws://localhost:8080

# Or using curl (WebSocket upgrade)
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  http://localhost:8080
```

**Success:** Connection established, receive session_start message

**Failure:** Check orchestrator logs:
```bash
docker logs orchestrator
# Or for local: check terminal 3 output
```

---

### Checkpoint 3: Worker Connected (Step 3)

**Check Redis worker registration:**

**For Docker:**
```bash
docker compose exec redis redis-cli

> KEYS worker:*
# Expected: 1) "worker:tts-worker-0"

> GET worker:tts-worker-0
# Expected: JSON with worker info

> TTL worker:tts-worker-0
# Expected: (integer) 25 (or similar, should refresh)
```

**For local development:**
```bash
redis-cli

> KEYS worker:*
# Expected: 1) "worker:tts-worker-0"
```

**Success:** Worker key exists and has TTL

**Failure:** Check worker logs:
```bash
docker logs tts-worker-0
# Or for local: check terminal 2 output
```

**Note:** If Redis connection fails, see [Redis Configuration Guide](REDIS_CONFIGURATION.md) for troubleshooting.

---

### Checkpoint 4: Audio Synthesis Working (Step 4-5)

**Check:**
```bash
# Run CLI client
just cli

# Send text
You: Test
```

**Success Indicators:**
- Session started message appears
- Progress dots (`.`) appear for each audio frame
- Session ended message appears
- Audio plays (440Hz tone) or files saved

**Failure:** See [Troubleshooting](#troubleshooting) below

---

## Troubleshooting

### Connection Refused

**Symptom:**
```
Connection failed: [Errno 111] Connection refused
```

**Causes & Solutions:**

1. **Orchestrator not running**
   ```bash
   # Check container
   docker ps | grep orchestrator

   # Check logs
   docker logs orchestrator

   # Restart if needed
   docker compose restart orchestrator
   ```

2. **Wrong port**
   ```bash
   # Verify port in config
   grep "port:" configs/orchestrator.yaml
   # Should be 8080

   # Use correct port
   just cli HOST="ws://localhost:8080"
   ```

3. **Firewall blocking**
   ```bash
   # Test port accessibility
   nc -zv localhost 8080

   # Allow port (if needed)
   sudo ufw allow 8080/tcp
   ```

---

### Redis Connection Issues

**Symptom:**
```
redis.exceptions.ConnectionError: Error connecting to redis:6379
```

**Docker Environment:**

Redis runs on Docker's internal network. Services should use:
- `REDIS_URL=redis://redis:6379` (NOT `redis:6380`)

**Check configuration:**
```bash
# Verify environment variables
docker compose config | grep REDIS_URL
# Should show: REDIS_URL=redis://redis:6379

# Test Redis connectivity from orchestrator
docker compose exec orchestrator redis-cli -h redis -p 6379 ping
# Expected: PONG
```

**Local Development:**

If running services locally, ensure Redis is accessible:
```bash
# Test connection
redis-cli -h localhost -p 6379 ping
# Expected: PONG

# Check if Redis is running
sudo systemctl status redis-server
# Or: docker ps | grep redis
```

**See:** [Redis Configuration Guide](REDIS_CONFIGURATION.md) for detailed troubleshooting

---

### No Audio Frames Received

**Symptom:**
```
🔗 Session started: ...
[No audio frames, session hangs]
```

**Causes & Solutions:**

1. **Worker not connected**
   ```bash
   # Check worker status
   docker ps | grep tts-worker

   # Check worker logs
   docker logs tts-worker-0

   # Verify registration
   docker compose exec redis redis-cli
   > KEYS worker:*
   ```

2. **Redis not accessible**
   ```bash
   # Check Redis
   docker ps | grep redis

   # Test connection (from inside Docker network)
   docker compose exec orchestrator redis-cli -h redis -p 6379 ping
   # Expected: PONG

   # Restart if needed
   docker compose restart redis
   ```

3. **gRPC connection failure**
   ```bash
   # Check orchestrator → worker connectivity
   docker logs orchestrator | grep -i grpc

   # Verify worker address in config
   grep "static_worker_addr" configs/orchestrator.docker.yaml
   # Should be: grpc://tts0:7001 (Docker Compose)

   # Or for local dev:
   grep "static_worker_addr" configs/orchestrator.yaml
   # Should be: grpc://localhost:7001
   ```

**Quick Fix:**
```bash
# Restart all services
docker compose down
docker compose up --build
```

---

### Container Exits Immediately

**Symptom:**
```
tts-worker exited with code 1
```

**Diagnosis:**
```bash
# Check logs
docker logs tts-worker-0

# Common issues:
# - Port already in use
# - Configuration file error
# - Missing voicepacks directory
# - GPU not available (if using GPU image)
```

**Solutions:**

1. **Port conflict**
   ```bash
   # Find process using port
   sudo lsof -i :7001

   # Kill process or change port
   # Edit docker-compose.yml or worker.yaml
   ```

2. **Configuration error**
   ```bash
   # Validate YAML syntax
   python -c "import yaml; yaml.safe_load(open('configs/worker.yaml'))"

   # Check required fields
   grep "default_model_id" configs/worker.yaml
   ```

3. **GPU not available**
   ```bash
   # Test GPU in container
   docker run --rm --gpus all nvidia/cuda:12.8.0-base nvidia-smi

   # If fails, see Docker Setup Guide
   ```

---

### sounddevice Not Available

**Symptom:**
```
WARNING: sounddevice not available, audio will be saved to file
```

**Cause:** `sounddevice` not installed (audio playback library)

**Solution:**
```bash
# Install sounddevice
uv pip install sounddevice

# Or with pip
pip install sounddevice

# Verify
python -c "import sounddevice as sd; print('✓ sounddevice available')"
```

**Alternative:** Use file output mode (audio saved to `.wav` files)

---

### Permission Denied (Docker)

**Symptom:**
```
Got permission denied while trying to connect to the Docker daemon socket
```

**Cause:** User not in docker group

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or use:
newgrp docker

# Verify
docker run hello-world
```

**See:** [Docker Setup Guide - User Permissions](setup/DOCKER_SETUP.md#user-permissions-linux)

---

## Next Steps

### Explore Features

**Test Control Commands:**
```
You: Long text to synthesize...
You: /pause   # Test pause functionality
You: /resume  # Test resume
You: /stop    # Stop current synthesis
You: /quit    # Exit client
```

**Measure Latency:**
```bash
# Run with verbose logging
python -m src.client.cli_client --host ws://localhost:8080 --verbose

# Look for timing information in logs
```

**Review Logs:**
```bash
# Orchestrator logs
docker logs orchestrator -f

# Worker logs
docker logs tts-worker-0 -f

# Redis logs
docker logs redis-tts -f
```

---

### Learn More

**Documentation:**
- [WebSocket Protocol Specification](WEBSOCKET_PROTOCOL.md) - Build custom clients
- [CLI Client Usage Guide](CLI_CLIENT_GUIDE.md) - Advanced CLI usage
- [Configuration Reference](CONFIGURATION_REFERENCE.md) - Customize settings
- [Docker Setup Guide](setup/DOCKER_SETUP.md) - Docker troubleshooting
- [Redis Configuration Guide](REDIS_CONFIGURATION.md) - Redis setup and troubleshooting

**Architecture:**
- [Architecture Diagrams](architecture/ARCHITECTURE.md) - System design
- [Component READMEs](../src/README.md) - Code structure

**Development:**
- [Testing Guide](TESTING_GUIDE.md) - Run and write tests
- [CLAUDE.md](../CLAUDE.md) - Development environment
- [Contributing Guide](../CONTRIBUTING.md) - Contribution guidelines

---

### Upgrade to Real TTS

**Current:** M2 - Mock worker (440Hz sine wave)

**Coming Soon:**
- **M4:** Model Manager with dynamic loading
- **M5:** Piper adapter (CPU-based TTS)
- **M6:** CosyVoice 2 adapter (GPU streaming TTS)
- **M7:** XTTS-v2 adapter (GPU with voice cloning)
- **M8:** Sesame/Unsloth adapter (LoRA support)

**Stay Updated:**
```bash
# Pull latest changes
git pull origin main

# Rebuild containers
docker compose down
docker compose up --build
```

---

## Common Workflows

### Development Workflow

```bash
# 1. Start infrastructure
just redis

# 2. Start worker (with code changes)
# Edit src/tts/worker.py
just run-tts-mock

# 3. Start orchestrator (with code changes)
# Edit src/orchestrator/server.py
just run-orch

# 4. Test changes
just cli

# 5. Run quality checks
just lint
just typecheck
just test

# 6. Commit changes
git add .
git commit -m "feat: add feature X"
```

### Testing Workflow

```bash
# Run all tests
just test

# Run specific test file
pytest tests/unit/test_vad.py

# Run integration tests (requires Docker)
pytest tests/integration/

# With coverage
pytest --cov=src tests/

# Verbose output
pytest -vv tests/
```

### Production Deployment (Docker Compose)

```bash
# 1. Clone on production server
git clone <repo> /opt/tts-system
cd /opt/tts-system

# 2. Configure for production
cp configs/orchestrator.yaml configs/orchestrator.prod.yaml
# Edit production settings

# 3. Start with production config
docker compose -f docker-compose.prod.yml up -d

# 4. Monitor logs
docker compose logs -f

# 5. Health checks
curl http://localhost:8080/health
# Expected: {"status":"healthy"}

# 6. Stop/restart
docker compose restart
docker compose down
```

---

## Performance Targets

**M2 Baseline:**
- **Barge-in latency:** < 50ms (pause command to synthesis stop)
- **WebSocket latency:** < 10ms (message delivery)
- **Frame rate:** 50 fps (20ms frames @ 48kHz)
- **Concurrent sessions:** 3 per worker (mock adapter)

**Measuring Performance:**
```bash
# Run with verbose logging
python -m src.client.cli_client --verbose

# Send text and observe timestamps
You: Test message

# Check logs for:
# - Session start time
# - First frame received time (FAL)
# - Frame receive intervals
# - Pause/resume latency
```

---

## Getting Help

### Check Logs

```bash
# All services
docker compose logs

# Specific service
docker compose logs orchestrator
docker compose logs tts-worker-0
docker compose logs redis-tts

# Follow logs (tail -f)
docker compose logs -f

# Last 100 lines
docker compose logs --tail=100
```

### Run Health Checks

```bash
# Orchestrator health (coming in P0-2)
curl http://localhost:8080/health

# Worker health (gRPC health check)
# Requires grpcurl: https://github.com/fullstorydev/grpcurl
grpcurl -plaintext localhost:7001 grpc.health.v1.Health/Check

# Redis health (from inside Docker network)
docker compose exec redis redis-cli ping
# Expected: PONG
```

### Common Issues Links

- [Docker Setup Troubleshooting](setup/DOCKER_SETUP.md#common-errors-and-resolutions)
- [Redis Configuration Guide](REDIS_CONFIGURATION.md) - Port conflicts, connectivity
- [WebSocket Protocol Errors](WEBSOCKET_PROTOCOL.md#troubleshooting)
- [CLI Client Issues](CLI_CLIENT_GUIDE.md#troubleshooting)
- [Configuration Problems](CONFIGURATION_REFERENCE.md#validation)

### Community Support

- **GitHub Issues:** https://github.com/your-org/full-duplex-voice-chat/issues
- **Discussions:** https://github.com/your-org/full-duplex-voice-chat/discussions
- **Discord:** [Link to Discord server]

---

## Changelog

**v0.2.1 (M2) - 2025-10-06:**
- Updated Redis configuration to use internal Docker networking
- Removed external port mapping to prevent conflicts
- Added Redis Configuration Guide
- Updated troubleshooting for Redis connectivity

**v0.2.0 (M2) - 2025-10-05:**
- Initial Quick Start guide
- Docker Compose setup
- Local development setup
- Mock TTS worker
- WebSocket CLI client
- Verification checkpoints
- Troubleshooting guide

**Future:**
- Browser client quick start (M3+)
- Real TTS adapter setup (M4-M8)
- Multi-GPU deployment (M9+)
- Production deployment guide (M10+)
