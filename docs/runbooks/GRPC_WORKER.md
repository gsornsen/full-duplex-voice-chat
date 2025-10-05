# Runbook: gRPC Worker Connectivity

**Time to Resolution:** < 10 minutes
**Severity:** Critical (blocks all TTS functionality)

## Symptom

Client cannot connect to TTS worker via gRPC. Common error messages:
- `connection refused`
- `deadline exceeded`
- `worker unavailable`
- `failed to connect to all addresses`

## Common Causes

1. Worker process not running
2. Wrong worker address in `orchestrator.yaml`
3. Network connectivity issues (firewall, routing)
4. Port conflicts (another process using port 7001)
5. Worker crashed during startup or model loading
6. gRPC stubs not generated or mismatched versions

## Diagnostic Steps

### 1. Verify Worker Process is Running

```bash
# Check Docker containers
docker ps | grep tts-worker

# Expected output: tts-worker-0 container running
# If not running, check exit code:
docker ps -a | grep tts-worker
```

**For local processes:**
```bash
ps aux | grep worker.py
# Or check with just:
just status
```

### 2. Check Worker Logs for Startup Errors

```bash
# Docker logs
docker logs tts-worker-0 --tail=50

# Local process logs
tail -f logs/worker.log
```

**Look for:**
- `gRPC server listening on port 7001` (success indicator)
- `Address already in use` (port conflict)
- `CUDA out of memory` (GPU allocation failure)
- `Model not found` (voicepack missing)
- `Failed to load proto` (gRPC stubs issue)

### 3. Test gRPC Health Endpoint

```bash
# Install grpc-health-probe if not present
wget https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/v0.4.24/grpc_health_probe-linux-amd64
chmod +x grpc_health_probe-linux-amd64
sudo mv grpc_health_probe-linux-amd64 /usr/local/bin/grpc-health-probe

# Check worker health
grpc-health-probe -addr=localhost:7001

# Expected: status: SERVING
# Error: Connection refused → worker not running
# Error: status: NOT_SERVING → worker started but unhealthy
```

### 4. Verify Network Connectivity

```bash
# Test port accessibility
telnet localhost 7001
# Or:
nc -zv localhost 7001

# Expected: Connection successful
# Error: Connection refused → port not listening
```

**For Docker:**
```bash
# Check container port mappings
docker port tts-worker-0

# Verify container is on correct network
docker inspect tts-worker-0 | grep -A 10 Networks
```

### 5. Check Orchestrator Configuration

```bash
# Verify worker address in config
grep -A 5 "routing:" configs/orchestrator.yaml

# Expected:
# routing:
#   static_worker_addr: "grpc://localhost:7001"
```

**Common mistakes:**
- Missing `grpc://` prefix
- Wrong port (e.g., 7002 instead of 7001)
- Using container name instead of host when running locally
- Using `localhost` when worker is in separate container

### 6. Test with grpcurl (Advanced)

```bash
# Install grpcurl
go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest

# List available services
grpcurl -plaintext localhost:7001 list

# Expected: tts.TTSService
# Call health check
grpcurl -plaintext localhost:7001 grpc.health.v1.Health/Check
```

## Resolution Strategies

### Worker Not Running

**Start worker with Docker:**
```bash
docker compose up tts-worker-0
# Or:
just run-tts-sesame
```

**Start worker locally:**
```bash
# Ensure proto stubs are generated
just gen-proto

# Start with default config
uv run python -m src.tts.worker_main

# Or with custom config:
uv run python -m src.tts.worker_main --config configs/worker.yaml
```

### Wrong Worker Address

Edit `configs/orchestrator.yaml`:
```yaml
routing:
  static_worker_addr: "grpc://localhost:7001"  # For local worker
  # OR
  static_worker_addr: "grpc://tts-worker-0:7001"  # For Docker container
```

**Important:** Use container name when both orchestrator and worker run in Docker. Use `localhost` when orchestrator runs locally.

### Port Conflict

**Find process using port:**
```bash
lsof -i :7001
# Or:
ss -tuln | grep :7001
```

**Kill conflicting process:**
```bash
kill -9 <PID>
```

**Or change worker port in config:**
```yaml
# configs/worker.yaml
grpc:
  port: 7002  # Use different port
```

**Update orchestrator config accordingly:**
```yaml
# configs/orchestrator.yaml
routing:
  static_worker_addr: "grpc://localhost:7002"
```

### Firewall Blocking Port

**Check firewall status (Linux):**
```bash
sudo ufw status
```

**Allow gRPC port:**
```bash
sudo ufw allow 7001/tcp
```

**For Docker, ensure port is exposed:**
```yaml
# docker-compose.yml
services:
  tts-worker-0:
    ports:
      - "7001:7001"  # Expose to host
```

### Worker Crashed During Startup

**Common crash causes and fixes:**

**1. CUDA Out of Memory:**
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size or unload other GPU processes
# Or use CPU-only worker (Piper adapter)
```

**2. Model Not Found:**
```bash
# Verify voicepacks directory
ls -la voicepacks/

# Download required model
# See: docs/VOICE_PACKS.md
```

**3. Proto Stubs Missing:**
```bash
# Regenerate gRPC stubs
just gen-proto

# Verify generation
ls src/rpc/generated/
# Should contain: tts_pb2.py, tts_pb2_grpc.py
```

**4. Python Dependencies:**
```bash
# Reinstall dependencies
uv sync

# Verify gRPC is installed
uv run python -c "import grpc; print(grpc.__version__)"
```

### Network Issues (Docker)

**Verify containers are on same network:**
```bash
docker network inspect tts-network

# Both orchestrator and worker should be listed
```

**Recreate network:**
```bash
docker compose down
docker network prune
docker compose up
```

## Prevention

**1. Run pre-flight check before starting:**
```bash
./scripts/preflight-check.sh
```

**2. Use connection test tool:**
```bash
./scripts/test-connection.py
```

**3. Monitor worker health:**
```bash
# Continuous health check
watch -n 5 grpc-health-probe -addr=localhost:7001
```

**4. Set up proper logging:**
```yaml
# configs/worker.yaml
log_level: DEBUG  # For troubleshooting
```

**5. Use healthchecks in docker-compose:**
```yaml
services:
  tts-worker-0:
    healthcheck:
      test: ["CMD", "grpc-health-probe", "-addr=:7001"]
      interval: 10s
      timeout: 5s
      retries: 3
```

## Quick Checklist

- [ ] Worker process running? (`docker ps` or `ps aux`)
- [ ] Worker logs show "listening on port 7001"?
- [ ] Health check passes? (`grpc-health-probe -addr=localhost:7001`)
- [ ] Port accessible? (`nc -zv localhost 7001`)
- [ ] Config has correct address? (`grep worker_address configs/orchestrator.yaml`)
- [ ] Proto stubs generated? (`ls src/rpc/generated/tts_pb2.py`)
- [ ] No port conflicts? (`lsof -i :7001`)

## Related Runbooks

- [Redis Connection Failures](REDIS.md)
- [Port Conflicts](PORT_CONFLICTS.md)
- [Environment Setup](ENVIRONMENT.md)
- [Docker Setup](../setup/DOCKER_SETUP.md)

## Still Having Issues?

1. Enable debug logging in both orchestrator and worker
2. Capture full logs: `docker logs tts-worker-0 > worker.log 2>&1`
3. Run connection test: `./scripts/test-connection.py`
4. Check GitHub issues: https://github.com/your-org/full-duplex-voice-chat/issues
5. Review TDD Section 4.2 (gRPC Communication): `docs/TDD.md#42-grpc-communication`
