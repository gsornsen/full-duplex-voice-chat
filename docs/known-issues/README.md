# Known Issues and Workarounds

**Last Updated**: 2025-10-09

This document tracks known operational issues, workarounds, and troubleshooting guidance for the full-duplex voice chat system.

---

## Critical Issues

### 1. gRPC Segfault in Integration Tests (WSL2)

**Severity**: High (blocks integration testing in WSL2)
**Status**: Mitigated with process isolation
**Affected Components**: Integration tests using gRPC
**Platforms**: WSL2, potentially other Linux environments

**Problem**:
Integration tests using gRPC encounter segmentation faults during test teardown due to grpc-python's background threads accessing garbage-collected event loops.

**Symptoms**:
- Segfaults during `gc` (garbage collection) in pytest fixture teardown
- Error: `Fatal Python error: Segmentation fault` during `Garbage-collecting`
- Inconsistent failures, typically after running multiple tests
- Affects Python 3.12 and 3.13

**Workaround**:
Use `pytest-forked` for process isolation:

```bash
# Recommended: Use justfile command
just test-integration

# Manual invocation
uv run pytest tests/integration/ --forked -v
```

**Results with workaround**:
- M1 Integration Tests: 16/16 tests PASS
- Full Pipeline Tests: 6/8 tests PASS (2 timeouts unrelated to segfault)

**Documentation**: See [grpc-segfault.md](grpc-segfault.md) for complete details.

**Upstream Issue**: https://github.com/grpc/grpc/issues/37714

**Long-term Solution**: Monitor upstream grpc-python releases for fix.

---

## Test Issues

### 2. WebSocket Test Timeouts

**Severity**: Medium (affects 2 integration tests)
**Status**: Under investigation
**Affected Tests**:
- `test_sequential_messages_same_session`
- `test_system_stability_under_load`

**Problem**:
Two integration tests in `tests/integration/test_full_pipeline.py` timeout inconsistently.

**Symptoms**:
- Tests hang waiting for responses
- Timeout after 30 seconds
- Other WebSocket tests pass consistently

**Current Status**:
- Not blocking M0-M2 completion
- May be related to async timing or cleanup issues
- Tests are non-critical stress/edge case scenarios

**Workaround**:
Run tests individually or with increased timeout:

```bash
# Skip failing tests
uv run pytest tests/integration/test_full_pipeline.py -k "not sequential and not stability"

# Increase timeout
uv run pytest tests/integration/test_full_pipeline.py --timeout=60
```

**Investigation Plan**:
1. Add detailed logging to failing tests
2. Check for resource cleanup issues
3. Validate async generator cleanup
4. Consider test isolation improvements

---

## Environment Issues

### 3. Port Conflicts

**Severity**: Low (easy to diagnose and fix)
**Status**: Documented
**Affected Components**: Orchestrator, TTS workers, Redis, LiveKit

**Problem**:
Multiple services may conflict on default ports.

**Default Ports**:
- Orchestrator WebSocket: 8080
- Orchestrator LiveKit: 8081
- TTS Worker gRPC: 7001, 7002, etc.
- Redis: 6379
- LiveKit Server: 7880, 7881, 7882
- Caddy HTTPS: 443

**Symptoms**:
- `Address already in use` errors
- Services fail to start
- Connection refused errors

**Diagnosis**:
```bash
# Check port usage
sudo lsof -i :8080
sudo lsof -i :7001

# Or with netstat
netstat -tulpn | grep 8080
```

**Solutions**:

1. **Stop conflicting services**:
```bash
# Kill process using port
sudo kill -9 $(sudo lsof -t -i:8080)

# Stop Docker containers
docker compose down
```

2. **Change ports in configuration**:
```bash
# configs/orchestrator.yaml
transport:
  websocket:
    host: "0.0.0.0"
    port: 8090  # Changed from 8080

# Or via environment variable
export ORCHESTRATOR_WS_PORT=8090
```

3. **Use Docker networking**:
Docker Compose handles port mapping automatically - conflicts only occur if host ports are already in use.

---

### 4. CUDA Version Mismatches

**Severity**: High (prevents GPU workers from running)
**Status**: Documented
**Affected Components**: TTS workers (GPU adapters)

**Problem**:
PyTorch CUDA version must match system CUDA toolkit for GPU support.

**Symptoms**:
- `CUDA not available` errors
- GPU workers fail to initialize
- Torch sees no GPUs despite nvidia-smi showing them

**Diagnosis**:
```bash
# Check system CUDA
nvcc --version
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

**Solution**:
Install correct PyTorch version:

```bash
# For CUDA 12.8 (recommended)
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# For CUDA 11.8
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Current Recommendation**: Use CUDA 12.8 with PyTorch 2.7.0.

---

### 5. WSL2 GPU Access

**Severity**: Medium (WSL2-specific setup required)
**Status**: Documented
**Affected Components**: GPU workers in WSL2

**Problem**:
WSL2 requires specific configuration to access NVIDIA GPUs.

**Requirements**:
- Windows 11 or Windows 10 with WSL2
- NVIDIA GPU driver on Windows host (525.60 or later)
- No NVIDIA driver needed in WSL2 (uses host driver)

**Diagnosis**:
```bash
# In WSL2, check GPU visibility
nvidia-smi

# Should show GPU(s) if properly configured
```

**Solution**:
1. Install latest NVIDIA driver on Windows host
2. Enable WSL2 GPU support (usually automatic)
3. Verify with `nvidia-smi` in WSL2
4. If not working, update WSL kernel:
```bash
wsl --update
```

**Known Limitation**: WSL2 performance is ~10-20% lower than native Linux for GPU workloads.

---

## Configuration Issues

### 6. Redis Connection Failures

**Severity**: High (breaks worker discovery)
**Status**: Documented
**Affected Components**: Orchestrator, TTS workers

**Problem**:
Services fail to connect to Redis for service discovery.

**Symptoms**:
- `Connection refused` errors
- Workers not registering
- Orchestrator can't find workers

**Diagnosis**:
```bash
# Check Redis is running
docker ps | grep redis

# Test Redis connection
redis-cli -h localhost -p 6379 ping
# Should respond: PONG
```

**Solutions**:

1. **Start Redis**:
```bash
just redis
# Or manually:
docker run -d --name redis -p 6379:6379 redis:7
```

2. **Check configuration**:
```yaml
# configs/orchestrator.yaml
redis:
  url: "redis://localhost:6379"

# Or environment variable
export REDIS_URL="redis://localhost:6379"
```

3. **Network issues** (Docker):
If using Docker Compose, use service name:
```yaml
redis:
  url: "redis://redis:6379"  # Use service name, not localhost
```

---

### 7. LiveKit Configuration Issues

**Severity**: Medium (affects WebRTC transport)
**Status**: Documented
**Affected Components**: Orchestrator LiveKit transport

**Problem**:
LiveKit transport requires proper API keys and URL configuration.

**Symptoms**:
- LiveKit connection failures
- Token generation errors
- WebRTC clients can't connect

**Diagnosis**:
```bash
# Check LiveKit server status
curl http://localhost:7880

# Verify environment variables
echo $LIVEKIT_URL
echo $LIVEKIT_API_KEY
echo $LIVEKIT_API_SECRET
```

**Solutions**:

1. **Generate API keys**:
```bash
# If running LiveKit server locally
livekit-server generate-keys

# Or use default dev keys from docker-compose.yml
```

2. **Configure in orchestrator**:
```yaml
# configs/orchestrator.yaml
transport:
  livekit:
    url: "ws://localhost:7880"
    api_key: "devkey"
    api_secret: "secret"
```

3. **Verify server is running**:
```bash
docker compose ps livekit
# Should show "running" status
```

---

## Performance Issues

### 8. High Memory Usage During Tests

**Severity**: Low (expected behavior)
**Status**: Documented
**Affected Components**: Integration tests

**Problem**:
Integration tests spawn multiple processes and can use 4-8 GB RAM.

**Symptoms**:
- High memory usage during test runs
- Potential OOM on systems with < 8 GB RAM

**Explanation**:
This is expected behavior:
- `--forked` mode spawns process per test
- Each process loads full test environment
- Multiple async loops and gRPC channels

**Solutions**:

1. **Run tests serially**:
```bash
uv run pytest tests/integration/ --forked -v -n 1
```

2. **Run specific tests**:
```bash
uv run pytest tests/integration/test_m1_worker_integration.py --forked -v
```

3. **Increase system memory** or close other applications during testing.

---

## Quick Troubleshooting Guide

### Service Won't Start

1. **Check ports**:
```bash
sudo lsof -i :<port>
```

2. **Check logs**:
```bash
docker compose logs <service>
# Or for local run:
journalctl -u <service> -f
```

3. **Verify dependencies**:
```bash
# Redis running?
docker ps | grep redis

# Configuration valid?
uv run python -c "from src.orchestrator.config import OrchestratorConfig; OrchestratorConfig.from_yaml('configs/orchestrator.yaml')"
```

### Tests Failing

1. **Run single test**:
```bash
uv run pytest tests/path/to/test.py::test_name -v
```

2. **Check for port conflicts**:
```bash
# Integration tests use ephemeral ports, but check Redis
sudo lsof -i :6379
```

3. **Use process isolation for gRPC tests**:
```bash
just test-integration
```

4. **Check environment**:
```bash
# CUDA available for GPU tests?
python -c "import torch; print(torch.cuda.is_available())"

# Redis accessible?
redis-cli ping
```

### GPU Not Working

1. **Verify GPU visibility**:
```bash
nvidia-smi
```

2. **Check PyTorch**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

3. **Check CUDA versions**:
```bash
nvcc --version  # System CUDA
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA
```

4. **Reinstall PyTorch with correct CUDA**:
```bash
uv add torch --index-url https://download.pytorch.org/whl/cu128
```

### Connection Issues

1. **Check service status**:
```bash
docker compose ps
```

2. **Verify networking**:
```bash
# Can orchestrator reach Redis?
docker compose exec orchestrator redis-cli -h redis ping

# Can worker reach orchestrator?
curl http://localhost:8080/health
```

3. **Check firewall** (if remote connections):
```bash
sudo ufw status
sudo iptables -L
```

---

## Reporting New Issues

If you encounter a new issue:

1. **Check existing issues**: Review this document first
2. **Gather diagnostics**:
   - Error messages and stack traces
   - Relevant log files
   - Configuration files (redact secrets)
   - Environment details (OS, Python version, CUDA version)
   - Steps to reproduce

3. **Create issue** with:
   - Clear title summarizing the problem
   - Detailed description
   - Reproduction steps
   - Expected vs actual behavior
   - Diagnostic information

4. **Include workarounds** if you find any

---

## Additional Resources

- **Development Guide**: [../DEVELOPMENT.md](../DEVELOPMENT.md)
- **Testing Guide**: [../TESTING_GUIDE.md](../TESTING_GUIDE.md)
- **Current Status**: [../CURRENT_STATUS.md](../CURRENT_STATUS.md)
- **Claude AI Guidance**: [../../CLAUDE.md](../../CLAUDE.md)

---

## Issue Index

| Issue | Severity | Status | Workaround Available |
|-------|----------|--------|---------------------|
| gRPC Segfault (WSL2) | High | Mitigated | Yes (process isolation) |
| WebSocket Test Timeouts | Medium | Under Investigation | Partial (skip tests) |
| Port Conflicts | Low | Documented | Yes (change ports) |
| CUDA Version Mismatch | High | Documented | Yes (reinstall PyTorch) |
| WSL2 GPU Access | Medium | Documented | Yes (driver update) |
| Redis Connection | High | Documented | Yes (start Redis) |
| LiveKit Config | Medium | Documented | Yes (set env vars) |
| High Memory in Tests | Low | Expected | Yes (run serially) |

---

**Last Review**: 2025-10-09
**Next Review**: When new issues discovered or existing issues resolved
