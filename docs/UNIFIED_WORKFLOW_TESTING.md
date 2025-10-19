# Unified Workflow Testing Guide

**Last Updated**: 2025-10-17
**Status**: Test suite implemented, execution pending phases 1-5 completion

---

## Overview

This guide documents the comprehensive integration test suite for the unified development workflow. The test suite validates Docker Compose infrastructure, model switching, service discovery, environment variable precedence, and end-to-end workflows.

**Test File**: `/home/gerald/git/full-duplex-voice-chat/tests/integration/test_unified_workflow.py`

---

## Test Categories

### 1. Infrastructure Tests (3 tests)

**Purpose**: Verify Docker Compose services start correctly and maintain clean state.

| Test | Description | Duration | Status |
|------|-------------|----------|--------|
| `test_dev_infra_only` | Start infrastructure (Redis, LiveKit, Caddy), verify health | < 60s | ✅ Implemented |
| `test_dev_idempotent` | Run `just dev` twice, verify no conflicts | < 90s | ✅ Implemented |
| `test_dev_clean_state` | Cleanup leaves no dangling containers/volumes | < 45s | ✅ Implemented |

**Validation Criteria**:
- ✅ Redis container starts and responds to PING
- ✅ LiveKit server serves health endpoint (http://localhost:7880/)
- ✅ Caddy admin API accessible (http://localhost:2019/config/)
- ✅ No duplicate containers after second run
- ✅ All containers removed after `docker compose down`

**Run Command**:
```bash
pytest tests/integration/test_unified_workflow.py::TestInfrastructure -v
```

---

### 2. Model Switching Tests (3 tests)

**Purpose**: Validate hot-swapping TTS models during runtime.

| Test | Description | Duration | Status |
|------|-------------|----------|--------|
| `test_model_switch_piper_to_cosyvoice` | Switch from Piper to CosyVoice, verify worker registration | < 10s | 🔄 Pending implementation |
| `test_model_switch_during_session` | Switch models mid-conversation, verify graceful transition | < 15s | 🔄 Pending implementation |
| `test_concurrent_model_profiles` | Verify only one TTS worker runs at a time | < 20s | 🔄 Pending implementation |

**Validation Criteria**:
- 🔄 Piper worker registers in Redis with capabilities
- 🔄 Switch command unloads Piper, loads CosyVoice
- 🔄 Orchestrator detects new worker within 2s
- 🔄 Active sessions complete before switch
- 🔄 No port conflicts between workers

**Run Command** (when implemented):
```bash
pytest tests/integration/test_unified_workflow.py::TestModelSwitching -v
```

**Implementation Requirements**:
- Model switching API in TTS worker
- Orchestrator support for dynamic worker discovery
- Docker Compose profiles for model selection

---

### 3. Service Discovery Tests (3 tests)

**Purpose**: Validate orchestrator discovers TTS workers via Redis and Docker DNS.

| Test | Description | Duration | Status |
|------|-------------|----------|--------|
| `test_orchestrator_starts_without_tts` | Orchestrator starts even if TTS worker down | < 30s | 🔄 Pending implementation |
| `test_orchestrator_discovers_tts` | Orchestrator finds TTS worker via network alias | < 5s | 🔄 Pending implementation |
| `test_orchestrator_handles_tts_restart` | Orchestrator reconnects after worker restart | < 45s | 🔄 Pending implementation |

**Validation Criteria**:
- 🔄 Orchestrator container starts successfully without TTS worker
- 🔄 Health check passes (HTTP /health endpoint)
- 🔄 TTS worker registers in Redis with network alias
- 🔄 gRPC connection established via Docker DNS
- 🔄 Active sessions fail gracefully on worker crash
- 🔄 Orchestrator reconnects when worker restarts

**Run Command** (when implemented):
```bash
pytest tests/integration/test_unified_workflow.py::TestServiceDiscovery -v
```

**Implementation Requirements**:
- Orchestrator graceful degradation without TTS worker
- Redis-based worker registration with TTL
- Docker network alias configuration

---

### 4. Environment Variable Tests (3 tests)

**Purpose**: Validate configuration precedence and model selection.

| Test | Description | Duration | Status |
|------|-------------|----------|--------|
| `test_default_model_env_var` | DEFAULT_MODEL env var loads correct model | < 30s | 🔄 Pending implementation |
| `test_env_var_precedence` | CLI > ENV > config > default precedence | < 60s | 🔄 Pending implementation |
| `test_model_specific_env_files` | .env.models/.env.piper loads correctly | < 45s | 🔄 Pending implementation |

**Validation Criteria**:
- 🔄 DEFAULT_MODEL=piper-en-us-lessac-medium loads Piper
- 🔄 DEFAULT_MODEL=cosyvoice2-en-base loads CosyVoice
- 🔄 Invalid model name logs error and falls back
- 🔄 CLI --default-model overrides env var
- 🔄 .env.models/.env.piper loads Piper-specific config

**Run Command** (when implemented):
```bash
pytest tests/integration/test_unified_workflow.py::TestEnvironmentVariables -v
```

**Implementation Requirements**:
- Environment variable parsing in TTS worker
- Configuration precedence logic
- Model-specific env file support

---

### 5. End-to-End Workflow Tests (3 tests)

**Purpose**: Validate complete workflows from text to audio.

| Test | Description | Duration | Status |
|------|-------------|----------|--------|
| `test_full_workflow_piper` | Full workflow with Piper adapter (CPU) | < 60s | 🔄 Pending implementation |
| `test_full_workflow_cosyvoice` | Full workflow with CosyVoice adapter (GPU) | < 90s | 🔄 Pending implementation |
| `test_model_switch_preserves_session` | Switch models mid-conversation | < 45s | 🔄 Pending implementation |

**Validation Criteria**:
- 🔄 `just dev` starts all services
- 🔄 CLI client connects via WebSocket
- 🔄 Speech synthesis request succeeds
- 🔄 Audio frames received (20ms @ 48kHz, 1920 bytes)
- 🔄 Barge-in triggers on VAD speech detection
- 🔄 FAL p95 < 300ms (GPU), < 500ms (Piper CPU)
- 🔄 Session ID preserved across model switch
- 🔄 Conversation history maintained

**Run Command** (when implemented):
```bash
pytest tests/integration/test_unified_workflow.py::TestEndToEndWorkflow -v
```

**Implementation Requirements**:
- Full Docker Compose stack with all services
- CLI client automation for testing
- Session persistence during model switch

---

### 6. CI Integration Tests (3 tests)

**Purpose**: Validate GitHub Actions CI compatibility.

| Test | Description | Duration | Status |
|------|-------------|----------|--------|
| `test_ci_docker_availability` | Docker and Docker Compose available in CI | < 5s | ✅ Implemented |
| `test_ci_cleanup_no_dangling_services` | CI cleanup removes all containers | < 30s | 🔄 Pending implementation |
| `test_ci_model_matrix` | CI matrix tests multiple models | Varies | 🔄 Pending implementation |

**Validation Criteria**:
- ✅ `docker --version` succeeds
- ✅ `docker compose version` succeeds
- ✅ Docker daemon running
- 🔄 All containers stopped after test run
- 🔄 All volumes removed (except named)
- 🔄 Matrix includes: mock, piper, cosyvoice
- 🔄 CosyVoice only on GPU runners

**Run Command**:
```bash
pytest tests/integration/test_unified_workflow.py::TestCIIntegration -v
```

**Implementation Requirements**:
- CI workflow integration with new test suite
- Matrix strategy for different models
- GPU runner configuration for CosyVoice tests

---

## Execution Order

**Recommended test execution order** (based on dependencies):

1. **Infrastructure Tests** → Validate Docker Compose basics
2. **Model Switching Tests** → Validate TTS worker hot-swap
3. **Service Discovery Tests** → Validate orchestrator-worker connection
4. **Environment Variable Tests** → Validate configuration loading
5. **End-to-End Tests** → Validate complete workflows
6. **CI Integration Tests** → Validate GitHub Actions compatibility

**Sequential Execution**:
```bash
# Run all tests in recommended order
pytest tests/integration/test_unified_workflow.py \
    --verbose \
    --tb=short \
    -m integration \
    --order-scope=module
```

**Parallel Execution** (when safe):
```bash
# Run infrastructure and CI tests in parallel (no conflicts)
pytest tests/integration/test_unified_workflow.py::TestInfrastructure \
       tests/integration/test_unified_workflow.py::TestCIIntegration \
    -v -n 2
```

---

## Environment Setup

### Prerequisites

**Required**:
- Docker 28.x with Docker Compose v2
- Python 3.12+ with `uv` package manager
- Redis client library (`redis-py`)
- Project dependencies installed (`uv sync --extra dev`)

**Optional** (for GPU tests):
- NVIDIA GPU with CUDA 12.1+ support
- NVIDIA Container Runtime
- CosyVoice 2 model weights

### Environment Variables

**Required for all tests**:
```bash
# Redis connection (default)
REDIS_URL=redis://localhost:6379

# LiveKit configuration
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=devsecret1234567890abcdefghijklmn
```

**Required for model-specific tests**:
```bash
# Default model selection
DEFAULT_MODEL=piper-en-us-lessac-medium  # or cosyvoice2-en-base

# GPU configuration (CosyVoice only)
CUDA_VISIBLE_DEVICES=0
```

### Docker Compose Profiles

**Infrastructure only** (Redis, LiveKit, Caddy):
```bash
docker compose up -d redis livekit caddy
```

**Full stack with Piper** (default):
```bash
docker compose up -d
```

**Full stack with CosyVoice** (GPU):
```bash
docker compose --profile cosyvoice up -d
```

---

## Test Execution Examples

### Run All Implemented Tests

```bash
# All infrastructure and CI tests (currently implemented)
pytest tests/integration/test_unified_workflow.py \
    -v \
    -m "integration and docker" \
    -k "Infrastructure or test_ci_docker_availability"
```

### Run Specific Test Category

```bash
# Infrastructure tests only
pytest tests/integration/test_unified_workflow.py::TestInfrastructure -v

# Model switching tests (when implemented)
pytest tests/integration/test_unified_workflow.py::TestModelSwitching -v --no-skip

# Service discovery tests (when implemented)
pytest tests/integration/test_unified_workflow.py::TestServiceDiscovery -v --no-skip
```

### Run Individual Test

```bash
# Single test with verbose output
pytest tests/integration/test_unified_workflow.py::TestInfrastructure::test_dev_infra_only -vv

# With debugging output
pytest tests/integration/test_unified_workflow.py::TestInfrastructure::test_dev_infra_only \
    -vv \
    --log-cli-level=DEBUG \
    -s
```

### Run with Coverage

```bash
# Infrastructure tests with coverage
pytest tests/integration/test_unified_workflow.py::TestInfrastructure \
    -v \
    --cov=src/orchestrator \
    --cov=src/tts \
    --cov-report=html \
    --cov-report=term
```

---

## Troubleshooting

### Common Issues

**1. Docker Compose Timeout**

**Symptom**: Test fails with "Docker Compose failed to start within 90s"

**Solutions**:
- Check Docker daemon is running: `docker ps`
- Check disk space: `df -h`
- Pull images manually: `docker compose pull`
- Increase timeout in test code (edit `timeout=90` → `timeout=180`)

**2. Port Conflicts**

**Symptom**: "Address already in use" error

**Solutions**:
- Check existing containers: `docker ps -a`
- Stop conflicting services: `docker compose down`
- Check host ports: `netstat -tuln | grep -E '6379|7880|8443|8444'`
- Kill processes on conflicting ports: `sudo lsof -ti:6379 | xargs kill -9`

**3. Redis Connection Refused**

**Symptom**: "ConnectionError: Error connecting to Redis"

**Solutions**:
- Verify Redis container running: `docker compose ps redis`
- Check Redis health: `docker compose exec redis redis-cli ping`
- Verify port mapping: `docker compose port redis 6379`
- Check firewall rules: `sudo ufw status`

**4. LiveKit Health Check Fails**

**Symptom**: "LiveKit health check did not pass within 30s"

**Solutions**:
- Check LiveKit logs: `docker compose logs livekit`
- Verify config file: `cat configs/livekit.yaml`
- Check port availability: `curl http://localhost:7880/`
- Restart LiveKit: `docker compose restart livekit`

**5. Test Segfault on WSL2**

**Symptom**: Segmentation fault during gRPC tests

**Solutions**:
- Use `pytest-forked`: `pytest --forked tests/integration/test_unified_workflow.py`
- See: [GRPC_SEGFAULT_WORKAROUND.md](../GRPC_SEGFAULT_WORKAROUND.md)
- Run in Docker: `docker compose exec orchestrator pytest tests/integration/`

---

## Performance Targets

### Infrastructure Startup

| Metric | Target | Measurement |
|--------|--------|-------------|
| Cold start (no cache) | < 60s | `docker compose up -d` (first run) |
| Warm start (cached) | < 30s | `docker compose up -d` (subsequent) |
| Health check convergence | < 30s | All services healthy |

### Model Switching

| Metric | Target | Measurement |
|--------|--------|-------------|
| Unload model | < 2s | `ModelManager.unload_model()` |
| Load model | < 3s | `ModelManager.load_model()` |
| Total switch time | < 5s | Unload + load |
| Worker discovery | < 2s | Redis registration → orchestrator query |

### Service Discovery

| Metric | Target | Measurement |
|--------|--------|-------------|
| Worker registration | < 1s | TTS worker → Redis |
| Orchestrator discovery | < 2s | Redis query → gRPC connection |
| Worker failure detection | < 5s | Redis TTL expiration |
| Reconnection after restart | < 10s | Worker down → restart → connection |

### End-to-End Workflows

| Metric | Target (Piper) | Target (CosyVoice) | Measurement |
|--------|----------------|-------------------|-------------|
| First Audio Latency (FAL) | < 500ms | < 300ms | Text → first audio frame |
| Frame jitter (p95) | < 10ms | < 10ms | Inter-frame delay variance |
| Synthesis RTF | ~1.0 | < 0.2 | Real-time factor (audio_duration / wall_time) |

---

## CI/CD Integration

### GitHub Actions Workflow

**File**: `.github/workflows/test-unified-workflow.yml` (to be created)

```yaml
name: Unified Workflow Tests

on:
  pull_request:
    paths:
      - 'docker-compose.yml'
      - 'Procfile.dev'
      - 'src/orchestrator/**'
      - 'src/tts/**'
      - 'tests/integration/test_unified_workflow.py'

jobs:
  infrastructure-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync --extra dev
      - name: Run infrastructure tests
        run: |
          pytest tests/integration/test_unified_workflow.py::TestInfrastructure -v

  model-switching-tests:
    runs-on: ubuntu-latest
    needs: infrastructure-tests
    strategy:
      matrix:
        model: [piper, cosyvoice]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Run model switching tests
        run: |
          DEFAULT_MODEL=${{ matrix.model }} \
          pytest tests/integration/test_unified_workflow.py::TestModelSwitching -v

  e2e-tests-gpu:
    runs-on: [self-hosted, gpu]
    needs: infrastructure-tests
    if: github.event_name == 'pull_request' && contains(github.event.pull_request.labels.*.name, 'gpu-required')
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Run GPU workflow tests
        run: |
          pytest tests/integration/test_unified_workflow.py::TestEndToEndWorkflow::test_full_workflow_cosyvoice -v
```

### Test Matrix Strategy

**CPU Runners** (ubuntu-latest):
- Infrastructure tests (all)
- Model switching tests (mock, piper)
- Service discovery tests (all)
- Environment variable tests (all)
- CI integration tests (all)

**GPU Runners** (self-hosted):
- Model switching tests (cosyvoice)
- End-to-end tests (cosyvoice)
- Performance validation tests (GPU-specific)

---

## Test Coverage Goals

### Current Status

| Category | Implemented | Pending | Total | Coverage |
|----------|-------------|---------|-------|----------|
| Infrastructure | 3 | 0 | 3 | 100% |
| Model Switching | 0 | 3 | 3 | 0% |
| Service Discovery | 0 | 3 | 3 | 0% |
| Environment Variables | 0 | 3 | 3 | 0% |
| End-to-End Workflows | 0 | 3 | 3 | 0% |
| CI Integration | 1 | 2 | 3 | 33% |
| **Total** | **4** | **14** | **18** | **22%** |

### Target Coverage

**Phase 1** (Infrastructure + CI): 4/18 tests (22%) ✅ Complete
**Phase 2** (Service Discovery): 7/18 tests (39%)
**Phase 3** (Model Switching): 10/18 tests (56%)
**Phase 4** (Environment Variables): 13/18 tests (72%)
**Phase 5** (End-to-End): 18/18 tests (100%)

---

## Coordination with Agents

### Agent Assignments

**@devops-engineer**:
- Docker Compose test fixtures
- Infrastructure test implementation
- CI workflow integration
- Cleanup and resource management

**@backend-developer**:
- Orchestrator connection tests
- Service discovery implementation
- WebSocket/gRPC integration tests
- Session persistence validation

**@python-pro**:
- Worker configuration tests
- Environment variable precedence
- Model switching logic
- Test code quality and coverage

**@test-automator** (this agent):
- Test suite design and implementation
- Test execution coordination
- Coverage reporting
- Failure analysis and reporting

### Communication Protocol

**Test failures should be reported to**:
1. **Infrastructure failures** → @devops-engineer
2. **Service discovery failures** → @backend-developer
3. **Configuration failures** → @python-pro
4. **Test framework issues** → @test-automator

**Request format**:
```json
{
  "agent": "test-automator",
  "status": "test_failure",
  "category": "infrastructure",
  "test_name": "test_dev_infra_only",
  "error": "Redis health check timeout after 30s",
  "assigned_to": "devops-engineer",
  "priority": "high"
}
```

---

## Next Steps

### Immediate (Current Sprint)

1. **Run Infrastructure Tests**:
   ```bash
   pytest tests/integration/test_unified_workflow.py::TestInfrastructure -v
   ```

2. **Document Results**: Update test coverage table with pass/fail status

3. **Fix Failures**: Coordinate with @devops-engineer for any infrastructure issues

### Short-term (Next Sprint)

4. **Implement Service Discovery Tests**: After orchestrator Redis integration complete

5. **Implement Model Switching Tests**: After model switching API available

6. **Update CI Workflows**: Add unified workflow tests to GitHub Actions

### Medium-term (Next 2-4 Weeks)

7. **Implement Environment Variable Tests**: After configuration precedence logic complete

8. **Implement End-to-End Tests**: After full Docker Compose stack ready

9. **Performance Validation**: Benchmark against targets, optimize bottlenecks

### Long-term (Next 1-3 Months)

10. **GPU Test Suite**: CosyVoice-specific tests on GPU runners

11. **Multi-GPU Tests**: Scale-out validation (M13 milestone)

12. **Production Deployment Tests**: Smoke tests for production configs

---

## Appendix

### Helper Functions

The test suite includes helper functions for common operations:

**`wait_for_redis(host, port, timeout)`**:
- Wait for Redis to be available
- Timeout after specified seconds
- Raises TimeoutError if unavailable

**`wait_for_http(url, expected_status, timeout)`**:
- Wait for HTTP endpoint to respond
- Validate status code
- Timeout after specified seconds

**`get_container_status(compose_file, service)`**:
- Get Docker Compose service status
- Returns dict with running/health/ports
- Use for health check validation

### Test Markers

The test suite uses pytest markers for filtering:

- `@pytest.mark.integration`: Integration tests (requires external services)
- `@pytest.mark.docker`: Requires Docker and Docker Compose
- `@pytest.mark.redis`: Requires Redis container
- `@pytest.mark.livekit`: Requires LiveKit container
- `@pytest.mark.slow`: Tests taking > 30s

**Filter examples**:
```bash
# Run only fast tests
pytest -v -m "integration and not slow"

# Run Docker-dependent tests
pytest -v -m "docker"

# Run Redis-specific tests
pytest -v -m "redis"
```

---

**Maintained by**: @test-automator
**Last Review**: 2025-10-17
**Next Review**: After Phase 2 completion
