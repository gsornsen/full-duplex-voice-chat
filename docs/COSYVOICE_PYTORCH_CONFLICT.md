# CosyVoice 2 PyTorch Version Conflict - Technical Analysis & Solutions

**Last Updated**: 2025-10-17
**Status**: Blocking M6 Implementation
**Impact**: Critical - Affects GPU TTS adapter deployment strategy

---

## Executive Summary

CosyVoice 2 requires **PyTorch 2.3.1 + CUDA 12.1**, while the main project uses **PyTorch 2.7.0 + CUDA 12.8**. This version incompatibility creates a blocking issue for M6 (CosyVoice 2 Adapter) implementation. This document analyzes the conflict, its impact, and provides concrete solution strategies.

**Recommended Approach**: Docker Container Isolation (Option 2)

---

## 1. The Version Conflict

### Current Environment (Main Project)

```yaml
Platform:
  - PyTorch: 2.7.0
  - CUDA: 12.8
  - Python: 3.13.x
  - cuDNN: 9.x (bundled with CUDA 12.8)
  - Base Container: nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

Dependencies (Existing):
  - livekit: Tested with PyTorch 2.7.0
  - Whisper ASR: Compatible with PyTorch 2.7.0
  - Piper TTS: CPU-only (no PyTorch dependency)
  - torchaudio: 2.7.0 (matched with PyTorch)
```

### CosyVoice 2 Requirements

```yaml
Platform:
  - PyTorch: 2.3.1 (STRICT)
  - CUDA: 12.1
  - Python: 3.8-3.11 (3.13 not tested upstream)
  - cuDNN: 8.9.x

Dependencies (CosyVoice 2 Specific):
  - matcha-tts: Pinned to PyTorch 2.3.x APIs
  - WeTextProcessing: Requires specific torch.ops APIs
  - modelscope: May have version constraints
  - onnxruntime-gpu: Specific CUDA version binding
```

### Official CosyVoice 2 Repository

```bash
# From: https://github.com/FunAudioLLM/CosyVoice
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice

# Installation (from official README)
conda create -n cosyvoice python=3.8
conda activate cosyvoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

**requirements.txt excerpt**:
```
torch==2.3.1
torchaudio==2.3.1
matcha-tts @ git+https://github.com/shivammehta25/Matcha-TTS.git@main
WeTextProcessing
modelscope
onnxruntime-gpu
```

---

## 2. Why This Matters

### Binary Incompatibility

PyTorch releases are **ABI-incompatible** across minor versions:

1. **CUDA Bindings**: PyTorch 2.3.1 compiled against CUDA 12.1 headers/libs; PyTorch 2.7.0 against CUDA 12.8
2. **C++ ABI Changes**: Internal C++ APIs changed between 2.3 and 2.7
3. **Kernel Implementations**: cuDNN operator implementations differ
4. **Shared Object Dependencies**: Different `.so` file versions (libcudnn, libcublas, libcudart)

**Consequence**: Installing CosyVoice 2 in a PyTorch 2.7.0 environment will:
- Fail during pip install (version conflict)
- Runtime segfaults if forced (mismatched binaries)
- Undefined behavior (silent corruption, wrong results)

### API Changes (2.3.1 → 2.7.0)

Notable changes that may affect CosyVoice 2:

```python
# torch.compile() improvements (2.4+)
# - New inductor backends
# - Changed default config parameters

# torch.ops API changes (2.5+)
# - WeTextProcessing depends on specific torch.ops APIs
# - Breaking changes in custom ops registration

# SDPA (Scaled Dot-Product Attention) changes (2.6+)
# - Memory format changes
# - Flash Attention 3 integration (incompatible with FA2 used in CosyVoice)

# Python 3.13 support (2.7+)
# - CosyVoice not tested with Python 3.13 (uses 3.8-3.11)
```

### Dependency Tree Conflicts

```
Project Dependencies (PyTorch 2.7.0):
├─ livekit
│  └─ depends on torch 2.x (flexible)
├─ whisper (openai-whisper)
│  └─ torchaudio 2.7.0 (implicitly requires torch 2.7.0)
├─ transformers
│  └─ torch >= 2.0 (flexible, but optimized for latest)
└─ accelerate
   └─ torch >= 2.0 (flexible)

CosyVoice 2 Dependencies (PyTorch 2.3.1):
├─ torch==2.3.1 (PINNED - strict requirement)
├─ torchaudio==2.3.1 (PINNED)
├─ matcha-tts
│  └─ requires torch 2.3.x (uses deprecated APIs removed in 2.4+)
└─ WeTextProcessing
   └─ torch.ops APIs specific to 2.3.x
```

**Conflict**: Cannot satisfy both `torch==2.3.1` and `torch>=2.7.0` in single environment.

---

## 3. Impact on M6 Implementation Timeline

### Original M6 Scope

From [.claude/modules/milestones.md](../.claude/modules/milestones.md):

```markdown
## M6: CosyVoice 2 Adapter (GPU TTS)

**Goal**: Implement CosyVoice 2 adapter for expressive GPU TTS with low latency.

**Tasks**:
1. CosyVoice 2 model integration (matcha-tts + WeTextProcessing)
2. Native sample rate → 48 kHz resampling
3. 20 ms frame repacketization
4. Emotion control mapping (zero-shot prompting)
5. Integration tests with Model Manager
6. Performance benchmarking (FAL < 300 ms p95)

**Original Estimate**: 5-7 days
```

### Revised Estimate with Version Conflict

**New Estimate**: 8-12 days (60-70% overhead)

**Additional Tasks**:
1. **Environment Isolation Setup** (+2-3 days)
   - Docker image for PyTorch 2.3.1 environment
   - gRPC communication across container boundaries
   - CUDA device passthrough configuration
   - Volume mounting for voicepacks
   - Redis service discovery updates

2. **Deployment Complexity** (+1-2 days)
   - Separate Dockerfile.tts-cosyvoice
   - docker-compose.yml updates for multi-container setup
   - Kubernetes manifests (if applicable)
   - Health check endpoints

3. **Testing Strategy Changes** (+1-2 days)
   - Cannot run CosyVoice tests in main test suite
   - Separate pytest environment
   - Mock-based unit tests (no real model)
   - Integration tests require Docker runtime
   - CI/CD pipeline adjustments

4. **Documentation** (+0.5-1 day)
   - Setup instructions for developers
   - Docker-specific troubleshooting
   - Environment variable reference
   - Multi-environment dependency management

**Risk Factors**:
- CUDA device passthrough issues (WSL2, Docker)
- gRPC performance overhead across containers
- Volume mount permissions (model files)
- Redis network configuration
- PyTorch 2.3.1 + CUDA 12.1 availability (may need custom build)

---

## 4. Solution Strategies

### Option 1: Separate Virtual Environment (Isolated venv)

**Approach**: Create dedicated Python venv with PyTorch 2.3.1 for CosyVoice worker only.

**Implementation**:

```bash
# Main project (PyTorch 2.7.0)
uv venv .venv-main
source .venv-main/bin/activate
uv pip install -e .  # PyTorch 2.7.0

# CosyVoice worker (PyTorch 2.3.1)
uv venv .venv-cosyvoice
source .venv-cosyvoice/bin/activate
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-cosyvoice.txt
```

**Project Structure**:
```
full-duplex-voice-chat/
├─ .venv-main/          # PyTorch 2.7.0 (orchestrator, Whisper, Piper)
├─ .venv-cosyvoice/     # PyTorch 2.3.1 (CosyVoice worker only)
├─ pyproject.toml       # Main dependencies (PyTorch 2.7.0)
├─ requirements-cosyvoice.txt  # CosyVoice-specific (PyTorch 2.3.1)
└─ justfile             # Commands for both envs
```

**justfile additions**:
```make
# CosyVoice worker (separate venv)
run-tts-cosyvoice:
    .venv-cosyvoice/bin/python -m src.tts.worker \
        --adapter cosyvoice2 \
        --port 7002 \
        --default_model cosyvoice2-en-base
```

**Pros**:
- ✅ Simple to set up (no Docker required)
- ✅ Fast local development iteration
- ✅ Both environments on same filesystem (easy model access)
- ✅ Direct GPU access (no container overhead)
- ✅ Can run unit tests independently

**Cons**:
- ❌ Manual environment switching (error-prone)
- ❌ Dependency duplication (2x disk space)
- ❌ `uv` doesn't manage multiple venvs well
- ❌ CI complexity (need to install both envs)
- ❌ Doesn't match production deployment model
- ❌ Risk of accidentally using wrong venv
- ❌ Python 3.13 incompatibility (CosyVoice needs 3.8-3.11)

**Verdict**: ⚠️ **Not Recommended** - Too fragile for production; doesn't match deployment model.

---

### Option 2: Docker Container Isolation (RECOMMENDED)

**Approach**: Run CosyVoice worker in dedicated Docker container with PyTorch 2.3.1.

**Implementation**:

**Dockerfile.tts-cosyvoice**:
```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install Python 3.10 (CosyVoice tested version)
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    git wget curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Install PyTorch 2.3.1 + CUDA 12.1
RUN pip3 install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Clone and install CosyVoice
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /app/third_party/CosyVoice && \
    cd /app/third_party/CosyVoice && \
    pip install -r requirements.txt

# Copy project code
COPY src/ /app/src/
COPY pyproject.toml /app/

# Install minimal project dependencies (gRPC, Redis, etc.)
# NOTE: Skip PyTorch (already installed) and other conflicting deps
RUN pip3 install grpcio grpcio-tools redis pydantic pyyaml soundfile numpy

EXPOSE 7002

ENTRYPOINT ["python3", "-m", "src.tts.worker", "--adapter", "cosyvoice2", "--port", "7002"]
```

**docker-compose.yml**:
```yaml
version: "3.9"
services:
  redis:
    image: redis:7
    ports: ["6379:6379"]
    networks: [voice-demo]

  orchestrator:
    build:
      context: .
      dockerfile: Dockerfile.orchestrator  # PyTorch 2.7.0
    environment:
      - REDIS_URL=redis://redis:6379
    ports: ["8080:8080"]
    depends_on: [redis]
    networks: [voice-demo]

  tts-piper:
    build:
      context: .
      dockerfile: Dockerfile.tts  # CPU, no PyTorch conflict
    environment:
      - REDIS_URL=redis://redis:6379
    command: ["python3", "-m", "src.tts.worker", "--adapter", "piper", "--port", "7001"]
    depends_on: [redis]
    networks: [voice-demo]

  tts-cosyvoice:
    build:
      context: .
      dockerfile: Dockerfile.tts-cosyvoice  # PyTorch 2.3.1 ISOLATED
    environment:
      - REDIS_URL=redis://redis:6379
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./voicepacks/cosyvoice2:/models/cosyvoice2:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    depends_on: [redis]
    networks: [voice-demo]

networks:
  voice-demo:
    driver: bridge
```

**Local Development**:
```bash
# Start all services
docker compose up --build

# Or start CosyVoice worker only
docker compose up tts-cosyvoice

# View logs
docker compose logs -f tts-cosyvoice

# Shell into container for debugging
docker compose exec tts-cosyvoice bash
```

**.env.cosyvoice** (example):
```bash
# CosyVoice 2 Worker Configuration
PYTORCH_VERSION=2.3.1
CUDA_VERSION=12.1
PYTHON_VERSION=3.10

# Model paths (inside container)
COSYVOICE_MODEL_PATH=/models/cosyvoice2/en-base
COSYVOICE_VOICEPACK_DIR=/models/cosyvoice2

# Worker settings
WORKER_PORT=7002
REDIS_URL=redis://redis:6379
DEVICE=cuda:0

# Model Manager (TTL, etc.)
DEFAULT_MODEL_ID=cosyvoice2-en-base
PRELOAD_MODEL_IDS=
TTL_MS=600000
RESIDENT_CAP=2
```

**Pros**:
- ✅ **Complete dependency isolation** (zero conflict risk)
- ✅ **Matches production deployment** (Docker Swarm/K8s)
- ✅ Easy GPU device assignment (`CUDA_VISIBLE_DEVICES`)
- ✅ Network isolation (gRPC over Docker network)
- ✅ Reproducible builds (pinned base image + dependencies)
- ✅ Can run multiple workers with different PyTorch versions
- ✅ CI/CD friendly (Docker layer caching)
- ✅ Health checks, restarts, resource limits
- ✅ Volume mounts for voicepacks (shared read-only)

**Cons**:
- ⚠️ Requires Docker GPU runtime (nvidia-container-toolkit)
- ⚠️ Slower build times (first build ~5-10 min)
- ⚠️ More complex local dev setup
- ⚠️ gRPC adds ~1-2 ms latency (negligible for TTS)
- ⚠️ Debugging requires container shell access
- ⚠️ WSL2 GPU passthrough setup required

**Mitigation**:
- Use Docker layer caching (BuildKit)
- Provide `just` commands for common operations
- Document WSL2 GPU setup (one-time)
- Use volume mounts for fast code iteration
- Enable hot-reload inside container (watchfiles)

**Verdict**: ✅ **RECOMMENDED** - Best balance of isolation, production parity, and maintainability.

---

### Option 3: Conditional Import with Graceful Degradation (Mock Mode)

**Approach**: Allow CosyVoice adapter to run without actual CosyVoice installed; use separate production environment.

**Implementation**:

**src/tts/adapters/adapter_cosyvoice2.py**:
```python
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from third_party.CosyVoice import CosyVoice2  # type: ignore
    COSYVOICE_AVAILABLE = True
except ImportError:
    logger.warning("CosyVoice not installed. Running in MOCK mode.")
    COSYVOICE_AVAILABLE = False
    CosyVoice2 = None


class CosyVoice2Adapter:
    def __init__(self, model_id: str, device: str = "cuda:0"):
        self.model_id = model_id
        self.device = device
        self._model: Optional[CosyVoice2] = None

        if COSYVOICE_AVAILABLE:
            self._model = CosyVoice2(model_id, device=device)
        else:
            logger.warning(
                f"CosyVoice2Adapter({model_id}) initialized in MOCK mode. "
                "Install CosyVoice in production environment."
            )

    def synthesize(self, text: str, **kwargs):
        if self._model is None:
            # Return silence (mock mode)
            logger.debug(f"MOCK synthesize: {text}")
            return self._generate_mock_audio(text)

        return self._model.synthesize(text, **kwargs)

    def _generate_mock_audio(self, text: str):
        """Generate silent audio for testing."""
        import numpy as np
        duration_ms = len(text) * 50  # Rough estimate
        samples = int(48000 * duration_ms / 1000)
        return np.zeros(samples, dtype=np.float32)
```

**pyproject.toml**:
```toml
[project.optional-dependencies]
cosyvoice = [
    "torch==2.3.1",
    "torchaudio==2.3.1",
    # ... other CosyVoice deps
]
```

**justfile**:
```make
# Main dev environment (no CosyVoice)
install:
    uv pip install -e .

# Production CosyVoice environment (separate venv/container)
install-cosyvoice:
    pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -e ".[cosyvoice]"
```

**Testing Strategy**:
```python
# tests/unit/test_cosyvoice_adapter.py
import pytest
from src.tts.adapters.adapter_cosyvoice2 import CosyVoice2Adapter, COSYVOICE_AVAILABLE


@pytest.mark.skipif(not COSYVOICE_AVAILABLE, reason="CosyVoice not installed")
def test_cosyvoice_real_synthesis():
    adapter = CosyVoice2Adapter("cosyvoice2-en-base")
    audio = adapter.synthesize("Hello world")
    assert len(audio) > 0


def test_cosyvoice_mock_mode():
    """Test that mock mode doesn't crash."""
    adapter = CosyVoice2Adapter("cosyvoice2-en-base")
    audio = adapter.synthesize("Hello world")
    assert audio is not None  # Should return mock audio
```

**CI/CD**:
```yaml
# .github/workflows/ci.yml
jobs:
  test-main:
    runs-on: ubuntu-latest
    steps:
      - run: uv pip install -e .
      - run: pytest -v  # CosyVoice tests skipped

  test-cosyvoice:
    runs-on: ubuntu-latest
    container: nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
    steps:
      - run: pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
      - run: pip install -e ".[cosyvoice]"
      - run: pytest -v -m cosyvoice  # Only CosyVoice tests
```

**Pros**:
- ✅ Enables development/testing without CosyVoice installed
- ✅ Graceful degradation (no crashes)
- ✅ Simple local setup for non-CosyVoice work
- ✅ Can run most tests without GPU
- ✅ Clear separation of concerns

**Cons**:
- ❌ Cannot test real CosyVoice synthesis locally
- ❌ Still need separate production environment
- ❌ Mock audio != real behavior (limited testing)
- ❌ Requires careful `skipif` markers in tests
- ❌ Production deployment still complex
- ❌ Risk of false confidence (tests pass in mock mode, fail in prod)

**Verdict**: ⚠️ **Conditional Recommendation** - Good for enabling development without CosyVoice, but must be paired with Docker isolation for production. Use as **complement** to Option 2, not replacement.

---

### Option 4: Downgrade PyTorch to 2.3.1 (NOT RECOMMENDED)

**Approach**: Downgrade entire project to PyTorch 2.3.1 + CUDA 12.1.

**Implementation**:
```toml
# pyproject.toml
[project]
dependencies = [
    "torch==2.3.1",
    "torchaudio==2.3.1",
    # ... rest of deps
]
```

**Pros**:
- ✅ Simplest for CosyVoice (no isolation needed)
- ✅ Single environment for all workers
- ✅ No Docker complexity

**Cons**:
- ❌ **Loses PyTorch 2.7.0 features**:
  - torch.compile() improvements (2.4+)
  - Better Flash Attention support (2.6+)
  - Python 3.13 support (2.7+)
  - Performance optimizations (2.5-2.7)
- ❌ **May break other components**:
  - Whisper optimized for latest PyTorch
  - LiveKit may have issues with old PyTorch
  - Transformers/Accelerate expect recent versions
- ❌ **Blocks future upgrades**:
  - Stuck on old PyTorch until CosyVoice updates
  - Cannot use new ML features
- ❌ **Security/bug fixes**:
  - Missing 4+ months of PyTorch patches
  - Known vulnerabilities in 2.3.1

**Verdict**: ❌ **NOT RECOMMENDED** - Technical debt, breaks upgrade path, loses features.

---

## 5. Recommended Solution: Docker Isolation (Option 2)

### Implementation Plan

**Phase 1: Container Setup (1-2 days)**

1. Create `Dockerfile.tts-cosyvoice`:
   - Base: `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
   - Python 3.10
   - PyTorch 2.3.1 + CUDA 12.1
   - CosyVoice installation
   - Minimal project dependencies (gRPC, Redis only)

2. Update `docker-compose.yml`:
   - Add `tts-cosyvoice` service
   - Configure GPU reservation
   - Set up volume mounts for voicepacks
   - Network configuration

3. Create `.env.cosyvoice` template

**Phase 2: Adapter Implementation (3-4 days)**

1. Implement `src/tts/adapters/adapter_cosyvoice2.py`:
   - CosyVoice model loading
   - Text → audio synthesis
   - Native sample rate → 48 kHz resampling
   - 20 ms frame repacketization
   - Emotion control mapping

2. Integration with Model Manager:
   - Load/unload lifecycle
   - TTL-based eviction
   - Warmup routine

3. gRPC service implementation in worker

**Phase 3: Testing (2-3 days)**

1. Unit tests (mock mode):
   - Adapter interface compliance
   - Frame repacketization
   - Sample rate conversion

2. Integration tests (Docker):
   - End-to-end synthesis
   - Model Manager integration
   - Performance benchmarks (FAL < 300 ms)

3. CI/CD updates:
   - Docker image build caching
   - Separate test job for CosyVoice

**Phase 4: Documentation (1 day)**

1. Update `CLAUDE.md` with PyTorch version note
2. Update `project_documentation/TDD.md` with deployment strategy
3. Create `.env.cosyvoice` example
4. Update `docs/MULTI_GPU.md` with CosyVoice container setup
5. Add troubleshooting guide for Docker GPU passthrough

### Development Workflow

**Local Development**:
```bash
# Terminal 1: Start Redis + Orchestrator (PyTorch 2.7.0)
docker compose up redis orchestrator

# Terminal 2: Start CosyVoice worker (PyTorch 2.3.1, isolated)
docker compose up tts-cosyvoice

# Terminal 3: Run CLI client
just cli

# Hot-reload for code changes (volume mounted)
# Edit src/tts/adapters/adapter_cosyvoice2.py → auto-restart
```

**Testing**:
```bash
# Main test suite (no CosyVoice)
just test

# CosyVoice integration tests (requires Docker)
docker compose run --rm tts-cosyvoice pytest tests/integration/test_cosyvoice.py -v

# Performance benchmarks
docker compose run --rm tts-cosyvoice pytest tests/performance/test_cosyvoice_fal.py -v
```

**Production Deployment**:
```bash
# Docker Swarm
docker stack deploy -c docker-compose.yml voice-demo

# Kubernetes (converted from Compose)
kompose convert
kubectl apply -f .
```

### Risk Mitigation

**Risk 1: WSL2 GPU Passthrough**

*Issue*: WSL2 requires nvidia-container-toolkit and proper driver setup.

*Mitigation*:
- Document one-time setup in `docs/setup/WSL2_GPU_DOCKER.md`
- Provide automated setup script
- Fallback to cloud GPU instance (vast.ai, Lambda Labs)

**Risk 2: Docker Build Time**

*Issue*: First build takes 5-10 minutes (CosyVoice clone + dependencies).

*Mitigation*:
- Use BuildKit caching
- Pre-built base image on Docker Hub (if public)
- CI cache layers between builds

**Risk 3: gRPC Overhead**

*Issue*: Container networking adds latency.

*Mitigation*:
- Use host network mode (`network_mode: host`) for lowest latency
- Benchmark: expect <2 ms overhead (negligible for TTS)
- Optimize gRPC settings (keepalive, buffer sizes)

**Risk 4: Volume Mount Permissions**

*Issue*: voicepacks/ directory permissions mismatch.

*Mitigation*:
- Run container with host UID/GID
- Use Docker volume with proper ownership
- Document `chown` workaround if needed

---

## 6. Alternative: Hybrid Approach (Option 2 + Option 3)

**Best of Both Worlds**: Docker isolation for production + mock mode for development.

**Strategy**:
1. **Production**: Use Docker container (Option 2)
2. **Development**: Use mock mode (Option 3) for fast iteration
3. **CI**: Run both mock tests (fast) and Docker integration tests (complete)

**Implementation**:
- Implement both `adapter_cosyvoice2.py` (with conditional import)
- Dockerfile for production deployment
- Mock tests run in main CI job (2 min)
- Docker integration tests run in separate job (10 min, cached)

**Developer Experience**:
```bash
# Fast local iteration (no Docker)
just test  # Mock mode, <30 sec

# Full integration test (Docker)
just test-cosyvoice-docker  # ~2 min (with cache)

# Production deployment
docker compose up  # Docker isolation
```

**Verdict**: ✅ **Excellent Compromise** - Fast development + production-ready deployment.

---

## 7. Documentation Updates Required

### CLAUDE.md

**Current** (lines 167-172):
```markdown
**Platform:**
- CUDA 12.8 + PyTorch 2.7.0 for GPU workers
- Docker 28.x with NVIDIA container runtime
- Redis for worker service discovery
- **WSL2 Note**: gRPC tests require `--forked` flag
```

**Updated**:
```markdown
**Platform:**
- CUDA 12.8 + PyTorch 2.7.0 for main workers (Orchestrator, Whisper, XTTS, Sesame)
- ⚠️ **M6 CosyVoice 2**: Requires PyTorch 2.3.1 + CUDA 12.1 (isolated Docker container)
  - See [docs/COSYVOICE_PYTORCH_CONFLICT.md](docs/COSYVOICE_PYTORCH_CONFLICT.md)
- Docker 28.x with NVIDIA container runtime
- Redis for worker service discovery
- **WSL2 Note**: gRPC tests require `--forked` flag
```

### project_documentation/TDD.md

Add new section under **M6: CosyVoice 2 Adapter**:

```markdown
## M6: CosyVoice 2 Adapter (GPU TTS)

### PyTorch Version Constraint

**Issue**: CosyVoice 2 requires PyTorch 2.3.1 + CUDA 12.1, incompatible with main project (PyTorch 2.7.0 + CUDA 12.8).

**Solution**: Docker container isolation (Dockerfile.tts-cosyvoice).

**Deployment Strategy**:
1. **Production**: Isolated Docker container with PyTorch 2.3.1
2. **Development**: Mock mode (conditional import) for fast iteration
3. **Testing**:
   - Unit tests: Mock mode (main CI pipeline)
   - Integration tests: Docker container (separate CI job)

**Configuration**:
- Base Image: `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
- Python: 3.10
- Environment: `.env.cosyvoice` (see docs/COSYVOICE_PYTORCH_CONFLICT.md)

**Testing Strategy**:
```bash
# Mock unit tests (fast)
pytest tests/unit/test_cosyvoice_adapter.py

# Docker integration tests (complete)
docker compose run --rm tts-cosyvoice pytest tests/integration/test_cosyvoice.py
```

**Documentation**:
- [docs/COSYVOICE_PYTORCH_CONFLICT.md](../docs/COSYVOICE_PYTORCH_CONFLICT.md) - Version conflict analysis
- [.env.cosyvoice](../.env.cosyvoice) - Environment template
```

### .env.cosyvoice Example

Create new file:

```bash
# =============================================================================
# CosyVoice 2 Worker Environment Configuration
# =============================================================================
# This worker runs in an isolated Docker container with PyTorch 2.3.1 + CUDA 12.1
# due to incompatibility with the main project (PyTorch 2.7.0 + CUDA 12.8).
#
# See docs/COSYVOICE_PYTORCH_CONFLICT.md for detailed explanation.

# -----------------------------------------------------------------------------
# PyTorch & CUDA Configuration
# -----------------------------------------------------------------------------
PYTORCH_VERSION=2.3.1
CUDA_VERSION=12.1
PYTHON_VERSION=3.10

# -----------------------------------------------------------------------------
# Model Paths (Inside Container)
# -----------------------------------------------------------------------------
# Voicepacks are volume-mounted from host:
#   volumes:
#     - ./voicepacks/cosyvoice2:/models/cosyvoice2:ro

COSYVOICE_MODEL_PATH=/models/cosyvoice2/en-base
COSYVOICE_VOICEPACK_DIR=/models/cosyvoice2

# Alternative: Download from Hugging Face on startup
# COSYVOICE_HF_REPO=FunAudioLLM/CosyVoice-300M
# COSYVOICE_HF_CACHE=/models/hf_cache

# -----------------------------------------------------------------------------
# Worker Configuration
# -----------------------------------------------------------------------------
WORKER_PORT=7002
WORKER_NAME=tts-cosyvoice@0
REDIS_URL=redis://redis:6379

# GPU Device Assignment
CUDA_VISIBLE_DEVICES=0

# Audio Output Settings
SAMPLE_RATE=48000
FRAME_MS=20

# -----------------------------------------------------------------------------
# Model Manager Configuration
# -----------------------------------------------------------------------------
DEFAULT_MODEL_ID=cosyvoice2-en-base

# Optional: Preload additional models at startup
# PRELOAD_MODEL_IDS=cosyvoice2-en-expressive,cosyvoice2-zh-base

# Model Lifecycle (TTL-based eviction)
TTL_MS=600000              # 10 minutes idle → unload
MIN_RESIDENCY_MS=120000    # Keep at least 2 minutes
EVICT_CHECK_INTERVAL_MS=30000  # Check every 30 seconds
RESIDENT_CAP=2             # Max 2 models resident
MAX_PARALLEL_LOADS=1       # Load models sequentially

# -----------------------------------------------------------------------------
# CosyVoice-Specific Settings
# -----------------------------------------------------------------------------
# Emotion Control (zero-shot prompting)
ENABLE_EMOTION_CONTROL=true
DEFAULT_EMOTION=neutral

# Inference Optimization
USE_FLASH_ATTENTION=true   # Flash Attention 2 (PyTorch 2.3.1 compatible)
USE_TORCH_COMPILE=false    # Disable torch.compile() (slower first run)
USE_FP16=true              # Half precision for faster inference

# WeTextProcessing (text normalization)
ENABLE_TEXT_NORMALIZATION=true
TEXT_NORMALIZATION_LANG=en

# -----------------------------------------------------------------------------
# Logging & Debugging
# -----------------------------------------------------------------------------
LOG_LEVEL=INFO
ENABLE_PROFILING=false     # Enable NVTX profiling markers
ENABLE_METRICS=true        # Expose Prometheus metrics

# -----------------------------------------------------------------------------
# Resource Limits (Docker)
# -----------------------------------------------------------------------------
# Set in docker-compose.yml:
#   deploy:
#     resources:
#       limits:
#         memory: 8G
#       reservations:
#         memory: 4G
#         devices:
#           - capabilities: [gpu]
#             device_ids: ['0']

# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------
HEALTH_CHECK_INTERVAL=30   # Seconds
HEALTH_CHECK_TIMEOUT=10    # Seconds
```

---

## 8. References

### Official CosyVoice 2 Documentation

- **Repository**: https://github.com/FunAudioLLM/CosyVoice
- **Paper**: https://arxiv.org/abs/2407.05407
- **Demo**: https://funaudiollm.github.io/cosyvoice2/

### PyTorch Compatibility

- **PyTorch 2.3.1 Release Notes**: https://github.com/pytorch/pytorch/releases/tag/v2.3.1
- **PyTorch 2.7.0 Release Notes**: https://github.com/pytorch/pytorch/releases/tag/v2.7.0
- **CUDA Compatibility Matrix**: https://pytorch.org/get-started/locally/

### Docker GPU Runtime

- **nvidia-container-toolkit**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- **WSL2 GPU Support**: https://docs.nvidia.com/cuda/wsl-user-guide/

### Project Documentation

- [CLAUDE.md](../CLAUDE.md) - Project overview
- [project_documentation/TDD.md](../project_documentation/TDD.md) - Technical design
- [.claude/modules/milestones.md](../.claude/modules/milestones.md) - M6 milestone details
- [docs/MULTI_GPU.md](MULTI_GPU.md) - Multi-GPU deployment guide

---

## 9. Appendix: Quick Decision Matrix

| Criteria | Option 1: Separate venv | Option 2: Docker (RECOMMENDED) | Option 3: Mock Mode | Option 4: Downgrade |
|----------|-------------------------|--------------------------------|---------------------|---------------------|
| **Production Ready** | ❌ No | ✅ Yes | ⚠️ Partial | ❌ No |
| **Local Dev Experience** | ✅ Good | ⚠️ Moderate | ✅ Excellent | ✅ Good |
| **Dependency Isolation** | ⚠️ Partial | ✅ Complete | ⚠️ Partial | ❌ None |
| **CI/CD Complexity** | ⚠️ High | ⚠️ Moderate | ✅ Low | ✅ Low |
| **Deployment Complexity** | ❌ High | ✅ Low | ❌ High | ✅ Low |
| **Matches Prod Environment** | ❌ No | ✅ Yes | ❌ No | ⚠️ Partial |
| **GPU Access** | ✅ Direct | ✅ Passthrough | ✅ Direct | ✅ Direct |
| **Upgrade Path** | ✅ Good | ✅ Good | ✅ Good | ❌ Blocked |
| **Setup Time** | 1-2 hours | 3-4 hours | 30 min | 30 min |
| **Risk Level** | ⚠️ Medium | ✅ Low | ⚠️ Medium | ❌ High |

**Verdict**: Use **Option 2 (Docker)** for production + **Option 3 (Mock)** for fast development.

---

## 10. Next Steps

1. **Immediate** (before M6 implementation):
   - ✅ Review this document with team
   - ✅ Approve Docker isolation strategy
   - ✅ Update CLAUDE.md with PyTorch version note

2. **M6 Preparation** (1-2 days):
   - Create `Dockerfile.tts-cosyvoice`
   - Update `docker-compose.yml`
   - Test WSL2 GPU passthrough
   - Create `.env.cosyvoice` template

3. **M6 Implementation** (5-7 days):
   - Implement `adapter_cosyvoice2.py` with mock mode
   - Integration with Model Manager
   - Docker container testing
   - Performance benchmarking

4. **Documentation** (parallel with M6):
   - Update TDD.md with deployment strategy
   - Create WSL2 GPU setup guide
   - Update MULTI_GPU.md with CosyVoice example

---

**Questions or Issues?** See [.claude/agents/README.md](../.claude/agents/README.md) for agent coordination.
