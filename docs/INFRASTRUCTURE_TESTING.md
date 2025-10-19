# Infrastructure Testing Strategy

**Last Updated**: 2025-10-18

## Overview

This document explains our approach to testing Docker Compose infrastructure and development workflow tooling, and why these tests are **excluded from CI**.

## Problem Statement

### The Flakiness Issue

Infrastructure tests like `test_dev_infra_only` were experiencing chronic flakiness in GitHub Actions CI despite multiple tactical fixes:

1. **Initial failure**: Health checks timeout after 30s
2. **After timeout increase to 90s**: Still failing intermittently
3. **After Caddy health check change**: Still failing
4. **Pattern**: Unpredictable failures on shared CI runners

### Root Cause: Fundamental Architecture Mismatch

The problem is **not** the test implementation - it's a **category error**. Infrastructure tests validate **development workflow tooling** (Docker Compose, justfile commands) in a **CI environment**, which creates fundamental conflicts:

**Resource Constraints on Shared Runners:**
- GitHub Actions runners: 2 CPU cores, 7GB RAM
- Shared with concurrent jobs from multiple users
- Unpredictable resource availability

**Network Timing Variability:**
- Docker daemon startup varies wildly (5-90s)
- Image pulls depend on network speed and cache state
- Health checks have high variance on constrained runners

**Wrong Abstraction Layer:**
- CI should test **business logic**, not **infrastructure tooling**
- Docker Compose is a development convenience, not production deployment
- Infrastructure health already validated by docker-compose and service health checks

**Inherent Flakiness:**
- No amount of timeout tuning fixes the underlying issue
- Shared CI runners have fundamentally different performance characteristics than developer workstations

## Solution: Pytest Marker + Optional Local Execution

### Strategy

We created a new `@pytest.mark.infrastructure` marker for Docker Compose tests:

1. **Excluded from CI**: Both feature branch and PR CI exclude these tests
2. **Available locally**: Developers can run `just test-infrastructure` to validate workflow changes
3. **Zero CI flakiness**: Tests that can't be made deterministic in CI are simply not run there
4. **Optional scheduled testing**: Can be added to nightly/weekly workflow if desired

### Implementation

**Pytest Markers:**
```python
@pytest.mark.infrastructure  # Identifies infrastructure smoke tests
@pytest.mark.ci_skip         # Additional marker for clarity
```

**Test Exclusion:**
```bash
# CI runs this (excludes infrastructure tests)
pytest tests/ -v -m "not grpc and not infrastructure"

# Developers run this locally when modifying Docker Compose or justfile
pytest tests/ -v -m "infrastructure"
```

**Justfile Commands:**
```bash
# Standard test suite (excludes infrastructure)
just test

# Integration tests (excludes infrastructure)
just test-integration

# Infrastructure smoke tests (local only)
just test-infrastructure
```

## Test Categories

### What IS an Infrastructure Test?

Tests marked with `@pytest.mark.infrastructure`:

- **Docker Compose startup/shutdown**: `test_dev_infra_only`, `test_dev_clean_state`
- **Justfile command validation**: `test_dev_idempotent`
- **Development workflow tooling**: Hot-swapping models, container management
- **Multi-service orchestration**: Starting Redis + LiveKit + Caddy together

### What is NOT an Infrastructure Test?

Tests that remain in CI:

- **Business logic**: TTS synthesis, VAD processing, session management
- **API contracts**: gRPC protocol, WebSocket messages
- **Integration with services**: Redis service discovery, LiveKit Agent
- **Performance benchmarks**: First Audio Latency, frame timing

The distinction: **Infrastructure tests validate tooling; integration tests validate service behavior**.

## Running Tests

### In CI (Automatic)

CI automatically excludes infrastructure tests:

```yaml
# .github/workflows/pr-ci.yml
- name: Run pytest with coverage
  run: |
    uv run pytest tests/ \
      -v \
      -m "not grpc and not infrastructure" \
      --cov=src
```

**Result**: Zero flakiness from Docker Compose timing issues.

### Locally (Developer Workflow)

**Run infrastructure tests before committing workflow changes:**

```bash
# After modifying docker-compose.yml or justfile
just test-infrastructure

# Output shows Docker Compose health checks
# Tests take 60-120s depending on cold/warm start
```

**Run full test suite (excludes infrastructure):**

```bash
# Standard workflow (fast feedback)
just test                    # Unit tests only (~30s)
just test-integration        # Integration tests (~2-3 min)
just ci                      # Full CI suite (~5 min)

# All tests EXCEPT infrastructure
pytest tests/ -v -m "not infrastructure"
```

**Run infrastructure tests alongside integration tests:**

```bash
# Run everything (including infrastructure)
pytest tests/ -v

# Run only integration + infrastructure
pytest tests/integration/ -v -m "integration or infrastructure"
```

## When to Update Infrastructure Tests

### Modify Infrastructure Tests When:

1. **Adding new Docker Compose services**: Update `test_dev_infra_only` to include new health checks
2. **Changing justfile commands**: Add/update tests in `test_unified_workflow.py`
3. **Modifying container startup logic**: Validate new orchestration patterns
4. **Adding Docker Compose profiles**: Test profile switching behavior

### Don't Modify Infrastructure Tests For:

1. **Business logic changes**: Use unit/integration tests instead
2. **API changes**: Test via integration tests with running services
3. **Performance tuning**: Use performance benchmark tests
4. **Bug fixes in services**: Mock the infrastructure, test the logic

## Test Maintenance

### Keeping Infrastructure Tests Reliable

Infrastructure tests are **local smoke tests**, not CI gates:

1. **Run before committing workflow changes**: `just test-infrastructure`
2. **Accept longer execution times**: 60-120s is normal for cold Docker start
3. **Don't optimize for CI**: These tests won't run there
4. **Focus on developer experience**: Fast feedback for workflow changes

### Handling Failures

**If `just test-infrastructure` fails locally:**

1. **Check Docker daemon**: `docker ps` should work
2. **Clean state**: `just dev-clean` to remove old containers
3. **Review logs**: `just dev-logs` or `docker compose logs`
4. **Increase timeouts**: Acceptable for local testing (not CI)

**If you're tempted to add infrastructure tests to CI:**

**DON'T.** The flakiness will return. Instead:

1. **Mock the infrastructure**: Test business logic with mocked Docker/Compose
2. **Use integration tests**: Validate service behavior with running containers
3. **Add nightly workflow**: Scheduled run with higher timeout tolerance

## Design Rationale

### Why Not Mock Docker Compose?

**Mocking docker-compose commands defeats the purpose:**

- We want to validate **real Docker Compose behavior**
- Mocks test our assumptions, not the tooling itself
- Infrastructure tests are smoke tests for developer workflow

### Why Not Increase Timeouts in CI?

**Timeouts don't solve the root cause:**

- Even 5-minute timeouts fail on heavily loaded runners
- Wastes CI minutes on infrastructure that's already validated by docker-compose
- Creates false sense of reliability (passes 95% of time, blocks PRs 5% of time)

### Why Not Use Dedicated Runners?

**Cost vs. benefit:**

- Self-hosted runners cost $$/month
- Solves wrong problem (validates tooling, not business logic)
- Better to run infrastructure tests locally where they're fast and deterministic

## Alternative: Scheduled Infrastructure Validation

If you need CI validation of Docker Compose workflows, create a **separate scheduled workflow**:

```yaml
# .github/workflows/infrastructure-nightly.yml
name: Infrastructure Tests (Nightly)

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:     # Manual trigger

jobs:
  infrastructure:
    name: Docker Compose Smoke Tests
    runs-on: ubuntu-latest
    timeout-minutes: 30  # Higher timeout acceptable for scheduled jobs
    steps:
      - uses: actions/checkout@v4
      - name: Run infrastructure tests
        run: |
          uv run pytest tests/ -v -m "infrastructure"
```

**Benefits:**
- Doesn't block PRs
- Catches infrastructure regressions
- Higher timeout tolerance (scheduled, not blocking)
- Can alert on failure without blocking development

**When to use:**
- Large teams with frequent docker-compose.yml changes
- Production deployments use docker-compose (rare - most use K8s)
- Regulatory requirement for infrastructure validation

## Summary

**Infrastructure tests are valuable for local development validation** but unsuitable for CI due to inherent flakiness on shared runners.

**Solution:**
- ✅ New `@pytest.mark.infrastructure` marker
- ✅ Excluded from feature branch and PR CI (`-m "not infrastructure"`)
- ✅ Available locally via `just test-infrastructure`
- ✅ Zero CI flakiness, fast developer feedback

**Impact on Development Velocity:**
- **Before**: ~5-10% PR failures due to Docker Compose timeouts, wasted CI minutes, developer frustration
- **After**: 0% infrastructure-related CI failures, developers validate workflow changes locally in < 2 min

**Trade-off:**
- **Lost**: Automated validation of Docker Compose changes in CI
- **Gained**: Reliable CI, faster feedback, eliminated whack-a-mole timeout tuning

**Key Principle**: Test business logic in CI, test infrastructure tooling locally.
