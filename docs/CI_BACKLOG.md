# CI/CD Backlog and Future Improvements

**Last Updated**: 2025-10-25

This document tracks future improvements and investigations for the CI/CD pipeline.

## Current State

As of October 2025:
- **Status**: GPU tests skipped in CI using `@pytest.mark.gpu` marker
- **Test Coverage**: 647 tests run in CI (2 GPU tests excluded)
- **CI Platform**: GitHub Actions (free tier, CPU-only runners)
- **Local Testing**: GPU tests run manually on developer machines with CUDA

## Priority Items

### P0: GPU CI Runner Investigation

**Status**: Backlog (not started)

**Problem**:
Currently, 2 GPU-required tests are excluded from CI:
1. `test_process_gpu_acceleration` - Validates GPU-accelerated resampling
2. `test_push_gpu_processing` - Validates GPU-based frame buffering

These tests require actual CUDA-capable GPU hardware and cannot run on GitHub Actions free tier runners.

**Options to Consider**:

#### Option A: Self-Hosted GPU Runner
- **Pros**:
  - Full control over hardware
  - Can use existing development GPU
  - No per-minute costs
  - Fastest execution
- **Cons**:
  - Requires always-on machine or manual runner management
  - Network bandwidth costs (if using home internet)
  - Maintenance overhead
  - Security considerations (exposing runner to public repos)
- **Estimated Cost**: $0/month (using existing hardware) + electricity
- **Setup Time**: 2-4 hours

#### Option B: GitHub Actions GPU Runners (Paid)
- **Pros**:
  - Fully managed
  - Pay-per-use model
  - Scales with team size
  - Integrated with existing workflows
- **Cons**:
  - Higher per-minute cost
  - Limited GPU options
  - May have availability issues
- **Estimated Cost**: $0.07-0.16/minute (varies by GPU tier)
- **Monthly Estimate**: ~$50-100/month for moderate usage
- **Documentation**: https://docs.github.com/en/actions/using-github-hosted-runners/about-larger-runners

#### Option C: Third-Party GPU CI Services
Examples: CircleCI GPU runners, GitLab GPU runners, AWS CodeBuild with GPU

- **Pros**:
  - Specialized GPU infrastructure
  - Better GPU selection
  - Often better pricing than GitHub
- **Cons**:
  - Additional service to manage
  - Migration effort from GitHub Actions
  - Learning curve for new platform
- **Estimated Cost**: $30-80/month depending on provider

#### Option D: Conditional GPU Testing (Smart Scheduling)
- **Pros**:
  - Minimize GPU runner usage
  - Run GPU tests only when GPU code changes
  - Use path-based triggers
- **Cons**:
  - May miss GPU regressions
  - More complex workflow logic
  - Requires manual GPU test runs for some PRs
- **Estimated Cost**: Reduces any paid option by 60-80%

**Recommendation Timeline**:
- **Now**: Continue skipping GPU tests in CI (current approach)
- **Q1 2026**: Re-evaluate when team size > 3 or GPU code becomes critical path
- **Trigger for Action**: Any GPU-related production incident

**Decision Criteria**:
- Team size reaches 5+ developers
- GPU code changes become frequent (>10% of PRs)
- GPU-related bugs escape to production
- Budget allocation for CI infrastructure

### P1: Performance Test Baseline Collection

**Status**: Backlog

**Goal**: Establish performance baselines for TTS/ASR operations to detect regressions

**Requirements**:
- Dedicated performance testing environment
- Consistent hardware specs
- Historical data storage
- Automated regression detection

**Estimated Effort**: 1-2 weeks

### P2: Multi-GPU Test Coverage

**Status**: Backlog

**Goal**: Add integration tests for multi-GPU deployments (M13 milestone)

**Dependencies**:
- Multi-GPU infrastructure (physical or cloud)
- Test orchestration framework
- GPU allocation strategies

**Estimated Effort**: 2-3 weeks

## Test Statistics

### GPU Test Details

| Test Name | File | Lines | Purpose | Risk if Skipped |
|-----------|------|-------|---------|-----------------|
| `test_process_gpu_acceleration` | `tests/unit/tts/audio/test_streaming.py` | 148-171 | Validates GPU tensor operations for resampling | **Low** - CPU path well-tested, GPU is performance optimization |
| `test_push_gpu_processing` | `tests/unit/tts/audio/test_streaming.py` | 382-400 | Validates GPU tensor operations for frame buffering | **Low** - CPU path well-tested, GPU is performance optimization |

**Total GPU-only tests**: 2 (0.3% of test suite)

**Risk Assessment**:
- Both GPU tests validate performance optimizations, not core functionality
- CPU fallback paths are thoroughly tested (100+ CPU-based tests)
- GPU functionality is manually validated during development
- Production deployments use CPU workers (Piper) by default

### Performance Test Details

| Test Category | Count | Markers | Runs in CI |
|---------------|-------|---------|------------|
| Performance | 10 | `@pytest.mark.performance` | No (too slow) |
| GPU Required | 2 | `@pytest.mark.gpu` | No (no GPU) |
| Infrastructure | 15 | `@pytest.mark.infrastructure` | No (flaky on shared runners) |
| gRPC | 30 | `@pytest.mark.grpc` | No (WSL2 segfault issues) |
| Unit (fast) | 540 | `@pytest.mark.unit` | Yes |
| Integration | 50 | `@pytest.mark.integration` | Partial (Docker-based only) |

## References

- **pytest.ini**: Test marker definitions and CI exclusion patterns
- **pr-ci.yml**: GitHub Actions workflow with GPU test exclusion
- **pre-push hook**: Local validation before pushing
- **docs/INFRASTRUCTURE_TESTING.md**: Infrastructure test strategy

## Change Log

| Date | Change | Author | Reason |
|------|--------|--------|--------|
| 2025-10-25 | Created backlog, added GPU runner investigation | DevOps Engineer | Track GPU CI options for future |
