# Testing Strategy & CI/CD Pipeline

**Last Updated**: 2025-10-16

This document provides comprehensive guidance on testing strategy, test execution, and the CI/CD pipeline.

## Test Coverage Summary

**Total Tests**: 649 tests (as of 2025-10-16)

**Breakdown:**
- **Unit tests**: 500 tests
- **Integration tests**: 139 tests
- **Performance tests**: 10 tests

**By Milestone:**
- **M0-M5**: 113 tests (core infrastructure, VAD, Model Manager, Piper adapter)
- **M10 ASR**: 128 tests (Whisper + WhisperX adapters, audio buffer, performance)
- **M10 Polish**: 65 tests (RMS buffer, session timeout, multi-turn conversation)
- **Other**: ~343 tests (config validation, utilities, etc.)

## Unit Tests (`tests/unit/`)

**Coverage by component:**
- VAD edge detection (M3): 29/29 tests ✅
- Piper adapter logic (M5): 15/15 tests ✅
- ASR base protocol (M10): 23/23 tests ✅
- Audio buffer (M10): 41/41 tests ✅
- RMS buffer / adaptive noise gate (M10 Polish): 31/31 tests ✅
- Session timeout validation (M10 Polish): 18/18 tests ✅
- Model manager lifecycle (M4): load/unload/TTL/evict/LRU ✅
- TTS control semantics: PAUSE/RESUME/STOP ✅
- Audio framing: exact 20 ms cadence, 48 kHz ✅
- Audio resampling: 22050Hz → 48kHz (Piper), 8kHz-48kHz → 16kHz (Whisper) ✅
- Routing policy logic (M9+): planned

**Running unit tests:**
```bash
# All unit tests
just test-unit

# Specific module
uv run pytest tests/unit/orchestrator/ -v
uv run pytest tests/unit/tts/adapters/ -v
uv run pytest tests/unit/asr/ -v

# Specific test file
uv run pytest tests/unit/orchestrator/test_vad.py -v

# With coverage
uv run pytest tests/unit/ --cov=src --cov-report=term-missing
```

## Integration Tests (`tests/integration/`)

**Coverage by milestone:**
- M1 Worker Integration: 16/16 tests ✅ (with --forked mode)
- M3 VAD Integration: 8/8 tests ✅
- M3 Barge-in Integration: 37/37 tests ✅
- M5 Piper Integration: 10/10 tests ✅
- M10 Whisper ASR Integration: 28/28 tests ✅
- M10 Whisper Performance: 11/11 tests ✅
- M10 Polish Multi-Turn Conversation: 22/22 tests ✅
- Full pipeline WebSocket tests: 6/8 passing (2 timeout - under investigation)

**Validations:**
- Loopback WebSocket test (FAL + frame timing) ✅
- Barge-in timing validation (<50 ms) ✅
- Piper FAL validation (<500ms CPU baseline) ✅
- Whisper transcription latency (<1.5s CPU) ✅
- Preload defaults honored (M4+) ✅

**Running integration tests:**
```bash
# All integration tests (with --forked for gRPC safety)
just test-integration

# Specific test file
uv run pytest tests/integration/test_m3_barge_in.py --forked -v
uv run pytest tests/integration/test_m5_piper_integration.py --forked -v
uv run pytest tests/integration/test_m10_whisper_integration.py --forked -v
```

## gRPC Testing in WSL2

**Issue**: grpc-python has segfault issues in WSL2 during test teardown

**Solution**: Use `--forked` flag for process isolation

```bash
# Automatic (recommended)
just test-integration  # Uses --forked automatically

# Manual
uv run pytest tests/integration/ --forked -v
```

**Details**: See [../../GRPC_SEGFAULT_WORKAROUND.md](../../GRPC_SEGFAULT_WORKAROUND.md)

**Status**: 100% mitigated with pytest-forked, tests reliable

**Alternative**: Skip gRPC tests in WSL2 (automatic detection), run in Docker or native Linux

## Running Tests

### Quick Commands

```bash
# All tests (full suite)
just test

# Unit tests only (fast feedback)
just test-unit

# Integration tests only (with --forked)
just test-integration

# With coverage report
uv run pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View coverage

# Verbose output
uv run pytest tests/ -vv

# Stop on first failure
uv run pytest tests/ -x

# Run specific marker
uv run pytest tests/ -m "not slow"
```

### Advanced Options

```bash
# Parallel execution (use with caution - requires thread-safe fixtures)
uv run pytest tests/unit/ -n auto

# Debug mode (drop into pdb on failure)
uv run pytest tests/ --pdb

# Only failed tests from last run
uv run pytest tests/ --lf

# Failed tests first, then rest
uv run pytest tests/ --ff

# Timeout for long-running tests
uv run pytest tests/ --timeout=300

# Capture logs
uv run pytest tests/ --log-cli-level=DEBUG
```

## CI/CD Pipeline

The project uses a modern **three-tier CI/CD strategy** optimized for fast feedback and comprehensive quality gates.

### CI Architecture

**Three-Tier Strategy:**

1. **Feature Branch CI** (`.github/workflows/feature-ci.yml`)
   - **Triggers**: Push to `feature/*`, `feat/*`, `fix/*`, etc.
   - **Duration**: ~3-5 minutes (60-70% faster than full suite)
   - **Strategy**: Smart test selection based on changed files
   - **Status**: Informational (non-blocking)
   - **Purpose**: Fast feedback during development

2. **Pull Request CI** (`.github/workflows/pr-ci.yml`)
   - **Triggers**: PR creation/updates to main
   - **Duration**: ~10-15 minutes
   - **Strategy**: Full test suite (all 649 tests) + code coverage
   - **Status**: **REQUIRED** (blocks merge if failing)
   - **Purpose**: Comprehensive quality gate before merge

3. **Main Branch** (no CI)
   - **Rationale**: Quality guaranteed by PR gates
   - **Benefit**: 44% reduction in CI minutes

### Feature Branch CI (Smart Test Selection)

**Workflow**: `.github/workflows/feature-ci.yml`

**Change Detection Logic:**
```
Changed Files → Test Categories
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
src/orchestrator/** → orchestrator unit + integration tests
src/tts/** → TTS unit + integration tests
src/asr/** → ASR unit + integration tests
src/rpc/** → all integration tests
pyproject.toml → full test suite
uv.lock → full test suite
configs/** → all integration tests
*.md, docs/** → skip all (docs-only)
```

**Jobs:**
1. **detect-changes**: Analyzes git diff to determine which tests to run
2. **lint**: Runs ruff on all code (always enabled for code changes)
3. **typecheck**: Runs mypy with protobuf generation (always enabled)
4. **test**: Runs selected test suites based on change detection
5. **summary**: Aggregates results and provides feedback

**Performance:**
- With caching: 30 sec dependency install (vs 5 min cold)
- Selective tests: 1-3 min test execution (vs 8-10 min full suite)
- Total: 3-5 minutes end-to-end

**Example Output:**
```bash
# Orchestrator changes detected
✅ Lint (ruff) - 45s
✅ Typecheck (mypy) - 1m 20s
✅ Test (orchestrator + integration) - 2m 30s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: 4m 35s (vs 12m 15s full suite)
```

### Pull Request CI (Full Validation)

**Workflow**: `.github/workflows/pr-ci.yml`

**Quality Gates (ALL must pass):**

1. **Lint** (ruff)
   - All code must follow style guidelines
   - No unused imports, variables, or code
   - Enforces project conventions

2. **Type Check** (mypy)
   - Strict mode must pass
   - All type annotations correct
   - No `type: ignore` without justification

3. **Full Test Suite** (pytest)
   - All 649 tests must pass
   - Unit tests (500 tests)
   - Integration tests (139 tests)
   - Performance tests (10 tests)
   - Excludes gRPC tests on non-Docker runners

4. **Code Coverage** (codecov)
   - Overall: ≥80% coverage required
   - Patch (new code): ≥60% coverage required
   - Threshold: 2% drop allowed from base
   - Automatic PR comments with coverage diff

5. **Security Scan** (bandit)
   - Scans for common security issues
   - Reports vulnerabilities in artifacts
   - Informational (non-blocking)

6. **Dependency Check** (pip-audit)
   - Scans for known vulnerabilities in dependencies
   - Reports CVEs and security advisories
   - Informational (non-blocking)

7. **Build Check**
   - Verifies `uv.lock` is up-to-date
   - Tests frozen dependency installation
   - Ensures reproducible builds

### Codecov Integration

**Configuration**: `codecov.yml`

```yaml
coverage:
  status:
    project:
      default:
        target: 80%  # Overall coverage target
        threshold: 2%  # Allow 2% drop
    patch:
      default:
        target: 60%  # New code coverage target
        threshold: 5%  # Allow 5% variance
```

**Features:**
- Automated coverage reporting on PRs
- Coverage diff visualization (shows exactly which lines need tests)
- Historical coverage tracking
- Branch coverage badges

**Setup:**
1. Sign up at [codecov.io](https://codecov.io) with GitHub account
2. Enable Codecov GitHub App for your repository
3. Get upload token: Settings → Repository → Upload Token
4. Add secret to GitHub: Settings → Secrets → Actions → New secret
   - Name: `CODECOV_TOKEN`
   - Value: (token from Codecov)
5. Codecov will automatically comment on PRs with coverage reports

### Caching Strategy

**uv Dependency Caching:**
```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v3
  with:
    version: "latest"
    enable-cache: true
    cache-dependency-glob: "uv.lock"
```

**Benefits:**
- **Cache hit** (same uv.lock): 30 sec install (90% faster)
- **Partial hit** (some deps changed): 2-3 min (50% faster)
- **Cache miss** (new deps): 5-6 min (baseline)

**Expected cache hit rates:**
- Feature branches: 90-95% (same dependencies as main)
- After dependency updates: 70-80% (partial cache reuse)

**Protobuf Stub Caching:**
```yaml
- name: Cache protobuf stubs
  uses: actions/cache@v4
  with:
    path: src/rpc/generated
    key: protobuf-${{ hashFiles('src/rpc/tts.proto') }}
```

**Benefits:**
- Skips protobuf generation if `.proto` file unchanged
- Saves 10-15 seconds per job (typecheck, test)

### Branch Protection Rules

**Required status checks** (configured in GitHub):

```
Branch: main
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Require pull request reviews: 1 reviewer
✅ Require status checks to pass:
   - PR CI / lint
   - PR CI / typecheck
   - PR CI / test
   - PR CI / build
✅ Require branches to be up to date
✅ Do not allow force pushes
✅ Do not allow deletions
```

**Setting up branch protection:**
1. Go to: Settings → Branches → Add rule
2. Branch name pattern: `main`
3. Enable "Require a pull request before merging"
4. Enable "Require status checks to pass before merging"
5. Select required checks (see above)
6. Enable "Require branches to be up to date before merging"
7. Save changes

### Performance Metrics

**Baseline (old CI):**
- Duration: 10-15 minutes per run
- Runs on: every push (feature + main) + PRs
- Caching: none
- CI minutes/month: ~5,400 min

**Optimized (new CI):**
- Feature CI: 3-5 minutes (60-70% faster)
- PR CI: 10-15 minutes (same, but with coverage)
- Main CI: none (0 runs)
- Caching: 80-90% faster dependency install
- CI minutes/month: ~3,000 min (44% reduction)

**Cost savings:**
- Free tier (GitHub Actions): 2,000 min/month
- Estimated usage: 3,000 min/month
- Overage: 1,000 min @ $0.008/min = $8/month
- Previous: ~$27/month
- **Savings: $19/month (70% reduction)**

## Running CI Locally

**Feature CI equivalent:**
```bash
# Check what would run based on your changes
git diff --name-only main...HEAD

# Run affected test suites
just ci  # Runs: lint + typecheck + test
```

**PR CI equivalent:**
```bash
# Run full test suite with coverage
uv run pytest tests/ \
  -v \
  --cov=src \
  --cov-report=xml \
  --cov-report=term \
  --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

**Pre-commit hooks** (recommended):
```bash
# Install pre-commit hooks
uv pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## Troubleshooting CI Failures

### Lint Failures
```bash
# Auto-fix most issues
just fix

# Check remaining issues
just lint
```

### Type Check Failures
```bash
# Regenerate protobuf stubs
just gen-proto

# Run type check
just typecheck

# Common issues:
# - Missing type annotations: add type hints
# - Import errors: check module paths
# - Protobuf stubs: ensure gen-proto was run
```

### Test Failures
```bash
# Run specific test file
uv run pytest tests/unit/orchestrator/test_session.py -v

# Run with verbose output
uv run pytest tests/ -vv --tb=long

# Run with pdb on failure
uv run pytest tests/ --pdb

# For gRPC tests in WSL2
just test-integration  # Uses --forked flag
```

### Coverage Failures
```bash
# Generate coverage report locally
uv run pytest tests/ --cov=src --cov-report=term-missing

# Identify untested lines
uv run pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Common issues:
# - Add tests for new functionality
# - Remove dead code
# - Mark test-only code with # pragma: no cover
```

### Cache Issues

If CI is slower than expected:

1. Check cache hit rate in CI logs:
   ```
   Cache restored successfully
   Cache restored from key: Linux-uv-abc123...
   ```

2. Clear cache if corrupted (GitHub UI):
   - Settings → Actions → Caches → Delete caches

3. Force cache refresh (temporary):
   - Update `uv.lock`: `uv lock --upgrade`

## CI Monitoring

**Key metrics to track:**

1. **CI Duration**
   - Feature CI: Target < 5 min
   - PR CI: Target < 15 min
   - Trend: Should decrease over time with caching

2. **Cache Hit Rate**
   - Target: >90% for feature branches
   - Monitor: CI logs show cache restore success

3. **Test Flakiness**
   - Target: <1% flaky tests
   - Monitor: GitHub Actions logs for intermittent failures
   - Action: Investigate and fix flaky tests immediately

4. **Coverage Trend**
   - Target: Maintain or increase coverage
   - Monitor: Codecov dashboard
   - Alert: If coverage drops >2% from main

**GitHub Actions insights:**
- Navigate to: Actions tab → Workflow → View workflow runs
- Filter by: branch, status, time period
- Analyze: duration trends, failure rates, cache effectiveness

## Best Practices

### For Developers

1. **Run local checks before pushing:**
   ```bash
   just ci  # Runs lint + typecheck + test
   ```

2. **Keep feature branches small:**
   - Smaller changes = faster CI
   - Easier to review and debug failures

3. **Fix CI failures immediately:**
   - Feature CI failures are warnings
   - PR CI failures block merge (fix required)

4. **Monitor coverage:**
   - Add tests for new functionality
   - Aim for >80% coverage on new code

5. **Update dependencies carefully:**
   - Run `uv lock` locally first
   - Test thoroughly before pushing
   - Expect longer CI times after updates

### For Maintainers

1. **Monitor CI health:**
   - Check weekly: duration trends, failure rates
   - Investigate: flaky tests, slow jobs
   - Optimize: add caching, parallelize tests

2. **Keep workflows updated:**
   - Update actions versions regularly
   - Review GitHub Actions changelog
   - Test workflow changes on feature branches

3. **Manage secrets:**
   - Rotate Codecov token annually
   - Limit secret access to required jobs
   - Never log secrets in CI output

4. **Branch protection:**
   - Enforce required checks
   - Require code reviews
   - Keep main branch green always

## Future Improvements

**Planned enhancements (M11-M12):**

1. **Parallel test execution** (M11):
   - Use `pytest-xdist` for parallel testing
   - Target: 50% faster test execution
   - Risk: Requires thread-safe test fixtures

2. **Matrix testing** (M11):
   - Test multiple Python versions (3.12, 3.13)
   - Test multiple OS (Ubuntu, macOS, Windows)
   - Increases CI time but improves compatibility

3. **GPU CI runners** (M11):
   - Test GPU-dependent code (TTS adapters)
   - Requires self-hosted runners with NVIDIA GPUs
   - Cost: $1-2/hour for GPU instances

4. **Performance benchmarking** (M11):
   - Automated performance regression tests
   - Track metrics: latency, throughput, memory
   - Alert on >10% regression

5. **Deployment preview** (M12):
   - Automatic staging deployments for PRs
   - Test full stack in isolated environment
   - Requires Kubernetes or similar infrastructure

## References

- **Development Guide**: [development.md](development.md)
- **Architecture**: [architecture.md](architecture.md)
- **Testing Guide (docs)**: [../../docs/TESTING_GUIDE.md](../../docs/TESTING_GUIDE.md)
- **gRPC Workaround**: [../../GRPC_SEGFAULT_WORKAROUND.md](../../GRPC_SEGFAULT_WORKAROUND.md)
