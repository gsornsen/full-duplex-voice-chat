# UV Workspace Migration - Current Status

**Last Updated**: 2025-10-30
**Status**: ⚠️ Partially Complete - Requires Completion

## What Was Accomplished

### ✅ Workspace Structure Created
- `packages/proto/` - gRPC protocol definitions
- `packages/common/` - Shared utilities
- `packages/tts-base/` - TTS base classes
- `packages/orchestrator/` - Orchestrator service
- `packages/tts-piper/` - Piper TTS adapter
- `packages/tts-cosyvoice/` - CosyVoice TTS adapter

### ✅ Dependencies Configured
- Root `pyproject.toml` defines workspace
- Each package has its own `pyproject.toml`
- `uv.lock` generated (88KB, 56 packages)

### ✅ Code Migrated
- All source code copied to `packages/`
- Directory structure established
- `__init__.py` files created

### ⚠️ Import Migration - Partially Complete
- Test imports fixed (sed-based migration)
- Package imports fixed
- Patch statements fixed
- **Remaining**: Some edge cases in test fixtures

### ❌ Not Yet Complete
- src/ directory emptied but needs proper cleanup
- Not all tests passing yet (dependency resolution issues)
- Docker builds not yet updated for workspace
- Documentation references old structure

## Current Issues

### Issue 1: Package Dependencies

**Problem**: Orchest rator and TTS adapters are not workspace members (by design for isolation), but tests expect them installed.

**Solution Needed**:
```bash
# Install workspace members
uv sync

# Install service packages separately
uv pip install -e packages/orchestrator
uv pip install -e packages/tts-piper
# (tts-cosyvoice conflicts with proto - Docker only)
```

### Issue 2: Module Resolution

**Problem**: Some tests try to import `tts.adapters.adapter_mock` but adapters are in separate packages.

**Solution Needed**:
- Move adapter_mock to tts-base (shared test adapter)
- Or update tests to use correct package paths

### Issue 3: Test File Skips

Currently skipping:
- `tests/unit/test_cli_client.py` - client not in workspace
- `tests/unit/test_adapter_piper.py` - piper not installed
- `tests/unit/tts/` - adapter tests need packages installed

## Next Steps to Complete Migration

### Step 1: Fix Package Structure (30 min)
- Move `adapter_mock.py` to `tts-base/src/tts/adapters/`
- Ensure all shared test code is in workspace members
- Add missing `__init__.py` files

### Step 2: Install Packages (10 min)
```bash
# Workspace members
uv sync

# Service packages (non-conflicting)
uv pip install -e packages/orchestrator
uv pip install -e packages/tts-piper

# Skip cosyvoice (Docker only due to protobuf conflict)
```

### Step 3: Run Test Suite (30 min)
```bash
# Core tests
uv run pytest tests/unit/orchestrator/ -v

# All tests (skip cosyvoice)
uv run pytest tests/unit/ --ignore=tests/unit/tts/adapters/test_adapter_cosyvoice.py

# Integration tests
uv run pytest tests/integration/ --ignore=tests/integration/test_cosyvoice_integration.py
```

### Step 4: Update Dockerfiles (1 hour)
- Update COPY commands to use `packages/` instead of `src/`
- Ensure workspace packages are available in containers
- Test builds

### Step 5: Documentation (30 min)
- Update CLAUDE.md with new import paths
- Update development.md with workspace commands
- Create workspace usage guide

## Rollback Plan

If migration proves too complex:

```bash
# Restore original src/ structure
git checkout HEAD -- src/

# Remove workspace changes
git checkout HEAD -- pyproject.toml
rm -rf packages/

# Keep Docker optimizations (they're separate)
# Commit those independently
```

## Testing Commands

```bash
# Test workspace lock
uv lock

# Test package installation
uv sync
uv pip install -e packages/orchestrator

# Test imports
uv run python -c "from rpc.generated import tts_pb2; print('✓')"
uv run python -c "from orchestrator.config import Config; print('✓')"

# Test suite
uv run pytest tests/unit/orchestrator/ -v
```

## Success Criteria

- [ ] uv.lock includes all dependencies
- [ ] All workspace packages install cleanly
- [ ] All imports resolve correctly
- [ ] All unit tests pass
- [ ] Docker builds work with workspace
- [ ] Documentation updated

## Current Test Results

**Last Run**: 2025-10-30
**Results**: 40/84 orchestrator tests passing, 44 skipped
**Errors**: Module resolution issues in fixtures

## Questions to Resolve

1. Should adapter_mock be in tts-base or tts-piper?
2. Should CLI client become a workspace package?
3. How to handle plugins/ directory (grpc_tts, whisperx)?
4. Keep src/ for plugins that aren't migrated yet?

## Recommendations

1. **Complete the migration** - We're 80% there
2. **Move adapter_mock to tts-base** - It's a test utility
3. **Skip cosyvoice local install** - Docker only
4. **Create plugins package** - For grpc_tts and whisperx
5. **Document the new structure** - Clear import examples

---

**Status**: Migration infrastructure is solid, needs finishing touches to be fully functional.
