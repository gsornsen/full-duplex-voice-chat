# Test Repair Coordination Summary

## Executive Summary

The LiveKit transport integration debugging session successfully implemented dual transport support (WebSocket + LiveKit) for the orchestrator. However, several test files now need updates to align with the new implementation.

**Status**: Ready for test repair execution
**Estimated Time**: 4-5 hours
**Complexity**: Medium (config fixes, import updates, fixture changes)

---

## What Changed (Implementation)

### 1. Directory Rename
- **Old**: `src/orchestrator/livekit/`
- **New**: `src/orchestrator/livekit_utils/`
- **Reason**: Avoid circular import with `livekit` package
- **Impact**: Import errors in any test using old path

### 2. Dual Transport Architecture
- **Before**: WebSocket transport only
- **After**: Both WebSocket AND LiveKit transports running concurrently
- **Implementation**:
  ```python
  # Separate async loops for each transport
  async def websocket_loop(): ...
  async def livekit_loop(): ...

  # Both use same session handler
  session_manager = SessionManager(session)
  handle_session(session_manager, worker_client)
  ```

### 3. Configuration Enhancements
- Added environment variable overrides:
  - `REDIS_URL` → `redis.url`
  - `LIVEKIT_URL` → `transport.livekit.url`
  - `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- Schema enforces Pydantic models (not dicts)

### 4. Health Check Updates
- `HealthCheckHandler` now requires `worker_client` parameter
- Response includes `"status"` field

---

## Critical Bug Identified

### In: `tests/integration/conftest.py` (Line 374)

**Problem**: Config uses dict for redis instead of RedisConfig model

```python
# WRONG (current):
config = OrchestratorConfig(
    redis={"url": redis_container},  # ❌ Dict won't validate
    ...
)

# CORRECT (required):
config = OrchestratorConfig(
    redis=RedisConfig(url=redis_container),  # ✅ Proper Pydantic model
    ...
)
```

This will cause Pydantic validation errors when tests run.

---

## Test Repair Strategy

### Phase 1: Discovery (15 min)
- Run full test suite
- Categorize failures by type:
  - Import errors (livekit → livekit_utils)
  - Validation errors (config schema)
  - Assertion errors (test expectations)
  - Connection/timeout errors

### Phase 2: Fix Imports (30 min)
- Search for `from src.orchestrator.livekit`
- Replace with `from src.orchestrator.livekit_utils`
- Verify test collection succeeds

### Phase 3: Fix Configuration (45 min)
- Update `orchestrator_server` fixture in conftest.py:
  - Import `RedisConfig`
  - Use `RedisConfig(url=...)` instead of dict
  - Add `LiveKitConfig(enabled=False)` explicitly
  - Fix YAML serialization: `config.model_dump(mode='json')`

### Phase 4: Fix Server Lifecycle (60 min)
- Update health check fixtures (add `worker_client` param)
- Account for dual transport startup logs
- Verify shutdown cleans up both transports

### Phase 5: Fix Session Management (45 min)
- Ensure `SessionManager` wrapper is used
- Update session test expectations
- Verify text → audio flow works

### Phase 6: Validation (30 min)
- Run full CI: `just ci`
- Validate M2 completion criteria
- Update milestone checklist if complete

---

## Key Files to Fix

### High Priority (MUST FIX)
1. **`tests/integration/conftest.py`**
   - Lines 29-35: Import `RedisConfig`
   - Lines 363-376: Fix `orchestrator_server` fixture
   - Lines 378-383: Fix YAML serialization

### Medium Priority (LIKELY FIX)
2. **Any files importing from `src.orchestrator.livekit`**
3. **`tests/integration/test_full_pipeline.py`** - session expectations
4. **Health check test fixtures** (if exist)

### Low Priority (UNLIKELY TO CHANGE)
5. **`tests/integration/test_m1_worker_integration.py`** - direct gRPC tests
6. **Unit tests** - mostly isolated from changes

---

## Expected Outcomes

### Success Criteria (MUST PASS)
✅ All unit tests pass
✅ Worker integration tests pass
✅ Type checking passes (`mypy`)
✅ Linting passes (`ruff`)
✅ Full CI passes (`just ci`)

### Validation Criteria (SHOULD PASS)
✅ Full pipeline tests pass
✅ WebSocket e2e tests pass
✅ No test collection errors

### M2 Completion Criteria (MILESTONE)
✅ WS echo test: text→worker→audio→back
✅ WebRTC echo works locally (LiveKit)
✅ Redis keys reflect worker registration

---

## Coordination Approach

### Agent Roles

**@multi-agent-coordinator (this agent)**:
- Created comprehensive test repair plan
- Identified critical bug in conftest.py
- Documented all changes and expected fixes
- Prepared handoff package for @agent-python-pro
- Standing by for progress monitoring and blocker resolution

**@agent-python-pro (execution agent)**:
- Will execute 6-phase repair plan
- Fix imports, config, fixtures, expectations
- Run validation at each step
- Report progress and escalate blockers
- Ensure all tests pass and CI is green

### Communication Protocol
- Progress updates after each phase
- Immediate escalation of blockers
- Final validation report with M2 status
- Documentation of any deferred issues

---

## Risk Assessment

### High Risk (Mitigated)
✅ **Config serialization** - Documented exact fix (RedisConfig model)
✅ **Import errors** - Straightforward search/replace pattern
✅ **Dual transport** - Implementation is solid, tests just need updates

### Medium Risk (Manageable)
⚠️ **Health check changes** - Well-documented, clear fix path
⚠️ **Session wrapping** - May need careful fixture updates
⚠️ **Test expectations** - Update to match new behavior

### Low Risk (Minimal)
✅ **Worker gRPC tests** - Unchanged, should pass as-is
✅ **Type checking** - Import fixes will resolve
✅ **Linting** - Auto-fixable with `just fix`

---

## Deliverables

### Documentation Created
1. **`TEST_REPAIR_PLAN.md`** - Detailed technical repair plan
2. **`AGENT_HANDOFF.md`** - Comprehensive handoff to @agent-python-pro
3. **`test-repair-summary.md`** (this file) - Executive summary

### Test Logs (To Be Generated)
4. `test_failures.log` - Full test run output
5. `import_errors.log` - Import failure categorization
6. `assertion_errors.log` - Assertion failure categorization
7. `validation_errors.log` - Pydantic validation errors

### Updates (After Repair)
8. Modified test files (conftest.py, etc.)
9. Updated M2 milestone checklist (if validation complete)
10. Final validation report

---

## Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 1 | Discovery | 15 min | 🔜 Ready to start |
| 2 | Fix Imports | 30 min | ⏳ Pending Phase 1 |
| 3 | Fix Config | 45 min | ⏳ Pending Phase 2 |
| 4 | Fix Server | 60 min | ⏳ Pending Phase 3 |
| 5 | Fix Sessions | 45 min | ⏳ Pending Phase 4 |
| 6 | Validation | 30 min | ⏳ Pending Phase 5 |

**Total Estimated Time**: 4.5 hours (includes 1-hour buffer)

---

## Next Steps

### Immediate (Now)
1. ✅ Test repair plan created (this document)
2. ✅ Agent handoff prepared (`AGENT_HANDOFF.md`)
3. ✅ Critical bug identified (conftest.py config)
4. 🔜 Hand off to @agent-python-pro for execution

### Short Term (Today)
5. Execute Phase 1: Discovery and categorization
6. Execute Phase 2-5: Systematic repairs
7. Execute Phase 6: Final validation
8. Update M2 milestone if complete

### Follow-up
9. Archive repair documentation to `docs/runbooks/`
10. Document lessons learned
11. Update testing guidelines if needed

---

## Critical Reminders

### DO:
✅ Fix tests to match implementation (implementation is correct)
✅ Use proper Pydantic models (RedisConfig, not dict)
✅ Account for dual transport architecture
✅ Validate at each phase
✅ Report blockers immediately

### DON'T:
❌ Modify implementation to fix tests
❌ Skip validation steps
❌ Disable tests without documentation
❌ Deviate from project architecture
❌ Change server.py, config.py, or health.py

---

## Contact & Support

**Coordination Agent**: @multi-agent-coordinator
- Monitoring progress
- Available for blocker resolution
- Validates milestone completion
- Ensures architectural alignment

**Execution Agent**: @agent-python-pro
- Executes repair plan
- Reports progress
- Escalates blockers
- Delivers passing tests

---

## Quick Commands Reference

```bash
# Working directory
cd /home/gerald/git/full-duplex-voice-chat

# Discovery
uv run pytest tests/ -v --tb=short 2>&1 | tee test_failures.log

# Test specific file
uv run pytest tests/integration/conftest.py -v

# Full CI suite
just ci

# Type check
uv run mypy src/ tests/

# Lint check
uv run ruff check src/ tests/
```

---

**Status**: Test repair coordination complete. Ready for execution by @agent-python-pro.

**Next Action**: Begin Phase 1 (Discovery) - run test suite and categorize failures.

---

**Related Documentation:**
- [Known Issues](../known-issues/README.md)
- [Testing Guide](../TESTING_GUIDE.md)
- [Development Guide](../DEVELOPMENT.md)
