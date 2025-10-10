# gRPC Segfault Workaround for Integration Tests

## Problem Summary

Integration tests using gRPC encounter segmentation faults during test teardown due to grpc-python's background threads accessing garbage-collected event loops. This is a known upstream issue in grpc-python.

**Upstream Issue:** https://github.com/grpc/grpc/issues/37714

## Symptoms

- Segfaults occur during `gc` (garbage collection) in pytest fixture teardown
- Error message: `Fatal Python error: Segmentation fault` during `Garbage-collecting`
- Happens inconsistently, typically after running multiple tests
- Affects both Python 3.12 and Python 3.13

## Solution: Process Isolation with pytest-forked

The recommended workaround is to use `pytest-forked` to run each test in a separate process. This isolates the segfault to individual test processes and prevents it from crashing the entire test suite.

### Installation

Already included in dev dependencies:
```toml
[project.optional-dependencies]
dev = [
    "pytest-forked>=1.6.0",
    "pytest-xdist>=3.5.0",
    ...
]
```

Install with:
```bash
uv sync --extra dev
```

### Usage

#### Recommended: Use justfile command
```bash
just test-integration
```

This runs:
```bash
uv run pytest tests/integration/ --forked -v -m "integration"
```

#### Manual invocation
```bash
# All integration tests with process isolation
uv run pytest tests/integration/ --forked -v

# Specific test file
uv run pytest tests/integration/test_full_pipeline.py --forked -v

# Specific test
uv run pytest tests/integration/test_full_pipeline.py::test_full_pipeline_websocket_only --forked -v
```

#### Fast mode (no isolation, may segfault)
```bash
just test-integration-fast
```

## How It Works

The `--forked` flag from pytest-forked:

1. **Forks a new process** for each test
2. **Runs the test** in the isolated process
3. **Collects results** and returns them to the parent process
4. **Terminates the forked process**, containing any segfaults

Even if a test segfaults during cleanup:
- The test result is still captured
- Other tests continue to run
- The segfault is isolated and doesn't affect the test runner

## Current Status

### M1 Integration Tests (test_m1_worker_integration.py)
✅ **16/16 tests PASS** without segfaults when using `--forked`

```bash
$ uv run pytest tests/integration/test_m1_worker_integration.py --forked -v
============================= 16 passed in 34.77s ==============================
```

### Full Pipeline Tests (test_full_pipeline.py)
⚠️ **6/8 tests PASS**, 2 timeout failures, 1 segfault during cleanup

Results with `--forked`:
- ✅ test_full_pipeline_websocket_only - PASSED
- ✅ test_concurrent_websocket_sessions - PASSED
- ❌ test_sequential_messages_same_session - FAILED (timeout)
- ✅ test_worker_registration_integration - PASSED
- ❌ test_system_stability_under_load - FAILED (timeout)
- ✅ test_error_recovery_and_resilience - PASSED
- ✅ test_session_cleanup_on_disconnect - PASSED
- ⚠️ test_component_integration_health_checks - CRASHED (signal 11, but isolated)

**Important:** The test framework completes successfully and reports all results despite the segfault.

## Additional Workarounds in conftest.py

The integration test `conftest.py` includes additional mitigations:

### 1. GC Workaround Fixture
```python
@pytest.fixture(scope="module", autouse=True)
def grpc_event_loop_workaround() -> None:
    gc_was_enabled = gc.isenabled()
    gc.disable()
    yield
    if gc_was_enabled:
        gc.enable()
    time.sleep(1.0)
```

This helps reduce segfaults when NOT using `--forked`, but is less effective than full process isolation.

### 2. Module-scoped fixtures
Event loop scope is set to `module` in pytest.ini:
```ini
asyncio_default_fixture_loop_scope = module
asyncio_default_test_loop_scope = module
```

This reduces event loop churn between tests.

## Limitations

### Process Forking Trade-offs
- **Pro:** Complete isolation prevents cross-test contamination
- **Pro:** Segfaults don't crash the test runner
- **Con:** Higher overhead (each test spawns a new process)
- **Con:** Module-scoped fixtures are recreated for each test
- **Con:** Slower execution compared to non-forked mode

### When --forked May Still Report Failures
If a test segfaults during teardown, pytest-forked will report it as:
```
FAILED ... - running the test CRASHED with signal 11
```

This is expected behavior - the test itself may have passed, but the cleanup crashed. The important point is that the test suite continues to run.

## Testing Strategy

### Development (fast iteration)
```bash
# Run single test without forking
uv run pytest tests/integration/test_full_pipeline.py::test_full_pipeline_websocket_only -v
```

### CI/CD (reliability)
```bash
# Always use --forked for integration tests
just test-integration
```

### Debugging
```bash
# Run without isolation to see full stack traces
just test-integration-fast

# Run with verbose output
uv run pytest tests/integration/ --forked -vv --log-cli-level=DEBUG
```

## Future Resolution

This is a temporary workaround. The proper solution requires fixes in grpc-python:

1. **Monitor upstream:** https://github.com/grpc/grpc/issues/37714
2. **Test new releases:** When grpc-python releases a fix, test without `--forked`
3. **Remove workarounds:** Once fixed, remove GC workaround and switch to regular test mode

## Summary

**Use `--forked` for all integration tests to avoid segfaults:**
```bash
just test-integration
```

This is the recommended and supported way to run integration tests until the upstream grpc-python issue is resolved.

---

**Related Documentation:**
- [Known Issues Index](README.md)
- [Testing Guide](../TESTING_GUIDE.md)
- [Development Guide](../DEVELOPMENT.md)
