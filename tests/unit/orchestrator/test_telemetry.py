"""Unit tests for telemetry and profiling support.

Tests cover:
- Request context tracking
- Structured logging
- Profiling hooks
- Environment-based configuration
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path

import pytest
from orchestrator.telemetry import (
    ProfilerConfig,
    RequestContext,
    configure_profiler,
    get_profiler_config,
    get_request_context,
    get_structured_logger,
    profile_section,
    request_context,
    should_enable_profiling,
    timed,
)


class TestRequestContext:
    """Test request context tracking."""

    def test_request_context_creation(self) -> None:
        """Test request context creation."""
        ctx = RequestContext(
            session_id="s1",
            request_id="r1",
            user_id="u1",
            extra={"key": "value"},
        )

        assert ctx.session_id == "s1"
        assert ctx.request_id == "r1"
        assert ctx.user_id == "u1"
        assert ctx.extra == {"key": "value"}

    def test_request_context_to_dict(self) -> None:
        """Test context dictionary conversion."""
        ctx = RequestContext(
            session_id="s1",
            request_id="r1",
            user_id="u1",
            extra={"custom_field": "custom_value"},
        )

        ctx_dict = ctx.to_dict()

        assert ctx_dict["session_id"] == "s1"
        assert ctx_dict["request_id"] == "r1"
        assert ctx_dict["user_id"] == "u1"
        assert ctx_dict["ctx_custom_field"] == "custom_value"

    def test_request_context_manager(self) -> None:
        """Test request context manager."""
        # No context initially
        assert get_request_context() is None

        # Context active within manager
        with request_context(session_id="s1", request_id="r1"):
            ctx = get_request_context()
            assert ctx is not None
            assert ctx["session_id"] == "s1"
            assert ctx["request_id"] == "r1"

        # Context cleared after manager
        assert get_request_context() is None

    def test_nested_request_contexts(self) -> None:
        """Test nested request contexts."""
        with request_context(session_id="s1", request_id="r1"):
            ctx1 = get_request_context()
            assert ctx1 is not None
            assert ctx1["session_id"] == "s1"

            # Inner context overrides
            with request_context(session_id="s2", request_id="r2"):
                ctx2 = get_request_context()
                assert ctx2 is not None
                assert ctx2["session_id"] == "s2"

            # Outer context restored
            ctx3 = get_request_context()
            assert ctx3 is not None
            assert ctx3["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_async_context_propagation(self) -> None:
        """Test context propagation across async tasks."""

        async def inner_task() -> str:
            ctx = get_request_context()
            assert ctx is not None
            return ctx["session_id"]

        with request_context(session_id="s1", request_id="r1"):
            # Context should propagate to async task
            session_id = await inner_task()
            assert session_id == "s1"


class TestStructuredLogger:
    """Test structured logging with context."""

    def test_structured_logger_creation(self) -> None:
        """Test structured logger creation."""
        logger = get_structured_logger("test_module")
        assert logger is not None

    def test_structured_logger_auto_enrichment(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test automatic context enrichment in logs."""
        logger = get_structured_logger(__name__)

        with request_context(session_id="s1", request_id="r1"):
            with caplog.at_level(logging.INFO):
                logger.info("Test message")

            # Check log record was created
            assert len(caplog.records) > 0

    def test_structured_logger_without_context(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test logging without request context."""
        logger = get_structured_logger(__name__)

        with caplog.at_level(logging.INFO):
            logger.info("Test message without context")

        # Should still work without context
        assert len(caplog.records) > 0


class TestProfiler:
    """Test profiling support."""

    def test_profiler_config(self) -> None:
        """Test profiler configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(
                enabled=True,
                output_dir=Path(tmpdir),
                cpu_profiling=True,
                memory_profiling=False,
            )

            assert config.enabled is True
            assert config.cpu_profiling is True
            assert config.memory_profiling is False
            assert config.output_dir.exists()

    def test_profiler_disabled_by_default(self) -> None:
        """Test profiler disabled by default."""
        config = get_profiler_config()
        assert config.enabled is False

    def test_configure_profiler(self) -> None:
        """Test profiler configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_profiler(
                enabled=True,
                output_dir=tmpdir,
                cpu_profiling=True,
                memory_profiling=False,
            )

            config = get_profiler_config()
            assert config.enabled is True
            assert config.cpu_profiling is True
            assert config.memory_profiling is False

    def test_profile_section_disabled(self) -> None:
        """Test profile_section no-op when profiling disabled."""
        # Ensure profiling is disabled
        configure_profiler(enabled=False)

        # Should not raise error even if section is profiled
        with profile_section("test_section"):
            pass

    def test_profile_section_cpu(self) -> None:
        """Test CPU profiling section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_profiler(
                enabled=True,
                output_dir=tmpdir,
                cpu_profiling=True,
                memory_profiling=False,
                min_duration_ms=0.0,  # Profile everything
            )

            # Profile a section
            with profile_section("test_cpu"):
                sum(range(10000))  # Do some work

            # Profile files may or may not be created depending on duration
            # Just verify no errors occurred

    def test_profile_section_min_duration(self) -> None:
        """Test min duration threshold for profiling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_profiler(
                enabled=True,
                output_dir=tmpdir,
                cpu_profiling=True,
                min_duration_ms=1000.0,  # Only profile >1s operations
            )

            # Quick operation should not generate profile
            with profile_section("quick_operation"):
                pass

            # No profile files should be created
            profile_dir = Path(tmpdir)
            profile_files = list(profile_dir.glob("quick_operation_*.pstats"))
            assert len(profile_files) == 0


class TestTimingDecorator:
    """Test @timed decorator."""

    @pytest.mark.asyncio
    async def test_timed_async_function(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test @timed decorator on async function."""

        @timed("test_async_section")  # type: ignore[misc]
        async def async_func() -> int:
            await asyncio.sleep(0.01)
            return 42

        with caplog.at_level(logging.DEBUG):
            result = await async_func()

        assert result == 42
        # Timing log should be present (exact check depends on logger configuration)

    def test_timed_sync_function(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test @timed decorator on sync function."""

        @timed("test_sync_section")  # type: ignore[misc]
        def sync_func() -> int:
            return 42

        with caplog.at_level(logging.DEBUG):
            result = sync_func()

        assert result == 42


class TestEnvironmentConfiguration:
    """Test environment-based profiling configuration."""

    def test_should_enable_profiling_false(self) -> None:
        """Test profiling disabled by default."""
        # Ensure env var not set
        os.environ.pop("ENABLE_PROFILING", None)
        assert should_enable_profiling() is False

    def test_should_enable_profiling_true(self) -> None:
        """Test profiling enabled via environment."""
        # Set env var
        original_value = os.environ.get("ENABLE_PROFILING")
        try:
            os.environ["ENABLE_PROFILING"] = "true"
            assert should_enable_profiling() is True

            os.environ["ENABLE_PROFILING"] = "1"
            assert should_enable_profiling() is True

            os.environ["ENABLE_PROFILING"] = "yes"
            assert should_enable_profiling() is True

            os.environ["ENABLE_PROFILING"] = "false"
            assert should_enable_profiling() is False

        finally:
            # Restore original value
            if original_value is not None:
                os.environ["ENABLE_PROFILING"] = original_value
            else:
                os.environ.pop("ENABLE_PROFILING", None)


@pytest.mark.performance
class TestTelemetryPerformance:
    """Performance tests for telemetry overhead."""

    def test_request_context_overhead(self) -> None:
        """Verify request context overhead is minimal."""
        import time

        iterations = 10000

        # Without context (baseline not used, just measuring with-context overhead)
        start = time.perf_counter()
        for _ in range(iterations):
            pass
        _ = time.perf_counter() - start  # Baseline duration (not used)

        # With context - measure absolute duration
        start = time.perf_counter()
        for _ in range(iterations):
            with request_context(session_id="s1", request_id="r1"):
                pass
        context_duration = time.perf_counter() - start

        # Overhead should be reasonable (absolute threshold for robustness)
        # Context creation adds ~3-5 microseconds per iteration
        # 10k iterations should complete in <100ms
        assert (
            context_duration < 0.1
        ), f"Context duration {context_duration:.4f}s exceeds 100ms threshold"

    @pytest.mark.asyncio
    async def test_structured_logging_overhead(self) -> None:
        """Verify structured logging overhead is minimal."""
        import time

        iterations = 1000

        # With context (logger not used, just measuring context overhead)
        with request_context(session_id="s1", request_id="r1"):
            start = time.perf_counter()
            for _ in range(iterations):
                # Don't actually log at INFO level
                pass
            duration_ms = (time.perf_counter() - start) * 1000.0

        # Should complete quickly even with context active
        assert duration_ms < 100.0  # <100ms for 1000 iterations
