"""Unit tests for RMSBuffer circular buffer (M10 Polish Task 4).

Tests the RMSBuffer circular buffer used by the adaptive noise gate for
percentile-based noise floor estimation. Critical for validating noise gate
functionality and performance.

Coverage:
- Initialization and properties
- Push operations with wraparound
- Percentile calculations for noise floor estimation
- Full/empty state management
- Clear operations
- Performance validation (<1ms target)
"""

import time

import numpy as np

from src.orchestrator.audio.buffer import RMSBuffer


class TestRMSBufferInit:
    """Test RMSBuffer initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization with size parameter."""
        buffer = RMSBuffer(size=100)
        assert buffer.size == 100
        assert buffer.count == 0
        assert not buffer.is_full()
        assert buffer.utilization == 0.0

    def test_initialization_with_custom_size(self) -> None:
        """Test initialization with custom buffer size."""
        buffer = RMSBuffer(size=50)
        assert buffer.size == 50
        assert buffer.count == 0

    def test_initialization_with_minimum_size(self) -> None:
        """Test initialization with minimum size (edge case)."""
        buffer = RMSBuffer(size=1)
        assert buffer.size == 1
        assert buffer.count == 0

    def test_initialization_with_large_size(self) -> None:
        """Test initialization with large buffer size."""
        buffer = RMSBuffer(size=1000)
        assert buffer.size == 1000
        assert buffer.count == 0


class TestRMSBufferPush:
    """Test RMSBuffer push operations."""

    def test_push_single_value(self) -> None:
        """Test pushing a single value."""
        buffer = RMSBuffer(size=10)
        buffer.push(100.0)
        assert buffer.count == 1
        assert not buffer.is_full()
        assert buffer.utilization == 0.1

    def test_push_multiple_values(self) -> None:
        """Test pushing multiple values."""
        buffer = RMSBuffer(size=10)
        for i in range(5):
            buffer.push(float(i * 100))
        assert buffer.count == 5
        assert not buffer.is_full()
        assert buffer.utilization == 0.5

    def test_push_until_full(self) -> None:
        """Test pushing values until buffer is full."""
        buffer = RMSBuffer(size=10)
        for i in range(10):
            buffer.push(float(i * 100))
        assert buffer.count == 10
        assert buffer.is_full()
        assert buffer.utilization == 1.0

    def test_push_with_wraparound(self) -> None:
        """Test push with circular buffer wraparound."""
        buffer = RMSBuffer(size=5)
        # Fill buffer
        for i in range(5):
            buffer.push(float(i * 100))
        assert buffer.is_full()

        # Push more values (should wrap around)
        for i in range(5, 10):
            buffer.push(float(i * 100))

        # Buffer should still be full (count capped at size)
        assert buffer.is_full()
        assert buffer.count == 5

    def test_push_zero_values(self) -> None:
        """Test pushing zero values (silence)."""
        buffer = RMSBuffer(size=10)
        for _ in range(5):
            buffer.push(0.0)
        assert buffer.count == 5

    def test_push_negative_values_not_allowed(self) -> None:
        """Test that negative RMS values are handled (edge case)."""
        buffer = RMSBuffer(size=10)
        # RMS values should be non-negative, but buffer doesn't enforce
        buffer.push(-100.0)  # Should not raise error
        assert buffer.count == 1


class TestRMSBufferPercentile:
    """Test RMSBuffer percentile calculations."""

    def test_get_percentile_empty_buffer(self) -> None:
        """Test percentile calculation on empty buffer."""
        buffer = RMSBuffer(size=100)
        # Should return 0.0 for empty buffer
        percentile = buffer.get_percentile(0.5)
        assert percentile == 0.0

    def test_get_percentile_single_value(self) -> None:
        """Test percentile calculation with single value."""
        buffer = RMSBuffer(size=100)
        buffer.push(100.0)
        percentile = buffer.get_percentile(0.5)
        assert percentile == 100.0

    def test_get_percentile_25th(self) -> None:
        """Test 25th percentile calculation."""
        buffer = RMSBuffer(size=100)
        for i in range(100):
            buffer.push(float(i))  # Values: 0-99

        # 25th percentile should be around 25
        percentile = buffer.get_percentile(0.25)
        assert 20.0 <= percentile <= 30.0

    def test_get_percentile_50th_median(self) -> None:
        """Test 50th percentile (median) calculation."""
        buffer = RMSBuffer(size=100)
        for i in range(100):
            buffer.push(float(i))  # Values: 0-99

        # 50th percentile (median) should be around 50
        percentile = buffer.get_percentile(0.5)
        assert 45.0 <= percentile <= 55.0

    def test_get_percentile_75th(self) -> None:
        """Test 75th percentile calculation."""
        buffer = RMSBuffer(size=100)
        for i in range(100):
            buffer.push(float(i))  # Values: 0-99

        # 75th percentile should be around 75
        percentile = buffer.get_percentile(0.75)
        assert 70.0 <= percentile <= 80.0

    def test_get_percentile_noise_floor(self) -> None:
        """Test percentile-based noise floor estimation (realistic scenario)."""
        buffer = RMSBuffer(size=100)

        # Simulate realistic audio: 70% noise (~100 RMS), 30% speech (~1000 RMS)
        for _ in range(70):
            buffer.push(100.0 + np.random.uniform(-20, 20))
        for _ in range(30):
            buffer.push(1000.0 + np.random.uniform(-100, 100))

        # 25th percentile should capture noise floor (~100)
        noise_floor = buffer.get_percentile(0.25)
        assert 80.0 <= noise_floor <= 150.0, f"Noise floor {noise_floor} outside expected range"

    def test_get_percentile_all_same_values(self) -> None:
        """Test percentile with all identical values."""
        buffer = RMSBuffer(size=100)
        for _ in range(100):
            buffer.push(500.0)

        # All percentiles should return same value
        assert buffer.get_percentile(0.25) == 500.0
        assert buffer.get_percentile(0.5) == 500.0
        assert buffer.get_percentile(0.75) == 500.0


class TestRMSBufferProperties:
    """Test RMSBuffer properties."""

    def test_is_full_empty_buffer(self) -> None:
        """Test is_full() on empty buffer."""
        buffer = RMSBuffer(size=10)
        assert not buffer.is_full()

    def test_is_full_partially_filled(self) -> None:
        """Test is_full() on partially filled buffer."""
        buffer = RMSBuffer(size=10)
        for i in range(5):
            buffer.push(float(i))
        assert not buffer.is_full()

    def test_is_full_completely_filled(self) -> None:
        """Test is_full() on completely filled buffer."""
        buffer = RMSBuffer(size=10)
        for i in range(10):
            buffer.push(float(i))
        assert buffer.is_full()

    def test_utilization_empty(self) -> None:
        """Test utilization on empty buffer."""
        buffer = RMSBuffer(size=100)
        assert buffer.utilization == 0.0

    def test_utilization_half_full(self) -> None:
        """Test utilization on half-full buffer."""
        buffer = RMSBuffer(size=100)
        for i in range(50):
            buffer.push(float(i))
        assert buffer.utilization == 0.5

    def test_utilization_full(self) -> None:
        """Test utilization on full buffer."""
        buffer = RMSBuffer(size=100)
        for i in range(100):
            buffer.push(float(i))
        assert buffer.utilization == 1.0


class TestRMSBufferClear:
    """Test RMSBuffer clear operations."""

    def test_clear_empty_buffer(self) -> None:
        """Test clearing an empty buffer."""
        buffer = RMSBuffer(size=100)
        buffer.clear()
        assert buffer.count == 0
        assert not buffer.is_full()

    def test_clear_filled_buffer(self) -> None:
        """Test clearing a filled buffer."""
        buffer = RMSBuffer(size=100)
        for i in range(100):
            buffer.push(float(i))
        assert buffer.is_full()

        buffer.clear()
        assert buffer.count == 0
        assert not buffer.is_full()
        assert buffer.utilization == 0.0

    def test_clear_and_refill(self) -> None:
        """Test clearing and refilling buffer."""
        buffer = RMSBuffer(size=10)

        # Fill
        for i in range(10):
            buffer.push(float(i * 100))
        assert buffer.is_full()

        # Clear
        buffer.clear()
        assert buffer.count == 0

        # Refill
        for i in range(5):
            buffer.push(float(i * 200))
        assert buffer.count == 5
        assert not buffer.is_full()


class TestRMSBufferRepr:
    """Test RMSBuffer repr."""

    def test_repr_empty(self) -> None:
        """Test repr on empty buffer."""
        buffer = RMSBuffer(size=100)
        repr_str = repr(buffer)
        assert "RMSBuffer" in repr_str
        assert "size=100" in repr_str
        assert "count=0" in repr_str

    def test_repr_filled(self) -> None:
        """Test repr on filled buffer."""
        buffer = RMSBuffer(size=50)
        for i in range(25):
            buffer.push(float(i))

        repr_str = repr(buffer)
        assert "RMSBuffer" in repr_str
        assert "size=50" in repr_str
        assert "count=25" in repr_str


class TestRMSBufferPerformance:
    """Test RMSBuffer performance requirements."""

    def test_push_performance(self) -> None:
        """Test push operation performance (<1ms target per frame)."""
        buffer = RMSBuffer(size=100)

        # Measure time for 1000 push operations
        start_time = time.perf_counter()
        for i in range(1000):
            buffer.push(float(i))
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Average time per push should be <1ms
        avg_time_ms = elapsed_ms / 1000
        assert avg_time_ms < 1.0, f"Push took {avg_time_ms:.3f}ms (target: <1ms)"

    def test_percentile_performance(self) -> None:
        """Test percentile calculation performance."""
        buffer = RMSBuffer(size=100)

        # Fill buffer
        for i in range(100):
            buffer.push(float(i))

        # Measure percentile calculation time
        start_time = time.perf_counter()
        for _ in range(100):
            _ = buffer.get_percentile(0.25)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Average time per percentile calculation should be reasonable
        avg_time_ms = elapsed_ms / 100
        assert avg_time_ms < 5.0, f"Percentile took {avg_time_ms:.3f}ms"

    def test_full_pipeline_performance(self) -> None:
        """Test full pipeline: push + percentile (realistic scenario)."""
        buffer = RMSBuffer(size=100)

        # Simulate 50fps audio processing for 2 seconds (100 frames)
        start_time = time.perf_counter()
        for i in range(100):
            buffer.push(float(i * 10))
            if buffer.is_full():
                _ = buffer.get_percentile(0.25)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should complete in <100ms for 100 frames
        assert elapsed_ms < 100.0, f"Pipeline took {elapsed_ms:.1f}ms (target: <100ms)"
