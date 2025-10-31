"""Unit tests for metrics collection and Prometheus export.

Tests cover:
- Metrics collection (counters, gauges, histograms)
- Prometheus format export
- Percentile calculation
- Thread safety
- Summary statistics
"""

import threading
import time

import pytest

from orchestrator.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
    get_metrics_collector,
)


class TestHistogram:
    """Test histogram metric for latency distributions."""

    def test_histogram_observe(self) -> None:
        """Test observing values in histogram."""
        hist = Histogram(name="test_latency", help="Test latency metric")

        # Observe values
        hist.observe(0.050)  # 50ms
        hist.observe(0.100)  # 100ms
        hist.observe(0.200)  # 200ms

        assert hist.count == 3
        assert abs(hist.sum - 0.350) < 0.001  # Floating point tolerance

        # Check bucket counts
        bucket_50ms = next(b for b in hist.buckets if b.le == 0.050)
        assert bucket_50ms.count == 1  # Only first value <= 50ms

        bucket_100ms = next(b for b in hist.buckets if b.le == 0.100)
        assert bucket_100ms.count == 2  # First two values <= 100ms

        bucket_200ms = next(b for b in hist.buckets if b.le == 0.200)
        assert bucket_200ms.count == 3  # All values <= 200ms

    def test_histogram_quantile(self) -> None:
        """Test quantile calculation."""
        hist = Histogram(name="test_latency", help="Test latency metric")

        # No data -> None
        assert hist.quantile(0.50) is None
        assert hist.quantile(0.95) is None

        # Add 100 samples distributed evenly
        for i in range(100):
            hist.observe(i / 1000.0)  # 0ms to 99ms

        # p50 should be around 50ms
        p50 = hist.quantile(0.50)
        assert p50 is not None
        assert 0.020 < p50 < 0.070  # 40-60ms range

        # p95 should be around 95ms
        p95 = hist.quantile(0.95)
        assert p95 is not None
        assert 0.090 < p95 < 0.100  # 90-100ms range

        # p99 should be around 99ms
        p99 = hist.quantile(0.99)
        assert p99 is not None
        assert 0.095 < p99 < 0.105  # 95-105ms range


class TestCounter:
    """Test counter metric."""

    def test_counter_increment(self) -> None:
        """Test counter increment."""
        counter = Counter(name="test_counter", help="Test counter")

        assert counter.value == 0.0

        counter.inc()
        assert counter.value == 1.0

        counter.inc(5.0)
        assert counter.value == 6.0


class TestGauge:
    """Test gauge metric."""

    def test_gauge_set(self) -> None:
        """Test gauge set value."""
        gauge = Gauge(name="test_gauge", help="Test gauge")

        assert gauge.value == 0.0

        gauge.set(42.0)
        assert gauge.value == 42.0

        gauge.set(100.0)
        assert gauge.value == 100.0

    def test_gauge_increment_decrement(self) -> None:
        """Test gauge increment and decrement."""
        gauge = Gauge(name="test_gauge", help="Test gauge")

        gauge.set(10.0)

        gauge.inc(5.0)
        assert gauge.value == 15.0

        gauge.dec(3.0)
        assert gauge.value == 12.0


class TestMetricsCollector:
    """Test metrics collector."""

    def test_synthesis_metrics(self) -> None:
        """Test synthesis metrics collection."""
        collector = MetricsCollector()

        # Record synthesis start
        collector.record_synthesis_start("req1")
        assert collector._gauges["synthesis_active"].value == 1.0

        # Simulate 100ms synthesis
        time.sleep(0.1)

        # Record completion
        collector.record_synthesis_complete("req1")
        assert collector._gauges["synthesis_active"].value == 0.0
        assert collector._counters["synthesis_total"].value == 1.0

        # Check latency recorded
        hist = collector._histograms["synthesis_latency_seconds"]
        assert hist.count == 1
        assert 0.09 < hist.sum < 0.15  # ~100ms

    def test_synthesis_error(self) -> None:
        """Test synthesis error recording."""
        collector = MetricsCollector()

        collector.record_synthesis_start("req1")
        collector.record_synthesis_complete("req1", error=True)

        assert collector._counters["synthesis_errors_total"].value == 1.0
        assert collector._counters["synthesis_total"].value == 0.0

    def test_worker_stats(self) -> None:
        """Test worker statistics update."""
        collector = MetricsCollector()

        # 2 active, 1 idle, 3 total
        collector.update_worker_stats(active_workers=2, idle_workers=1, total_workers=3)

        assert collector._gauges["workers_active"].value == 2.0
        assert collector._gauges["workers_idle"].value == 1.0

        # Utilization = 2/3 = 66.67%
        util = collector._gauges["worker_utilization_percent"].value
        assert 66.0 < util < 67.0

    def test_session_metrics(self) -> None:
        """Test session lifecycle metrics."""
        collector = MetricsCollector()

        # Start session
        collector.record_session_start()
        assert collector._gauges["sessions_active"].value == 1.0

        # End session
        collector.record_session_end(duration_seconds=120.0, message_count=5, barge_in_count=2)

        assert collector._gauges["sessions_active"].value == 0.0
        assert collector._histograms["session_duration_seconds"].count == 1
        assert collector._histograms["session_messages"].count == 1
        assert collector._histograms["session_barge_ins"].count == 1

    def test_barge_in_metrics(self) -> None:
        """Test barge-in event recording."""
        collector = MetricsCollector()

        # Record barge-in within SLA (<50ms)
        collector.record_barge_in(0.030)  # 30ms
        assert collector._counters["sla_violations_total"].value == 0.0

        # Record barge-in violating SLA (>50ms)
        collector.record_barge_in(0.080)  # 80ms
        assert collector._counters["sla_violations_total"].value == 1.0

    def test_queue_depth_tracking(self) -> None:
        """Test synthesis queue depth tracking."""
        collector = MetricsCollector()

        collector.set_synthesis_queue_depth(5)
        assert collector._gauges["synthesis_queue_depth"].value == 5.0

        collector.set_synthesis_queue_depth(10)
        assert collector._gauges["synthesis_queue_depth"].value == 10.0

    def test_prometheus_export(self) -> None:
        """Test Prometheus format export."""
        collector = MetricsCollector()

        # Add some data
        collector._counters["synthesis_total"].inc(10)
        collector._gauges["workers_active"].set(2)
        collector._histograms["synthesis_latency_seconds"].observe(0.100)

        # Export
        output = collector.export_prometheus()

        # Validate format
        assert "# HELP synthesis_total" in output
        assert "# TYPE synthesis_total counter" in output
        assert "synthesis_total 10.0" in output

        assert "# HELP workers_active" in output
        assert "# TYPE workers_active gauge" in output
        assert "workers_active 2" in output

        assert "# HELP synthesis_latency_seconds" in output
        assert "# TYPE synthesis_latency_seconds histogram" in output
        assert "synthesis_latency_seconds_bucket" in output
        assert "synthesis_latency_seconds_sum" in output
        assert "synthesis_latency_seconds_count 1" in output

    def test_summary_statistics(self) -> None:
        """Test summary statistics generation."""
        collector = MetricsCollector()

        # Add synthesis data
        for i in range(100):
            collector.record_synthesis_start(f"req{i}")
            time.sleep(0.001)  # 1ms
            collector.record_synthesis_complete(f"req{i}")

        # Get summary
        summary = collector.get_summary()

        assert summary["synthesis_total"] == 100.0
        assert summary["synthesis_errors"] == 0.0
        assert summary["synthesis_latency_p50_ms"] is not None
        assert summary["synthesis_latency_p95_ms"] is not None
        assert summary["synthesis_latency_p99_ms"] is not None

    def test_thread_safety(self) -> None:
        """Test thread-safe metrics collection."""
        collector = MetricsCollector()
        errors = []

        def worker_thread(thread_id: int) -> None:
            try:
                for i in range(100):
                    request_id = f"thread{thread_id}_req{i}"
                    collector.record_synthesis_start(request_id)
                    time.sleep(0.001)
                    collector.record_synthesis_complete(request_id)
            except Exception as e:
                errors.append(e)

        # Spawn 10 threads
        threads = [threading.Thread(target=worker_thread, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Verify no errors
        assert len(errors) == 0

        # Verify counts
        assert collector._counters["synthesis_total"].value == 1000.0

    def test_singleton_metrics_collector(self) -> None:
        """Test global singleton metrics collector."""
        # Get collector twice, should be same instance
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        assert collector1 is collector2

        # Verify it's functional (check increment works)
        before_value = collector1._counters["synthesis_total"].value
        collector1._counters["synthesis_total"].inc()
        after_value = collector2._counters["synthesis_total"].value
        assert after_value == before_value + 1.0


@pytest.mark.performance
class TestMetricsPerformance:
    """Performance tests for metrics collection overhead."""

    def test_metrics_overhead(self) -> None:
        """Verify metrics collection overhead <1ms per operation."""
        collector = MetricsCollector()

        # Warm up
        for i in range(10):
            collector.record_synthesis_start(f"warmup{i}")
            collector.record_synthesis_complete(f"warmup{i}")

        # Measure overhead
        start = time.perf_counter()
        iterations = 1000

        for i in range(iterations):
            collector.record_synthesis_start(f"perf{i}")
            collector.record_synthesis_complete(f"perf{i}")

        duration = time.perf_counter() - start
        avg_overhead_ms = (duration / iterations) * 1000.0

        # Should be <1ms per operation
        assert avg_overhead_ms < 1.0, f"Overhead {avg_overhead_ms:.3f}ms exceeds 1ms target"

    def test_export_performance(self) -> None:
        """Verify export latency <100ms."""
        collector = MetricsCollector()

        # Add substantial data
        for i in range(10000):
            collector._histograms["synthesis_latency_seconds"].observe(i / 1000000.0)

        # Measure export time
        start = time.perf_counter()
        output = collector.export_prometheus()
        duration_ms = (time.perf_counter() - start) * 1000.0

        assert duration_ms < 100.0, f"Export {duration_ms:.1f}ms exceeds 100ms target"
        assert len(output) > 0  # Ensure export succeeded
