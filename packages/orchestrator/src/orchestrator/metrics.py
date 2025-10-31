"""Prometheus-compatible metrics for TTS pipeline observability.

This module provides comprehensive metrics collection for monitoring:
- Synthesis performance (latency, throughput, queue depth)
- Worker utilization (active/idle time, rotation patterns)
- Parallel synthesis effectiveness (worker selection, load distribution)
- Session health (duration, message count, barge-in events)
- SLA compliance (p95 latency against targets)

Metrics are collected in-memory with minimal overhead (<1% latency impact)
and exposed via /metrics endpoint in Prometheus exposition format.

Architecture:
    Application → MetricsCollector → PrometheusExporter → /metrics endpoint
                       ↓
                  In-memory storage (thread-safe)
                       ↓
                  Aggregation (p50/p95/p99)

Performance targets:
- Collection overhead: <1ms per operation
- Memory overhead: <10MB for 10k samples
- Export latency: <100ms for full scrape
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric types following Prometheus conventions."""

    COUNTER = "counter"  # Monotonically increasing (e.g., total_requests)
    GAUGE = "gauge"  # Can go up or down (e.g., active_workers)
    HISTOGRAM = "histogram"  # Distribution (e.g., latency_seconds)
    SUMMARY = "summary"  # Similar to histogram but with quantiles


@dataclass
class HistogramBucket:
    """Histogram bucket for latency distributions."""

    le: float  # Upper bound (less-than-or-equal)
    count: int = 0  # Number of observations <= le


@dataclass
class Histogram:
    """Histogram metric for tracking distributions.

    Automatically maintains buckets for percentile calculation (p50, p95, p99).
    Uses fixed bucket boundaries for consistent memory footprint.
    """

    name: str
    help: str
    labels: dict[str, str] = field(default_factory=dict)

    # Fixed bucket boundaries (in base units, e.g., seconds)
    # Covers 1ms to 10s range for TTS latency
    buckets: list[HistogramBucket] = field(
        default_factory=lambda: [
            HistogramBucket(le=0.001),  # 1ms
            HistogramBucket(le=0.005),  # 5ms
            HistogramBucket(le=0.010),  # 10ms
            HistogramBucket(le=0.020),  # 20ms
            HistogramBucket(le=0.050),  # 50ms
            HistogramBucket(le=0.100),  # 100ms
            HistogramBucket(le=0.200),  # 200ms
            HistogramBucket(le=0.300),  # 300ms (FAL SLA)
            HistogramBucket(le=0.500),  # 500ms
            HistogramBucket(le=1.000),  # 1s
            HistogramBucket(le=2.000),  # 2s
            HistogramBucket(le=5.000),  # 5s
            HistogramBucket(le=10.000),  # 10s
            HistogramBucket(le=float("inf")),  # +Inf
        ]
    )

    sum: float = 0.0  # Sum of all observed values
    count: int = 0  # Total number of observations

    def observe(self, value: float) -> None:
        """Record an observation.

        Args:
            value: Observed value (in base units, e.g., seconds)
        """
        self.sum += value
        self.count += 1

        # Update bucket counts
        for bucket in self.buckets:
            if value <= bucket.le:
                bucket.count += 1

    def quantile(self, q: float) -> float | None:
        """Calculate quantile (e.g., 0.95 for p95).

        Uses bucket interpolation for approximate quantile.
        Note: bucket.count values are CUMULATIVE (not per-bucket).

        Args:
            q: Quantile to calculate (0.0 to 1.0)

        Returns:
            Approximate quantile value, or None if no data
        """
        if self.count == 0:
            return None

        target_rank = int(q * self.count)

        # Find bucket containing target rank (bucket counts are cumulative)
        prev_count = 0
        for i, bucket in enumerate(self.buckets):
            if bucket.count >= target_rank:
                # Found target bucket
                if i == 0:
                    # First bucket, return midpoint
                    return bucket.le / 2.0
                else:
                    # Interpolate within this bucket
                    prev_bucket = self.buckets[i - 1]
                    bucket_count = bucket.count - prev_count  # Samples in this bucket

                    if bucket_count == 0:
                        return bucket.le

                    # Linear interpolation within bucket
                    rank_in_bucket = target_rank - prev_count
                    bucket_width = bucket.le - prev_bucket.le
                    return prev_bucket.le + (rank_in_bucket / bucket_count) * bucket_width

            prev_count = bucket.count

        # Target rank beyond all buckets, return max
        return self.buckets[-1].le


@dataclass
class Counter:
    """Counter metric (monotonically increasing)."""

    name: str
    help: str
    labels: dict[str, str] = field(default_factory=dict)
    value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        """Increment counter.

        Args:
            amount: Amount to increment by (default: 1.0)
        """
        self.value += amount


@dataclass
class Gauge:
    """Gauge metric (can go up or down)."""

    name: str
    help: str
    labels: dict[str, str] = field(default_factory=dict)
    value: float = 0.0

    def set(self, value: float) -> None:
        """Set gauge value.

        Args:
            value: New gauge value
        """
        self.value = value

    def inc(self, amount: float = 1.0) -> None:
        """Increment gauge.

        Args:
            amount: Amount to increment by (default: 1.0)
        """
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement gauge.

        Args:
            amount: Amount to decrement by (default: 1.0)
        """
        self.value -= amount


class MetricsCollector:
    """Thread-safe metrics collector with Prometheus-compatible output.

    Collects application metrics and provides export in Prometheus
    exposition format for scraping by monitoring systems.

    Thread-safety: All public methods are thread-safe via mutex.
    """

    _instance: "MetricsCollector | None" = None

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._lock = threading.RLock()

        # Metrics storage (keyed by metric name)
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}

        # Timing helpers for measuring durations
        self._timing_contexts: dict[str, float] = {}

        # Initialize core metrics
        self._init_synthesis_metrics()
        self._init_worker_metrics()
        self._init_session_metrics()
        self._init_sla_metrics()

        logger.info("MetricsCollector initialized")

    def _init_synthesis_metrics(self) -> None:
        """Initialize synthesis performance metrics."""
        # Latency histogram
        self._histograms["synthesis_latency_seconds"] = Histogram(
            name="synthesis_latency_seconds",
            help="TTS synthesis latency in seconds (text received to audio ready)",
        )

        # Throughput counter
        self._counters["synthesis_total"] = Counter(
            name="synthesis_total",
            help="Total number of synthesis requests completed",
        )

        # Errors counter
        self._counters["synthesis_errors_total"] = Counter(
            name="synthesis_errors_total",
            help="Total number of synthesis errors",
        )

        # Queue depth gauge
        self._gauges["synthesis_queue_depth"] = Gauge(
            name="synthesis_queue_depth",
            help="Current number of sentences queued for synthesis",
        )

        # Active synthesis gauge
        self._gauges["synthesis_active"] = Gauge(
            name="synthesis_active",
            help="Number of sentences currently being synthesized",
        )

    def _init_worker_metrics(self) -> None:
        """Initialize worker pool metrics."""
        # Worker utilization gauge (percentage)
        self._gauges["worker_utilization_percent"] = Gauge(
            name="worker_utilization_percent",
            help="Worker pool utilization percentage (0-100)",
        )

        # Active workers gauge
        self._gauges["workers_active"] = Gauge(
            name="workers_active",
            help="Number of workers currently synthesizing",
        )

        # Idle workers gauge
        self._gauges["workers_idle"] = Gauge(
            name="workers_idle",
            help="Number of workers idle (waiting for work)",
        )

        # Worker tasks completed counter
        self._counters["worker_tasks_total"] = Counter(
            name="worker_tasks_total",
            help="Total number of tasks completed by worker pool",
        )

        # Worker errors counter
        self._counters["worker_errors_total"] = Counter(
            name="worker_errors_total",
            help="Total number of worker errors",
        )

    def _init_session_metrics(self) -> None:
        """Initialize session lifecycle metrics."""
        # Session duration histogram
        self._histograms["session_duration_seconds"] = Histogram(
            name="session_duration_seconds",
            help="Session duration in seconds",
        )

        # Active sessions gauge
        self._gauges["sessions_active"] = Gauge(
            name="sessions_active",
            help="Number of active sessions",
        )

        # Message count histogram
        self._histograms["session_messages"] = Histogram(
            name="session_messages",
            help="Number of messages per session",
        )

        # Barge-in count histogram
        self._histograms["session_barge_ins"] = Histogram(
            name="session_barge_ins",
            help="Number of barge-in events per session",
        )

        # Barge-in latency histogram
        self._histograms["barge_in_latency_seconds"] = Histogram(
            name="barge_in_latency_seconds",
            help="Barge-in latency in seconds (speech detection to PAUSE)",
        )

    def _init_sla_metrics(self) -> None:
        """Initialize SLA compliance metrics."""
        # SLA violations counter
        self._counters["sla_violations_total"] = Counter(
            name="sla_violations_total",
            help="Total number of SLA violations",
        )

        # SLA compliance gauge (percentage)
        self._gauges["sla_compliance_percent"] = Gauge(
            name="sla_compliance_percent",
            help="SLA compliance percentage over last 1000 requests",
        )

    # === Synthesis metrics ===

    def record_synthesis_start(self, request_id: str) -> None:
        """Record synthesis start for latency tracking.

        Args:
            request_id: Unique identifier for this synthesis request
        """
        with self._lock:
            self._timing_contexts[f"synthesis_{request_id}"] = time.monotonic()
            self._gauges["synthesis_active"].inc()

    def record_synthesis_complete(self, request_id: str, error: bool = False) -> None:
        """Record synthesis completion and calculate latency.

        Args:
            request_id: Unique identifier for this synthesis request
            error: Whether synthesis failed with error
        """
        with self._lock:
            # Calculate latency
            start_time = self._timing_contexts.pop(f"synthesis_{request_id}", None)
            if start_time is not None:
                latency = time.monotonic() - start_time
                self._histograms["synthesis_latency_seconds"].observe(latency)

            # Update counters
            if error:
                self._counters["synthesis_errors_total"].inc()
            else:
                self._counters["synthesis_total"].inc()

            self._gauges["synthesis_active"].dec()

    def set_synthesis_queue_depth(self, depth: int) -> None:
        """Update synthesis queue depth.

        Args:
            depth: Current queue depth
        """
        with self._lock:
            self._gauges["synthesis_queue_depth"].set(float(depth))

    # === Worker metrics ===

    def update_worker_stats(
        self, active_workers: int, idle_workers: int, total_workers: int
    ) -> None:
        """Update worker pool statistics.

        Args:
            active_workers: Number of workers currently processing
            idle_workers: Number of workers idle
            total_workers: Total number of workers in pool
        """
        with self._lock:
            self._gauges["workers_active"].set(float(active_workers))
            self._gauges["workers_idle"].set(float(idle_workers))

            # Calculate utilization percentage
            if total_workers > 0:
                utilization = (active_workers / total_workers) * 100.0
                self._gauges["worker_utilization_percent"].set(utilization)

    def record_worker_task_complete(self, error: bool = False) -> None:
        """Record worker task completion.

        Args:
            error: Whether task failed with error
        """
        with self._lock:
            if error:
                self._counters["worker_errors_total"].inc()
            else:
                self._counters["worker_tasks_total"].inc()

    # === Session metrics ===

    def record_session_start(self) -> None:
        """Record session start."""
        with self._lock:
            self._gauges["sessions_active"].inc()

    def record_session_end(
        self, duration_seconds: float, message_count: int, barge_in_count: int
    ) -> None:
        """Record session end with statistics.

        Args:
            duration_seconds: Total session duration
            message_count: Number of messages exchanged
            barge_in_count: Number of barge-in events
        """
        with self._lock:
            self._gauges["sessions_active"].dec()
            self._histograms["session_duration_seconds"].observe(duration_seconds)
            self._histograms["session_messages"].observe(float(message_count))
            self._histograms["session_barge_ins"].observe(float(barge_in_count))

    def record_barge_in(self, latency_seconds: float) -> None:
        """Record barge-in event with latency.

        Args:
            latency_seconds: Latency from speech detection to PAUSE completion
        """
        with self._lock:
            self._histograms["barge_in_latency_seconds"].observe(latency_seconds)

            # Check SLA violation (target: <50ms = 0.050s)
            if latency_seconds > 0.050:
                self._counters["sla_violations_total"].inc()

    # === SLA tracking ===

    def check_synthesis_sla(self, latency_seconds: float, sla_seconds: float = 0.300) -> None:
        """Check synthesis latency against SLA and record violation.

        Args:
            latency_seconds: Measured latency
            sla_seconds: SLA threshold (default: 300ms for FAL)
        """
        with self._lock:
            if latency_seconds > sla_seconds:
                self._counters["sla_violations_total"].inc()

    # === Export ===

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus exposition format.

        Returns:
            Metrics in Prometheus text format for scraping

        Format:
            # HELP metric_name Description
            # TYPE metric_name type
            metric_name{label="value"} value timestamp
        """
        with self._lock:
            lines: list[str] = []

            # Export counters
            for counter in self._counters.values():
                lines.append(f"# HELP {counter.name} {counter.help}")
                lines.append(f"# TYPE {counter.name} counter")
                labels_str = self._format_labels(counter.labels)
                lines.append(f"{counter.name}{labels_str} {counter.value}")

            # Export gauges
            for gauge in self._gauges.values():
                lines.append(f"# HELP {gauge.name} {gauge.help}")
                lines.append(f"# TYPE {gauge.name} gauge")
                labels_str = self._format_labels(gauge.labels)
                lines.append(f"{gauge.name}{labels_str} {gauge.value}")

            # Export histograms
            for histogram in self._histograms.values():
                lines.append(f"# HELP {histogram.name} {histogram.help}")
                lines.append(f"# TYPE {histogram.name} histogram")

                labels_str = self._format_labels(histogram.labels)

                # Export buckets
                for bucket in histogram.buckets:
                    bucket_labels = {**histogram.labels, "le": str(bucket.le)}
                    bucket_labels_str = self._format_labels(bucket_labels)
                    lines.append(f"{histogram.name}_bucket{bucket_labels_str} {bucket.count}")

                # Export sum and count
                lines.append(f"{histogram.name}_sum{labels_str} {histogram.sum}")
                lines.append(f"{histogram.name}_count{labels_str} {histogram.count}")

            return "\n".join(lines) + "\n"

    def _format_labels(self, labels: dict[str, str]) -> str:
        """Format labels for Prometheus output.

        Args:
            labels: Label dictionary

        Returns:
            Formatted label string (e.g., '{label1="value1",label2="value2"}')
        """
        if not labels:
            return ""

        label_pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(label_pairs) + "}"

    # === Summary statistics ===

    def get_summary(self) -> dict[str, float | None]:
        """Get summary statistics for monitoring dashboard.

        Returns:
            Dictionary with key metrics and percentiles
        """
        with self._lock:
            synthesis_hist = self._histograms["synthesis_latency_seconds"]
            barge_in_hist = self._histograms["barge_in_latency_seconds"]

            # Calculate percentiles (quantile returns float | None)
            p50 = synthesis_hist.quantile(0.50)
            p95 = synthesis_hist.quantile(0.95)
            p99 = synthesis_hist.quantile(0.99)
            barge_in_p95 = barge_in_hist.quantile(0.95)

            return {
                # Synthesis metrics
                "synthesis_total": self._counters["synthesis_total"].value,
                "synthesis_errors": self._counters["synthesis_errors_total"].value,
                "synthesis_active": self._gauges["synthesis_active"].value,
                "synthesis_queue_depth": self._gauges["synthesis_queue_depth"].value,
                "synthesis_latency_p50_ms": p50 * 1000 if p50 is not None else None,
                "synthesis_latency_p95_ms": p95 * 1000 if p95 is not None else None,
                "synthesis_latency_p99_ms": p99 * 1000 if p99 is not None else None,
                # Worker metrics
                "workers_active": self._gauges["workers_active"].value,
                "workers_idle": self._gauges["workers_idle"].value,
                "worker_utilization_percent": self._gauges["worker_utilization_percent"].value,
                "worker_tasks_total": self._counters["worker_tasks_total"].value,
                "worker_errors": self._counters["worker_errors_total"].value,
                # Session metrics
                "sessions_active": self._gauges["sessions_active"].value,
                "session_barge_in_p95_ms": (
                    barge_in_p95 * 1000 if barge_in_p95 is not None else None
                ),
                # SLA metrics
                "sla_violations": self._counters["sla_violations_total"].value,
            }


# Global metrics collector singleton
_metrics_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector singleton.

    Returns:
        Global MetricsCollector instance

    Thread-safety: Safe for concurrent access.
    """
    global _metrics_collector

    if _metrics_collector is None:
        with _collector_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector()

    return _metrics_collector
