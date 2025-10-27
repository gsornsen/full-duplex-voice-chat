"""Unit tests for WorkerSelector module.

Tests selection algorithms for load balancing.
"""

import time

import pytest

from src.orchestrator.registry import WorkerRegistration
from src.orchestrator.worker_selector import WorkerSelector


@pytest.fixture
def sample_workers():
    """Create sample worker registrations for testing."""
    current_time = time.time()

    return [
        WorkerRegistration(
            name="worker-0",
            addr="grpc://localhost:7001",
            capabilities={"streaming": True},
            resident_models=[],
            metrics={"queue_depth": 0, "rtf": 0.5},
            last_heartbeat_ts=current_time,
        ),
        WorkerRegistration(
            name="worker-1",
            addr="grpc://localhost:7002",
            capabilities={"streaming": True},
            resident_models=[],
            metrics={"queue_depth": 3, "rtf": 0.3},
            last_heartbeat_ts=current_time,
        ),
        WorkerRegistration(
            name="worker-2",
            addr="grpc://localhost:7003",
            capabilities={"streaming": True},
            resident_models=[],
            metrics={"queue_depth": 1, "rtf": 0.4},
            last_heartbeat_ts=current_time,
        ),
    ]


@pytest.fixture
def selector():
    """Create WorkerSelector instance."""
    return WorkerSelector()


# =============================================================================
# Round-Robin Selection Tests
# =============================================================================


def test_round_robin_cycles_through_workers(selector, sample_workers):
    """Test round-robin cycles through workers in order."""
    selections = []

    for _ in range(6):
        worker = selector.round_robin(sample_workers)
        selections.append(worker.name)

    # Should cycle through workers: 0, 1, 2, 0, 1, 2
    assert selections == [
        "worker-0", "worker-1", "worker-2",
        "worker-0", "worker-1", "worker-2",
    ]


def test_round_robin_single_worker(selector):
    """Test round-robin with single worker."""
    worker = WorkerRegistration(
        name="solo-worker",
        addr="grpc://localhost:7001",
        capabilities={},
        resident_models=[],
        metrics={},
        last_heartbeat_ts=time.time(),
    )

    # Should always return the same worker
    for _ in range(5):
        selected = selector.round_robin([worker])
        assert selected.name == "solo-worker"


def test_round_robin_empty_list_error(selector):
    """Test round-robin raises error with empty list."""
    with pytest.raises(ValueError, match="Cannot select from empty worker list"):
        selector.round_robin([])


# =============================================================================
# Least-Loaded Selection Tests
# =============================================================================


def test_least_loaded_selects_lowest_queue(selector, sample_workers):
    """Test least-loaded selects worker with lowest queue depth."""
    selected = selector.least_loaded(sample_workers)

    # worker-0 has queue_depth=0 (lowest)
    assert selected.name == "worker-0"
    assert selected.metrics["queue_depth"] == 0


def test_least_loaded_with_missing_metrics(selector):
    """Test least-loaded handles missing metrics."""
    workers = [
        WorkerRegistration(
            name="worker-no-metrics",
            addr="grpc://localhost:7001",
            capabilities={},
            resident_models=[],
            metrics={},  # No queue_depth metric
            last_heartbeat_ts=time.time(),
        ),
        WorkerRegistration(
            name="worker-with-metrics",
            addr="grpc://localhost:7002",
            capabilities={},
            resident_models=[],
            metrics={"queue_depth": 5},
            last_heartbeat_ts=time.time(),
        ),
    ]

    selected = selector.least_loaded(workers)

    # Should select worker with metrics (not missing)
    assert selected.name == "worker-with-metrics"


def test_least_loaded_all_equal(selector):
    """Test least-loaded when all workers have same queue depth."""
    workers = [
        WorkerRegistration(
            name=f"worker-{i}",
            addr=f"grpc://localhost:700{i}",
            capabilities={},
            resident_models=[],
            metrics={"queue_depth": 2},  # All same
            last_heartbeat_ts=time.time(),
        )
        for i in range(3)
    ]

    selected = selector.least_loaded(workers)

    # Should select first worker with min value
    assert selected.name in [w.name for w in workers]


def test_least_loaded_empty_list_error(selector):
    """Test least-loaded raises error with empty list."""
    with pytest.raises(ValueError, match="Cannot select from empty worker list"):
        selector.least_loaded([])


# =============================================================================
# Least-Latency Selection Tests
# =============================================================================


def test_least_latency_selects_lowest_rtf(selector, sample_workers):
    """Test least-latency selects worker with lowest RTF."""
    selected = selector.least_latency(sample_workers)

    # worker-1 has rtf=0.3 (lowest)
    assert selected.name == "worker-1"
    assert selected.metrics["rtf"] == 0.3


def test_least_latency_with_missing_metrics(selector):
    """Test least-latency handles missing metrics."""
    workers = [
        WorkerRegistration(
            name="worker-no-metrics",
            addr="grpc://localhost:7001",
            capabilities={},
            resident_models=[],
            metrics={},  # No rtf metric
            last_heartbeat_ts=time.time(),
        ),
        WorkerRegistration(
            name="worker-with-metrics",
            addr="grpc://localhost:7002",
            capabilities={},
            resident_models=[],
            metrics={"rtf": 0.5},
            last_heartbeat_ts=time.time(),
        ),
    ]

    selected = selector.least_latency(workers)

    # Should select worker with metrics
    assert selected.name == "worker-with-metrics"


def test_least_latency_empty_list_error(selector):
    """Test least-latency raises error with empty list."""
    with pytest.raises(ValueError, match="Cannot select from empty worker list"):
        selector.least_latency([])


# =============================================================================
# Random Selection Tests
# =============================================================================


def test_random_selection_variability(selector, sample_workers):
    """Test random selection produces variable results."""
    selections = set()

    # Make 20 selections
    for _ in range(20):
        worker = selector.random(sample_workers)
        selections.add(worker.name)

    # Should select at least 2 different workers (probabilistic)
    assert len(selections) >= 2


def test_random_selection_single_worker(selector):
    """Test random selection with single worker."""
    worker = WorkerRegistration(
        name="solo-worker",
        addr="grpc://localhost:7001",
        capabilities={},
        resident_models=[],
        metrics={},
        last_heartbeat_ts=time.time(),
    )

    # Should always return the same worker
    for _ in range(5):
        selected = selector.random([worker])
        assert selected.name == "solo-worker"


def test_random_selection_empty_list_error(selector):
    """Test random raises error with empty list."""
    with pytest.raises(ValueError, match="Cannot select from empty worker list"):
        selector.random([])


# =============================================================================
# Weighted Selection Tests
# =============================================================================


def test_weighted_selection_combines_metrics(selector, sample_workers):
    """Test weighted selection combines queue and latency metrics."""
    # Default weights: queue=0.7, latency=0.3
    selected = selector.weighted_selection(sample_workers)

    # worker-0: queue=0, rtf=0.5
    # worker-1: queue=3, rtf=0.3 (best RTF but high queue)
    # worker-2: queue=1, rtf=0.4 (balanced)

    # Normalized scores:
    # worker-0: 0.7*(0/3) + 0.3*(0.5/0.5) = 0.0 + 0.3 = 0.3
    # worker-1: 0.7*(3/3) + 0.3*(0.3/0.5) = 0.7 + 0.18 = 0.88
    # worker-2: 0.7*(1/3) + 0.3*(0.4/0.5) = 0.23 + 0.24 = 0.47

    # Should select worker-0 (lowest score)
    assert selected.name == "worker-0"


def test_weighted_selection_custom_weights(selector, sample_workers):
    """Test weighted selection with custom weights."""
    # Heavy weight on latency
    selected = selector.weighted_selection(
        sample_workers,
        queue_weight=0.2,
        latency_weight=0.8,
    )

    # Should prioritize low RTF (worker-1 has rtf=0.3)
    assert selected.name == "worker-1"


def test_weighted_selection_invalid_weights(selector, sample_workers):
    """Test weighted selection rejects invalid weights."""
    with pytest.raises(ValueError, match="Weights must sum to 1.0"):
        selector.weighted_selection(
            sample_workers,
            queue_weight=0.5,
            latency_weight=0.3,  # Doesn't sum to 1.0
        )


def test_weighted_selection_empty_list_error(selector):
    """Test weighted selection raises error with empty list."""
    with pytest.raises(ValueError, match="Cannot select from empty worker list"):
        selector.weighted_selection([])


# =============================================================================
# Performance Tests
# =============================================================================


def test_selection_performance_benchmark(selector, sample_workers):
    """Test selection algorithms meet performance targets (<100μs)."""
    import timeit

    # Test round-robin performance
    def test_round_robin():
        selector.round_robin(sample_workers)

    rr_time = timeit.timeit(test_round_robin, number=1000) / 1000
    assert rr_time < 0.0001  # <100μs per call

    # Test least-loaded performance
    def test_least_loaded():
        selector.least_loaded(sample_workers)

    ll_time = timeit.timeit(test_least_loaded, number=1000) / 1000
    assert ll_time < 0.0001  # <100μs per call

    # Test least-latency performance
    def test_least_latency():
        selector.least_latency(sample_workers)

    lat_time = timeit.timeit(test_least_latency, number=1000) / 1000
    assert lat_time < 0.0001  # <100μs per call


def test_selection_with_large_worker_pool(selector):
    """Test selection algorithms scale to large worker pools."""
    # Create 100 workers
    workers = [
        WorkerRegistration(
            name=f"worker-{i}",
            addr=f"grpc://localhost:{7000+i}",
            capabilities={},
            resident_models=[],
            metrics={"queue_depth": i, "rtf": 0.3 + (i * 0.01)},
            last_heartbeat_ts=time.time(),
        )
        for i in range(100)
    ]

    # Should select efficiently
    selected = selector.least_loaded(workers)
    assert selected.name == "worker-0"  # Lowest queue

    selected = selector.least_latency(workers)
    assert selected.name == "worker-0"  # Lowest RTF


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_all_algorithms_with_two_workers(selector):
    """Test all algorithms work with two workers."""
    workers = [
        WorkerRegistration(
            name="worker-a",
            addr="grpc://localhost:7001",
            capabilities={},
            resident_models=[],
            metrics={"queue_depth": 1, "rtf": 0.4},
            last_heartbeat_ts=time.time(),
        ),
        WorkerRegistration(
            name="worker-b",
            addr="grpc://localhost:7002",
            capabilities={},
            resident_models=[],
            metrics={"queue_depth": 2, "rtf": 0.3},
            last_heartbeat_ts=time.time(),
        ),
    ]

    # Round-robin should alternate
    assert selector.round_robin(workers).name == "worker-a"
    assert selector.round_robin(workers).name == "worker-b"

    # Least-loaded should select worker-a
    assert selector.least_loaded(workers).name == "worker-a"

    # Least-latency should select worker-b
    assert selector.least_latency(workers).name == "worker-b"

    # Random should select one of them
    assert selector.random(workers).name in ["worker-a", "worker-b"]


def test_selector_state_isolation(selector, sample_workers):
    """Test selector maintains state correctly across calls."""
    # Make several round-robin selections
    selections1 = [selector.round_robin(sample_workers).name for _ in range(3)]

    # Round-robin counter should maintain state
    assert selections1 == ["worker-0", "worker-1", "worker-2"]

    # Make more selections (should continue from counter state)
    selections2 = [selector.round_robin(sample_workers).name for _ in range(3)]
    assert selections2 == ["worker-0", "worker-1", "worker-2"]


def test_selector_thread_safety_simulation(selector, sample_workers):
    """Test selector works correctly with interleaved calls."""
    # Simulate concurrent calls by interleaving different strategies
    results = []

    for i in range(10):
        if i % 3 == 0:
            worker = selector.round_robin(sample_workers)
        elif i % 3 == 1:
            worker = selector.least_loaded(sample_workers)
        else:
            worker = selector.least_latency(sample_workers)

        results.append(worker.name)

    # Should complete without errors
    assert len(results) == 10
    assert all(name in ["worker-0", "worker-1", "worker-2"] for name in results)
